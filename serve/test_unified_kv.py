"""Native telemetry and error routing; no GPU, model or conversation database."""
import queue
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock
from serve.unified_kv import KvStatus
from serve.server import StrataEngine


class Telemetry(unittest.TestCase):
    def test_eviction_is_not_conversation_completion(self):
        status = KvStatus()
        status.consume("KVSESSION session=1 generation=3 epoch=7 state=resident tokens=1024 bytes=65536")
        recognized, invalid = status.consume("KVEVICT session=1 generation=3 epoch=8 tokens=1024 bytes=65536 sequence=1 reason=pressure")
        self.assertEqual((recognized, invalid), (True, 1))
        snap = status.snapshot()
        self.assertEqual(snap['sessions']['1']['state'], 'evicted')
        self.assertEqual(snap['sessions']['1']['bytes'], 0)
        self.assertEqual(snap['evictions'][0]['tokens'], 1024)
        self.assertEqual(status.consume('BDONE 1 0 length 1'), (False, None))
        status.consume('KVCACHE session=1 generation=3 epoch=9 result=miss replay=1')
        self.assertEqual(status.snapshot()['sessions']['1']['state'], 'active')

    def test_commit_and_physical_budget_are_observable_without_history_storage(self):
        status = KvStatus()
        status.consume('KVCOMMIT session=0 generation=1 epoch=2 cells=256 bytes=4194304 pool_bytes=8388608 latency_us=43')
        status.consume('KVMEM physical_available_after_headroom=16777216 admissible_bytes=12582912')
        snap = status.snapshot()
        self.assertEqual(snap['pool']['committed_bytes'], 8388608)
        self.assertEqual(snap['sessions']['0']['bytes'], 4194304)
        self.assertEqual(snap['memory']['admissible_bytes'], 12582912)
        self.assertNotIn('history', snap)

    def test_workspace_admission_does_not_claim_the_destination_is_active_yet(self):
        status = KvStatus()
        status.consume('KVSESSION session=1 generation=1 epoch=2 state=resident tokens=1024 bytes=4096')
        handled, invalid = status.consume('KVCACHE session=1 generation=1 epoch=2 state=resident tokens=1024 bytes=4096 result=resident_or_new')
        self.assertTrue(handled)
        self.assertIsNone(invalid)
        self.assertEqual(status.snapshot()['sessions']['1']['state'], 'resident')
        _, invalid = status.consume('KVCOMMIT session=1 generation=1 epoch=3 bytes=8192 pool_bytes=16384')
        self.assertEqual(invalid, 1)
        self.assertEqual(status.snapshot()['sessions']['1']['state'], 'active')
        self.assertNotIn('tokens', status.snapshot()['sessions']['1'])  # old history is not attributed to a new epoch

    def test_stale_event_cannot_invalidate_new_generation(self):
        status = KvStatus()
        status.consume('KVRESET session=0 generation=4 epoch=10')
        self.assertEqual(status.consume('KVEVICT session=0 generation=3 epoch=9 sequence=1'), (True, None))
        self.assertEqual(status.snapshot()['sessions']['0']['generation'], 4)

    def test_bounded_events_and_sequence_gap(self):
        status = KvStatus()
        for seq in range(3, 303):
            status.consume(f'KVEVICT session=0 generation=1 epoch={seq} sequence={seq}')
        snap = status.snapshot()
        self.assertEqual(len(snap['evictions']), 256)
        self.assertEqual(snap['missed_events'], 2)
        snap['sessions']['0']['state'] = 'corrupt'
        self.assertEqual(status.snapshot()['sessions']['0']['state'], 'evicted')

    def test_pump_consumes_telemetry_and_routes_slot_error(self):
        engine = StrataEngine('not-started', [], lazy=True)
        engine.info = {'batch_session_offset': 1}
        engine.slot_cv = threading.Condition()
        engine.slot_held = [[1, 2], [3, 4]]
        engine.slot_busy = [False, True]
        engine.lines = queue.Queue()
        engine.slot_q = [queue.Queue(), queue.Queue()]
        engine.proc = mock.Mock()
        engine.proc.stdout = iter([
            'KVEVICT session=1 generation=1 epoch=2 tokens=2 bytes=64 sequence=1\n',
            'BERR 1 kv_pool_full: active caches protected\n',
            'DONE 1 1 0 1 stop\n'])
        engine._pump()
        self.assertEqual(engine.slot_held, [[], [3, 4]])
        self.assertEqual(engine.slot_busy, [False, True])  # event does not release active ownership
        self.assertTrue(engine.slot_q[1].get_nowait().startswith('BERR 1'))
        self.assertTrue(engine.lines.get_nowait().startswith('DONE'))
        self.assertIsNone(engine.lines.get_nowait())


FAKE = r'''
import sys
mode=sys.argv[sys.argv.index('--mode')+1]
print('INFO unified_kv=1 kv_protocol=1 batch_slots=4 slot_cache=1 batch_session_offset=0',flush=True)
print('KVPOOL enabled=1 token_limit=500000 committed_bytes=0 resident_tokens=0',flush=True)
print('READY 250000 stop',flush=True)
first=True
for line in sys.stdin:
 if line.startswith('QUIT'): break
 if line.startswith('BGEN '):
  b=int(line.split()[1])
  print('T 90\nDONE 1 1 0 1 length\nBADM %d 1'%b,flush=True)
  print('BERR %d kv_pool_full: active caches protected'%b,flush=True)
 elif line.startswith('GEN '):
  if first and mode=='refuse': print('ERR kv_pool_full: replay needs shared capacity',flush=True)
  else: print('T 90\nDONE 1 1 0 1 stop',flush=True)
  first=False
'''


class PipeErrors(unittest.TestCase):
    def engine(self, mode):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        script = Path(temp.name) / 'native.py'
        script.write_text('#!' + sys.executable + '\n' + FAKE)
        script.chmod(0o700)
        engine = StrataEngine(str(script), ['--batch', '4', '--mode', mode])
        self.addCleanup(engine.close)
        return engine

    def test_capacity_refusal_is_error_and_process_can_recover(self):
        e = self.engine('refuse')
        with self.assertRaisesRegex(ValueError, 'kv_pool_full'):
            list(e.generate([1], 2, {}, threading.Event()))
        self.assertTrue(e.alive())
        self.assertEqual(list(e.generate([1], 2, {}, threading.Event())), [90])
        self.assertEqual(e.unified_kv.snapshot()['pool']['token_limit'], 500000)

    def test_batch_allocation_failure_does_not_hang_or_report_length(self):
        e = self.engine('batch')
        # Force batch admission beside another request; no inference is simulated
        # in that other slot. The native BERR belongs only to the admitted slot.
        e.slot_busy[0] = True
        e.slot_claims.add(0)
        with mock.patch.object(e, '_may_go_solo', return_value=False):
            with self.assertRaisesRegex(ValueError, 'kv_pool_full'):
                list(e.generate_batched([1], 8, {}, threading.Event()))
        self.assertTrue(e.alive())
        self.assertEqual(sum(e.slot_busy), 1)


if __name__ == '__main__':
    unittest.main()
