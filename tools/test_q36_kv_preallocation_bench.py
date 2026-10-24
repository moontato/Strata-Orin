"""Protocol/measurement negative controls, not a model or performance test."""
import unittest
from tools.q36_kv_preallocation_bench import request
from tools.unified_kv_validate import Native, NativeError


class Engine:
    def __init__(self, allocation_delta=0, done='DONE 2 2 10 20 length 1 2 0', driver_delta=0):
        self.commits = []
        self.sent = []
        self.lines = iter(['T 10', 'T 11', done])
        self.calls = 0
        self.allocation_delta = allocation_delta
        self.driver_delta = driver_delta

    def status(self):
        bank = dict(preallocated=1, physical_allocations=8, map_calls=0, initialized_pages=0,
                    mapping_us=0, initialization_us=0, checkpoint_operations=0, checkpoint_copy_us=0,
                    driver_allocations=8, driver_allocation_attempts=8, driver_releases=0)
        if self.calls:
            bank.update(physical_allocations=8 + self.allocation_delta, mapping_us=3, initialization_us=5,
                        driver_allocations=8 + self.driver_delta)
        self.calls += 1
        return {'bank': bank}

    def send(self, line):
        self.sent.append(line)

    def recv(self):
        return next(self.lines)


class Measurement(unittest.TestCase):
    def test_sampling_and_cost_categories(self):
        e = Engine()
        row = request(e, [80, 198], 2)
        self.assertEqual(row['tokens'], [10, 11])
        self.assertEqual((row['prefill_ms'], row['decode_ms']), (10, 20))
        self.assertEqual(row['bank_delta']['physical_allocations'], 0)
        self.assertEqual(row['bank_delta']['mapping_us'], 3)
        for key in ('seed=1234', 'temperature=1.0', 'top_p=0.95', 'top_k=20',
                    'penalty_present=1.5', 'penalty_last_n=64', 'sampled_mtp=1'):
            self.assertIn(key, e.sent[0])

    def test_physical_allocation_after_ready_is_not_a_pass(self):
        with self.assertRaisesRegex(NativeError, 'increased after readiness'):
            request(Engine(allocation_delta=1), [80, 198], 2)

    def test_untracked_driver_allocation_is_not_hidden_by_bank_counter(self):
        with self.assertRaisesRegex(NativeError, 'increased after readiness'):
            request(Engine(driver_delta=1), [80, 198], 2)

    def test_shortened_output_is_not_a_comparable_benchmark(self):
        with self.assertRaisesRegex(NativeError, 'exact output cap'):
            request(Engine(done='DONE 2 2 10 20 stop 0 2 0'), [80, 198], 2)

    def test_reset_keeps_lazy_zero_and_banked_held_invariants_separate(self):
        for bank, held, fail in (({}, 0, False), ({}, 4096, True),
                                 ({'preallocated': 1, 'bank_bytes': 8192, 'mapped_bytes': 0}, 8192, False),
                                 ({'preallocated': 1, 'bank_bytes': 8192, 'mapped_bytes': 4096}, 8192, True),
                                 ({'preallocated': 1, 'bank_bytes': 8192, 'mapped_bytes': 0}, 0, True)):
            n = Native.__new__(Native)
            n.info = {'unified_kv': 1}
            n.sessions = {0: {'generation': 1}}
            n.status = lambda: {'pool': {'committed_bytes': held}, 'bank': bank}
            n.send = lambda line: None
            n.recv = lambda: 'KVRESET session=0 generation=2 epoch=3'
            if fail:
                with self.assertRaises(NativeError):
                    n.reset()
            else:
                n.reset()


if __name__ == '__main__':
    unittest.main()
