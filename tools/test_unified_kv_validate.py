"""Validation-driver demultiplexing tests, no native engine or GPU."""
import unittest
from tools.unified_kv_validate import Native, NativeError, fields, prompt


class Protocol(unittest.TestCase):
    def native(self, script):
        n = Native.__new__(Native)
        n.context = 8192
        n.sent = []
        n.send = n.sent.append
        lines = iter(script)
        def recv(allow_error=False):
            line = next(lines)
            if line.startswith(('ERR ', 'BERR ')) and not allow_error:
                raise NativeError(line)
            return line
        n.recv = recv
        n.status = lambda: {'pool': {'committed_bytes': 4096}}
        return n

    def test_interleaved_admission_and_slot_output(self):
        n = self.native(['T 10', 'DONE 1 2 10 1 length 0 0 0', 'BADM 0 1', 'BT 0 11',
                         'T 20', 'DONE 1 2 11 1 length 0 0 0', 'BADM 1 1',
                         'BT 0 12', 'BDONE 0 3 length 2', 'BT 1 21', 'BT 1 22', 'BDONE 1 3 length 3'])
        result = n.batch({0: [80, 198], 1: [81, 198]}, 3)
        self.assertEqual(result['slots'][0]['tokens'], [10,11,12])
        self.assertEqual(result['slots'][1]['tokens'], [20,21,22])
        self.assertEqual(result['slots'][1]['prefill_ms'], 11)
        self.assertEqual(len(n.sent), 2)

    def test_no_silent_shortening_on_failed_admission(self):
        n = self.native(['T 10', 'DONE 1 2 10 1 length 0 0 0', 'BADM 0 0'])
        with self.assertRaisesRegex(NativeError, 'silently shortened'):
            n.batch({0: [80, 198]}, 32)

    def test_allocation_error_is_not_a_completion(self):
        n = self.native(['T 10', 'DONE 1 2 10 1 length 0 0 0', 'BADM 0 1', 'BERR 0 kv_pool_full'])
        with self.assertRaisesRegex(NativeError, 'kv_pool_full'):
            n.batch({0: [80, 198]}, 32)

    def test_context_is_never_shrunk_to_fit(self):
        n = self.native([])
        with self.assertRaisesRegex(NativeError, 'must not be shortened'):
            n.batch({0: [80] * 8192}, 32)
        self.assertEqual(n.sent, [])

    def test_cases_have_distinct_prefixes_for_uncached_ab(self):
        self.assertNotEqual(prompt(0, 4096, 1)[0], prompt(0, 4096, 99)[0])
        self.assertEqual(len(prompt(0, 250000-32, 1)), 249968)
        self.assertEqual(fields('KVPOOL committed_bytes=4194304 enabled=1'),
                         {'committed_bytes': 4194304, 'enabled': 1})


if __name__ == '__main__':
    unittest.main()
