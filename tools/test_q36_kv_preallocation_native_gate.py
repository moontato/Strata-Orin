"""Native-gate coverage/error policy using tiny process mocks, never a GPU/model."""
import contextlib
import io
import json
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path
from tools.q36_kv_preallocation_native_gate import (DEVICE_CASES, HOST_CASES, INT8_PASS,
                                                  NATIVE_TESTS, run_native_gates, vmm_coverage)


def native_log(unsupported=()):
    unsupported = set(unsupported)
    lines = []
    for case in sorted(DEVICE_CASES | HOST_CASES | {'host_incremental'}):
        status = 'unsupported' if case in unsupported else 'passed'
        line = f'UNIFIED_KV_COVERAGE case={case} host_required={int(case not in DEVICE_CASES)} status={status}'
        if status == 'unsupported':
            line += ' rejection=passed reason=unsupported_flat_address'
        if case.startswith('qsa_ring_'):
            line += ' ring_table=passed'
        lines.append(line)
    lines.append('UNIFIED_KV_VMM_TEST PASS device_vmm=passed graph_reuse=passed qsa_device_passed=8 '
                 f'qsa_host_passed={8-len(HOST_CASES & unsupported)} '
                 f'qsa_host_unsupported={len(HOST_CASES & unsupported)} '
                 'draft_ring_tables_passed=4 '
                 f'host_incremental={"unsupported" if "host_incremental" in unsupported else "passed"} '
                 f'host_rejections={len(unsupported)}')
    return '\n'.join(lines) + '\n'


class Coverage(unittest.TestCase):
    def test_full_host_positive_coverage(self):
        report = vmm_coverage(native_log())
        self.assertEqual(len(report['device_passed']), 8)
        self.assertEqual(len(report['host_positive_passed']), 9)
        self.assertFalse(report['host_positive_unsupported'])

    def test_host_unsupported_is_separate_and_device_ring_checks_still_pass(self):
        report = vmm_coverage(native_log(HOST_CASES | {'host_incremental'}))
        self.assertEqual(len(report['device_passed']), 8)
        self.assertEqual(len(report['draft_ring_tables_passed']), 4)
        self.assertFalse(report['host_positive_passed'])
        self.assertEqual(len(report['host_positive_unsupported']), 9)
        self.assertEqual(report['host_positive_unsupported'], report['host_rejection_passed'])

    def test_mixed_per_registration_outcomes(self):
        report = vmm_coverage(native_log({'qsa_stream_INT8', 'qsa_ring_Q4_0'}))
        self.assertEqual(len(report['host_positive_passed']), 7)
        self.assertEqual(report['host_positive_unsupported'], ['qsa_ring_Q4_0', 'qsa_stream_INT8'])

    def test_no_case_can_disappear(self):
        for case in ('qsa_device_INT8', 'qsa_stream_INT8', 'qsa_ring_INT8', 'qsa_draft_resident_INT8', 'host_incremental'):
            text = '\n'.join(line for line in native_log().splitlines() if f'case={case} ' not in line)
            with self.subTest(case=case), self.assertRaises(ValueError):
                vmm_coverage(text)

    def test_device_only_cannot_be_unsupported(self):
        with self.assertRaises(ValueError):
            vmm_coverage(native_log({'qsa_device_INT8'}))

    def test_host_rejection_and_device_ring_proof_are_required(self):
        text = native_log(HOST_CASES | {'host_incremental'})
        for old, new in (('rejection=passed', 'rejection=failed'),
                         ('reason=unsupported_flat_address', 'reason=backend'),
                         (' ring_table=passed', '')):
            with self.subTest(old=old), self.assertRaises(ValueError):
                vmm_coverage(text.replace(old, new))

    def test_counts_and_graph_coverage_are_not_optional(self):
        for old, new in (('qsa_device_passed=8', 'qsa_device_passed=0'),
                         ('graph_reuse=passed', 'graph_reuse=unsupported'),
                         ('host_rejections=0', 'host_rejections=9')):
            with self.subTest(old=old), self.assertRaises(ValueError):
                vmm_coverage(native_log().replace(old, new))

    def test_duplicate_summary_or_case_is_not_accepted(self):
        text = native_log()
        for line in (text.splitlines()[0], text.splitlines()[-1]):
            with self.subTest(line=line), self.assertRaises(ValueError):
                vmm_coverage(text + line + '\n')


class NativeExecution(unittest.TestCase):
    def run_gate(self, root, fail_test=None, exit_code=0, missing_marker=None):
        build = root / 'bin'
        build.mkdir()
        outputs = {NATIVE_TESTS[0]: native_log(HOST_CASES | {'host_incremental'}),
                   NATIVE_TESTS[1]: INT8_PASS + '\n', NATIVE_TESTS[2]: 'KV_PREALLOCATION_TEST PASS\n'}
        for test in NATIVE_TESTS:
            output = '' if missing_marker == test else outputs[test]
            if fail_test == test:
                output += 'KV host backing cannot satisfy flat device addressing\n'
            path = build / test
            path.write_text('#!/bin/sh\nprintf %s ' + shlex.quote(output) + '\nexit ' +
                            str(exit_code if test == fail_test else 0) + '\n')
            path.chmod(0o700)
        with contextlib.redirect_stdout(io.StringIO()):
            return run_native_gates(build, root / 'results')

    def test_unsupported_host_does_not_skip_int8_or_preallocation(self):
        with tempfile.TemporaryDirectory() as temp:
            report = self.run_gate(Path(temp))
            self.assertEqual([report['native_tests'][t]['status'] for t in NATIVE_TESTS], ['passed']*3)
            self.assertEqual(len(report['vmm_coverage']['host_positive_unsupported']), 9)

    def test_any_nonzero_native_exit_is_fatal_even_with_pass_markers(self):
        for test in NATIVE_TESTS:
            for code in (1, 77):
                with self.subTest(test=test, code=code), tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)
                    with self.assertRaises(subprocess.CalledProcessError):
                        self.run_gate(root, fail_test=test, exit_code=code)
                    report = json.loads((root / 'results/native-coverage.json').read_text())
                    self.assertEqual(report['native_tests'][test]['status'], 'failed')
                    for later in NATIVE_TESTS[NATIVE_TESTS.index(test)+1:]:
                        self.assertEqual(report['native_tests'][later]['status'], 'pending')

    def test_zero_exit_without_mandatory_marker_is_fatal(self):
        for test in NATIVE_TESTS:
            with self.subTest(test=test), tempfile.TemporaryDirectory() as temp:
                with self.assertRaises(ValueError):
                    self.run_gate(Path(temp), missing_marker=test)


if __name__ == '__main__':
    unittest.main()
