#!/usr/bin/env python3
"""Mandatory native gates; explicit host-positive UNSUPPORTED is coverage, not exit-code forgiveness.
Called inside q36_kv_preallocation_gate.sh's service-stop window. No model is loaded here.
"""
import argparse
import json
import subprocess
from pathlib import Path

FORMATS = ('FP16', 'INT8', 'Q4_0', 'K8V4')
DEVICE_CASES = {prefix + fmt for prefix in ('qsa_device_', 'qsa_draft_resident_') for fmt in FORMATS}
HOST_CASES = {prefix + fmt for prefix in ('qsa_stream_', 'qsa_ring_') for fmt in FORMATS}
NATIVE_TESTS = ('unified_kv_vmm_test', 'q36_kv_int8_test', 'kv_preallocation_cuda_test')
INT8_PASS = ('Q36 INT8: quantization, FP64 stored-KV reference, prefill staging, batch rows, '
             'main/draft copy, VMM accounting, LRU and graph replay passed')


def fields(words):
    out = {}
    for word in words:
        key, sep, value = word.partition('=')
        if not sep or not key or not value or key in out:
            raise ValueError('malformed/duplicate native coverage field: ' + word)
        out[key] = value
    return out


def vmm_coverage(text):
    cases, summaries = {}, []
    for line in text.splitlines():
        if line.startswith('UNIFIED_KV_COVERAGE '):
            item = fields(line.split()[1:])
            case = item.get('case')
            if not case or case in cases:
                raise ValueError('missing/duplicate native coverage case')
            cases[case] = item
        elif line.startswith('UNIFIED_KV_VMM_TEST '):
            if not line.startswith('UNIFIED_KV_VMM_TEST PASS '):
                raise ValueError('native VMM test did not pass')
            summaries.append(fields(line.split()[2:]))
    if len(summaries) != 1 or set(cases) != DEVICE_CASES | HOST_CASES | {'host_incremental'}:
        raise ValueError('missing/extra native VMM/QSA coverage or summary')
    unsupported, host_passed = [], []
    for case, item in cases.items():
        requires_host = case not in DEVICE_CASES
        if item.get('host_required') != str(int(requires_host)):
            raise ValueError('wrong host placement classification: ' + case)
        status = item.get('status')
        if status == 'unsupported':
            if (not requires_host or item.get('rejection') != 'passed' or
                    item.get('reason') != 'unsupported_flat_address'):
                raise ValueError('unsupported coverage lacks explicit safe host rejection: ' + case)
            unsupported.append(case)
        elif status == 'passed':
            if requires_host:
                host_passed.append(case)
        else:
            raise ValueError('missing/invalid coverage outcome: ' + case)
        if case.startswith('qsa_ring_') and item.get('ring_table') != 'passed':
            raise ValueError('device ring-table coverage is mandatory: ' + case)
    expected = dict(device_vmm='passed', graph_reuse='passed', qsa_device_passed='8',
                    qsa_host_passed=str(len(HOST_CASES & set(host_passed))),
                    qsa_host_unsupported=str(len(HOST_CASES & set(unsupported))),
                    draft_ring_tables_passed='4', host_incremental=cases['host_incremental']['status'],
                    host_rejections=str(len(unsupported)))
    if summaries[0] != expected:
        raise ValueError('native summary disagrees with mandatory case coverage')
    return {'device_passed': sorted(DEVICE_CASES), 'host_positive_passed': sorted(host_passed),
            'host_positive_unsupported': sorted(unsupported), 'host_rejection_passed': sorted(unsupported),
            'draft_ring_tables_passed': sorted(case for case in HOST_CASES if case.startswith('qsa_ring_')),
            'summary': summaries[0]}


def run_native_gates(build, results):
    build, results = Path(build).resolve(), Path(results).resolve()
    results.mkdir(parents=True, exist_ok=True)
    report = {'native_tests': {test: {'status': 'pending'} for test in NATIVE_TESTS}}
    def save():
        (results / 'native-coverage.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    for test in NATIVE_TESTS:
        try:
            with (results / (test + '.log')).open('w') as log:
                # All nonzero exits, including 1 and CUDA skip=77, remain failures.
                subprocess.run([str(build / test)], stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=600)
            text = (results / (test + '.log')).read_text()
            if test == 'unified_kv_vmm_test':
                report['vmm_coverage'] = vmm_coverage(text)
            elif test == 'q36_kv_int8_test' and INT8_PASS not in text.splitlines():
                raise ValueError('mandatory INT8 pass marker missing')
            elif test == 'kv_preallocation_cuda_test' and 'KV_PREALLOCATION_TEST PASS' not in text.splitlines():
                raise ValueError('mandatory preallocation pass marker missing')
            report['native_tests'][test] = {'status': 'passed'}
        except Exception as error:
            report['native_tests'][test] = {'status': 'failed', 'error': str(error)}
            save()
            raise
        save()
    print('KV_NATIVE_DEVICE_GATES PASS')
    coverage = report['vmm_coverage']
    print('KV_NATIVE_HOST_COVERAGE positive_passed=' + str(len(coverage['host_positive_passed'])) +
          ' positive_unsupported=' + str(len(coverage['host_positive_unsupported'])) +
          ' rejection_passed=' + str(len(coverage['host_rejection_passed'])))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', required=True, type=Path)
    p.add_argument('--results', required=True, type=Path)
    args = p.parse_args()
    run_native_gates(args.build, args.results)


if __name__ == '__main__':
    main()
