#!/usr/bin/env python3
"""Isolated native off/on measurement. Run ONLY inside the documented service-stop window.
Reports load-to-READY, combined KVCOMMIT, bank mapping/initialization, prefill and decode
separately. No speedup claim is inferred. Numerical qualification uses the native gates.
"""
import argparse
import json
import os
import time
from pathlib import Path
try:
    from .unified_kv_validate import Native, NativeError, prompt
except ImportError:  # direct script execution
    from unified_kv_validate import Native, NativeError, prompt


def request(engine, ids, count):
    before = engine.status()
    start_commit = len(engine.commits)
    started = time.monotonic()
    engine.send(f'GEN {count} temperature=1.0 top_p=0.95 top_k=20 penalty_present=1.5 '
                'penalty_last_n=64 seed=1234 sampled_mtp=1 ' + ','.join(map(str, ids)))
    tokens, first = [], None
    while True:
        line = engine.recv()
        if line.startswith('T '):
            tokens.append(int(line.split()[1]))
            if first is None:
                first = time.monotonic() - started
        if line.startswith('DONE '):
            words = line.split()
            if words[5] != 'length' or len(tokens) != count:
                raise NativeError('benchmark ended before its exact output cap; not a comparable measurement')
            break
    wall = time.monotonic() - started
    after = engine.status()
    bank = after.get('bank', {})
    if bank and any(bank[key] != before['bank'][key] for key in
                    ('physical_allocations', 'driver_allocations', 'driver_allocation_attempts', 'driver_releases')):
        raise NativeError('physical allocation/release counters increased after readiness')
    return {'tokens': tokens, 'prompt_tokens': len(ids), 'wall_seconds': wall,
            'first_token_seconds': first, 'prefill_ms': float(words[3]), 'decode_ms': float(words[4]),
            'accepted': int(words[6]), 'offered': int(words[7]),
            'kvcommit_us': sum(c['latency_us'] for c in engine.commits[start_commit:]),
            'bank_delta': {key: bank[key] - before['bank'][key] for key in
                           ('physical_allocations', 'driver_allocations', 'driver_allocation_attempts', 'driver_releases',
                            'map_calls', 'initialized_pages', 'mapping_us',
                            'initialization_us', 'checkpoint_operations', 'checkpoint_copy_us')} if bank else {},
            'status': after}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--pairs', type=int, default=5)
    p.add_argument('--lengths', default='4096,32768')
    p.add_argument('--gen', type=int, default=96)
    o = p.parse_args()
    if os.environ.get('STRATA_KV_REVIEW_WINDOW') != '1':
        p.error('stop/reap the deployed service using docs/Q36_KV_PREALLOCATION.md first')
    if o.pairs < 1 or o.gen < 1:
        p.error('pairs and gen must be positive')
    o.out.mkdir(parents=True, exist_ok=True)
    args = ['--model', o.model, '--max-context', '262144', '--batch', '4', '--kv-pool-tokens', '500000',
            '--kv', 'int8', '--mtp', '2', '--batch-mtp', '--mtp-sampled', '--chunk', '512',
            '--kv-checkpoint-mib', '1024', '--kv-checkpoint-interval', '1024']
    report = {'args': args, 'runs': []}
    try:
        for pair in range(o.pairs):
            arms = {}
            for mode in (['off', 'on'] if pair % 2 == 0 else ['on', 'off']):
                prefix = o.out / f'{pair}-{mode}'
                e = Native(o.exe, args + (['--kv-preallocate'] if mode == 'on' else []), prefix, 3600)
                try:
                    row = {'pair': pair, 'mode': mode, 'startup_seconds': e.startup_seconds, 'requests': []}
                    if bool(e.status().get('bank', {}).get('preallocated')) != (mode == 'on'):
                        raise NativeError('preallocation flag was ignored or implicitly enabled')
                    for case, length in enumerate(map(int, o.lengths.split(','))):
                        e.reset()
                        row['requests'].append(request(e, prompt(0, length, case + 1), o.gen))
                    e.reset()  # also assert the bank survives repeated end/reset cycles
                    arms[mode] = row
                    report['runs'].append(row)
                finally:
                    e.close()  # waits/reaps before constructing the next model instance
                text = Path(str(prefix) + '.stderr.log').read_text()
                if 'every routed expert in VRAM' not in text or 'hybrid experts:' in text or 'given back' in text:
                    raise NativeError('both arms must preserve full-GPU placement and packed copies')
            for old, new in zip(arms['off']['requests'], arms['on']['requests']):
                if old['tokens'] != new['tokens']:
                    raise NativeError('off/on sampled output differs at identical settings/seed')
        report['result'] = 'KV_PREALLOCATION_BENCH PASS'
    finally:
        (o.out / 'measurements.json').write_text(json.dumps(report, indent=2) + '\n')
    print(report['result'])


if __name__ == '__main__':
    main()
