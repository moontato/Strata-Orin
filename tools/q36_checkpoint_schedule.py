#!/usr/bin/env python3
"""Audit saved native SLOT CSV fixtures using actual token prefixes (no model load).
The independent native implementation is --kv-schedule-test, not this script.
Never use a case name, message count or character offset to classify continuation.
"""
import argparse
import hashlib
import json
from pathlib import Path


def audit(text):
    previous = {}
    group, phase, rows = 'unnamed', 'warmup', []
    for line in text.splitlines():
        if not line:
            continue
        if line.startswith('# case '):
            group, previous = line[7:], {}
            continue
        if line.startswith('# phase '):
            phase = line[8:]
            if phase not in ('warmup', 'steady'):
                raise ValueError('invalid phase')
            continue
        slot, csv = line.split(' ', 1)
        if slot not in ('0', '1', '2'):
            raise ValueError('slot 3 is reference-only')
        ids = list(map(int, csv.split(',')))
        if not ids or min(ids) < 0:
            raise ValueError('invalid IDs')
        old = previous.get(slot, [])
        common = next((i for i, (a, b) in enumerate(zip(old, ids)) if a != b), min(len(old), len(ids)))
        rows.append(dict(case=group, phase=phase, slot=int(slot), tokens=len(ids), previous=len(old),
                         lcp=common, classification='new' if not old else 'identical' if ids == old else 'append' if common == len(old) else 'rewrite',
                         ids_sha256=hashlib.sha256(csv.encode()).hexdigest()))
        previous[slot] = ids
    if len(rows) < 2:
        raise ValueError('need multiple turns')
    return dict(native_oracle='--kv-schedule-test', independent_native_implemented=True,
                source_sha256=hashlib.sha256(text.encode()).hexdigest(), turns=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    Path(a.out).write_text(json.dumps(audit(Path(a.fixture).read_text()), indent=2) + '\n')


if __name__ == '__main__':
    main()
