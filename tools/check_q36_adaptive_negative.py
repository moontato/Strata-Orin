#!/usr/bin/env python3
"""Production-header mutations must FAIL the sequence regression, not skip it."""
import os
from pathlib import Path
import resource
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
HEADER = Path('strata/core/q36_prefix_checkpoint.hpp')
MUTATIONS = {
    'ineffective-retention': ('if (policy!=Policy::adaptive || !chunk || incoming<=start || points.size()<3)',
                            'if (true)'),
    'reuse-after-divergence': ('p.pos<=common && p.pos<incoming', 'p.pos<incoming'),
    'stale-generation': ('p.generation==generation && p.epoch==epoch', 'p.epoch==epoch'),
    'stale-epoch': ('p.generation==generation && p.epoch==epoch', 'p.generation==generation'),
    'publish-partial-payload': ('points[index]={};return old;', 'return old;'),
    'append-only-trains': ('!chunk || !common || common>=previous', '!chunk || !common || common>previous'),
    'unattainable-target': ('const auto end=start+(bound-start)/chunk*chunk;', 'const auto end=bound;'),
}


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    for name, (old, new) in MUTATIONS.items():
        with tempfile.TemporaryDirectory(prefix='q36-adaptive-negative-') as directory:
            root = Path(directory)
            path = root / HEADER
            path.parent.mkdir(parents=True)
            text = (ROOT / 'include' / HEADER).read_text()
            assert text.count(old) == 1, (name, 'mutation not unique')
            path.write_text(text.replace(old, new))
            exe = root / 'test'
            subprocess.run([os.environ.get('CXX', 'g++'), '-std=c++20', '-O1', '-I' + str(root),
                            '-Iinclude', 'tests/core/q36_adaptive_checkpoint_test.cpp', '-o', str(exe)],
                           cwd=ROOT, check=True)
            run = subprocess.run([str(exe)], capture_output=True, timeout=60)
            if run.returncode >= 0:
                raise AssertionError(f'{name}: expected assertion/signal failure, got {run.returncode}')
            if b'Assertion' not in run.stderr:
                raise AssertionError(f'{name}: unrelated failure: {run.stderr!r}')
            print(f'NEGATIVE_CONTROL PASS {name} exit={run.returncode} assertion={run.stderr.decode().strip()}')


if __name__ == '__main__':
    main()
