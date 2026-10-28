#!/usr/bin/env python3
"""CPU-only deliberate diagnostic mutations must fail assertions, not just crash.
No source edit. Set TMPDIR to development/staging scratch before invoking.
"""
import os
from pathlib import Path
import resource
import subprocess
import tempfile
ROOT = Path(__file__).resolve().parents[1]
HEADER = Path('strata/core/q36_cache_diagnostics.hpp')
MUTATIONS = {
    'ignore-token-divergence': ('held[d.lcp]==ids[d.lcp]', 'true'),
    'identical-is-proper-extension': ('held.size()<ids.size()', 'held.size()<=ids.size()'),
    'ignore-incomplete-image': ('else if(k.cell<(int64_t)held.size()) d.complete=false;', 'else if(k.cell<(int64_t)held.size()) d.complete=true;'),
    'ignore-image-identity': ('d.images&=x==y;', 'd.images&=true;'),
    'ignore-placement': ('d.placement|=x.cell!=y.cell || x.n!=y.n;', 'd.placement=false;'),
    'ignore-grid': ('d.grid|=x.nx!=y.nx || x.ny!=y.ny;', 'd.grid=false;'),
    'ignore-row-bytes': ('d.rows|=x.rows!=y.rows;', 'd.rows=false;'),
    'ignore-supplied-key': ('d.key|=x.keyed!=y.keyed || (x.keyed && x.key!=y.key);', 'd.key=false;'),
}


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    original = (ROOT / 'include' / HEADER).read_text()
    for name, (old, new) in MUTATIONS.items():
        assert original.count(old) == 1, name
        with tempfile.TemporaryDirectory(prefix='q36-cache-negative-') as temp:
            root = Path(temp); target = root / HEADER; target.parent.mkdir(parents=True); target.write_text(original.replace(old, new))
            exe = root / 'test'
            subprocess.run([os.environ.get('CXX', 'g++'), '-std=c++17', '-O1', '-UNDEBUG', '-I' + str(root),
                            str(ROOT / 'tests/core/q36_cache_diagnostic_test.cpp'), '-o', str(exe)], check=True, capture_output=True)
            run = subprocess.run([str(exe)], capture_output=True)
            if run.returncode >= 0 or b'Assertion' not in run.stderr:
                raise AssertionError(f'{name}: expected specific assertion failure, got {run.returncode}, {run.stderr!r}')
            print('CACHE_DIAGNOSTIC_NEGATIVE PASS', name, 'exit=' + str(run.returncode))


if __name__ == '__main__':
    main()
