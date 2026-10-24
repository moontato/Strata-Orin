#!/usr/bin/env python3
"""Compile/run deliberate allocator mutations in temporary overlays, never edit source.
Each mutated production path MUST fail the production-allocator regression.
"""
import os
import resource
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MUTATIONS = {
    'bank-exhaustion-allocates': ('unified_kv_bank.hpp', 'if (free_.empty()) return 0;',
                                'if (free_.empty()) return vmm_chunk_new();'),
    'recycled-pages-not-zeroed': ('unified_kv_vmm.hpp',
                                'cudaMemset(f.b->base() + (uint64_t) f.chunk * gran, 0, (size_t) gran)',
                                'cudaSuccess'),
    'bank-charged-as-fresh-memory': ('unified_kv.hpp',
                                   '? held <= byte_limit_ && extra / bank.granularity <= bank.free_pages',
                                   '? held <= byte_limit_ && extra / bank.granularity <= bank.free_pages && available() >= extra'),
    'reservation-tail-not-reclaimed': ('unified_kv_vmm.hpp', 'if (!bank_) return true;', 'return true;'),
    'borrowed-handle-released-on-failure': ('unified_kv_vmm.hpp', 'range.map_borrowed(c, h)',
                                          'range.map_range(c, c + 1, [h] { return h; })'),
}


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    for name, (header, old, new) in MUTATIONS.items():
        with tempfile.TemporaryDirectory(prefix='kv-negative-') as d:
            temp = Path(d)
            path = temp / 'strata/core' / header
            path.parent.mkdir(parents=True)
            text = (ROOT / 'include/strata/core' / header).read_text()
            assert text.count(old) == 1, (name, 'mutation no longer unique')
            path.write_text(text.replace(old, new))
            exe = temp / 'test'
            subprocess.run([os.environ.get('CXX', 'g++'), '-std=c++20', '-O1', '-pthread',
                            '-DSTRATA_KV_FAKE_DRIVER', '-I' + str(temp), '-Itests/fake_cuda', '-Iinclude',
                            'tests/core/kv_preallocation_test.cpp', 'src/core/vmm.cpp',
                            'tests/fake_cuda/driver.cpp', '-o', str(exe)], cwd=ROOT, check=True)
            result = subprocess.run([str(exe)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
            if not result.returncode:
                raise AssertionError(f'{name}: mutation incorrectly passed')
            print(f'NEGATIVE_CONTROL PASS {name} exit={result.returncode}')


if __name__ == '__main__':
    main()
