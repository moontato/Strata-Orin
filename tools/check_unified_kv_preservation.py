#!/usr/bin/env python3
"""Check protected Jetson/compute sources; optionally compare CUDA kernel bytes.

CUBIN section names include NVCC's anonymous-namespace source-path hash. Only
that part of the NAME is normalized. The compared machine-code BYTES are not.
This is not an inference, numerical-parity or performance test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess

PROTECTED = ['src/kernels', 'include/strata/kernels', 'src/prefill', 'include/strata/prefill',
             'src/platform', 'include/strata/platform', 'src/qwen35/q36_kernels.cuh',
             'src/core/expert_cache.cpp', 'include/strata/artifact']


def kernel_hashes(path):
    data = Path(path).read_bytes()
    if data[:6] != b'\x7fELF\x02\x01':
        raise ValueError('expected a little-endian ELF64 CUDA cubin')
    offset = struct.unpack_from('<Q', data, 40)[0]
    size, count, strings_index = struct.unpack_from('<HHH', data, 58)
    sections = [struct.unpack_from('<IIQQQQIIQQ', data, offset + i * size) for i in range(count)]
    strings = sections[strings_index]
    names = data[strings[4]:strings[4] + strings[5]]
    result = {}
    for section in sections:
        name = names[section[0]:names.find(b'\0', section[0])].decode()
        if not name.startswith('.text.'):
            continue
        name = re.sub(r'_GLOBAL__N__[0-9a-f]{8}', '_GLOBAL__N__SOURCE_PATH_HASH', name)
        if name in result:
            raise ValueError('normalization produced a duplicate kernel name')
        result[name] = hashlib.sha256(data[section[4]:section[4] + section[5]]).hexdigest()
    if not result:
        raise ValueError('cubin has no kernel text sections')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--base', default='ec8149d1')
    p.add_argument('--baseline-cubin', type=Path)
    p.add_argument('--candidate-cubin', type=Path)
    a = p.parse_args()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(a.repo), *args], text=True).strip()
    subprocess.run(['git', '-C', str(a.repo), 'merge-base', '--is-ancestor', a.base, 'HEAD'], check=True)
    changed = git('diff', '--name-only', a.base, '--', *PROTECTED).splitlines()
    report = {'base': git('rev-parse', a.base), 'head': git('rev-parse', 'HEAD'),
              'branch': git('branch', '--show-current'), 'protected_source_changes': changed}
    ok = not changed and report['branch'] == 'feature/unified-kv'
    if bool(a.baseline_cubin) != bool(a.candidate_cubin):
        p.error('supply both cubin paths')
    if a.baseline_cubin:
        old, new = kernel_hashes(a.baseline_cubin), kernel_hashes(a.candidate_cubin)
        different = sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k))
        report.update(baseline_kernels=len(old), candidate_kernels=len(new),
                      identical_kernel_text_sections=sum(old[k] == new.get(k) for k in old),
                      differing_kernel_text_sections=different)
        ok = ok and not different
    report['passed'] = ok
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
