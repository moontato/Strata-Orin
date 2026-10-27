#!/usr/bin/env python3
"""Source invariants only; NOT CUDA compilation, SASS or model qualification."""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = '3f3bc4e85e1d44b3501f0dcb8cd73471632753cf'
ENGINE = 'src/qwen35/q36_engine.cu'


def original(path):
    return subprocess.check_output(['git', 'show', BASE + ':' + path], cwd=ROOT, text=True)


def body(text, marker):
    start = text.index(marker)
    # Replace strings/comments without changing byte positions before brace scan.
    lexical = re.sub(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*.*?\*/',
                     lambda m: ' ' * len(m[0]), text, flags=re.S)
    at = lexical.index('{', start); depth = 0
    for i in range(at, len(text)):
        if lexical[i] == '{': depth += 1
        if lexical[i] == '}':
            depth -= 1
            if not depth: return text[start:i+1]
    raise ValueError('unterminated function: ' + marker)


def main():
    old, new = original(ENGINE), (ROOT / ENGINE).read_text()
    methods = ['void f32mm(', 'void mm(', 'void layer(', 'void forward(', 'void forward_mtp(',
               'int64_t process(', 'void run_rows(', 'void commit_rows(', 'void mtp_rows(',
               'int spec_window(', 'void spec_commit(', 'int spec_step(', 'void checkpoint_transfer(',
               'bool checkpoint_available(', 'void set_images(', 'void clear_images(', 'void chunk_pictures(']
    for marker in methods:
        assert body(old, marker) == body(new, marker), marker
        print('SOURCE_IDENTICAL', marker)
    for path in ['src/qwen35/q36_kernels.cuh', 'src/qwen35/q36_kv_int8.cuh',
                 'src/qwen35/q36_vision_staging.inc', 'src/qwen35/q36_vision_regression.inc',
                 'include/strata/core/q36_vision_preallocation.hpp', 'include/strata/core/q36_sampled_mtp.hpp',
                 'include/strata/core/unified_kv_layout.hpp', 'include/strata/core/unified_kv_vmm.hpp',
                 'serve/server.py', 'serve/runconfig.py']:
        assert original(path) == (ROOT / path).read_text(), path
        print('SOURCE_IDENTICAL', path)
    old_header = original('include/strata/core/q36_prefix_checkpoint.hpp')
    header = (ROOT / 'include/strata/core/q36_prefix_checkpoint.hpp').read_text()
    assert body(old_header, 'static size_t capacity(') == body(header, 'static size_t capacity(')
    # Existing sampled text / no image / no batch / no concurrent request guard.
    assert body(old, 'const bool sampled_spec =') == body(new, 'const bool sampled_spec =')
    print('SOURCE_IDENTICAL checkpoint capacity / sampled restrictions')
    assert 'Policy checkpoint_policy = strata::core::Q36PrefixCheckpoints::Policy::fixed;' in new
    assert 'std::array<size_t,Engine::kVerify> common_cache' in new
    assert 'if(!geni && fresh>1 && (!reuse || restored)) E.checkpoint_capture();' in new
    print('PASS fixed default, corrected Serve scope, text-only checkpoint capture')


if __name__ == '__main__':
    main()
