"""Compile a minimal host scope reproduction using the actual cache declaration.
This does NOT compile Serve::request_body(), the CUDA TU, or a model path.
"""
import os
from pathlib import Path
import re
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


def scope_source(unqualified=False):
    text = (ROOT / 'src/qwen35/q36_engine.cu').read_text()
    engine = text.split('struct Engine {', 1)[1].split('struct Serve {', 1)[0]
    constant = re.search(r'static constexpr int kVerify\s*=\s*\d+;', engine).group()
    serve = text.split('struct Serve {', 1)[1]
    declarations = re.findall(
        r'std::array<size_t,[^>]+> common_cache\{\};const size_t\* common_tokens=nullptr;', serve)
    if len(declarations) != 1:
        raise AssertionError('expected exactly one request-path cache declaration')
    declaration = declarations[0]
    if unqualified:
        declaration = declaration.replace('Engine::kVerify', 'kVerify')
    return f'''#include <array>
#include <cstddef>
using std::size_t;
struct Engine {{ {constant} }};
struct Serve {{
    size_t request_scope_probe() {{
        {declaration}
        static_assert(decltype(common_cache){{}}.size() == Engine::kVerify);
        common_tokens = common_cache.data();
        return common_tokens[Engine::kVerify - 1];
    }}
}};
'''


def compile_scope(source):
    return subprocess.run([os.environ.get('CXX', 'g++'), '-std=c++20', '-Wall', '-Wextra',
                           '-Werror', '-pedantic-errors', '-fsyntax-only', '-x', 'c++', '-'],
                          input=source, text=True, capture_output=True, timeout=30)


@unittest.skipUnless(shutil.which(os.environ.get('CXX', 'g++')), 'host C++ compiler unavailable')
class CheckpointScope(unittest.TestCase):
    def test_actual_declaration_in_serve_scope_compiles(self):
        result = compile_scope(scope_source())
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_unqualified_engine_constant_is_rejected(self):
        result = compile_scope(scope_source(unqualified=True))
        self.assertNotEqual(result.returncode, 0, 'unqualified member unexpectedly compiled')
        self.assertIn('kVerify', result.stderr)
        self.assertRegex(result.stderr, r'not declared|undeclared')


if __name__ == '__main__':
    unittest.main()
