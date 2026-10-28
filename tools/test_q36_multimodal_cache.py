"""CPU/mock ONLY. No CUDA, encoder, model or HTTP service operated here."""
import ast
import copy
import json
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
import textwrap
from types import SimpleNamespace
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from tools.check_q36_integration_preservation import body
from tools import q36_multimodal_cache_fixture as fixture
from serve import q36_cache_diagnostics as diag
from serve.server import StrataEngine
BASE = '4130bdfdb43bdb273412cc3530660bd19ec14602'


def original(path):
    return subprocess.check_output(['git', 'show', BASE + ':' + path], cwd=ROOT, text=True)


def test_matching_and_numeric_source_invariants():
    current = (ROOT / 'src/qwen35/q36_engine.cu').read_text(); old = original('src/qwen35/q36_engine.cu')
    # Image eligibility/metadata methods intentionally evolve; numerical bodies do not.
    for marker in ['int64_t process(', 'void forward(', 'void forward_mtp(',
                   'void layer(', 'void run_rows(', 'void mtp_rows(', 'void preallocate_kv(',
                   'void checkpoint_transfer(', 'int spec_window(']:
        assert body(current, marker) == body(old, marker), marker
    for path in ['include/strata/core/q36_prefix_checkpoint.hpp', 'include/strata/core/unified_kv_layout.hpp',
                 'include/strata/core/unified_kv_vmm.hpp',
                 'include/strata/core/q36_sampled_mtp.hpp', 'src/qwen35/q36_kv_int8.cuh', 'src/qwen35/q36_kernels.cuh',
                 'src/qwen35/q36_vision_staging.inc', 'serve/frontend.py', 'serve/runconfig.py']:
        assert (ROOT / path).read_text() == original(path), path
    followup = 'f934b85b42e79cc2a7c4ec0ca9da7685674e1ebd'
    on_followup = subprocess.run(['git', 'merge-base', '--is-ancestor', followup, 'HEAD'], cwd=ROOT, capture_output=True).returncode == 0
    protected_base = followup if on_followup else BASE
    for path in ['tools/q36_adaptive_integration_fixture.py', 'tools/test_q36_integration_ordinary.py', 'docs/Q36_ADAPTIVE_ORIN_VALIDATION.json']:
        old_file = subprocess.run(['git', 'show', protected_base + ':' + path], cwd=ROOT, capture_output=True)
        if old_file.returncode == 0: assert (ROOT / path).read_bytes() == old_file.stdout
        else: assert not (ROOT / path).exists()
    assert 'bool cache_diagnostics = false;' in current
    assert 'if(!geni && fresh>1 && (!reuse || restored)) E.checkpoint_capture();' in current
    assert 'if (!reuse && (!geni || image_checkpoint_mode) && E.checkpoint_count)' in current
    assert 'const bool image_checkpoint_mode=geni && E.image_checkpoints;' in current
    assert 'bool image_checkpoints=false;' in current


def test_prepare_and_send_bodies_unchanged_except_read_only_probe():
    old = ast.parse(original('serve/server.py'))
    new = ast.parse((ROOT / 'serve/server.py').read_text())
    def method(module, cls, name):
        node = next(c for c in module.body if isinstance(c, ast.ClassDef) and c.name == cls)
        return next(f for f in node.body if isinstance(f, ast.FunctionDef) and f.name == name)
    for cls, name in [('StrataEngine', '_send'), ('Service', 'prepare')]:
        a, b = method(old, cls, name), method(new, cls, name)
        b.decorator_list = []
        if name == 'prepare':
            def is_probe(node):
                return isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'q36_cache_diagnostics.note_prepare'
            assert sum(is_probe(n) for n in b.body) == 1
            b.body = [n for n in b.body if not is_probe(n)]
        assert ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False)
    gate = (ROOT / 'src/qwen35/q36_image_cache_regression.inc').read_text()
    assert 'checkpoint_schedule_regression' not in gate and 'cudaMemcpyHostToDevice' not in gate
    assert 'independent_image_reference=0' in gate


def test_actual_native_selectors_and_new_probe_compile_execute(tmp_path):
    current = (ROOT / 'src/qwen35/q36_engine.cu').read_text()
    inc = tmp_path / 'selectors.inc'; inc.write_text(body(current, 'size_t match_len(') + '\n' + body(current, 'int pick_session('))
    baseline = tmp_path / 'baseline-selectors.inc'; old = original('src/qwen35/q36_engine.cu')
    baseline.write_text(body(old, 'size_t match_len(') + '\n' + body(old, 'int pick_session('))
    exe = tmp_path / 'probe'
    subprocess.run(['g++', '-std=c++17', '-Wall', '-Wextra', '-Werror', '-pedantic-errors', '-pthread',
                    '-I' + str(ROOT / 'include'), '-DMATCH_INC="' + str(inc) + '"', '-DBASE_MATCH_INC="' + str(baseline) + '"',
                    str(ROOT / 'tests/core/q36_cache_native_probe_test.cpp'), '-o', str(exe)], check=True, capture_output=True)
    sve = tmp_path / 'PRIVATE_PATH_SENTINEL.sve'; sve.write_bytes(struct.pack('<5i', 0x31455653, 4, 2, 2, 2) + bytes(32))
    result = subprocess.run([str(exe), str(sve)], check=True, capture_output=True, text=True)
    assert 'CPU stand-ins) PASS' in result.stdout
    assert 'PRIVATE_PATH_SENTINEL' not in result.stderr
    assert 'mode=explicit_batch explicit_destination=2' in result.stderr
    assert 'rows_diff=1 key_diff=0' in result.stderr
    assert 'idle_backed_candidates=0' in result.stderr
    assert 'state=active' in result.stderr and 'checkpoint_fallback=image_request_bypass' in result.stderr
    assert 'details=0' in result.stderr  # stale epoch doesn't claim row/grid/key equality
    assert max(map(len, result.stderr.splitlines())) < 1200


def test_python_original_selector_body_unchanged_and_off_on_neutral(monkeypatch, capsys):
    source = original('serve/server.py')
    start = source.index('    def pick_slot('); stop = source.index('    def slots_view(', start)
    old = ast.parse(textwrap.dedent(source[start:stop])).body[0]
    import inspect
    current = ast.parse(textwrap.dedent(inspect.getsource(StrataEngine.pick_slot.__wrapped__))).body[0]
    current.decorator_list = []
    assert ast.dump(old, include_attributes=False) == ast.dump(current, include_attributes=False)
    rng = random.Random(19)
    for _ in range(100):
        engine = object.__new__(StrataEngine); engine.batch = 4; engine.slot_order = [0, 1, 2, 3]
        engine.slot_busy = [bool(rng.randrange(2)) for _ in range(4)]
        engine.slot_held = [[rng.randrange(5) for _ in range(rng.randrange(8))] for _ in range(4)]
        engine.slot_used = [rng.random() for _ in range(4)]; engine.slot_gs = 0; engine.slot_groups = 1
        engine.slot_group = [0] * 4; prompt = [rng.randrange(5) for _ in range(10)]
        before = copy.deepcopy(engine.__dict__)
        monkeypatch.delenv('STRATA_Q36_CACHE_DIAGNOSTICS', raising=False)
        off = engine.pick_slot(prompt); assert capsys.readouterr().err == ''
        monkeypatch.setenv('STRATA_Q36_CACHE_DIAGNOSTICS', '1')
        on = engine.pick_slot(prompt)
        assert off == on and {k: engine.__dict__[k] for k in before} == before
        lines = capsys.readouterr().err.splitlines(); assert len(lines) == 5
        assert all(x.startswith('Q36CACHE_PY ') for x in lines)


def test_dispatch_read_only_private_command_and_slot_log_budget(monkeypatch, capsys):
    monkeypatch.setenv('STRATA_Q36_CACHE_DIAGNOSTICS', '1')
    sent = []
    @diag.dispatch_probe
    def send(_engine, text): sent.append(text); return 'unchanged-result'
    text = 'BGENI 2 1 img_keys=PRIVATE_KEY /PRIVATE_IMAGE_PATH 99123456,99123457'
    assert send(None, text) == 'unchanged-result' and sent == [text]
    log = capsys.readouterr().err
    assert 'PRIVATE' not in log and '99123456' not in log
    data = json.loads(log.removeprefix('Q36CACHE_PY '))
    assert data['protocol'] == 'BGENI' and data['destination'] == 2 and data['expanded_tokens'] == 2
    calls = []
    @diag.slot_probe
    def select(_engine, _prompt): calls.append(1); return 2
    engine = SimpleNamespace(slot_order=[])
    for _ in range(30): assert select(engine, [99123456]) == 2
    lines = capsys.readouterr().err.splitlines()
    assert len(calls) == 30 and len(lines) == 9 and 'slot_suppressed' in lines[-1]


def test_python_sink_failure_does_not_change_selection(monkeypatch):
    @diag.slot_probe
    def selected(_engine, _prompt): return 2
    monkeypatch.setenv('STRATA_Q36_CACHE_DIAGNOSTICS', '1')
    engine = SimpleNamespace(slot_order=[])
    monkeypatch.setattr(diag.sys, 'stderr', SimpleNamespace(write=lambda *_: (_ for _ in ()).throw(OSError('sink'))))
    assert selected(engine, [99123456]) == 2


class ToyTokenizer:
    control_tokens = ['<|im_start|>', '<|im_end|>', '<|vision_start|>', '<|image_pad|>', '<|vision_end|>']
    def encode(self, text, parse_special=False, **kwargs):
        import re
        special = dict(zip(self.control_tokens, [248045, 248044, 248053, 248056, 248054]))
        out = []
        for part in re.split('(' + '|'.join(map(re.escape, self.control_tokens)) + ')', text):
            out += [special[part]] if parse_special and part in special else [ord(c) + 1000 for c in part]
        return out


TEMPLATE = '''{% for m in messages %}<|im_start|>{{m.role}}\n{% if m.content is string %}{{m.content}}{% else %}{% for p in m.content %}{% if p.type == 'image' %}<|vision_start|><|image_pad|><|vision_end|>{% else %}{{p.text}}{% endif %}{% endfor %}{% endif %}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}'''


def test_actual_prepare_pipeline_with_fake_tokenizer_encoder_io_only(tmp_path, monkeypatch, capsys):
    from tools import strata_tokenizer, gguf_reader
    monkeypatch.setattr(strata_tokenizer.Tokenizer, 'from_gguf', staticmethod(lambda _: ToyTokenizer()))
    monkeypatch.setattr(gguf_reader, 'GGUFFile', lambda _: SimpleNamespace(metadata={'tokenizer.chat_template': TEMPLATE}))
    model = tmp_path / 'model'; model.write_bytes(b'NOT A MODEL; CPU MOCK ONLY')
    fixture.make_images(tmp_path / 'images')
    a, b = tmp_path / 'a.sve', tmp_path / 'b.sve'
    for p, value in ((a, 0.25), (b, 0.5)):
        p.write_bytes(struct.pack('<5i', 0x31455653, 4, 2, 2, 2048) + struct.pack('<f', value) * (4 * 2048))
    manifest = fixture.prepare(model, tmp_path / 'images/red-square.png', tmp_path / 'images/blue-square.png', a, b, tmp_path / 'prepared')
    assert len(manifest['rows']) == 10 and manifest['python_prepare_off_on_exact']
    assert manifest['rows'][1]['audit']['classification'] == 'exact_prefix'
    assert manifest['rows'][-1]['audit']['classification'] == 'token_prefix_image_mismatch'
    assert manifest['rows'][-1]['images'][0]['n'] == manifest['rows'][-2]['images'][0]['n']
    assert {r['kind'] for r in manifest['rows']} >= {'native_constructed_not_API', 'real_Service.prepare'}
    log = capsys.readouterr().err
    assert 'red-square' not in log and 'file://' not in log and 'payload_sha256' not in log and 'Public stable' not in log
    assert 'expanded_lcp' in log and 'prepared_not_native_retained' in log
    from tools.q36_multimodal_cache_gate import validate_fixture
    validate_fixture(tmp_path / 'prepared', manifest)
    malformed = copy.deepcopy(manifest); malformed['rows'][1]['ids'][0] += 1
    with pytest.raises(ValueError, match='native plan differs'):
        validate_fixture(tmp_path / 'prepared', malformed)
    # This manifest was produced using fake IO/tokenizer: NEVER hardware evidence.


def test_token_vs_image_classification_not_message_count():
    image = [dict(cell=1, n=2, identity=91)]
    old = ([1, 9, 9, 2], image)
    assert fixture.classify(old, [1, 9, 9, 2, 3], image)['classification'] == 'exact_prefix'
    assert fixture.classify(old, [4, 9, 9, 2, 3], image)['classification'] == 'token_rewrite'
    assert fixture.classify(old, [1, 9, 9, 2, 3], [dict(cell=1, n=2, identity=92)])['classification'] == 'token_prefix_image_mismatch'
    assert not fixture.classify(([1, 9], []), [1, 9, 9, 2], image)['complete_image_boundary']
