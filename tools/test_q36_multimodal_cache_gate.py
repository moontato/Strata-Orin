"""Worker control flow tests with Python stand-ins, never an LLM/service."""
import json
from pathlib import Path
import sys
import time
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import q36_multimodal_cache_gate as gate


def transcript():
    return '\n'.join(['IMAGE_CACHE_DIAGNOSTIC_TEST PASS independent_image_reference=0', 'KV_PREALLOCATION_MODEL PASS'] +
        [f'IMAGE_CACHE_TURN case=public turn={i+1} slot=0 reused=0 read=10 elapsed_ms=3 schedule_and_state_recorded=1' for i in range(10)]) + '\n'


def test_exact_raw_comparison_and_schedule_signature(tmp_path):
    a, b = tmp_path / 'a', tmp_path / 'b'; a.write_bytes(b'x' * 100000); b.write_bytes(a.read_bytes())
    assert gate.exact(a, b)
    b.write_bytes(b'x' * 99999 + b'y'); assert not gate.exact(a, b)
    assert gate.signature(transcript()) == gate.signature(transcript().replace('elapsed_ms=3', 'elapsed_ms=99'))
    assert gate.signature(transcript()) != gate.signature(transcript().replace('reused=0', 'reused=1'))


def test_owned_child_timeout_and_unrelated_native_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, 'no_models', lambda: None)
    with pytest.raises(TimeoutError):
        gate.run([sys.executable, '-c', 'import time;time.sleep(10)'], tmp_path, 'timeout', time.monotonic() + .03)
    record = json.loads((tmp_path / 'timeout.exit.json').read_text())
    assert record['exit'] < 0 and record['error'] == 'TimeoutError'
    with pytest.raises(RuntimeError, match='native gate failed'):
        gate.run([sys.executable, '-c', 'raise SystemExit(1)'], tmp_path, 'fail', time.monotonic() + 5)


def test_preflight_before_stop_and_off_on_order_fail_fast(tmp_path, monkeypatch):
    stage = tmp_path / 'stage'; stage.mkdir(); fixtures = stage / 'fixture'; fixtures.mkdir()
    binary = tmp_path / 'review' / 'build' / 'strata-q36'; binary.parent.mkdir(parents=True)
    binary.write_bytes(b'CPU MOCK ONLY --kv-image-cache-test --kv-cache-diagnostics')
    model = stage / 'model'; model.write_bytes(b'CPU MOCK ONLY')
    data = fixtures / 'image-cache.txt'; data.write_text('CPU MOCK ONLY\n')
    st = model.stat()
    manifest = dict(kind='q36-image-cache-v1', real_tokenizer_template=True, python_prepare_off_on_exact=True,
                    model=dict(path=str(model.resolve()), size=st.st_size, mtime_ns=st.st_mtime_ns),
                    files={'image-cache.txt': gate.digest(data)})
    (fixtures / 'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(gate, 'ROOT', binary.parent.parent); monkeypatch.setattr(gate, 'REVIEW', binary.parent.parent)
    monkeypatch.setattr(gate, 'STAGE', stage); monkeypatch.setattr(gate.shutil, 'disk_usage', lambda _: type('Disk', (), dict(free=9 * 1024**3))())
    calls = []
    monkeypatch.setattr(gate, 'validate_fixture', lambda *_: None)  # control flow mock, not fixture evidence
    monkeypatch.setattr(gate, 'no_models', lambda: calls.append('no_models'))
    args = ['gate', '--engine', str(binary), '--model', str(model), '--fixtures', str(fixtures), '--out', str(stage / 'results')]
    monkeypatch.setattr(sys, 'argv', args + ['--preflight'])
    gate.main(); assert calls == [] and not (stage / 'results').exists()
    monkeypatch.setenv('STRATA_MM_CACHE_REVIEW_WINDOW', '1')
    def run(cmd, out, arm, deadline):
        calls.append(arm)
        assert 0 < deadline - time.monotonic() <= 300
        assert ('--kv-cache-diagnostics' in cmd) == (arm == 'on')
        for flag, value in [('--max-context', '262144'), ('--batch', '4'), ('--kv-pool-tokens', '524288'),
                            ('--kv-checkpoint-policy', 'adaptive'), ('--kv-vision-max-tokens', '1024')]:
            assert cmd[cmd.index(flag) + 1] == value
        Path(cmd[cmd.index('--test-state-dump') + 1]).write_bytes(b'identical RAW snapshot / schedule')
        return transcript() + ('' if arm == 'off' else '\n'.join(['Q36CACHE candidate reason=exact_prefix', 'reason=image_mismatch',
            'mode=explicit_batch', 'mode=engine_selected', 'checkpoint_fallback=image_request_bypass']))
    monkeypatch.setattr(gate, 'run', run); monkeypatch.setattr(sys, 'argv', args)
    gate.main(); assert calls == ['no_models', 'off', 'on']
    report = json.loads((stage / 'results/result.json').read_text())
    assert report['status'] == 'PASS_DIAGNOSTIC_NEUTRALITY_ONLY' and not report['independent_image_reference']
    calls.clear()
    def fail(cmd, out, arm, deadline): calls.append(arm); raise RuntimeError('first arm failed')
    monkeypatch.setattr(gate, 'run', fail); monkeypatch.setattr(sys, 'argv', args[:-1] + [str(stage / 'failed')])
    with pytest.raises(RuntimeError): gate.main()
    assert calls == ['no_models', 'off']


def test_no_service_or_live_worktree_operations_in_worker():
    source = (ROOT / 'tools/q36_multimodal_cache_gate.py').read_text()
    assert 'systemctl' not in source and 'pkill' not in source
    assert 'os.WNOWAIT' in source and 'next_progress = started + 20' in source
    assert 'deadline = time.monotonic() + 300' in source
    assert 'worktrees/multimodal-cache' in source and 'strata-checkpoints' in source
