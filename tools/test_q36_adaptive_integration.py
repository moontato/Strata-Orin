import json
from pathlib import Path
import sys
import subprocess
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import q36_adaptive_integration_fixture as fixture
import q36_adaptive_integration_gate as gate
from q36_checkpoint_schedule import audit


def encode(messages):
    return list(''.join(m['role'] + ':' + m['content'] + '\n' for m in messages).encode())


def test_actual_token_audit_not_case_names():
    result = audit('# case append\n0 1,2,3\n0 1,4\n0 1,4,5\n')
    assert [r['classification'] for r in result['turns']] == ['new', 'rewrite', 'append']
    assert [r['lcp'] for r in result['turns']] == [0, 1, 2]
    assert audit('0 1,2\n0 1,2\n')['turns'][1]['classification'] == 'identical'
    result = audit('# case peer\n0 1,2\n1 8,9\n0 1,2,3\n1 8,9,10\n')
    assert [r['classification'] for r in result['turns']] == ['new', 'new', 'append', 'append']


def test_short_fixture_cpu_standin_only():
    replay, smoke, metrics, messages = fixture.generate(encode)
    assert len(metrics['turns']) == 20
    assert max(r['tokens'] for r in metrics['turns']) <= 8192
    assert {r['case'] for r in metrics['turns']} == {'temporary_user', 'system_context', 'ordinary', 'interleaved'}
    assert len(audit(smoke)['turns']) == 2
    assert {r['phase'] for r in metrics['turns']} == {'warmup', 'steady'}
    assert any(r['classification'] == 'append' for r in metrics['turns'])
    assert 'Temporary public fetch_url content' in messages['temporary_user'][1][1][1]['content']
    assert 'Temporary public fetch_url content' not in messages['temporary_user'][2][1][1]['content']
    assert audit(replay) == metrics


def test_reject_false_append_claim():
    # Fake encoder destroys every prefix. Generator must fail instead of labelling
    # a rewritten prompt append-only based on its message structure.
    def wrong(messages):
        ids = encode(messages)
        ids[0] = len(messages)
        return ids
    with pytest.raises(ValueError):
        fixture.generate(wrong)


def measurements():
    return '\n'.join('SCHEDULE_MEASURE case=user turn=1 phase=' + p +
                     ' reused=1024 replayed=44 copy_us=2 copy_bytes=9 prefill_ms=3 latency_ms=4'
                     for p in ('warmup', 'steady')) + '\n'


def test_report_keeps_phases_and_different_schedules_separate():
    report = gate.summarize(measurements() + 'SCHEDULE_LOGITS comparison=different_schedule differing=8 exact=0\n')
    assert report['phases']['warmup']['reused'] == 1024
    assert report['phases']['steady']['copy_us'] == 2
    assert report['different_schedule_logit_failures'] == 1
    with pytest.raises(ValueError):
        gate.summarize(measurements().splitlines()[0])


def setup_fixture(tmp_path):
    model = tmp_path / 'model.gguf'; model.write_bytes(b'CPU MOCK ONLY')
    binary = tmp_path / 'engine'; binary.write_bytes(b'CPU MOCK ONLY')
    fixtures = tmp_path / 'fixtures'; fixtures.mkdir()
    replay, smoke, metrics, _ = fixture.generate(encode)
    (fixtures / 'replay.csv').write_text(replay)
    (fixtures / 'reference-smoke.csv').write_text(smoke)
    st = model.stat()
    manifest = dict(kind='q36-adaptive-integration-v1', byte_test_only=False,
                    model=dict(path=str(model.resolve()), size=st.st_size, mtime_ns=st.st_mtime_ns), audit=metrics,
                    files={p.name: gate.digest(p) for p in fixtures.iterdir()})
    (fixtures / 'manifest.json').write_text(json.dumps(manifest))
    return model, binary, fixtures


def test_stage_order_policy_oracles_and_early_stop(tmp_path, monkeypatch):
    model, binary, fixtures = setup_fixture(tmp_path)
    calls = []
    def run(command, out, name, timeout, negative=None):
        if '--test-state-dump' in command:
            data = b'adaptive-own-schedule' if 'adaptive' in command else b'baseline-fixed'
            Path(command[command.index('--test-state-dump') + 1]).write_bytes(data)
        calls.append((name, command, negative))
        return measurements()
    monkeypatch.setattr(gate, 'guard', lambda: None)
    monkeypatch.setattr(gate, 'run', run)
    output = tmp_path / 'out'
    args = ['gate', '--baseline', str(binary), '--engine', str(binary), '--model', str(model), '--fixtures', str(fixtures),
            '--vision-max-tokens', '1024', '--out', str(output)]
    monkeypatch.setattr(sys, 'argv', args)
    gate.main()
    assert [c[0] for c in calls] == ['01-baseline-reference', '02-negative-conv', '02-negative-gdn', '02-negative-hlast',
                                    '03-replay-baseline', '03-replay-fixed', '03-replay-adaptive']
    for _, cmd, _ in calls:
        for flag, value in [('--max-context', '262144'), ('--batch', '4'), ('--kv-pool-tokens', '524288'), ('--mtp', '2')]:
            assert cmd[cmd.index(flag) + 1] == value
        assert {'--kv-preallocate', '--batch-mtp', '--mtp-sampled', '--vision'} <= set(cmd)
    assert json.loads((output / 'report.json').read_text())['production_benefit'] == 'NOT_CLAIMED'
    calls.clear()
    def fail(*args, **kwargs):
        calls.append(args[2]); raise RuntimeError('matching oracle failed')
    monkeypatch.setattr(gate, 'run', fail)
    monkeypatch.setattr(sys, 'argv', args[:-1] + [str(tmp_path / 'failed')])
    with pytest.raises(RuntimeError, match='oracle failed'):
        gate.main()
    assert calls == ['01-baseline-reference']


def test_guard_requires_explicit_window(monkeypatch):
    monkeypatch.delenv('STRATA_CHECKPOINT_REVIEW_WINDOW', raising=False)
    with pytest.raises(RuntimeError, match='approved'):
        gate.guard()


def test_run_rejects_wrong_negative_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, 'guard', lambda: None)
    script = tmp_path / 'fake.py'
    script.write_text("print('unrelated failure'); raise SystemExit(1)\n")
    with pytest.raises(RuntimeError, match='negative control'):
        gate.run([sys.executable, str(script)], tmp_path, 'wrong', 10, negative='conv')
    assert json.loads((tmp_path / 'wrong.exit.json').read_text())['exit'] == 1


def test_timeout_keeps_exit_record_and_reaps_owned_child(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, 'guard', lambda: None)
    with pytest.raises(subprocess.TimeoutExpired):
        gate.run([sys.executable, '-c', 'import time;time.sleep(10)'], tmp_path, 'timeout', 0.02)
    record = json.loads((tmp_path / 'timeout.exit.json').read_text())
    assert record['error'] == 'TimeoutExpired' and record['exit'] < 0


  # runtime settings are exact, not changed defaults

