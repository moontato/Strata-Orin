"""Focused review regressions. No model, GPU, or production service."""
import json, os, signal, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace
from tools import q36_image_http_smoke as smoke
from serve.server import Service


def test_metrics_real_service_contract_and_proxy_bypass(monkeypatch):
    svc = SimpleNamespace(status_lock=threading.Lock(), status={}, live_reqs={},
        history=[{'id': i} for i in range(15)], totals={}, engine=SimpleNamespace(),
        loaded=lambda: True, model='test', reported_ctx=lambda: 1, vision=None,
        conv_log=SimpleNamespace(poll=lambda *a: {}))
    assert len(Service.metrics(svc)['requests']) == 12
    assert len(Service.metrics(svc, all_requests=True)['requests']) == 15
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_GET(self):
            body = json.dumps(Service.metrics(svc, all_requests='requests=all' in self.path)).encode()
            self.send_response(200); self.end_headers(); self.wfile.write(body)
    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    monkeypatch.setenv('http_proxy', 'http://127.0.0.1:1')
    monkeypatch.setenv('no_proxy', '')
    try:
        assert len(smoke.Client(server.server_port, timeout=2).metrics()['requests']) == 15
    finally:
        server.shutdown(); thread.join(); server.server_close()


def test_real_solo_and_batch_summaries_and_unknown_restore():
    text = ('KVCACHE session=0 generation=1 epoch=1 state=active tokens=10 bytes=100 reserved_tokens=10 result=new replay=0\n'
            'strata serve: session 0: prompt 10 tokens = 0 reused + 10 read in 2 ms (5.0 tok/s), 1 generated in 1 ms (1.0 tok/s), drafts accepted 0 of 0 (stop)\n'
            'KVREPLAY session=0 generation=2 epoch=2 tokens=8 reason=prefix_checkpoint\n'
            'KVCACHE session=0 generation=2 epoch=2 state=active tokens=12 bytes=120 reserved_tokens=12 result=checkpoint replay=0\n'
            'strata batch: slot 0 takes a prompt of 12 tokens = 8 reused + 4 read in 3 ms (1.3 tok/s)\n')
    rows, counts = smoke.correlate([{'prompt_total': 10}, {'prompt_total': 12}], smoke.parse_engine_log(text))
    assert counts == {'unmatched_requests': 0, 'unmatched_engine_lines': 0}
    assert [r['engine']['kind'] for r in rows] == ['request', 'slot']
    assert all(r['restored'] is None and r['result'] is None for r in rows)


@__import__('pytest').mark.skipif(not Path('/proc/self/stat').exists(),reason='real /proc unavailable')
def test_missing_leader_with_live_group_is_not_signalled(tmp_path):
    # The leader creates a descendant in its process group, then exits. Its parent
    # reaps it before the probe. A ready file makes the descendant lifetime explicit.
    ready = tmp_path / 'ready'
    descendant = "import pathlib,sys,time; pathlib.Path(sys.argv[1]).write_text('ready'); time.sleep(60)"
    leader_code = "import subprocess,sys; subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]]); import time; time.sleep(.2)"
    leader = subprocess.Popen([sys.executable, '-c', leader_code, descendant, str(ready)], start_new_session=True)
    try:
        smoke.register_group(tmp_path, leader, 'test')
        leader.wait(timeout=5)
        for _ in range(100):
            if ready.exists(): break
            time.sleep(.02)
        assert ready.exists()
        result = smoke.reap_recorded_groups(tmp_path, time.monotonic()+1)[0]
        assert result['kind'] == 'leader_missing_group_present_not_signalled'
        assert result['group_empty'] is False and result['operator_cgroup_required']
        os.killpg(leader.pid, 0)  # probe: still present and was not signalled
    finally:
        # Test owns this PGID and its descendant; clean up the deliberate fixture.
        try: os.killpg(leader.pid, signal.SIGKILL)
        except ProcessLookupError: pass


def test_real_run_requires_exact_source_root_even_with_window_environment(tmp_path):
    cmd = [sys.executable, str(Path(smoke.__file__)), '--engine', 'x', '--baseline-engine', 'x',
           '--model', 'x', '--source-config', 'x', '--server-root', 'x', '--out', str(tmp_path/'out')]
    p = subprocess.run(cmd, capture_output=True, text=True,
        env=dict(os.environ, STRATA_IMAGE_HTTP_REVIEW_WINDOW='1'))
    assert p.returncode == 2 and 'STRATA_IMAGE_HTTP_ROOT' in p.stderr
