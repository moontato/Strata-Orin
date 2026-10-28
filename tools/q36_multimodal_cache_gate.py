#!/usr/bin/env python3
"""Future APPROVED isolated Orin window only. No service/config operations.
Two sequential LLM arms, one binary, identical input/schedule; <=300s total model
budget, progress every20s. No encoder/HTTP server, projection study or image oracle.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STAGE = Path('/mnt/ssd/tmp/strata-checkpoints')
REVIEW = Path('/mnt/ssd/Strata-Orin/worktrees/multimodal-cache')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def exact(a, b):
    with Path(a).open('rb') as x, Path(b).open('rb') as y:
        while True:
            left, right = x.read(65536), y.read(65536)
            if left != right: return False
            if not left: return True


def first_difference(a, b):
    offset = 0
    with Path(a).open('rb') as x, Path(b).open('rb') as y:
        while True:
            left, right = x.read(65536), y.read(65536)
            if left != right:
                return offset + next((i for i, (u, v) in enumerate(zip(left, right)) if u != v), min(len(left), len(right)))
            if not left: return None
            offset += len(left)


def no_models():
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid(): continue
        try:
            name = (proc / 'comm').read_text().strip()
            cmd = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        except FileNotFoundError: continue
        if name in {'strata', 'strata-q36', 'strata-vision', 'llama-server', 'llama-cli', 'llama-bench'} or re.search(r'(?:^|\s)\S*serve/server\.py(?:\s|$)|-m serve\.server', cmd):
            raise RuntimeError('existing model/server/encoder detected; STOP without touching it')


def run(command, out, arm, deadline):
    no_models()
    if time.monotonic() >= deadline: raise TimeoutError('five-minute model budget exhausted before next arm')
    (out / (arm + '.command.json')).write_text(json.dumps(command, indent=2) + '\n')
    started = time.monotonic(); next_progress = started + 20
    print('MM_CACHE_PROGRESS arm=' + arm + ' stage=starting_model', flush=True)
    with (out / (arm + '.log')).open('wb') as log:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        error = None
        try:
            # WNOWAIT keeps leader PID/PGID reserved until owned-group cleanup.
            while os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None:
                now = time.monotonic()
                if now >= deadline: raise TimeoutError('five-minute model budget exhausted')
                if now >= next_progress:
                    print(f'MM_CACHE_PROGRESS arm={arm} elapsed_s={int(now-started)} remaining_s={max(0,int(deadline-now))}', flush=True)
                    next_progress = now + 20
                time.sleep(min(0.1, max(0, deadline-now)))
        except BaseException as exc:
            error = type(exc).__name__; raise
        finally:
            # Never kill process names or foreign PIDs. The operator also owns
            # the enclosing cgroup and must prove ALL descendants gone on exit.
            old_handlers = {s: signal.signal(s, signal.SIG_IGN) for s in (signal.SIGINT, signal.SIGTERM)}
            try:
                try: os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                rc = child.wait(timeout=20)
            finally:
                for s, handler in old_handlers.items(): signal.signal(s, handler)
            (out / (arm + '.exit.json')).write_text(json.dumps(dict(exit=rc, error=error, elapsed_s=time.monotonic()-started)) + '\n')
    text = (out / (arm + '.log')).read_text(errors='replace')
    if rc or 'IMAGE_CACHE_DIAGNOSTIC_TEST PASS' not in text or 'KV_PREALLOCATION_MODEL PASS' not in text or re.search(r'^(?:ERR|BERR) ', text, re.M):
        raise RuntimeError(arm + ': native gate failed; stop promptly and retain log')
    return text


def signature(text):
    result = []
    for line in text.splitlines():
        p = line.split()
        if not p: continue
        if p[0] in ('T', 'BT', 'BADM'): result.append(p)
        if p[0] == 'DONE': result.append(p[:3] + p[5:9] + p[14:16])
        if p[0] == 'IMAGE_CACHE_TURN': result.append([v for v in p if not v.startswith('elapsed_ms=')])
    if len([r for r in result if r[0] == 'IMAGE_CACHE_TURN']) != 10:
        raise ValueError('incomplete public replay')
    return result


def validate_fixture(directory, manifest):
    # CPU-only, including ACTUAL expanded IDs/grid/identity and control coverage.
    from tools.q36_multimodal_cache_fixture import metadata, classify
    rows = manifest['rows']
    expected = ['native_exact'] * 3 + ['rendered_history'] * 3 + ['system_rewrite'] * 2 + ['changed_image'] * 2
    if [r['case'] for r in rows] != expected: raise ValueError('incomplete bounded replay cases')
    lines = (directory / 'image-cache.txt').read_text().splitlines()
    actual, case = [], None
    for line in lines:
        if line.startswith('# case '): case = line[7:]; continue
        if line: actual.append((case, line.split('\t')))
    if len(actual) != 10: raise ValueError('native fixture row count differs')
    old, previous_case = None, None
    for row, (case, fields) in zip(rows, actual):
        if case != previous_case: old = None
        previous_case = case
        if fields != [row['mode'], row['expect'], row['key'], row['emb'], ','.join(map(str, row['ids']))]:
            raise ValueError('native plan differs from audited expanded IDs/metadata')
        ids = row['ids']; path = Path(row['emb']).resolve()
        if case != row['case'] or not 2 < len(ids) <= 4096 or not path.is_relative_to(directory.resolve()) or not re.fullmatch('[0-9a-f]{16}', row['key']):
            raise ValueError('invalid short native replay input')
        images = [metadata(ids, path, row['key'])]
        if images != row['images'] or classify(old, ids, images) != row['audit']:
            raise ValueError('expanded token/image audit mismatch')
        if row['expect'] == 'hit' and row['audit']['classification'] != 'exact_prefix':
            raise ValueError('false native constructed-prefix claim')
        old = (ids, images)
    if rows[7]['audit']['classification'] != 'token_rewrite' or rows[-1]['audit']['classification'] != 'token_prefix_image_mismatch':
        raise ValueError('rewrite/changed-image control missing')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine', required=True, type=Path); p.add_argument('--model', required=True, type=Path)
    p.add_argument('--fixtures', required=True, type=Path); p.add_argument('--out', required=True, type=Path)
    p.add_argument('--preflight', action='store_true', help='CPU checks only, allowed before approved service window')
    a = p.parse_args()
    if ROOT != REVIEW or (not a.preflight and os.environ.get('STRATA_MM_CACHE_REVIEW_WINDOW') != '1'):
        p.error('only an explicitly approved nested multimodal-cache review checkout/window')
    if not a.engine.resolve().is_relative_to(REVIEW) or not a.out.resolve().is_relative_to(STAGE) or a.out.resolve()==STAGE:
        p.error('isolated review binary and NEW staging/results path required; live checkpoint-policy build forbidden')
    if not a.fixtures.resolve().is_relative_to(STAGE): p.error('fixtures must be staged under strata-checkpoints')
    binary = a.engine.read_bytes()
    if b'--kv-image-cache-test' not in binary or b'--kv-cache-diagnostics' not in binary:
        p.error('build the new diagnostic executable')
    manifest = json.loads((a.fixtures / 'manifest.json').read_text())
    if manifest['kind'] != 'q36-image-cache-v1' or not manifest['real_tokenizer_template'] or not manifest['python_prepare_off_on_exact']:
        p.error('prepare/verify actual GGUF/template/encoder fixtures before the service window')
    for name, sha in manifest['files'].items():
        path = (a.fixtures / name).resolve()
        if not path.is_relative_to(a.fixtures.resolve()) or digest(path) != sha: p.error('fixture hash/path changed')
    validate_fixture(a.fixtures, manifest)
    st = a.model.stat()
    if manifest['model'] != dict(path=str(a.model.resolve()), size=st.st_size, mtime_ns=st.st_mtime_ns): p.error('model provenance differs')
    if shutil.disk_usage(STAGE).free < 8 * 1024**3: p.error('reserve >=8GiB free staging disk for raw state artifacts')
    if a.out.exists(): p.error('results directory already exists; preserve old artifacts')
    if a.preflight:
        print('Q36_MULTIMODAL_CACHE_PREFLIGHT PASS model_not_started=1'); return
    no_models(); a.out.mkdir(parents=True, exist_ok=False)
    def interrupted(signum, _frame): raise KeyboardInterrupt('signal ' + str(signum))
    signal.signal(signal.SIGINT, interrupted); signal.signal(signal.SIGTERM, interrupted)
    command = [str(a.engine.resolve()), '--model', str(a.model.resolve()), '--serve', '--vision', '--max-context', '262144',
               '--batch', '4', '--kv-pool-tokens', '524288', '--kv', 'int8', '--mtp', '2', '--batch-mtp', '--mtp-sampled',
               '--chunk', '512', '--kv-checkpoint-mib', '1024', '--kv-checkpoint-interval', '1024', '--kv-checkpoint-policy', 'adaptive',
               '--kv-preallocate', '--kv-vision-max-tokens', '1024', '--kv-image-cache-test', str((a.fixtures / 'image-cache.txt').resolve())]
    report = dict(binary_sha256=digest(a.engine), inputs=manifest, status='PENDING', independent_image_reference=False,
                  model_window_budget_s=300, cause_of_reported_OWUI_misses='NOT_ESTABLISHED')
    (a.out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    deadline = time.monotonic() + 300
    try:
        off = run(command + ['--test-state-dump', str((a.out / 'off.bin').resolve())], a.out, 'off', deadline)
        on = run(command + ['--test-state-dump', str((a.out / 'on.bin').resolve()), '--kv-cache-diagnostics'], a.out, 'on', deadline)
        if 'Q36CACHE candidate' in off: raise RuntimeError('diagnostics emitted while disabled')
        for marker in ('Q36CACHE candidate', 'reason=exact_prefix', 'reason=image_mismatch',
                       'mode=explicit_batch', 'mode=engine_selected', 'checkpoint_fallback=image_request_bypass'):
            if marker not in on: raise RuntimeError('missing diagnostic coverage: ' + marker)
        if signature(off) != signature(on):
            raise RuntimeError('OFF/ON emitted protocol/turn selection differs: EXACT required, stop')
        if not exact(a.out / 'off.bin', a.out / 'on.bin'):
            offset = first_difference(a.out / 'off.bin', a.out / 'on.bin')
            raise RuntimeError(f'OFF/ON state/logit/schedule artifact differs at byte={offset}: EXACT required, no tolerance')
        for name, sha in manifest['files'].items():
            if digest(a.fixtures / name) != sha: raise RuntimeError('fixture changed during replay')
        report.update(status='PASS_DIAGNOSTIC_NEUTRALITY_ONLY', exact_state_logits_output_schedule=True)
        print('Q36_MULTIMODAL_CACHE_NEUTRALITY PASS independent_image_reference=0 cause_of_live_misses=unresolved', flush=True)
    except BaseException as exc:
        report.update(status='FAIL', error=str(exc)); raise
    finally:
        (a.out / 'result.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__': main()
