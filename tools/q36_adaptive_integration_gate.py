#!/usr/bin/env python3
"""FUTURE approved Orin window only. Sequential native tests; no service control,
HTTP, config changes or deployment. One owned model process at a time. Each policy
uses its OWN independent schedule reference. Stop on first unexpected failure.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import time
from q36_checkpoint_schedule import audit


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def guard():
    if os.environ.get('STRATA_CHECKPOINT_REVIEW_WINDOW') != '1':
        raise RuntimeError('requires separately approved, isolated checkpoint review window')
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            comm = (entry / 'comm').read_text().strip()
            cmd = (entry / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        except FileNotFoundError:
            continue
        except PermissionError as exc:
            raise RuntimeError('cannot prove model processes absent') from exc
        if comm in {'strata', 'strata-q36', 'strata-vision', 'llama-server', 'llama-cli', 'llama-bench'} or re.search(r'(?:^|\s)(?:\S*/)?serve/server\.py(?:\s|$)|-m serve\.server', cmd):
            raise RuntimeError(f'existing model/server/encoder process {entry.name}: {comm}; abort without stopping it')


def wait_unreaped(child, timeout):
    # Linux/Orin: retain the leader (WNOWAIT) until its group is killed, so a
    # reaped/recycled PID cannot identify somebody else's process group.
    deadline = time.monotonic() + timeout
    while True:
        info = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        if info is not None:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(child.args, timeout)
        time.sleep(min(0.05, remaining))


def run(command, out, name, timeout, negative=None):
    guard()
    log = out / (name + '.log')
    (out / (name + '.command.json')).write_text(json.dumps(command, indent=2) + '\n')
    started = time.monotonic()
    with log.open('wb') as stream:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'interrupted by {signum}')
        old = {s: signal.signal(s, interrupted) for s in (signal.SIGINT, signal.SIGTERM)}
        error = None
        try:
            wait_unreaped(child, timeout)
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            # Own, still-unreaped process group ONLY, including descendants.
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            rc = child.wait()
            (out / (name + '.exit.json')).write_text(json.dumps(dict(exit=child.returncode, error=error,
                elapsed_seconds=time.monotonic() - started)) + '\n')
            for s, handler in old.items():
                signal.signal(s, handler)
    text = log.read_text(errors='replace')
    if negative:
        expected = ('SCHEDULE_CORRUPT category=' + negative,
                    'SCHEDULE_FIRST comparison=restored category=' + negative,
                    'SCHEDULE_FAILURE comparison=restored')
        if rc != 1 or not all(x in text for x in expected) or 'CHECKPOINT_SCHEDULE_TEST PASS' in text:
            raise RuntimeError(f'{name}: negative control did not reject corrupted checkpoint at restored boundary')
    elif rc != 0 or 'CHECKPOINT_SCHEDULE_TEST PASS' not in text or 'SCHEDULE_FAILURE' in text:
        raise RuntimeError(f'{name}: native correctness failed; STOP, preserve {log}')
    return text


def summarize(text):
    rows = []
    for line in text.splitlines():
        if line.startswith('SCHEDULE_MEASURE '):
            d = dict(x.split('=', 1) for x in line.split()[1:])
            for key in ('turn', 'reused', 'replayed', 'copy_us', 'copy_bytes'):
                d[key] = int(d[key])
            for key in ('prefill_ms', 'latency_ms'):
                d[key] = float(d[key])
            rows.append(d)
    if not rows:
        raise ValueError('no measurement rows')
    groups = {}
    for phase in ('warmup', 'steady'):
        selected = [r for r in rows if r['phase'] == phase]
        if not selected:
            raise ValueError('missing ' + phase)
        groups[phase] = dict(turns=len(selected), **{k: sum(r[k] for r in selected) for k in
                            ('reused', 'replayed', 'copy_us', 'copy_bytes', 'prefill_ms', 'latency_ms')},
                            median_latency_ms=statistics.median(r['latency_ms'] for r in selected))
    return dict(phases=groups, turns=rows,
                startup_warmup_seconds=[float(x) for x in re.findall(r'warmed up in ([0-9.]+) s', text)],
                different_schedule_logit_failures=len(re.findall(r'SCHEDULE_LOGITS comparison=different_schedule .*exact=0', text)),
                note='Diagnostic first-token latency includes tested host snapshots; reference excluded. Not production throughput.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--engine', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--fixtures', type=Path, required=True)
    p.add_argument('--vision-max-tokens', type=int, required=True, help='EXACT existing qualified cap; 0 means context default')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--timeout', type=int, default=1800, help='per bounded native invocation')
    a = p.parse_args()
    guard()
    if not 0 <= a.vision_max_tokens <= 262144:
        raise ValueError('invalid existing vision cap')
    manifest = json.loads((a.fixtures / 'manifest.json').read_text())
    if manifest['kind'] != 'q36-adaptive-integration-v1' or manifest['byte_test_only']:
        raise ValueError('real GGUF/template fixtures required, not byte/mock inputs')
    for name, sha in manifest['files'].items():
        if Path(name).name != name or digest(a.fixtures / name) != sha:
            raise ValueError('fixture hash/name mismatch: ' + name)
    if audit((a.fixtures / 'replay.csv').read_text()) != manifest['audit']:
        raise ValueError('rendered token audit changed')
    st = a.model.stat()
    if manifest['model'] != dict(path=str(a.model.resolve()), size=st.st_size, mtime_ns=st.st_mtime_ns):
        raise ValueError('model provenance mismatch')
    a.out.mkdir(parents=True, exist_ok=False)
    settings = ['--model', str(a.model.resolve()), '--max-context', '262144', '--chunk', '512', '--batch', '4',
                '--kv-pool-tokens', '524288', '--kv', 'int8', '--kv-preallocate', '--mtp', '2', '--batch-mtp',
                '--mtp-sampled', '--kv-checkpoint-mib', '1024', '--kv-checkpoint-interval', '1024', '--vision', '--serve']
    if a.vision_max_tokens:
        settings += ['--kv-vision-max-tokens', str(a.vision_max_tokens)]
    report = dict(settings=settings, fixture_manifest=manifest, binary_sha256={str(x.resolve()): digest(x) for x in (a.baseline, a.engine)},
                  validation='PENDING', production_benefit='NOT_CLAIMED')
    (a.out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    smoke = str((a.fixtures / 'reference-smoke.csv').resolve())
    # Stage 1: independent reference + corruption sensitivity on non-adaptive base.
    base = [str(a.baseline.resolve())] + settings
    run(base + ['--kv-schedule-test', smoke], a.out, '01-baseline-reference', a.timeout)
    for control in ('conv', 'gdn', 'hlast'):
        run(base + ['--kv-schedule-test', smoke, '--kv-schedule-corrupt', control], a.out,
            '02-negative-' + control, a.timeout, negative=control)
    # Stage 2: same-day baseline/fixed equality; adaptive compared to its OWN
    # producer/suffix oracle, never the different-schedule fixed dump.
    report['replay'] = {}
    for arm, binary, policy in [('baseline', a.baseline, None), ('fixed', a.engine, 'fixed'), ('adaptive', a.engine, 'adaptive')]:
        command = [str(binary.resolve())] + settings
        if policy:
            command += ['--kv-checkpoint-policy', policy]
        dump = a.out / (arm + '.bin')
        text = run(command + ['--kv-schedule-test', str((a.fixtures / 'replay.csv').resolve()),
                              '--test-state-dump', str(dump.resolve())], a.out, '03-replay-' + arm, a.timeout)
        report['replay'][arm] = summarize(text)
        if not dump.is_file() or not dump.stat().st_size:
            raise RuntimeError('missing exact logit/pick artifact')
        if arm == 'fixed' and (a.out / 'baseline.bin').read_bytes() != dump.read_bytes():
            raise RuntimeError('default/fixed differs from vision-base reference build under the same schedules')
        (a.out / 'report-partial.json').write_text(json.dumps(report, indent=2) + '\n')
    report['validation'] = 'PASS_FOR_THIS_TEXT_FIXTURE_ONLY'
    report['unqualified'] = ['image inference/encoder co-residency', 'HTTP/OWUI', 'concurrent active decoding',
                             'sampled MTP (run existing separate native gate)', 'long-context pressure', 'other backends']
    (a.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    (a.out / 'SHA256SUMS').write_text(''.join(digest(f) + '  ' + f.name + '\n' for f in sorted(a.out.iterdir()) if f.is_file()))
    print('Q36_ADAPTIVE_INTEGRATION_ORIN PASS (bounded text replay only; benefit not claimed)')


if __name__ == '__main__':
    main()
