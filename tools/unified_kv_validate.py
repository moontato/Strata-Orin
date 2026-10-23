#!/usr/bin/env python3
"""Native engine gate/benchmark. No tokenizer, GPU emulation, KV files or proxy.

Example (same arguments in both arms, graphs/MMQ/MTP left enabled):
  python3 tools/unified_kv_validate.py --exe build-orin/strata-q36 --compare \
    --lengths 4096 --gen 1024 --out results/kv-short -- \
    --model MODEL.gguf --max-context 8192 --batch 4 --mtp 2 --batch-mtp

Long-context and LRU commands are in docs/UNIFIED_KV.md. Results are raw native
protocol, timings and memory measurements, not evidence until run on the device.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time


class NativeError(RuntimeError):
    pass


def fields(line):
    result = {}
    for word in line.split()[1:]:
        key, sep, value = word.partition('=')
        if sep:
            result[key] = int(value) if value.isdigit() else value
    return result


def memory(pid):
    result = {}
    for path, keys in [(Path('/proc/meminfo'), {'MemAvailable', 'SwapFree'}),
                       (Path(f'/proc/{pid}/status'), {'VmRSS', 'VmSwap'})]:
        try:
            for line in path.read_text().splitlines():
                key, _, rest = line.partition(':')
                if key in keys:
                    result[key + '_bytes'] = int(rest.split()[0]) * 1024
        except (OSError, ValueError, IndexError):
            pass
    return result


class Native:
    def __init__(self, exe, args, prefix, timeout):
        self.timeout = timeout
        self.stderr = Path(str(prefix) + '.stderr.log').open('w')
        self.stdout = Path(str(prefix) + '.stdout.log').open('w')
        self.proc = subprocess.Popen([exe, '--serve', *args], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=self.stderr, text=True, bufsize=1)
        self.lines = queue.Queue()
        self.info, self.pool, self.sessions, self.physical = {}, {}, {}, {}
        self.events, self.commits = [], []
        def pump():
            for line in self.proc.stdout:
                self.lines.put(line.strip())
            self.lines.put(None)
        threading.Thread(target=pump, daemon=True).start()
        try:
            while True:
                line = self.recv()
                if line.startswith('READY '):
                    self.context = int(line.split()[1])
                    break
            if int(self.info.get('batch_slots', 0)) != 4:
                raise NativeError('gate requires exactly four native slots; no silently reduced batch')
        except BaseException:
            self.close()
            raise

    def send(self, line):
        self.proc.stdin.write(line + '\n')
        self.proc.stdin.flush()

    def recv(self, allow_error=False):
        try:
            line = self.lines.get(timeout=self.timeout)
        except queue.Empty:
            raise NativeError('native engine timed out; inspect stderr/thermals, do not call it a pass')
        if line is None:
            raise NativeError(f'native engine exited ({self.proc.poll()}); inspect stderr')
        self.stdout.write(line + '\n'); self.stdout.flush()
        if line.startswith('INFO '): self.info.update(fields(line))
        if line.startswith('KVPOOL '): self.pool = fields(line)
        if line.startswith('KVMEM '): self.physical = fields(line)
        if line.startswith('KVSESSION '):
            data = fields(line); self.sessions[int(data['session'])] = data
        if line.startswith('KVEVICT '): self.events.append(fields(line))
        if line.startswith('KVCOMMIT '): self.commits.append(fields(line))
        if line.startswith(('ERR ', 'BERR ')) and not allow_error:
            raise NativeError(line)
        return line

    def status(self):
        if not self.info.get('unified_kv'):
            return {'memory': memory(self.proc.pid)}
        self.send('KVSTATUS')
        while not self.recv().startswith('KVEND '):
            pass
        return {'pool': dict(self.pool), 'sessions': {str(k): dict(v) for k, v in self.sessions.items()},
                'memory': memory(self.proc.pid), 'physical_budget': dict(self.physical)}

    def reset(self):
        self.status()
        for slot, data in sorted(self.sessions.items()):
            self.send(f'KVRESET {slot} {data["generation"]}')
            while not self.recv().startswith('KVRESET '):
                pass
        status = self.status()
        if self.info.get('unified_kv') and status['pool']['committed_bytes'] != 0:
            raise NativeError('reset did not reclaim physical KV')

    def batch(self, prompts, gen):
        """Queue admissions together; tokens of other slots may arrive during prefill."""
        result = {b: {'tokens': [], 'prompt_tokens': len(ids)} for b, ids in prompts.items()}
        order = list(prompts)
        admission = 0
        done = set()
        started = time.monotonic()
        for b, ids in prompts.items():
            if len(ids) + gen > self.context:
                raise NativeError('workload exceeds logical context; it must not be shortened')
            self.send(f'BGEN {b} {gen} temperature=0 ' + ','.join(map(str, ids)))
        while len(done) < len(order) or admission < len(order):
            line = self.recv()
            f = line.split()
            if f[:1] == ['T']:
                row = result[order[admission]]
                row['tokens'].append(int(f[1])); row.setdefault('first_token_seconds', time.monotonic() - started)
            elif f[:1] == ['BT']:
                result[int(f[1])]['tokens'].append(int(f[2]))
            elif f[:1] == ['DONE']:
                b = order[admission]
                result[b].update(prefill_ms=float(f[3]), first_decode_ms=float(f[4]), admission_finish=f[5],
                                 reused_tokens=int(f[8]) if len(f) > 8 else 0, native_done=line)
            elif f[:1] == ['BADM']:
                b = int(f[1])
                if b != order[admission]: raise NativeError('admission ownership/order mismatch')
                if f[2] == '0':
                    done.add(b)
                    result[b]['finish'] = result[b]['admission_finish']
                    result[b]['finish_seconds'] = time.monotonic() - started
                admission += 1
            elif f[:1] == ['BDONE']:
                b = int(f[1]); done.add(b)
                result[b].update(finish=f[3], decode_ms=float(f[4]), native_bdone=line,
                                 finish_seconds=time.monotonic() - started)
                if f[3] not in ('stop', 'length'): raise NativeError('unexpected batch termination: ' + line)
        elapsed = time.monotonic() - started
        for b, row in result.items():
            if row.get('finish') == 'length' and len(row['tokens']) != gen:
                raise NativeError('successful length completion silently shortened output')
            row['prompt_sha256'] = hashlib.sha256(','.join(map(str, prompts[b])).encode()).hexdigest()
        return {'slots': result, 'wall_seconds': elapsed,
                'aggregate_end_to_end_tok_s': sum(len(r['tokens']) for r in result.values()) / elapsed,
                'status': self.status()}

    def close(self):
        if self.proc.poll() is None:
            try:
                for b in range(4): self.send(f'BSTOP {b}')
                self.send('STOP'); self.proc.stdin.close()
                self.proc.wait(timeout=30)
            except (OSError, subprocess.TimeoutExpired):
                self.proc.terminate()
                try: self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired: self.proc.kill(); self.proc.wait()
        self.stdout.close(); self.stderr.close()


def prompt(b, n, nonce=0):
    if n < 2: raise NativeError('use at least two prompt tokens')
    # Distinct case prefixes prevent the legacy checkpoint cache from giving
    # the reference arm hits after the unified arm's explicit KVRESET.
    return [80 + b + 8 * nonce, 198] + [198] * (n - 2)


def workload(engine, lengths, gen, mixed, eviction, pool_tokens):
    report = []
    for case, n in enumerate(lengths):
        engine.reset()
        row = engine.batch({0: prompt(0, n, case + 1)}, gen)
        row['case'] = f'solo-{n}'; report.append(row)
    if mixed:
        for case, total in enumerate(([250000,150000,70000,30000], [250000,250000,0,0],
                                     [125000]*4, [250000,10000,5000,0])):
            engine.reset()
            row = engine.batch({b: prompt(b, t-gen, 100 + case) for b, t in enumerate(total) if t}, gen)
            row['case'] = 'mixed-' + '-'.join(map(str, total)); report.append(row)
            if engine.info.get('unified_kv'):
                expected = sum(t-gen for t in total if t)
                if row['status']['pool']['resident_tokens'] < expected:
                    raise NativeError('mixed workload completed but requested residency was NOT achieved; inspect physical pressure/evictions')
    else:
        engine.reset()
        row = engine.batch({b: prompt(b, lengths[0], 99) for b in range(4)}, gen)
        row['case'] = 'four-parallel'; report.append(row)
    if eviction:
        if pool_tokens != 2 * engine.context:
            raise NativeError('deterministic LRU gate requires pool_tokens = 2 * max_context')
        engine.reset()
        c = engine.context
        sizes = {1: int(c*.60)-gen, 0: int(c*.28)-gen, 2: c-max(4*gen, c//250)-gen, 3: max(gen+2, int(c*.016))-gen}
        held = {}
        for b in [0,1,2,3]:
            ids = prompt(b, sizes[b], 200); row = engine.batch({b: ids}, gen)
            held[b] = ids + row['slots'][b]['tokens']; row['case'] = f'lru-fill-{b}'; report.append(row)
            if b == 1:
                # Touch 0 while there is room for Flash-Next's temporary main
                # copy. Victims must now be 1 then 0, NOT allocation order 0,1.
                row = engine.batch({0: held[0] + [198]}, gen)
                held[0] += [198] + row['slots'][0]['tokens']
                row['case'] = 'lru-touch-0'; report.append(row)
        start = len(engine.events)
        row = engine.batch({3: prompt(3, c-gen, 201)}, gen); row['case'] = 'lru-pressure'; report.append(row)
        victims = engine.events[start:]
        offset = int(engine.info.get('batch_session_offset', 0))
        if [v['session'] for v in victims[:2]] != [1+offset, offset]:
            raise NativeError('LRU order not proved (or memory pressure occurred before the gate): ' + str(victims))
        state = engine.status()['sessions'][str(1+offset)]
        engine.send(f'BGEN 1 {gen} kv_resume=1 kv_generation={state["generation"]} kv_epoch={state["epoch"]} ' + ','.join(map(str, held[1]+[198])))
        while True:
            line = engine.recv(allow_error=True)
            if line.startswith('ERR '):
                if 'kv_cache_miss' not in line: raise NativeError(line)
                break
            if line.startswith(('T ', 'DONE ', 'BADM ')): raise NativeError('evicted strict resume was accepted')
        row = engine.batch({1: held[1]+[198]}, gen)  # explicit full-history replay, no kv_resume claim
        row['case'] = 'lru-replay'; report.append(row)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--lengths', default='4096,32768,131072,249968')
    p.add_argument('--gen', type=int, default=32); p.add_argument('--pool-tokens', type=int, default=500000)
    p.add_argument('--pool-mib', type=int); p.add_argument('--timeout', type=float, default=3600)
    p.add_argument('--mixed', action='store_true'); p.add_argument('--eviction', action='store_true')
    p.add_argument('--compare', action='store_true', help='run without pool first; compare exact tokens')
    p.add_argument('--reference-exe', help='optional Jetson baseline binary for the non-pool arm (default: same binary)')
    p.add_argument('args', nargs=argparse.REMAINDER)
    o = p.parse_args()
    if o.gen < 1 or o.pool_tokens < 1 or o.timeout <= 0: p.error('generation, pool and timeout limits must be positive')
    args = o.args[1:] if o.args[:1] == ['--'] else o.args
    if '--kv-pool-tokens' in args or '--kv-pool-mib' in args or '--serve' in args:
        p.error('supply base engine args without --serve or KV pool flags')
    if o.compare and o.eviction: p.error('the eviction gate has no non-pool equivalent; run separately')
    if o.reference_exe and not o.compare: p.error('--reference-exe requires --compare')
    o.out.parent.mkdir(parents=True, exist_ok=True)
    report = {'platform': os.uname()._asdict() if hasattr(os, 'uname') and hasattr(os.uname(), '_asdict') else str(os.uname()) if hasattr(os, 'uname') else os.name,
              'command': [o.exe, '--serve', *args], 'pool_tokens': o.pool_tokens, 'runs': {}}
    try:
        for mode in (['legacy', 'unified'] if o.compare else ['unified']):
            extra = ['--kv-pool-tokens', str(o.pool_tokens)] if mode == 'unified' else []
            if mode == 'unified' and o.pool_mib: extra += ['--kv-pool-mib', str(o.pool_mib)]
            exe = o.reference_exe if mode == 'legacy' and o.reference_exe else o.exe
            report[mode+'_command'] = [exe, '--serve', *args, *extra]
            e = Native(exe, args + extra, Path(str(o.out) + '.' + mode), o.timeout)
            try:
                report['runs'][mode] = workload(e, [int(n) for n in o.lengths.split(',')], o.gen, o.mixed, o.eviction, o.pool_tokens)
            finally:
                report[mode+'_events'] = e.events; report[mode+'_commits'] = e.commits; report[mode+'_info'] = e.info
                e.close()
        if o.compare:
            for old, new in zip(report['runs']['legacy'], report['runs']['unified']):
                for b in old['slots']:
                    if old['slots'][b]['tokens'] != new['slots'][b]['tokens']:
                        raise NativeError(f'exact token mismatch: {old["case"]}, slot {b}; run teacher-forced logits gate')
        report['result'] = 'passed'
    except NativeError as error:
        report['result'] = 'not_passed'; report['error'] = str(error)
    o.out.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(report['result'], report.get('error', ''), o.out.with_suffix('.json'))
    return 0 if report['result'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
