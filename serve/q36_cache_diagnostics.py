"""Optional, bounded, content-free Q36 cache observations. Not a routing policy.
STRATA_Q36_CACHE_DIAGNOSTICS=1 enables Python preparation/slot observations;
the native flag is separate. No raw IDs, image/key digests, URLs or text logged.
"""
import functools
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time


def enabled():
    return os.environ.get('STRATA_Q36_CACHE_DIAGNOSTICS') == '1'


def lcp(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def emit(stage, **fields):
    try:
        print('Q36CACHE_PY ' + json.dumps(dict(stage=stage, **fields), sort_keys=True), file=sys.stderr, flush=True)
    except Exception:
        pass  # a diagnostic sink failure must not fail the request


def image_metadata(ids, encoded, keys, pad):
    # Preparation already validated these files. Only diagnostic errors are caught
    # by note_prepare; no extra admission condition or input mutation is introduced.
    if len(encoded) > 8:
        return None
    out, at = [], 0
    for i, (path, count) in enumerate(encoded):
        at = ids.index(pad, at)
        with Path(path).open('rb') as f:
            header = f.read(20)
            magic, n, nx, ny, width = struct.unpack('<5i', header)
            if magic != 0x31455653 or n != count or n != nx * ny or n * width * 4 > 64 * 1024 * 1024:
                return None
            h = hashlib.sha256()
            remaining = n * width * 4
            while remaining:
                chunk = f.read(min(65536, remaining))
                if not chunk:
                    return None
                h.update(chunk); remaining -= len(chunk)
            if f.read(1):
                return None
        out.append(dict(cell=at, rows=n, nx=nx, ny=ny, width=width, key=keys[i] if keys else None, payload=h.digest()))
        at += n
    return out


def image_flags(a, b):
    if a is None or b is None:
        return dict(details=False)
    return dict(details=True, count_diff=len(a) != len(b),
                placement_diff=any((x['cell'], x['rows']) != (y['cell'], y['rows']) for x, y in zip(a, b)),
                grid_diff=any((x['nx'], x['ny'], x['width']) != (y['nx'], y['ny'], y['width']) for x, y in zip(a, b)),
                key_diff=any(x['key'] != y['key'] for x, y in zip(a, b)),
                rows_diff=any(x['payload'] != y['payload'] for x, y in zip(a, b)))


def note_prepare(service, encoded_ids, ids, messages, kwargs, encoded, keys):
    if not enabled():
        return
    try:
        # Four bounded *prepared* prompts, NOT the native retained/generated stream.
        if len(ids) > 262144 or len(encoded_ids) > 262144:
            emit('prepare', unavailable=True, reason='diagnostic_bound'); return
        pad = service.tok.encode('<|image_pad|>', parse_special=True)[0] if encoded else -1
        meta = image_metadata(ids, encoded, keys, pad)
        history = getattr(service, '_q36_cache_observed', ())
        for index, old in enumerate(history[-4:]):
            emit('prepare_candidate', previous=index, comparison='prepared_not_native_retained',
                 old_tokens=len(old[1]), incoming=len(ids), unexpanded_lcp=lcp(old[0], encoded_ids), expanded_lcp=lcp(old[1], ids),
                 proper_token_prefix=len(old[1]) < len(ids) and lcp(old[1], ids) == len(old[1]), **image_flags(old[2], meta))
        emit('prepare', unexpanded=len(encoded_ids), expanded=len(ids), images=len(encoded),
             reasoning_messages=sum(bool(m.get('reasoning_content')) for m in messages),
             thinking_enabled=kwargs.get('enable_thinking', True) is not False, image_details=meta is not None)
        service._q36_cache_observed = (*history[-3:], (tuple(encoded_ids), tuple(ids), meta))
    except Exception:
        emit('prepare', unavailable=True)  # never log exception text/paths/content


def slot_probe(function):
    @functools.wraps(function)
    def observed(engine, prompt):
        selected = function(engine, prompt)  # original selector runs exactly once
        if enabled():
            try:
                # pick_slot can be polled repeatedly while all slots are busy.
                # At most eight complete observations + one suppression notice
                # per engine per20s; this budget NEVER feeds the selector.
                now = time.monotonic()
                start, count = getattr(engine, '_q36_cache_slot_budget', (now, 0))
                if now - start >= 20: start, count = now, 0
                engine._q36_cache_slot_budget = (start, count + 1)
                if count >= 8:
                    if count == 8: emit('slot_suppressed', reason='8_observations_per_20s')
                    return selected
                for slot in list(engine.slot_order)[:8]:
                    held = engine.slot_held[slot]
                    k = len(held) - 1 - held[::-1].index(engine.TURN_ID) if engine.TURN_ID in held else 0
                    emit('slot_candidate', slot=slot, busy=bool(engine.slot_busy[slot]), retained=len(held),
                         incoming=len(prompt), lcp=lcp(held, prompt),
                         complete_prefix=bool(held) and len(held) < len(prompt) and lcp(held, prompt) == len(held),
                         turn_prefix=k if 0 < k < len(prompt) and prompt[:k] == held[:k] else 0)
                emit('slot_selected', slot=selected, native_destination='explicit_for_BGEN_or_BGENI')
            except Exception:
                emit('slot_selected', unavailable=True)
        return selected  # diagnostics NEVER change the original selector result
    return observed


def dispatch_probe(function):
    @functools.wraps(function)
    def observed(engine, text):
        if enabled():
            try:
                kind = text.partition(' ')[0]
                if kind in ('GEN', 'BGEN', 'GENI', 'BGENI'):
                    parts = text.split(' ', 2)
                    count = text.rsplit(' ', 1)[-1].count(',') + 1
                    # Only counts/mode/destination, NEVER the serialized command.
                    emit('dispatch', protocol=kind, explicit_destination=kind.startswith('B'),
                         destination=int(parts[1]) if kind.startswith('B') else -1,
                         expanded_tokens=min(count, 262145), image_request=kind.endswith('I'),
                         keys_present=' img_keys=' in text)
            except Exception:
                emit('dispatch', unavailable=True)
        return function(engine, text)  # unchanged command, exactly one send
    return observed
