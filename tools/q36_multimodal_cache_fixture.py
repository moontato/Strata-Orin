#!/usr/bin/env python3
"""CPU-only public fixture preparation using real GGUF/template and stored actual
SVE1 encoder output. Calls real OpenAI conversion + Service.prepare; frozen encoder
IO does NOT test encoder/runtime routing/HTTP. No GPU, model load or downloads.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import threading
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from tools.q36_vision_fixture import sve


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_images(out):
    # Same public geometry, different actual pixels; no URL/network access.
    from PIL import Image
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    Image.new('RGB', (1024, 1024), (220, 20, 20)).save(out / 'red-square.png')
    Image.new('RGB', (1024, 1024), (20, 20, 220)).save(out / 'blue-square.png')


def identity(path, key):
    data = Path(path).read_bytes(); _, n, nx, ny, _ = struct.unpack('<5i', data[:20])
    h = 1469598103934665603
    for byte in struct.pack('<3q', n, nx, ny) + data[20:] + struct.pack('<Q', int(key, 16)):
        h = ((h ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return h


def metadata(ids, path, key):
    info = sve(path); pads = [i for i, t in enumerate(ids) if t == 248056]
    if len(pads) != info['tokens'] or pads != list(range(pads[0], pads[0] + len(pads))):
        raise ValueError('need exactly one actual expanded image-pad run')
    return dict(cell=pads[0], n=len(pads), nx=info['nx'], ny=info['ny'], key=key,
                payload_sha256=hashlib.sha256(Path(path).read_bytes()[20:]).hexdigest(), identity=identity(path, key))


def classify(old, ids, images):
    if old is None:
        return dict(classification='new', lcp=0, retained=0, incoming=len(ids), complete_image_boundary=True)
    previous, previous_images = old
    common = next((i for i, (x, y) in enumerate(zip(previous, ids)) if x != y), min(len(previous), len(ids)))
    complete = not any(x['cell'] < len(previous) < x['cell'] + x['n'] for x in images)
    prefix = [x for x in images if x['cell'] + x['n'] <= len(previous)]
    same_images = complete and [(x['cell'], x['n'], x['identity']) for x in prefix] == [(x['cell'], x['n'], x['identity']) for x in previous_images]
    proper = len(previous) < len(ids) and common == len(previous)
    label = 'exact_prefix' if proper and same_images else 'token_prefix_image_mismatch' if proper else 'identical_tokens' if previous == ids else 'token_rewrite'
    return dict(classification=label, lcp=common, retained=len(previous), incoming=len(ids),
                complete_image_boundary=complete, image_prefix_equal=same_images,
                note='comparison to previous fixture prompt, not proof of a live generated retained stream')


def frozen_service(tokenizer, renderer, directory, sources):
    from serve.server import Service
    from serve.frontend import literal_tags
    service = object.__new__(Service)
    service.engine = SimpleNamespace(max_context=262144)
    service.tok = tokenizer; service.template = renderer; service.prompts = None; service.effort_end = False
    service.literals = literal_tags(getattr(tokenizer, 'control_tokens', ()))
    service.embeddings = threading.local(); service.fifo = threading.Lock(); service.fit_max_tokens = False
    service._vision_up = lambda: None  # frozen encoder IO ONLY, no process started
    service.vision = SimpleNamespace(dir=Path(directory), encode_all=lambda images, cancel: [sources[x] for x in images])
    return service


def prepare(model, image_a, image_b, sve_a, sve_b, out, template=None):
    from tools.gguf_reader import GGUFFile
    from tools.strata_tokenizer import Tokenizer
    from serve.frontend import ChatTemplate, openai_to_messages
    model = Path(model).resolve(); out = Path(out).resolve()
    a, b = sve(sve_a), sve(sve_b)
    if (a['tokens'], a['nx'], a['ny']) != (b['tokens'], b['nx'], b['ny']) or digest(sve_a) == digest(sve_b):
        raise ValueError('changed-image control needs DIFFERENT actual embeddings with IDENTICAL grid/row geometry')
    for p in (out, sve_a, sve_b, image_a, image_b):
        if any(c.isspace() for c in str(p)):
            raise ValueError('native fixture paths must not contain whitespace')
    out.mkdir(parents=True, exist_ok=False)
    source = Path(template).read_text() if template else GGUFFile(model).metadata['tokenizer.chat_template']
    (out / 'template.jinja').write_text(source)
    tokenizer = Tokenizer.from_gguf(model); renderer = ChatTemplate(out / 'template.jinja')
    if tokenizer.encode('<|image_pad|>', parse_special=True) != [248056]:
        raise ValueError('requires actual Qwen3.6 image-pad vocabulary')
    directory = out / 'encoded'; directory.mkdir()
    sources = {}; records = []
    for image, emb in ((image_a, sve_a), (image_b, sve_b)):
        image = Path(image).resolve(); key = digest(image)[:32]; target = directory / (key + '.sve')
        shutil.copyfile(emb, target); sources[image.as_uri()] = (target, a['tokens']); records.append((target, key[:16]))
    service = frozen_service(tokenizer, renderer, directory, sources)
    first = dict(messages=[dict(role='system', content='Public stable system context. Describe this public synthetic square.'),
                dict(role='user', content=[dict(type='image_url', image_url=dict(url=Path(image_a).resolve().as_uri())),
                                          dict(type='text', text='What color is the square?')])],
                 chat_template_kwargs=dict(enable_thinking=False))
    second = copy.deepcopy(first); second['messages'] += [dict(role='assistant', content='The square is red.', reasoning_content='I examined the public square.'),
                                                         dict(role='user', content='Which color did I send?')]
    third = copy.deepcopy(second); third['messages'] += [dict(role='assistant', content='You sent red.', reasoning_content='The same historical image is present.'),
                                                       dict(role='user', content='Name that same color again.')]
    rewrite = copy.deepcopy(second); rewrite['messages'][0]['content'] = 'A changed earlier system context. Describe the public square.'
    old_env = os.environ.get('STRATA_Q36_CACHE_DIAGNOSTICS')
    prepared = []
    try:
        for index, req in enumerate((first, second, third, rewrite)):
            pair = []
            for diagnostic in ('0', '1'):
                os.environ['STRATA_Q36_CACHE_DIAGNOSTICS'] = diagnostic
                messages, tools, kwargs = openai_to_messages(copy.deepcopy(req))
                ids, thinking, maximum = service.prepare(messages, tools, kwargs, max_new=1)
                path = Path(service.embeddings.path); keys = list(service.embeddings.keys)
                pair.append((ids, thinking, maximum, keys, path.read_bytes())); path.unlink()
            if pair[0] != pair[1]:
                raise ValueError('Python diagnostics changed prepared IDs/keys/image file bytes')
            ids, _, _, keys, payload = pair[0]; target = out / f'prepared-{index}.sve'; target.write_bytes(payload)
            if not 2 < len(ids) <= 4096 or len(keys) != 1:
                raise ValueError('short fixture requires one image and <=4096 expanded tokens')
            prepared.append((ids, keys[0], target, [metadata(ids, target, keys[0])]))
    finally:
        if old_env is None: os.environ.pop('STRATA_Q36_CACHE_DIAGNOSTICS', None)
        else: os.environ['STRATA_Q36_CACHE_DIAGNOSTICS'] = old_env
    base, key, emb, im = prepared[0]
    suffix = tokenizer.encode('\nPublic native constructed continuation: amber birch cedar.\n', parse_special=False)
    if len(suffix) < 2:
        raise ValueError('native control requires multi-token suffix')
    changed_path, changed_key = records[1]
    rows = []
    def add(case, mode, expect, kind, ids, key, emb, images, old):
        if not 2 < len(ids) <= 4096:
            raise ValueError('constructed control exceeds the short replay bound')
        rows.append(dict(case=case, mode=mode, expect=expect, kind=kind, ids=ids, key=key, emb=str(emb), images=images,
                         audit=classify(old, ids, images)))
    add('native_exact', 'auto', 'miss', 'native_constructed_not_API', base, key, emb, im, None)
    add('native_exact', 'auto', 'hit', 'native_constructed_not_API', base + suffix, key, emb, im, (base, im))
    add('native_exact', '1', 'hit', 'native_constructed_not_API', base + suffix * 2, key, emb, im, (base + suffix, im))
    old = None
    for ids, k, path, images in prepared[:3]:
        add('rendered_history', 'auto', 'observe', 'real_Service.prepare_canned_public_history_not_HTTP', ids, k, path, images, old)
        old = (ids, images)
    add('system_rewrite', 'auto', 'miss', 'real_Service.prepare', base, key, emb, im, None)
    ids, k, path, images = prepared[3]
    add('system_rewrite', 'auto', 'miss', 'real_Service.prepare', ids, k, path, images, (base, im))
    add('changed_image', 'auto', 'miss', 'native_constructed_not_API', base, key, emb, im, None)
    changed = [metadata(base + suffix, changed_path, changed_key)]
    add('changed_image', 'auto', 'miss', 'native_constructed_not_API', base + suffix, changed_key, changed_path, changed, (base, im))
    if rows[1]['audit']['classification'] != 'exact_prefix' or rows[2]['audit']['classification'] != 'exact_prefix' or rows[7]['audit']['classification'] != 'token_rewrite':
        raise ValueError('actual expanded IDs did not isolate native continuation/system rewrite controls')
    if rows[-1]['audit']['classification'] != 'token_prefix_image_mismatch':
        raise ValueError('changed-image negative control does not isolate image identity')
    lines = []; case = None
    for row in rows:
        if row['case'] != case:
            case = row['case']; lines.append('# case ' + case)
        lines.append('\t'.join([row['mode'], row['expect'], row['key'], row['emb'], ','.join(map(str, row['ids']))]))
    (out / 'image-cache.txt').write_text('\n'.join(lines) + '\n')
    (out / 'public-requests.json').write_text(json.dumps([first, second, third, rewrite], indent=2) + '\n')
    st = model.stat()
    manifest = dict(kind='q36-image-cache-v1', real_tokenizer_template=True, frozen_actual_encoder_IO=True,
                    python_prepare_off_on_exact=True, independent_image_reference=False,
                    model=dict(path=str(model), size=st.st_size, mtime_ns=st.st_mtime_ns), rows=rows,
                    files={str(p.relative_to(out)): digest(p) for p in sorted(out.rglob('*')) if p.is_file()},
                    input_sha256={str(Path(p).resolve()): digest(p) for p in (image_a, image_b, sve_a, sve_b)},
                    note='public synthetic API preparation, not actual OWUI/HTTP or generated assistant history; native controls separate')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--make-images'); p.add_argument('--model'); p.add_argument('--image-a'); p.add_argument('--image-b')
    p.add_argument('--sve-a'); p.add_argument('--sve-b'); p.add_argument('--out'); p.add_argument('--template')
    a = p.parse_args()
    if a.make_images: make_images(a.make_images)
    else:
        if not all((a.model, a.image_a, a.image_b, a.sve_a, a.sve_b, a.out)):
            p.error('need model, image-a/b, sve-a/b and out')
        m = prepare(a.model, a.image_a, a.image_b, a.sve_a, a.sve_b, a.out, a.template)
        print(json.dumps({k: v for k, v in m.items() if k != 'rows'}, indent=2))
