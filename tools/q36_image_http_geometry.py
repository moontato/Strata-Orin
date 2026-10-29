"""CPU-only binding of the HTTP positive case to the GPU-tested public fixture.

No encoder, listener, native engine, service operation, or synthetic tokenizer.
The canned-history geometry case is separate from the generated-history smoke.
"""
import base64
import copy
import hashlib
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def common(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def legal_checkpoints(ids, cell, rows, chunk=512, interval=1024):
    """First fresh request, four payloads, empty learning; mirror Q36ImagePlan.

    Only existing non-final callbacks can be anchors. The recent payload follows
    the production interval rule. No invented callback or altered chunk shape.
    """
    safe = [p for p in range(chunk, len(ids), chunk) if not cell < p < cell + rows]
    if not safe:
        return []
    def nearest(wanted):
        return max((p for p in safe if p <= wanted), default=safe[0])
    anchors = [nearest(interval << i) for i in range(3)]
    points = [0, 0, 0, 0]
    for pos in range(chunk, len(ids) + chunk, chunk):
        pos = min(pos, len(ids))
        if cell < pos < cell + rows or pos in points:
            continue
        index = next((i for i, p in enumerate(anchors) if p == pos and not points[i]), None)
        if index is None:
            last = max(points)
            if pos >= interval and pos >= last and pos - last >= interval:
                index = 3
        if index is not None:
            points[index] = pos
    return sorted(p for p in points if p)


def audit_ids(ids, cell, rows):
    if len(ids) != 3 or not all(2 < len(x) <= 6144 for x in ids):
        raise ValueError("need the three bounded actual-GGUF public requests")
    end = cell + rows
    candidates = legal_checkpoints(ids[0], cell, rows)
    audits = []
    for i in (1, 2):
        lcp = common(ids[0], ids[i])
        available = [p for p in candidates if end <= p <= lcp and p < len(ids[i])]
        if not available or lcp >= min(len(ids[0]), len(ids[i])):
            raise ValueError("no retained legal after-image checkpoint before a genuine rewrite")
        audits.append(dict(turn=i + 1, lcp=lcp, image_end=end,
                           available=available, expected_min_reuse=max(available)))
    return dict(callback_chunk=512, checkpoint_interval=1024, checkpoint_count=4,
                initial_retained_checkpoints=candidates, rewrites=audits)


def file_identity(path):
    p = Path(path).resolve()
    s = p.stat()
    return dict(path=str(p), size=s.st_size, mtime_ns=s.st_mtime_ns)


def prepare(state_path, out, source_config, baseline_engine):
    from tools.gguf_reader import GGUFFile
    from tools.strata_tokenizer import Tokenizer
    from tools.q36_vision_fixture import sve
    from tools.q36_multimodal_cache_fixture import frozen_service
    from serve.frontend import ChatTemplate, openai_to_messages

    state_path = Path(state_path).resolve()
    state = json.loads(state_path.read_text())
    fixtures = Path(state['fixtures']).resolve()
    model = Path(state['model']).resolve()
    manifest_path = fixtures / 'manifest.json'
    if digest(manifest_path) != state['fixture_manifest_sha256']:
        raise ValueError('qualified fixture manifest changed')
    native = json.loads(manifest_path.read_text())
    if native['model'] != file_identity(model):
        raise ValueError('GGUF identity changed')
    for name, sha in native['files'].items():
        p = (fixtures / name).resolve()
        if not p.is_relative_to(fixtures) or digest(p) != sha:
            raise ValueError('qualified input changed: ' + name)

    requests = json.loads((fixtures / 'public-requests.json').read_text())[:3]
    rows = [r for r in native['rows'] if r['case'] == 'real_earlier_user_rewrite'][:3]
    if len(requests) != 3 or len(rows) != 3:
        raise ValueError('missing qualified public rewrite case')
    first_url = requests[0]['messages'][1]['content'][0]['image_url']['url']
    uri = urlsplit(first_url)
    if uri.scheme != 'file' or uri.netloc:
        raise ValueError('expected saved local public image')
    image = Path(unquote(uri.path)).resolve()
    prior_path = Path(state['diagnostic_fixtures']) / 'manifest.json'
    prior = json.loads(prior_path.read_text())
    if prior.get('input_sha256', {}).get(str(image)) != digest(image):
        raise ValueError('public PNG is not bound to the saved encoder evidence')
    emb = fixtures / 'red.sve'
    info = sve(emb)
    if (info['tokens'], info['nx'], info['ny'], info['width']) != (1024, 32, 32, 2048):
        raise ValueError('qualified image geometry changed')
    cfg = json.loads(Path(source_config).read_text())
    vision = cfg.get('vision') or {}
    if vision.get('gpu') is not True or vision.get('max_tokens') != 1024:
        raise ValueError('requires the qualified GPU encoder and 1024-row cap')
    if vision.get('min_tokens') not in (None, 0, 1024):
        raise ValueError('unreviewed encoder minimum token setting')
    if any(vision.get(k) for k in ('elastic', 'elastic_mib', 'elastic_idle_s', 'cuda_device')):
        raise ValueError('unreviewed encoder placement/elastic configuration')

    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    directory = out / 'encoded'
    directory.mkdir()
    key = digest(image)[:32]
    target = directory / (key + '.sve')
    target.write_bytes(emb.read_bytes())
    data_url = 'data:image/png;base64,' + base64.b64encode(image.read_bytes()).decode()
    for request in requests:
        parts = request['messages'][1]['content']
        if parts[0]['image_url']['url'] != first_url:
            raise ValueError('positive case unexpectedly changed its image')
        parts[0]['image_url']['url'] = data_url
    template = out / 'template.jinja'
    template.write_text(GGUFFile(model).metadata['tokenizer.chat_template'])
    gguf_tokenizer = Tokenizer.from_gguf(model)
    # Match serve.server.main, rather than assuming the deployed tokenizer export
    # and template still correspond to the GGUF inspected above.
    tpath = Path(cfg['tokenizer']).resolve()
    vocab_path, merges_path, types_path = [tpath / n for n in ('vocab.json', 'merges.txt', 'token_type.json')]
    vocab = json.loads(vocab_path.read_text())
    tokens = [None] * len(vocab)
    for token, index in vocab.items():
        tokens[index] = token
    tokenizer = Tokenizer(tokens, merges_path.read_text().split('\n'), json.loads(types_path.read_text()))
    if (tokenizer.tokens != gguf_tokenizer.tokens or tokenizer.token_types != gguf_tokenizer.token_types
            or tokenizer.ranks != gguf_tokenizer.ranks):
        raise ValueError('actual server tokenizer export differs from GGUF')
    server_template = tpath / 'chat_template.jinja'
    if not server_template.is_file():
        server_template = Path(state['trial']) / 'serve/chat_template.jinja'
    renderer = ChatTemplate(server_template)
    service = frozen_service(tokenizer, renderer, directory, {data_url: (target, 1024)})
    ids = []
    for request, row in zip(requests, rows):
        messages, tools, kwargs = openai_to_messages(copy.deepcopy(request))
        tokens, _, _ = service.prepare(messages, tools, kwargs, max_new=24)
        packed = Path(service.embeddings.path)
        packed.unlink()
        if tokens != row['ids'] or service.embeddings.keys != [row['key']]:
            raise ValueError('actual HTTP-body tokens/identity differ from GPU-tested fixture')
        ids.append(tokens)
    image_meta = rows[0]['images'][0]
    audit = audit_ids(ids, image_meta['cell'], image_meta['n'])
    (out / 'requests.json').write_text(json.dumps(requests, indent=2) + '\n')
    (out / 'ids.json').write_text(json.dumps(ids) + '\n')
    bound = {str(p.resolve()): digest(p) for p in (
        state_path, manifest_path, prior_path, image, emb, Path(source_config),
        Path(state['trial']) / 'serve/server.py', Path(state['trial']) / 'serve/frontend.py',
        vocab_path, merges_path, types_path, server_template,
        Path(state['build']) / 'strata-q36', Path(baseline_engine),
        Path(vision['exe']),
    )}
    report = dict(kind='q36-http-geometry-v2', cpu_only=True,
                  source_commit=state['commit'], engine_sha256=state['engine_sha256'],
                  model=file_identity(model), projector=file_identity(vision['mmproj']),
                  server_root=state['trial'], image=info, audit=audit,
                  request_prompt_tokens=list(map(len, ids)), bindings=bound,
                  files={p.name: digest(p) for p in (out / 'requests.json', out / 'ids.json', template)},
                  note='controlled canned history; generated-history cases are separate; no GPU run')
    if bound[str((Path(state['build']) / 'strata-q36').resolve())] != state['engine_sha256']:
        raise ValueError('qualified native binary changed')
    (out / 'geometry.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def load(path, model, engine, source_config, server_root, baseline_engine=None):
    """Recheck before plan/preflight and again inside the bounded GPU worker."""
    path = Path(path).resolve()
    report = json.loads(path.read_text())
    if report.get('kind') != 'q36-http-geometry-v2' or report.get('cpu_only') is not True:
        raise ValueError('CPU actual-GGUF geometry preflight required')
    if report['model'] != file_identity(model) or report['server_root'] != str(Path(server_root).resolve()):
        raise ValueError('model/server identity differs from geometry audit')
    if report['projector'] != file_identity(report['projector']['path']):
        raise ValueError('projector identity changed')
    required = [engine, source_config, Path(server_root) / 'serve/server.py']
    if baseline_engine is not None:
        required.append(baseline_engine)
    for p in required:
        if str(Path(p).resolve()) not in report['bindings']:
            raise ValueError('missing required audit binding')
    for p, sha in report['bindings'].items():
        if digest(p) != sha:
            raise ValueError('preflight binding changed: ' + p)
    for name, sha in report['files'].items():
        p = (path.parent / name).resolve()
        if not p.is_relative_to(path.parent) or digest(p) != sha:
            raise ValueError('prepared HTTP input changed')
    ids = json.loads((path.parent / 'ids.json').read_text())
    end = report['audit']['rewrites'][0]['image_end']
    cells = [i for i, t in enumerate(ids[0]) if t == 248056]
    if not cells or cells != list(range(cells[0], cells[0] + report['image']['tokens'])):
        raise ValueError('audited image-pad run changed')
    if cells[-1] + 1 != end or audit_ids(ids, cells[0], len(cells)) != report['audit']:
        raise ValueError('legal checkpoint audit changed')
    return report, json.loads((path.parent / 'requests.json').read_text())
