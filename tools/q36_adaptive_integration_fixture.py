#!/usr/bin/env python3
"""Short PUBLIC synthetic rendered-token replay. No HTTP, GPU or private text.
Hardware use requires --model (actual GGUF and template); --byte-test is NOT
model/rendering evidence. Continuation is classified only after tokenization.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from q36_checkpoint_schedule import audit


def generate(encode):
    memory = ''
    for i in range(1000):
        memory += f'Public memory record {i}: amber birch cedar; preserve this synthetic entry.\n'
        if len(encode([dict(role='system', content=memory)])) >= 3072:
            break
    else:
        raise ValueError('failed to construct bounded synthetic memory')

    downstream = ''
    while len(encode([dict(role='assistant', content=downstream)])) < 2304:
        downstream += 'Public downstream record: silver maple hazel; synthetic reference only.\n'

    def history(context='Context version A.', inserted=False, tail=0, peer=False):
        user = ('Temporary public fetch_url content: synthetic retrieved record.\n' * 16 if inserted else '')
        user += 'What public records were saved?' if not peer else 'Which synthetic records belong to peer chat?'
        messages = [dict(role='system', content=memory + context), dict(role='user', content=user),
                    dict(role='assistant', content='The synthetic records are saved.\n' + downstream)]
        for t in range(tail):
            messages += [dict(role='user', content=f'Continue public step {t}.'),
                         dict(role='assistant', content=f'Public step {t} recorded.')]
        return messages

    # Closed assistant turns avoid a generation-header assumption. Even then the
    # actual GGUF/template audit below, not this structure, determines appendness.
    temporary = [history(), history(inserted=True, tail=1), history(tail=2), history(inserted=True, tail=3), history(tail=4)]
    system = [history(), history(context='Context version B.', tail=1), history(tail=2),
              history(context='Context version B.', tail=3), history(tail=4)]
    ordinary = [history(tail=t) for t in range(4)]
    interleaved = [(0, history(tail=0)), (1, history(peer=True)),
                   (0, history(inserted=True, tail=1)), (1, history(tail=1, peer=True)),
                   (0, history(tail=2)), (1, history(tail=2, peer=True))]
    cases = {'temporary_user': list(enumerate(temporary)), 'system_context': list(enumerate(system)),
             'ordinary': list(enumerate(ordinary))}
    cases = {name: [(0, messages) for _, messages in rows] for name, rows in cases.items()}
    cases['interleaved'] = interleaved
    encoded = {name: [(slot, encode(messages)) for slot, messages in rows] for name, rows in cases.items()}
    text = []
    for name, rows in encoded.items():
        text.append('# case ' + name)
        for i, (slot, ids) in enumerate(rows):
            if not ids or len(ids) > 8192:
                raise ValueError('fixture exceeds short gate bound (8192 tokens)')
            text.append('# phase ' + ('warmup' if i < 2 else 'steady'))
            text.append(str(slot) + ' ' + ','.join(map(str, ids)))
    replay = '\n'.join(text) + '\n'
    metrics = audit(replay)
    for name in ('temporary_user', 'system_context'):
        rows = [r for r in metrics['turns'] if r['case'] == name]
        if not all(r['classification'] == 'rewrite' and r['lcp'] >= 1024 for r in rows[1:]):
            raise ValueError(f'{name}: actual tokenization did not produce eligible rewrites')
    if not any(r['classification'] == 'append' for r in metrics['turns'] if r['case'] == 'ordinary'):
        raise ValueError('no actual append-only ordinary continuation; do not relabel a rewrite')
    smoke = ['# case reference_smoke', '# phase warmup']
    for slot, ids in encoded['temporary_user'][:2]:
        smoke.append(str(slot) + ' ' + ','.join(map(str, ids)))
    return replay, '\n'.join(smoke) + '\n', metrics, cases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--model'); g.add_argument('--byte-test', action='store_true')
    p.add_argument('--template'); p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='q36-integration-template-') as temp:
        if a.byte_test:
            encode = lambda m: list(''.join(x['role'] + ':' + x['content'] + '\n' for x in m).encode())
            template_sha = None; model = None
        else:
            from gguf_reader import GGUFFile
            from strata_tokenizer import Tokenizer
            from serve.frontend import ChatTemplate
            model_path = Path(a.model).resolve()
            tokenizer = Tokenizer.from_gguf(model_path)
            template_path = Path(a.template) if a.template else Path(temp) / 'chat.jinja'
            if not a.template:
                template_path.write_text(GGUFFile(model_path).metadata['tokenizer.chat_template'])
            template = ChatTemplate(template_path)
            (a.out / 'template.jinja').write_bytes(template_path.read_bytes())
            template_sha = hashlib.sha256(template_path.read_bytes()).hexdigest()
            st = model_path.stat(); model = dict(path=str(model_path), size=st.st_size, mtime_ns=st.st_mtime_ns)
            encode = lambda m: tokenizer.encode(template.render(m, tools=[], add_generation_prompt=False,
                                                                 enable_thinking=False), parse_special=True)
        replay, smoke, metrics, cases = generate(encode)
    for name, text in [('replay.csv', replay), ('reference-smoke.csv', smoke)]:
        (a.out / name).write_text(text)
    (a.out / 'messages.json').write_text(json.dumps(cases, indent=2) + '\n')
    manifest = dict(kind='q36-adaptive-integration-v1', byte_test_only=a.byte_test, model=model,
                    template_sha256=template_sha, audit=metrics,
                    files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(a.out.iterdir()) if p.is_file()})
    (a.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
