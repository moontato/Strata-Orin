#!/usr/bin/env python3
"""Synthetic multi-turn rendered-token fixtures. No private conversation text.
Byte-test is ONLY for generator tests; hardware must use the GGUF tokenizer.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]


def cases():
    # Character lengths deliberately match the report, never used as token hints.
    s1 = ''.join(f'{i:05d}:a,b;c|d!e?f\n' for i in range(1000))[:16852]
    s2 = s1[:13917] + 'Z' + s1[13918:]
    assert len(s1) == len(s2) == 16852
    u = ('A synthetic question for checkpoint retention. ' * 2)[:52]
    u2 = ('Another synthetic question. ' * 8)[:128]
    aug = ('Synthetic retrieved context.\n' * 100)[:1200] + u
    aug = (aug + '\n' + 'Synthetic added context.\n' * 100)[:2459]
    aug2 = ('Synthetic added context.\n' * 200)[:2507] + u2
    assert len(aug) == 2459 and u in aug and len(aug2) == 2635
    tools = [{'type': 'function', 'function': {'name': 'lookup_record',
              'description': 'Return synthetic records.',
              'parameters': {'type': 'object', 'properties': {}}}}]

    def prompt(system=s1, user=u, result='Synthetic record.\n' * 600, tail=False):
        messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user},
                    {'role': 'assistant', 'content': '', 'tool_calls': [
                        {'id': 'synthetic1', 'type': 'function', 'function':
                         {'name': 'lookup_record', 'arguments': '{}'}}]},
                    {'role': 'tool', 'tool_call_id': 'synthetic1', 'content': result}]
        if tail:
            messages += [{'role': 'user', 'content': 'Continue the synthetic discussion.'}]
        return {'messages': messages, 'tools': tools}

    a, b, c = prompt(), prompt(user=aug, tail=True), prompt(system=s2, tail=True)
    # B has an additional call/result; C retains it with the restored exact user.
    extra = [{'role': 'assistant', 'content': '', 'tool_calls': [
                 {'id': 'synthetic2', 'type': 'function', 'function':
                  {'name': 'lookup_record', 'arguments': '{}'}}]},
             {'role': 'tool', 'tool_call_id': 'synthetic2', 'content': 'Additional result.\n' * 500}]
    b['messages'] += copy.deepcopy(extra)
    c['messages'] += copy.deepcopy(extra) + [{'role': 'user', 'content': 'Now continue.'}]
    combined = [a, b, c, b, c, b, c]
    user = [a, prompt(user=aug), a, prompt(user=aug), a, prompt(user=aug), a]
    system = [a, prompt(system=s2), a, prompt(system=s2), a, prompt(system=s2), a]
    second = [prompt(user=u2), prompt(user=aug2), prompt(user=u2)] * 3
    append = [prompt()]
    # Deliberately request a closed transcript, not a replaceable generation
    # header, for the strict rendered-token append-only negative control.
    append[0]['add_generation_prompt'] = False
    for i in range(6):
        nxt = copy.deepcopy(append[-1])
        nxt['messages'] += [{'role': 'assistant', 'content': f'Synthetic answer {i}.'},
                            {'role': 'user', 'content': f'Next synthetic step {i}.'}]
        append.append(nxt)
    removed = prompt()
    removed['messages'] = removed['messages'][:2]
    summary = prompt()
    summary['messages'][2:] = [{'role': 'assistant', 'content': 'Short synthetic summary.'},
                               {'role': 'user', 'content': 'Continue from the summary.'}]
    rewrites = [a, prompt(result='Changed record.\n' * 700), removed, a, summary, a, summary, a]
    changed_tools = prompt()
    changed_tools['tools'] = copy.deepcopy(tools)
    changed_tools['tools'][0]['function']['description'] = 'Changed synthetic schema description.'
    template_move = prompt()
    template_move['template_variant'] = True
    moving = [a, b, a, b, changed_tools, a, changed_tools, a, template_move, a]
    return {'user': user, 'user_zero_offset': second, 'system': system, 'combined': combined,
            'append': append, 'tool_edit_summary': rewrites, 'tools_template': moving}


def encode_cases(encode):
    result = {}
    for name, sequence in cases().items():
        result[name] = [encode(p) for p in sequence]
    return result


def write_fixture(out, encoded):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for name, sequence in encoded.items():
        lines = ['# case ' + name]
        metrics[name] = []
        previous = []
        for i, ids in enumerate(sequence):
            common = next((j for j, (x, y) in enumerate(zip(previous, ids)) if x != y),
                          min(len(previous), len(ids)))
            metrics[name].append({'turn': i, 'tokens': len(ids), 'common_tokens': common,
                                  'append_only': bool(previous) and common == len(previous)})
            lines.append(str(i % 2) + ' ' + ','.join(map(str, ids)))
            previous = ids
        (out / (name + '.csv')).write_text('\n'.join(lines) + '\n')
    (out / 'rendered-metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    (out / 'synthetic-messages.json').write_text(json.dumps(cases(), indent=2) + '\n')
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--model')
    group.add_argument('--byte-test', action='store_true')
    p.add_argument('--template')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    with tempfile.TemporaryDirectory(prefix='q36-adaptive-template-') as scratch:
        if a.byte_test:
            # Deterministic stand-in only, NOT a model or append-only rendering test.
            encode = lambda item: list(json.dumps(item, sort_keys=True).encode())
        else:
            from strata_tokenizer import Tokenizer
            from gguf_reader import GGUFFile
            from serve.frontend import ChatTemplate
            tokenizer = Tokenizer.from_gguf(a.model)
            path = Path(a.template) if a.template else Path(scratch) / 'template.jinja'
            if not a.template:
                source = GGUFFile(Path(a.model)).metadata.get('tokenizer.chat_template')
                if not isinstance(source, str):
                    raise ValueError('GGUF lacks template; pass --template')
                path.write_text(source)
            template = ChatTemplate(path)
            alternate = Path(scratch) / 'alternate-template.jinja'
            alternate.write_text('Synthetic alternate template framing.\n' + path.read_text())
            alternate_template = ChatTemplate(alternate)
            def encode(item):
                chosen = alternate_template if item.get('template_variant') else template
                return tokenizer.encode(chosen.render(item['messages'], tools=item['tools'],
                    add_generation_prompt=item.get('add_generation_prompt', True),
                    enable_thinking=False), parse_special=True)
        metrics = write_fixture(a.out, encode_cases(encode))
    print(json.dumps({'byte_test_only': a.byte_test, 'metrics': metrics}, indent=2))


if __name__ == '__main__':
    main()
