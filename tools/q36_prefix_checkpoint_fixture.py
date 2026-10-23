#!/usr/bin/env python3
"""One checkpoint regression fixture: retrieved memory changes inside the system
message after 13,917 unchanged CHARACTERS. Token offsets are computed, not guessed.
Writes two CSV lines consumed by strata-q36 --kv-checkpoint-test.
"""
import argparse
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from serve.frontend import ChatTemplate


def fixture(tokenizer, template):
    common=('Stable context uses fixed facts and never changes.\n'*400)[:13917]
    assert len(common)==13917
    tail='\nThe remaining background is unchanged and belongs to the same system message.\n'*256
    def prompt(memory):
        return template.render([{'role':'system','content':common+memory+tail},
                                {'role':'user','content':'Reply briefly with the current memory record.'}],
                               enable_thinking=False)
    a=tokenizer.encode(prompt('A retrieved memory: red.'),parse_special=True)
    b=tokenizer.encode(prompt('B retrieved memory: blue.'),parse_special=True)
    shared=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
    if not 0<shared<min(len(a),len(b)):
        raise ValueError('fixture did not create an internal token mismatch')
    return a,b,shared


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,help='GGUF first shard carrying the actual tokenizer/template')
    p.add_argument('--template',help='override only if the HTTP server uses a different exported template')
    p.add_argument('--out',required=True)
    a=p.parse_args()
    from strata_tokenizer import Tokenizer
    from gguf_reader import GGUFFile
    tokenizer=Tokenizer.from_gguf(a.model)
    with tempfile.TemporaryDirectory(prefix='q36-checkpoint-template-') as scratch:
        path=Path(a.template) if a.template else Path(scratch)/'chat_template.jinja'
        if not a.template:
            source=GGUFFile(Path(a.model)).metadata.get('tokenizer.chat_template')
            if not isinstance(source,str):raise ValueError('GGUF lacks a chat template; pass --template')
            path.write_text(source,encoding='utf-8')
        original,revised,common=fixture(tokenizer,ChatTemplate(path))
    out=Path(a.out)
    out.write_text(','.join(map(str,original))+'\n'+','.join(map(str,revised))+'\n')
    print(f'system_character_mismatch=13917 common_tokens={common} original_tokens={len(original)} revised_tokens={len(revised)} fixture={out}')
    print('Use an interval smaller than common_tokens; --chunk 512 --kv-checkpoint-interval 1024 is the focused gate.')


if __name__=='__main__':main()
