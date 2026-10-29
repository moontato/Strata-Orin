#!/usr/bin/env python3
"""CPU-only public fixture preparation from ACTUAL encoder SVE1 files/GGUF.
No byte-token fallback, no fabricated embeddings. All output goes to --out.
"""
import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def sve(path):
    data=Path(path).read_bytes()
    if len(data)<20:raise ValueError('short encoder file')
    magic,n,nx,ny,width=struct.unpack('<5i',data[:20])
    if magic!=0x31455653 or not(0<n<=1024) or nx<=0 or ny<=0 or nx*ny!=n or width!=2048 or len(data)!=20+n*width*4:
        raise ValueError('need one actual SVE1 image, <=1024 rows, width2048, valid grid')
    values=array.array('f');values.frombytes(data[20:])
    if not all(math.isfinite(v) for v in values) or not any(v!=0 for v in values):raise ValueError('encoder output is nonfinite/all zero')
    return dict(tokens=n,nx=nx,ny=ny,width=width,delta=max(nx,ny)-n)

def prepare(model,emb_a,emb_b,out):
    from tools.gguf_reader import GGUFFile
    from tools.strata_tokenizer import Tokenizer
    from serve.frontend import ChatTemplate
    model=Path(model).resolve();emb_a=Path(emb_a).resolve();emb_b=Path(emb_b).resolve();out=Path(out).resolve()
    for p in (emb_a,emb_b,out):
        if any(c.isspace() for c in str(p)):raise ValueError('native fixture paths must not contain whitespace')
    a,b=sve(emb_a),sve(emb_b)
    if a['delta']==b['delta']:raise ValueError('use different image aspect ratios/grids for positional isolation')
    out.mkdir(parents=True,exist_ok=False);template=out/'template.jinja';template.write_text(GGUFFile(model).metadata['tokenizer.chat_template'])
    renderer=ChatTemplate(template);tokenizer=Tokenizer.from_gguf(model)
    if tokenizer.encode('<|image_pad|>',parse_special=True)!=[248056]:raise ValueError('wrong model image-pad token')
    def render(image,tail):
        content=([dict(type='image',source='public-fixture.png')] if image else [])+[dict(type='text',text=tail)]
        text=renderer.render([dict(role='user',content=content)],tools=[],add_generation_prompt=True,enable_thinking=False)
        if image:
            if text.count('<|image_pad|>')!=1:raise ValueError('template did not emit exactly one image placeholder')
            text=text.replace('<|image_pad|>','<|image_pad|>'*image['tokens'])
        return tokenizer.encode(text,parse_special=True)
    task='State the dominant color if there is a picture. Then count from 1 to 100, separated by commas. Do not stop early.'
    # Long public text tail makes cancellation at1536 occur AFTER image A but
    # BEFORE prompt end. The actual tokenizer verifies all boundaries below.
    ia=render(a,('Public fixture padding: amber birch cedar delta.\n'*160)+task)
    ib=render(b,task)
    text=render(None,('Public text-only checkpoint/control: number 17, number 29.\n'*160)+task)
    pads=[i for i,t in enumerate(ia) if t==248056]
    if not pads or not(pads[0]<512<pads[-1]+1<1536<len(ia)) or len(text)<1025 or max(len(ia),len(ib),len(text))+256>262144:
        raise ValueError('actual rendered fixture does not cover cancellation/checkpoint geometry; do not fake token IDs')
    fixture=out/'vision.txt';fixture.write_text(str(emb_a)+'\n'+','.join(map(str,ia))+'\n'+str(emb_b)+'\n'+','.join(map(str,ib))+'\n'+','.join(map(str,text))+'\n')
    st=model.stat();manifest=dict(model=dict(path=str(model),size=st.st_size,mtime_ns=st.st_mtime_ns),render_verified=True,
        image_a=a,image_b=b,tokens=dict(image_a=len(ia),image_b=len(ib),text=len(text)),image_a_pad_begin=pads[0],image_a_pad_end=pads[-1]+1,
        fixture=str(fixture),files={str(p):digest(p) for p in (emb_a,emb_b,template,fixture)},note='actual encoder/GGUF outputs; no model inference qualified by preparation')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest

def make_images(out):
    import zlib
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    def png(name,w,h,color):
        def chunk(kind,data):return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
        raw=b''.join(b'\0'+bytes(color)*w for _ in range(h))
        data=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b'')
        (out/name).write_bytes(data)
    png('red-square.png',1024,1024,(220,20,20));png('blue-wide.png',1024,512,(20,20,220))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--make-images');p.add_argument('--model');p.add_argument('--a');p.add_argument('--b');p.add_argument('--out');a=p.parse_args()
    if a.make_images:make_images(a.make_images)
    else:
        if not all((a.model,a.a,a.b,a.out)):p.error('need model, a, b, out')
        print(json.dumps(prepare(a.model,a.a,a.b,a.out),indent=2))
