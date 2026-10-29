#!/usr/bin/env python3
"""CPU preparation ONLY: actual GGUF/template + stored real SVE inputs.
Public API histories are distinguished from constructed native token/grid controls.
No encoder/model/service invocation, prompt normalization, downloads or OWUI claims.
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
ROOT=Path(os.environ.get('STRATA_IMAGE_CHECKPOINT_REVIEW_ROOT',Path(__file__).resolve().parents[1])).resolve();sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from tools.q36_multimodal_cache_fixture import frozen_service,metadata,classify,digest
from tools.q36_vision_fixture import sve

def prepare(model,image_a,image_b,sve_a,sve_b,diagnostic,out):
 from tools.gguf_reader import GGUFFile
 from tools.strata_tokenizer import Tokenizer
 from serve.frontend import ChatTemplate,openai_to_messages
 model=Path(model).resolve();out=Path(out).resolve();diagnostic=Path(diagnostic).resolve()
 a,b=sve(sve_a),sve(sve_b)
 if (a['tokens'],a['nx'],a['ny'])!=(1024,32,32) or a!=b or digest(sve_a)==digest(sve_b):raise ValueError('need distinct actual1024-row32x32 encoder outputs, width2048')
 out.mkdir(parents=True,exist_ok=False);(out/'encoded').mkdir()
 for p in (out,image_a,image_b,sve_a,sve_b):
  if any(c.isspace() for c in str(p)):raise ValueError('no whitespace in native fixture paths')
 source=GGUFFile(model).metadata['tokenizer.chat_template'];(out/'template.jinja').write_text(source)
 tokenizer=Tokenizer.from_gguf(model);renderer=ChatTemplate(out/'template.jinja')
 if tokenizer.encode('<|image_pad|>',parse_special=True)!=[248056]:raise ValueError('wrong actual image-pad vocabulary')
 sources={}
 for name,img,emb in [('red',image_a,sve_a),('blue',image_b,sve_b)]:
  target=out/(name+'.sve');shutil.copyfile(emb,target)
  encoded=out/'encoded'/(digest(img)[:32]+'.sve');shutil.copyfile(emb,encoded);sources[Path(img).resolve().as_uri()]=(encoded,1024)
 service=frozen_service(tokenizer,renderer,out/'encoded',sources)
 base=dict(messages=[dict(role='system',content='Public stable archive facts: amber birch cedar delta.\n'*90),
  dict(role='user',content=[dict(type='image_url',image_url=dict(url=Path(image_a).resolve().as_uri())),dict(type='text',text='Public stable facts after the picture: amber birch cedar delta.\n'*110)]),
  dict(role='assistant',content='The public square is red.',reasoning_content='Public canned reasoning about the same square.'),
  dict(role='user',content='Earlier question version A: describe the color carefully.'),dict(role='assistant',content='Red.'),dict(role='user',content='Summarize the same public image history.')],chat_template_kwargs=dict(enable_thinking=False))
 requests=[base]
 for version in ('B','C'):
  req=copy.deepcopy(base);req['messages'][3]['content']=f'Earlier question version {version}: describe the color carefully.';requests.append(req)
 rows=[];public=[]
 def add(case,slot,phase,expect,kind,ids,key,emb,old=None):
  ids=list(ids);images=[metadata(ids,emb,key)]
  if not 2<len(ids)<=6144:raise ValueError('bounded native prompt length exceeded')
  audit=classify(old,ids,images)
  rows.append(dict(case=case,slot=str(slot),phase=phase,expect=expect,kind=kind,ids=ids,key=key,emb=str(Path(emb).resolve()),images=images,audit=audit))
  return (ids,images)
 old=None
 for i,req in enumerate(requests):
  messages,tools,kwargs=openai_to_messages(copy.deepcopy(req));ids,_,_=service.prepare(messages,tools,kwargs,max_new=1)
  path=Path(service.embeddings.path);target=out/f'api-{i}.sve';shutil.copyfile(path,target);path.unlink();key=service.embeddings.keys[0]
  previous=old;old=add('real_earlier_user_rewrite',0 if i<2 else 1,'steady' if i==2 else 'warmup','zero' if i==0 else 'after','real_Service.prepare_public_earlier_user_rewrite',ids,key,target,old)
  if previous and (rows[-1]['audit']['classification']!='token_rewrite' or rows[-1]['audit']['lcp']<(rows[-1]['images'][0]['cell']+1024+511)//512*512+1):
   raise ValueError('actual rendered earlier-user rewrite lacks a complete attainable after-image checkpoint; do not fake qualification')
  public.append(req)
 # Actual earlier system/header rewrite: legitimate short LCP, not a routing defect.
 req=copy.deepcopy(requests[-1]);req['messages'][0]['content']='Changed earlier system message.\n'+req['messages'][0]['content'];messages,tools,kwargs=openai_to_messages(req)
 ids,_,_=service.prepare(messages,tools,kwargs,max_new=1);path=Path(service.embeddings.path);target=out/'api-system.sve';shutil.copyfile(path,target);path.unlink()
 add('real_earlier_user_rewrite',1,'steady','zero','real_Service.prepare_system_rewrite',ids,service.embeddings.keys[0],target,old);public.append(req)
 # Native controls use real vocabulary IDs/actual rows, NOT API histories.
 prefix=tokenizer.encode('Public native structural fixture: amber birch cedar delta.\n'*200,parse_special=False)[:700]
 tail=tokenizer.encode('Public native suffix: count and name the red color.\n'*220,parse_special=False)[:2200]
 if len(prefix)!=700 or len(tail)<1800:raise ValueError('insufficient real-token native control')
 native=prefix+[248056]*1024+tail;key='1234567890abcdef';red=out/'red.sve';blue=out/'blue.sve'
 suffix=tokenizer.encode('\nPublic exact continuation.\n',parse_special=False)
 old=add('native_boundaries',0,'warmup','zero','constructed_native',native,key,red)
 old=add('native_boundaries',0,'warmup','exact','constructed_native',native+suffix,key,red,old)
 old=add('native_boundaries',1,'steady','exact','constructed_native_cross_slot_exact',native+suffix*2,key,red,old)
 changed=list(native);changed[600]=tokenizer.encode('DIFFERENT',parse_special=False)[0]
 add('native_boundaries',1,'steady','before','constructed_native_before_image_rewrite',changed,key,red,old)
 for case,emb,ids in [('changed_pixels',blue,native+suffix),('changed_grid',out/'grid.sve',native+suffix),('changed_placement',red,prefix+prefix[:10]+[248056]*1024+tail)]:
  if case=='changed_grid':
   data=red.read_bytes();h=list(struct.unpack('<5i',data[:20]));h[2:4]=[16,64];emb.write_bytes(struct.pack('<5i',*h)+data[20:]) # explicitly synthetic POSITIONAL control
  old=add(case,0,'warmup','zero','constructed_native',native,key,red)
  add(case,1,'steady','before','constructed_native_identity_negative',ids,key,emb,old)
 # Preserve the diagnostic's REAL rendered header/reasoning follow-ups, including
 # legitimate zero reuse. They are not rewritten or frontend-normalized here.
 prior=json.loads((diagnostic/'manifest.json').read_text())
 for name,sha in prior['files'].items():
  if digest(diagnostic/name)!=sha:raise ValueError('diagnostic input hash changed')
 old=None
 for i,row in enumerate([r for r in prior['rows'] if r['case']=='rendered_history']):
  target=out/f'prior-rendered-{i}.sve';shutil.copyfile(row['emb'],target)
  old=add('rendered_header_reasoning',0,'warmup' if i==0 else 'steady','zero' if i==0 else 'observe',row['kind'],row['ids'],row['key'],target,old)
 if len(rows)>24:raise ValueError('too many short gate requests')
 lines=[];last=None
 for row in rows:
  if row['case']!=last:last=row['case'];lines.append('# case '+last)
  lines.append('\t'.join([row['slot'],row['phase'],row['expect'],row['key'],row['emb'],','.join(map(str,row['ids']))]))
 (out/'image-checkpoints.txt').write_text('\n'.join(lines)+'\n');(out/'public-requests.json').write_text(json.dumps(public,indent=2)+'\n')
 st=model.stat();manifest=dict(kind='q36-image-checkpoints-v1',real_tokenizer_template=True,frozen_actual_encoder_IO=True,
  model=dict(path=str(model),size=st.st_size,mtime_ns=st.st_mtime_ns),rows=rows,
  files={str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*')) if p.is_file()},
  actual_OWUI=dict(token_divergence='UNMEASURED',savings='UNMEASURED'),note='grid/header/native controls are synthetic; public Service.prepare histories are not actual OWUI traffic')
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for x in ('model','image-a','image-b','sve-a','sve-b','diagnostic','out'):p.add_argument('--'+x,required=True)
 a=p.parse_args();m=prepare(a.model,a.image_a,a.image_b,a.sve_a,a.sve_b,a.diagnostic,a.out);print(json.dumps({k:v for k,v in m.items() if k!='rows'},indent=2))
