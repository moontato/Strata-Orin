#!/usr/bin/env python3
"""Focused host mutation controls. No native engine, listener, model or services."""
import copy
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import types
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools import q36_image_http_geometry as geometry

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'tools/q36_image_http_smoke.py'
GEOMETRY=ROOT/'tools/q36_image_http_geometry.py'
SOLO='strata serve: session 0: prompt 12 tokens = 8 reused + 4 read in 3 ms (1.3 tok/s), 2 generated in 5 ms (4.0 tok/s), drafts accepted 0 of 0 (length)\n'
RESTORE='KVREPLAY session=0 generation=2 epoch=2 tokens=8 reason=prefix_checkpoint\n'

def module(source,path):
 ns=dict(__name__='q36_mutation_probe',__file__=str(path))
 exec(compile(source,str(path),'exec'),ns)
 return types.SimpleNamespace(**ns)

def metrics(m,tmp):
 c=m.Client(11501);c._json=lambda method,path:path
 return c.metrics()=='/metrics?requests=all'

def span_usage(m,tmp):
 return not m.audit_log_span(SOLO.encode(),[dict(usage=dict(prompt_tokens=99))])['exact']

def later(m,tmp):
 return not m.audit_log_span((SOLO+RESTORE).encode(),[dict(usage=dict(prompt_tokens=12))])['exact']

def burst(m,tmp):
 raw=(SOLO+SOLO.replace('session 0','session 1')).encode()
 a=m.audit_log_span(raw,[dict(usage=dict(prompt_tokens=12))]*2)
 return a['exact'] and not a['attributed']

def reuse(m,tmp):
 checks,_=m.arm_checks('candidate_on',[],[],dict(unmatched_requests=0,unmatched_engine_lines=0),
  dict(images=True,loaded=True),dict(live=dict(state='idle')),'',list(m.QUALIFIED)+[m.IMAGE_FLAG],[])
 check=next(c for c in checks if c['name']=='qualified_after_image_restore')
 return check['required'] and not check['ok']

def config(m,tmp):
 cfg=dict(exe='/qualified/native',args=[],vision={});path=tmp/'config-candidate_on.json';path.write_text(json.dumps(cfg))
 a=m.Arm(tmp,'candidate_on',cfg,sys.executable,tmp,11501)
 return a.command[a.command.index('--config')+1]==str(path)

def cleanup(m,tmp):
 (tmp/'owned-groups.jsonl').write_text(json.dumps(dict(pid=987654321,starttime='1',arm='a'))+'\n')
 old=os.killpg;probes=[]
 def probe(pid,sig):probes.append(sig)
 os.killpg=probe
 try:
  result=m.reap_recorded_groups(tmp,time.monotonic()+.1)
 finally:os.killpg=old
 return probes==[0] and not result[0]['group_empty']

def marker(m,tmp):
 ok,_=m.verdict(False,False,0,True,'PASS_BOUNDED_IMAGE_HTTP_SMOKE',True,preflight=True)
 return not ok

def image_boundary(m,tmp):
 a=[7]*1088+[248056]*1024+[9]*1596;b=a.copy();b[1600]=5
 try:m.audit_ids([a,b,b],1088,1024)
 except ValueError:return True
 return False

CONTROLS=[
 ('all_metrics',SOURCE,"'/metrics?requests=all'","'/metrics'",metrics),
 ('response_usage',SOURCE,"Counter(e['prompt'] for e in summaries)!=Counter(prompts)","False",span_usage),
 ('future_restore',SOURCE,"e['session']!=session or events.index(e)>index","e['session']!=session",later),
 ('burst_identity',SOURCE,"out=dict(attributed=len(answers)==1,exact=True","out=dict(attributed=True,exact=True",burst),
 ('positive_reuse_required',SOURCE,"for r in positive])","for r in positive],required=False)",reuse),
 ('config_path',SOURCE,"'--config',str(self.config_path)","'--config',str(cfg)",config),
 ('leader_absence',SOURCE,"kind='leader_missing_group_present_not_signalled',\n    group_empty=False","kind='leader_missing_group_present_not_signalled',\n    group_empty=True",cleanup),
 ('mode_binding',SOURCE,"if report_status!=accepted:","if False:",marker),
 ('after_image_boundary',GEOMETRY,"end <= p <= lcp","p <= lcp",image_boundary),
]

def main():
 results=[]
 for name,path,old,new,probe in CONTROLS:
  source=path.read_text()
  if source.count(old)!=1:raise RuntimeError(name+': mutation anchor changed')
  with tempfile.TemporaryDirectory(prefix='q36-http-mutation-') as d:
   tmp=Path(d)
   kept=bool(probe(module(source,path),tmp))
   detected=not bool(probe(module(source.replace(old,new),path),tmp))
   results.append(dict(control=name,baseline_pass=kept,mutation_detected=detected))
  if not kept or not detected:raise RuntimeError('mutation control failed: '+name)
 print(json.dumps(dict(status='PASS',native_model_or_service_started=False,controls=results),indent=2))
 return 0

if __name__=='__main__':sys.exit(main())
