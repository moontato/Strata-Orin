#!/usr/bin/env python3
"""Bounded HTTP integration smoke for the OPT-IN image-aware Qwen3.6 checkpoints.
Runs ONLY inside the separately approved operator isolation window, with EVERY file
under /mnt/ssd/tmp/strata-checkpoints: a TEMPORARY loopback server (own config, own
port, own engine log) against ONE already-built engine at a time. It never touches the
production config, port, OWUI, service units or any worktree.

What this test does: the resident GPU encoder gets a real image part through
POST /v1/chat/completions; the assistant replies the API actually returned (content,
reasoning_content and tool_calls, unchanged) are carried into the following requests;
slot/session choice and reuse are read from the ENGINE'S OWN log lines (strata serve,
KVCACHE, KVREPLAY) correlated with the server's per-request records; the sequence
covers a stable image history, an earlier-user rewrite after the image, an early
system rewrite, changed image pixels, a four-client interleave and text-only
generation after images.

What this test does NOT claim: it does NOT re-check restored-state or logit exactness
and never inspects GPU tensors - that is the completed independent native oracle
(IMAGE_CHECKPOINT_ORIN_GATE PASS), which is NOT repeated here. Arms are compared for
rendered-token count and answer text and every difference is REPORTED, never required
to be zero: baseline/candidate-off/candidate-on take different prefill schedules.

Supervisor + worker, reusing the reviewed watchdog mechanisms of
q36_image_checkpoint_gate.py: ONE absolute window deadline bounds every phase (server
start, requests, log reading, reporting) with a process-group kill; ONE cleanup
allowance covers waiting, group killing, reaping and the emptiness proof and is never
re-armed; SIGINT/SIGTERM/SIGHUP are recorded and routed through that same teardown; a
result that cannot be recorded exits nonzero without restarting cleanup. The
supervisor signals only the group IT created, so the operator-owned enclosing cgroup
is still required, and only ONE arm runs at a time.
"""
import argparse,base64,hashlib,json,os,re,signal,socket,subprocess,sys,threading,time,urllib.error,urllib.request,zlib
sys.dont_write_bytecode=True
from http.server import BaseHTTPRequestHandler,HTTPServer
from pathlib import Path
ROOT=Path(os.environ.get('STRATA_IMAGE_HTTP_ROOT',Path(__file__).resolve().parents[1])).resolve();sys.path.insert(0,str(ROOT))
from tools.q36_multimodal_cache_gate import no_models,digest
try: from q36_image_checkpoint_gate import Budget,CleanupState,teardown,group_kill,group_probe,worker_status
except ImportError: from tools.q36_image_checkpoint_gate import Budget,CleanupState,teardown,group_kill,group_probe,worker_status

STAGE=Path('/mnt/ssd/tmp/strata-checkpoints')
QUALIFIED=['--serve','--vision','--max-context','262144','--batch','4','--kv-pool-tokens','524288','--kv','int8',
 '--mtp','2','--batch-mtp','--mtp-sampled','--kv-checkpoint-mib','1024','--kv-checkpoint-interval','1024',
 '--kv-checkpoint-policy','adaptive','--kv-preallocate','--kv-vision-max-tokens','1024','--chunk','512']
IMAGE_FLAG='--kv-image-checkpoints'
ARMS=('baseline','candidate_off','candidate_on')
WINDOW_DEFAULT=1800;WINDOW_MAX=3600;CLEANUP_ALLOWANCE_DEFAULT=20;CLEANUP_ALLOWANCE_MAX=60
PRODUCTION_PORT=11436;PRODUCTION_HEALTH='http://100.100.10.3:11436/health';PRODUCTION_HOST='100.100.10.3'
PROGRESS_S=20
SELFTESTS=('normal','no_encoder','zero_stable_reuse','no_restore','engine_error','divergence','hung_server',
 'cross_talk','missing_record','slow_server','stuck_group')
PASS_MARKER='IMAGE_HTTP_SERIAL_SMOKE PASS concurrency=NOT_TESTED native_oracle=NOT_REPEATED gpu_tensor_exactness=NOT_CHECKED actual_OWUI_savings=UNMEASURED'
PREFLIGHT_MARKER='IMAGE_HTTP_SMOKE_PREFLIGHT PASS model_not_started=1'
PLAN_MARKER='IMAGE_HTTP_PLAN_ONLY PASS nothing_started=1'
SESSION=re.compile(r'KVCACHE session=(\d+) generation=(\d+) epoch=(\d+) state=(\w+) tokens=(\d+) bytes=(\d+) '
 r'reserved_tokens=(\d+) result=(\w+) replay=(\d)')
SERVE_LINE=re.compile(r'strata serve: session (\d+): prompt (\d+) tokens = (\d+) reused \+ (\d+) read in ([\d.]+) ms'
 r'.*\((\w+)\)')
BATCH_LINE=re.compile(r'strata batch: slot (\d+) takes a prompt of (\d+) tokens = (\d+) reused \+ (\d+) read in ([\d.]+) ms')
RESTORED=re.compile(r'KVREPLAY session=(\d+).*reason=prefix_checkpoint')
PREFIX_RESTORED=re.compile(r'Qwen3\.6 prefix checkpoint: source=(\d+) session=(\d+) restored_tokens=(\d+)')
ENGINE_FAULT=re.compile(r'^(?:ERR|BERR) |IMAGE_CHECKPOINT_FAILURE|KV_FAILURE|terminate called',re.M)

class Conversation:
 """One client's conversation as the client keeps it. Assistant replies are stored
 exactly as the API returned them and re-sent unchanged, so reasoning_content and
 tool_calls are carried back as supplied. Images name a local PNG the client uploads."""
 def __init__(self,marker,system='You answer in one short factual sentence.'):
  self.marker=marker;self.system=system;self.turns=[]
 def copy(self,marker=None,system=None):
  c=Conversation(marker or self.marker,system or self.system);c.turns=[dict(t) for t in self.turns];return c
 def user(self,text,image=None):self.turns.append({'role':'user','text':text,'image':image});return self
 def reply(self,message):self.turns.append({'role':'assistant','message':message});return self
 def rewrite_first_user(self,text):
  """Rewrite the FIRST user turn and drop everything after it: the shape a UI hits when
  an earlier question is edited AFTER an image was already sent. The image part of that
  turn stays, and the request still ends on a user message (a chat template rejects a
  history that ends on an assistant turn)."""
  for index,t in enumerate(self.turns):
   if t['role']=='user':
    t['text']=text;self.turns=self.turns[:index+1];return self
  raise ValueError('no earlier user turn to rewrite')
 def replace_image(self,image):
  for t in self.turns:
   if t['role']=='user' and t.get('image'):t['image']=image;return self
  raise ValueError('no image turn to change')
 def user_turns(self):return sum(1 for t in self.turns if t['role']=='user')
 def render(self,image_files):
  out=[{'role':'system','content':self.system}]
  for t in self.turns:
   if t['role']=='assistant':
    m=t['message'];msg={'role':'assistant'}
    for key in ('content','reasoning_content','tool_calls'):
     if key in m:msg[key]=__import__('copy').deepcopy(m[key])
    out.append(msg);continue
   parts=[{'type':'text','text':self.marker+' '+t['text']}]
   if t.get('image'):
    raw=Path(image_files[t['image']]).read_bytes()
    parts.append({'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(raw).decode()}})
   out.append({'role':'user','content':parts})
  return out

class Client:
 """Loopback only, no API key (a private test endpoint). The read timeout is bounded by
  the window deadline, so no request can outlive the run's one absolute budget."""
 def __init__(self,port,timeout=600,deadline=None):
  self.base='http://127.0.0.1:%d'%port;self.timeout=timeout;self.deadline=deadline
  self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 def _timeout(self):
  if self.deadline is None:return self.timeout
  return max(1.0,min(self.timeout,self.deadline-time.monotonic()))
 def _json(self,method,path,body=None):
  data=json.dumps(body).encode() if body is not None else None
  req=urllib.request.Request(self.base+path,data=data,headers={'Content-Type':'application/json'})
  with self.opener.open(req,timeout=self._timeout()) as response:return json.load(response)
 def health(self):return self._json('GET','/health')
 def metrics(self):return self._json('GET','/metrics?requests=all')
 def ask(self,model,messages,max_tokens):
  return self._json('POST','/v1/chat/completions',{'model':model,'messages':messages,'max_tokens':max_tokens,
   'temperature':0,'stream':False,'chat_template_kwargs':{'enable_thinking':True}})

def plan(run_marker):
 """Serial image-history requests only; concurrency is outside this qualification.
 Each step names its conversation and, for the copy
 steps, the conversation it starts from, so an assistant reply is only ever carried into
 a later request of the SAME conversation."""
 return [
  dict(step='stable_image/1',conv='stable',op='user',text='%s what colour square is this'%run_marker,image='A'),
  dict(step='stable_image/2',conv='stable',op='user',text='%s is the square centred'%run_marker),
  dict(step='earlier_user_rewrite/1',conv='rewrite',op='rewrite_first_user',source='stable',
   text='%s identify the picture edited'%run_marker),
  dict(step='system_rewrite/1',conv='system',op='rewrite_system',source='stable',
   system='You answer in exactly two words.',text='%s same history, new first instruction'%run_marker),
  dict(step='changed_pixels/1',conv='pixels',op='replace_image',source='stable',image='B',
   text='%s same question, other picture'%run_marker),
  dict(step='text_only_after_images/1',conv='text',op='user',text='%s one sentence about the moon'%run_marker),
  dict(step='text_only_after_images/2',conv='text',op='user',text='%s and one about iron'%run_marker)]

def expectations(step):
 """What the step is allowed to reuse, stated up front so a zero is never hidden and a
 reuse is never assumed: 'prefix_or_checkpoint' must be positive, the others are
 reported as measured with their expected direction."""
 if step=='stable_image/2':return 'positive_required'
 if step=='earlier_user_rewrite/1':return 'prefix_only_reported'
 if step=='system_rewrite/1':return 'near_zero_reported'
 if step=='changed_pixels/1':return 'image_rows_zero_reported'
 if step.startswith('interleave_four/2') or step=='text_only_after_images/2':return 'positive_expected_reported'
 return 'fresh_reported'

def parse_engine_log(text):
 """The engine's own per-request facts: which session served it, how much of the prompt
 was reused, and whether a prefix/image checkpoint was restored for it."""
 events=[]
 for line in text.splitlines():
  if (m:=SESSION.search(line)):events.append(dict(kind='kvcache',session=int(m.group(1)),state=m.group(4),
   result=m.group(8),replay=int(m.group(9))))
  elif (m:=SERVE_LINE.search(line)):events.append(dict(kind='request',session=int(m.group(1)),prompt=int(m.group(2)),
   reused=int(m.group(3)),read=int(m.group(4)),prompt_ms=float(m.group(5)),finish=m.group(6)))
  elif (m:=BATCH_LINE.search(line)):events.append(dict(kind='slot',session=int(m.group(1)),prompt=int(m.group(2)),
   reused=int(m.group(3)),read=int(m.group(4)),prompt_ms=float(m.group(5)),finish='length'))
  elif (m:=PREFIX_RESTORED.search(line)):
   events.append(dict(kind='restored',session=int(m.group(2)),source=int(m.group(1)),tokens=int(m.group(3))))
  elif (m:=RESTORED.search(line)):
   tokens=re.search(r' tokens=(\d+)',line)
   events.append(dict(kind='restored',session=int(m.group(1)),tokens=int(tokens.group(1)) if tokens else None))
 return events

def correlate(records,events):
 """Pair each server request record with the engine line naming the same prompt size (the
 server's own prompt_total IS the engine's ids.size()); order breaks ties. Anything
 unmatched is REPORTED: the smoke never invents a slot or a reuse count."""
 # Production solo and batch admissions emit different summary lines. Treat both as
 # summaries; if duplicates or ordering make pairing ambiguous, counts expose the
 # mismatch rather than silently discarding a batch admission.
 requests=[e for e in events if e['kind'] in ('request','slot')];used=[False]*len(requests);rows=[]
 for r in records:
  match=None
  for i,e in enumerate(requests):
   if not used[i] and e['prompt']==r.get('prompt_total'):match=i;break
  if match is None:rows.append(dict(record=r,engine=None));continue
  used[match]=True;e=requests[match]
  # A session may serve many requests, and mixed stdout/stderr ordering does not
  # establish event boundaries. Do not attribute another request's restore or result.
  rows.append(dict(record=r,engine=e,restored=None,result=None))
 return rows,dict(unmatched_requests=sum(1 for r in rows if r['engine'] is None),
  unmatched_engine_lines=sum(1 for u in used if not u))

def audit_log_span(raw, answers):
 """Authoritative bracket audit. No pairing by arrival order or reused session."""
 from collections import Counter
 bad=lambda why:dict(attributed=False,exact=False,reason=why,summaries=[],sessions=[],events=[])
 if not isinstance(raw,bytes) or (raw and not raw.endswith(b'\n')):return bad('partial_log_line')
 events=parse_engine_log(raw.decode('utf-8','replace'))
 summaries=[e for e in events if e['kind'] in ('request','slot')]
 prompts=[(a.get('usage') or {}).get('prompt_tokens') for a in answers]
 if any(type(n) is not int for n in prompts):return bad('missing_response_usage')
 if len(summaries)!=len(answers) or Counter(e['prompt'] for e in summaries)!=Counter(prompts):
  return bad('ambiguous_or_usage_mismatch')
 out=dict(attributed=len(answers)==1,exact=True,reason=None if len(answers)==1 else 'concurrent_burst',
  summaries=summaries,sessions=sorted({e['session'] for e in summaries}),events=events)
 if len(answers)!=1:return out  # burst evidence only, never per-client attribution
 summary=summaries[0];index=events.index(summary);session=summary['session']
 provenance=[e for e in events if e['kind'] in ('kvcache','restored')]
 if any(e['session']!=session or events.index(e)>index for e in provenance):
  return bad('provenance_outside_serial_request')
 caches=[e for e in provenance if e['kind']=='kvcache']
 if len(caches)>1:return bad('ambiguous_cache_events')
 result=caches[0]['result'] if caches else None
 # The HTTP server consumes stdout KVREPLAY/KVCACHE internally; the native
 # stderr prefix-checkpoint line is the authoritative restore evidence here.
 out.update(restored=any(e['kind']=='restored' and e.get('tokens')==summary['reused'] for e in provenance),result=result)
 return out


def attribute_spans(windows,answers,engine_bytes,out,arm):
 rows=[];groups=[];audits=[];by_step={a['step']:a for a in answers}
 for i,window in enumerate(windows):
  raw=engine_bytes[window['log_start']:window['log_end']]
  path=out/('%s-span-%02d.log'%(arm,i));path.write_bytes(raw)
  audit=audit_log_span(raw,[by_step[step] for step in window['steps']])
  audits.append(dict(steps=window['steps'],start=window['log_start'],end=window['log_end'],path=str(path),audit=audit))
  if not audit['exact']:raise RuntimeError('unattributable log span: '+str(window['steps'])+' '+audit['reason'])
  if len(window['steps'])==1:
   summary=audit['summaries'][0]
   rows.append(dict(record=dict(_step=window['steps'][0],prompt_total=summary['prompt']),engine=summary,
    restored=audit['restored'],result=audit['result']))
  else:
   group='burst-'+str(i)
   groups.append(dict(group=group,steps=window['steps'],exact=True,records=audit['summaries']))
   for summary in audit['summaries']:
    rows.append(dict(record=dict(_group=group,prompt_total=summary['prompt']),engine=summary,restored=None,result=None))
 all_summaries=[e for e in parse_engine_log(engine_bytes.decode('utf-8','replace')) if e['kind'] in ('request','slot')]
 if len(all_summaries)!=len(rows):raise RuntimeError('request summaries outside bracketed spans')
 return rows,dict(unmatched_requests=0,unmatched_engine_lines=0),groups,audits

def png(path,rgb,size=64):
 """A deterministic solid-colour PNG: a public synthetic image, no external asset."""
 def chunk(tag,data):
  body=tag+data;return len(data).to_bytes(4,'big')+body+zlib.crc32(body).to_bytes(4,'big')
 raw=b''.join(b'\x00'+bytes(rgb)*size for _ in range(size))
 data=(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',size.to_bytes(4,'big')*2+b'\x08\x02\x00\x00\x00')
  +chunk(b'IDAT',zlib.compress(raw,6))+chunk(b'IEND',b''))
 validate_generated_png(data,rgb,size)
 Path(path).write_bytes(data)
 return Path(path)

def validate_generated_png(data,rgb,size=64):
 """CPU guard for OUR RGB fixtures, run in preflight before services stop."""
 if data[:8]!=b'\x89PNG\r\n\x1a\n':raise ValueError('PNG fixture signature invalid')
 offset=8;chunks=[];payload=b''
 while offset<len(data):
  if offset+12>len(data):raise ValueError('PNG fixture truncated')
  n=int.from_bytes(data[offset:offset+4],'big');tag=data[offset+4:offset+8]
  end=offset+8+n
  if end+4>len(data):raise ValueError('PNG fixture chunk length invalid')
  body=data[offset+8:end];crc=int.from_bytes(data[end:end+4],'big')
  if zlib.crc32(tag+body)!=crc:raise ValueError('PNG fixture CRC invalid')
  chunks.append(tag)
  if tag==b'IHDR' and body!=size.to_bytes(4,'big')*2+b'\x08\x02\x00\x00\x00':
   raise ValueError('PNG fixture IHDR invalid')
  if tag==b'IDAT':payload+=body
  if tag==b'IEND' and body:raise ValueError('PNG fixture IEND invalid')
  offset=end+4
 if chunks!=[b'IHDR',b'IDAT',b'IEND']:raise ValueError('PNG fixture chunk sequence invalid')
 expected=b''.join(b'\x00'+bytes(rgb)*size for _ in range(size))
 if zlib.decompress(payload)!=expected:raise ValueError('PNG fixture pixels invalid')
 return True

def vision_of(source):
 v=source.get('vision') if isinstance(source.get('vision'),dict) else {}
 allowed=('exe','mmproj','model','gpu','threads','max_tokens','min_tokens','elastic','elastic_mib','elastic_idle_s','cuda_device')
 return {k:v[k] for k in allowed if k in v}

def plan_config(source,engine,port,cfg_path,log_path,cwd,arm,model):
 """A TEMPORARY config. The source (production) config supplies ONLY the encoder section;
 every network, service, host/port, api_key, log, cwd, replica, sampling and engine-arg
 key of the source config is dropped and replaced by staging values."""
 args=list(QUALIFIED)+['--model',str(Path(model).resolve())]
 if arm=='candidate_on':args.append(IMAGE_FLAG)
 cfg={'exe':str(Path(engine).resolve()),'args':args,'cwd':str(cwd),'log':str(log_path),'host':'127.0.0.1',
  'port':int(port),'tokenizer':str(source.get('tokenizer') or (ROOT/'pack/full/tokenizer').resolve()),
  'model_name':str(source.get('model_name') or 'strata-q36'),'vision':vision_of(source)}
 if not cfg['vision'].get('exe'):raise ValueError('the source config has no vision encoder section')
 Path(cfg_path).write_text(json.dumps(cfg,indent=2)+'\n')
 return cfg

def free_port():
 for port in range(11500,11600):
  if port==PRODUCTION_PORT:continue
  with socket.socket() as s:
   try:s.bind(('127.0.0.1',port));return port
   except OSError:continue
 raise RuntimeError('no free loopback port in 11500..11599')

def proc_starttime(pid):
 """The kernel start time of a pid: the recycled-PID guard. If it changed, the pid is
 somebody else's process now and must NOT be signalled."""
 try:fields=Path('/proc/%d/stat'%pid).read_text().split(')')[-1].split()
 except (OSError,IndexError):return None
 return fields[19] if len(fields)>19 else None

def register_group(out,child,arm):
 """Every temporary server group this run creates is recorded the moment it starts, so a
 worker that is killed or crashes cannot leave a model/encoder running behind it."""
 Path(out/'owned-groups.jsonl').open('a').write(json.dumps(dict(arm=arm,pid=child.pid,pgid=child.pid,
  starttime=proc_starttime(child.pid),registered=time.time()))+'\n')

def reap_recorded_groups(out,deadline):
 """Signal ONLY the groups this run registered, and only while their recorded starttime
 still matches. ONE deadline (the run's cleanup deadline), never a fresh allowance."""
 report=[]
 try:lines=(out/'owned-groups.jsonl').read_text().splitlines()
 except OSError:lines=[]
 for line in lines:
  try:entry=json.loads(line)
  except ValueError:
   report.append(dict(kind='unreadable_group_registration',group_empty=False,operator_cgroup_required=True));continue
  pid=entry.get('pid');starttime=entry.get('starttime')
  if not pid:continue
  if not Path('/proc/%d'%pid).exists():
   # A missing leader is not proof that its process group disappeared. Probe only;
   # never signal the numeric PGID after ownership of its leader has been lost.
   try:os.killpg(pid,0)
   except ProcessLookupError:
    report.append(dict(pid=pid,arm=entry.get('arm'),kind='group_absent_probe_only',group_empty=True));continue
   except PermissionError:pass
   report.append(dict(pid=pid,arm=entry.get('arm'),kind='leader_missing_group_present_not_signalled',
    group_empty=False,operator_cgroup_required=True));continue
  if starttime is None or proc_starttime(pid)!=starttime:
   report.append(dict(pid=pid,arm=entry.get('arm'),kind='pid_no_longer_ours_not_signalled',group_empty=False));continue
  try:os.killpg(pid,signal.SIGKILL);killed='present'
  except ProcessLookupError:killed='empty'
  except PermissionError:killed='denied'
  empty=killed=='empty'
  while not empty:
   if time.monotonic()>=deadline:break
   # an orphaned group leader only leaves the group once it is REAPED: if this process is
   # the subreaper it took the orphan, so try to reap it; otherwise init already has
   try:os.waitpid(pid,os.WNOHANG)
   except OSError:pass
   try:os.killpg(pid,0);time.sleep(0.05)
   except ProcessLookupError:empty=True
   except PermissionError:break
  report.append(dict(pid=pid,arm=entry.get('arm'),kind='killed_owned_group',group_empty=empty))
 return report

class Arm:
 """One arm: one temporary server (and its engine child) in ITS OWN process group."""
 def __init__(self,out,arm,cfg,python_bin,server_root,port):
  self.out=out;self.name=arm;self.cfg=cfg;self.port=port
  self.config_path=out/('config-'+arm+'.json')
  if json.loads(self.config_path.read_text())!=cfg:raise ValueError('temporary config differs from planned config')
  self.command=[python_bin,'-m','serve.server','--engine','strata','--config',str(self.config_path),'--host','127.0.0.1',
   '--port',str(port)]
  self.server_root=server_root;self.child=None;self.stream=None;self.stop_result=None
 def start(self,fake_plan=None):
  log=self.out/(self.name+'-server-stdout.txt')
  command=self.command if fake_plan is None else [sys.executable,str(Path(__file__).resolve()),'--fake-server',
   str(self.port),str(fake_plan),str(self.out/(self.name+'-engine.log'))]
  self.stream=log.open('wb')
  env={k:v for k,v in os.environ.items() if not k.startswith('STRATA_')}
  env.update(TMPDIR=str(self.out),PYTHONPATH=str(self.server_root),PYTHONDONTWRITEBYTECODE='1',
   PYTHONUNBUFFERED='1',STRATA_MMQ_KQ='1',
   LD_LIBRARY_PATH=str(Path(self.cfg['exe']).parent)+':/usr/local/cuda-13.2/lib64')
  self.child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=self.stream,stderr=subprocess.STDOUT,
   cwd=None if fake_plan is not None else str(self.server_root),start_new_session=True,
   env=env)
  register_group(self.out,self.child,self.name)
  return self.child
 def stop(self,deadline):
  """Graceful first, then the OWNED group only: signal while the leader is still
  unreaped (its PGID is then certainly still ours), reap the leader, then PROVE the
  group empty. ONE deadline for all of it, never a fresh allowance."""
  graceful=True
  if self.child.returncode is not None:
   # Someone already consumed the leader's status. Its numeric PID/PGID is
   # no longer ours to signal; only a presence probe is permitted.
   self.stop_result=dict(graceful=False,worker_reaped=True,
    owned_group_empty=group_probe(self.child)=='empty')
   self.stream.close()
   return self.stop_result
  # Popen.terminate() first polls and can REAP an already-exited leader. Use a
  # direct signal while this still-unreaped child reserves its PID/PGID.
  try:os.kill(self.child.pid,signal.SIGTERM)
  except OSError:graceful=False
  end=time.monotonic()+min(5.0,max(0.0,deadline-time.monotonic()))
  while time.monotonic()<end and worker_status(self.child) is None:time.sleep(0.05)
  # Still unreaped: this leader's PID/PGID cannot be recycled. Kill residual
  # descendants even if the leader has already exited, before consuming its status.
  if group_probe(self.child)!='empty':
   group_kill(self.child)
   if worker_status(self.child) is None:graceful=False
  reaped=False
  try:
   self.child.wait(timeout=max(0.0,deadline-time.monotonic()));reaped=True
  except subprocess.TimeoutExpired:pass
  empty=False
  while True:
   if group_probe(self.child)=='empty':empty=True;break
   if deadline-time.monotonic()<=0:break
   time.sleep(0.05)
  try:self.stream.close()
  except Exception:pass
  self.stop_result=dict(graceful=graceful,worker_reaped=reaped,owned_group_empty=empty)
  return self.stop_result

def wait_healthy(client,deadline,out,arm,child=None,stdout=None):
 """Poll /health until it reports what this arm requires: loaded model and the resident
 encoder (images=true). Bounded by the window deadline; progress every PROGRESS_S."""
 started=time.monotonic();next_progress=started+PROGRESS_S;last=None
 while True:
  if time.monotonic()>=deadline:raise TimeoutError('%s: server not healthy inside the window'%arm)
  try:last=client.health()
  except (urllib.error.URLError,OSError,ValueError) as exc:last=dict(error=type(exc).__name__)
  if child is not None and worker_status(child) is not None:
   # a server that already exited can never become healthy: report WHY, do not spin
   tail=''
   try:tail=' | '.join(Path(stdout).read_text(errors='replace').splitlines()[-6:])
   except OSError:pass
   raise RuntimeError('%s: the temporary server exited before READY: %s'%(arm,tail[:400]))
  if last.get('status')=='ok' and last.get('service')=='strata' and last.get('loaded') is True:
   if last.get('images') is not True:raise RuntimeError('%s: health reports images=%s: the image encoder is not resident'
    %(arm,last.get('images')))
   (out/(arm+'-health.json')).write_text(json.dumps(last,indent=2)+'\n');return last
  if time.monotonic()>=next_progress:
   print('HTTP_SMOKE_PROGRESS arm=%s stage=server_start waited_s=%d health=%s'%(arm,int(time.monotonic()-started),
    json.dumps(last)[:160]),flush=True)
   next_progress+=PROGRESS_S
  time.sleep(0.5)

def run_steps(client,model,steps,max_tokens,images,run_marker,deadline,arm,log_path=None,out=None):
 """Drive the sequence. The four first turns and the four second turns of the interleave
 run together (the engine has four slots); every other step is serial, so its server
 record maps to exactly one request."""
 convs={};answers=[];tokens={s['conv']:'%s-%s'%(run_marker,s['conv']) for s in steps}
 parallel=('interleave_four/1c1','interleave_four/1c2','interleave_four/1c3','interleave_four/1c4',
  'interleave_four/2c1','interleave_four/2c2','interleave_four/2c3','interleave_four/2c4')
 def conversation(step):
  if step['conv'] not in convs:
   source=convs.get(step.get('source'))
   # a derived conversation keeps the SOURCE token: only the intended change (the edited
   # text, the system prompt or the image bytes) then differs from the source history
   conv=source.copy() if source is not None else Conversation(tokens[step['conv']])
   convs[step['conv']]=conv
  return convs[step['conv']]
 def send(step):
  conv=conversation(step)
  op=step['op']
  if op=='user':conv.user(step['text'],step.get('image'))
  elif op=='rewrite_first_user':conv.rewrite_first_user(step['text'])
  elif op=='rewrite_system':
   conv.system=step['system']
   if step.get('text'):conv.user(step['text'])
  elif op=='replace_image':
   conv.replace_image(step['image'])
   if step.get('text'):conv.user(step['text'])
  elif op=='geometry':pass
  else:raise ValueError('unknown step operation '+op)
  started=time.monotonic();started_wall=time.time()
  if step.get('body') is not None:
   payload=__import__('copy').deepcopy(step['body'])
   payload.update(model=model,max_tokens=max_tokens,temperature=0,stream=False)
  else:
   payload=dict(model=model,messages=conv.render(images),max_tokens=max_tokens,temperature=0,stream=False,
    chat_template_kwargs=dict(enable_thinking=True))
  if out:
   stem=step['step'].replace('/','-')
   (out/(arm+'-'+stem+'-request.json')).write_text(json.dumps(payload,indent=2)+'\n')
  try:answer=client._json('POST','/v1/chat/completions',payload)
  except urllib.error.HTTPError as error:
   body=error.read(65536).decode('utf-8','replace')
   if out:(out/(arm+'-'+stem+'-error.json')).write_text(json.dumps(dict(status=error.code,body=body),indent=2)+'\n')
   raise RuntimeError('HTTP '+str(error.code)+': '+body) from error
  if out:
   (out/(arm+'-'+stem+'-response.json')).write_text(json.dumps(answer,indent=2)+'\n')
  choice=(answer.get('choices') or [{}])[0]
  message=choice.get('message') or {}
  conv.reply(message)
  return dict(step=step['step'],conv=step['conv'],op=op,user_turns=conv.user_turns(),
   started=started,started_wall=started_wall,client_elapsed_s=round(time.monotonic()-started,3),
   finish=choice.get('finish_reason'),usage=answer.get('usage'),
   content=message.get('content') or '',reasoning_chars=len(message.get('reasoning_content') or ''),
   tool_calls=len(message.get('tool_calls') or []),expected_prompt=step.get('expected_prompt'),
   markers=sorted(set(tokens.values())- {conv.marker}))
 # plan order is kept; only CONSECUTIVE parallel steps (the two groups of four) run
 # together, so each step's start time orders it the same way the server orders records
 # consecutive steps of the SAME burst go together; the two bursts of the interleave are
 # two separate runs, so each one is attributed to its own four server records
 def burst(step):return step.split('/')[1][0] if '/' in step and step in parallel else ''
 runs=[];index=0;windows=[]
 while index<len(steps):
  if steps[index]['step'] in parallel:
   run=[];key=burst(steps[index]['step'])
   while index<len(steps) and steps[index]['step'] in parallel and burst(steps[index]['step'])==key:
    run.append(steps[index]);index+=1
   runs.append(run)
  else:runs.append([steps[index]]);index+=1
 started=time.monotonic();next_progress=started+PROGRESS_S;done=0
 last_end=None
 for run in runs:
  if time.monotonic()>=deadline:raise TimeoutError('window exhausted before step %s'%run[0]['step'])
  found={};errors=[];group_start=time.time()
  span_start=log_path.stat().st_size if log_path and log_path.exists() else 0
  if last_end is not None:
   gap=log_path.read_bytes()[last_end:span_start]
   if parse_engine_log(gap.decode('utf-8','replace')):raise RuntimeError('late request events outside previous span')
  def go(step):
   try:found[step['step']]=send(step)
   except BaseException as exc:errors.append('%s: %s: %s'%(step['step'],type(exc).__name__,exc))
  threads=[threading.Thread(target=go,args=(s,)) for s in run]
  for t in threads:t.start()
  # bounded by the same window deadline as everything else, and visible every PROGRESS_S
  while any(t.is_alive() for t in threads):
   for t in threads:t.join(timeout=min(1.0,max(0.05,deadline-time.monotonic())))
   if time.monotonic()>=next_progress:
    print('HTTP_SMOKE_PROGRESS arm=%s stage=requests done=%d/%d last=%s elapsed_s=%d remaining_s=%d'%(
     arm,done,len(steps),run[0]['step'],int(time.monotonic()-started),max(0,int(deadline-time.monotonic()))),
     flush=True)
    next_progress+=PROGRESS_S
   if time.monotonic()>=deadline:raise TimeoutError('window exhausted during step %s'%','.join(s['step'] for s in run))
  done+=len(run)
  if errors:raise RuntimeError('request failed: '+'; '.join(errors))
  for s in run:
   if s['step'] not in found:raise RuntimeError('step %s returned nothing'%s['step'])
   answers.append(found[s['step']])
  # Collect only complete lines visible within a bounded wait. A missing or late
  # summary is deliberately unknown, not attributed from the next request's span.
  span_end=span_start
  if log_path:
   end=min(deadline,time.monotonic()+2.0)
   while time.monotonic()<end:
    if log_path.exists():
     data=log_path.read_bytes()[span_start:]
     span_end=span_start+(data.rfind(b'\n')+1 if b'\n' in data else 0)
     summaries=[e for e in parse_engine_log(data[:span_end-span_start].decode('utf-8','replace'))
      if e['kind'] in ('request','slot')]
     if len(summaries)==len(run) and data.endswith(b'\n'):
      time.sleep(min(.1,max(0,deadline-time.monotonic())))
      if log_path.stat().st_size==span_end:break
     elif len(summaries)>len(run):raise RuntimeError('too many request summaries in bounded span')
    time.sleep(.05)
  raw=log_path.read_bytes()[span_start:span_end] if log_path else b''
  audit=audit_log_span(raw,[found[s['step']] for s in run])
  if not audit['exact']:raise RuntimeError('request log span failed: '+audit['reason'])
  last_end=span_end
  windows.append(dict(steps=[s['step'] for s in run],start_wall=group_start,end_wall=time.time(),
   log_start=span_start,log_end=span_end))
 # the server-side view of the same requests; each record carries the request's START time
 snapshot=client.metrics()
 return answers,snapshot,windows

def arm_checks(arm,answers,rows,correlation,health,metrics_snapshot,engine_text,cfg_args=None,groups=()):
 """Per-arm checks. Required failures stop the run; the reuse directions of the rewrite
 and changed-image steps are REPORTED with the expectation named, never turned into a
 tolerance pass."""
 by_step={r['record'].get('_step'):r for r in rows}
 checks=[]
 def add(name,ok,value,required=True):checks.append(dict(name=name,ok=bool(ok),required=bool(required),value=value))
 add('encoder_resident',health.get('images') is True,health.get('images'))
 add('model_loaded',health.get('loaded') is True,health.get('loaded'))
 add('server_healthy_after_sequence',(metrics_snapshot.get('live') or {}).get('state') in ('idle','unloaded'),
  (metrics_snapshot.get('live') or {}).get('state'))
 add('all_requests_finished',all(a['finish'] in ('stop','length') for a in answers),
  sorted({str(a['finish']) for a in answers}))
 add('answer_present_every_step',all(a['content'] or a['reasoning_chars'] for a in answers),
  [a['step'] for a in answers if not (a['content'] or a['reasoning_chars'])])
 # only steps the harness could attribute to a server record are held to this; a
 # concurrent burst is accounted for by concurrent_groups_attributed instead
 carried=[a['step'] for a in answers if a['op']=='user' and a['user_turns']>1 and a['step'] in by_step
  and by_step[a['step']]['record'].get('prompt_total') is None]
 add('server_records_present',not carried,carried)
 add('server_record_count_matches_requests',len(rows)==len(answers),dict(records=len(rows),requests=len(answers)))
 add('correlation_complete',correlation['unmatched_requests']==0 and correlation['unmatched_engine_lines']==0,
  correlation)
 add('engine_log_without_fault',not ENGINE_FAULT.search(engine_text),
  (ENGINE_FAULT.search(engine_text) or None) and engine_text[max(0,ENGINE_FAULT.search(engine_text).start()):][:200])
 # the flag is the ONLY thing that differs between arms, and it is checked from the config
 # the arm actually ran with, not from the command line the harness meant to use
 add('image_flag_matches_the_running_config',(IMAGE_FLAG in (cfg_args or []))==(arm=='candidate_on'),
  dict(arm=arm,flag_in_config=IMAGE_FLAG in (cfg_args or [])))
 if arm!='candidate_on':
  add('no_image_checkpoint_lines_in_log','IMAGE_CHECKPOINT' not in engine_text,
   [l for l in engine_text.splitlines() if 'IMAGE_CHECKPOINT' in l][:4])
 stable=next((r for r in rows if r['record'].get('_step')=='stable_image/2'),None)
 stable_reuse=(stable or {}).get('engine',{}).get('reused') if stable else None
 add('stable_image_history_reuse_positive',stable_reuse is not None and stable_reuse>0,stable_reuse,required=False)
 if arm=='candidate_on':
  positive=[r for r in rows if r['record'].get('_step','').startswith('qualified_rewrite/') and
   r['record']['_step']!='qualified_rewrite/1']
  add('qualified_after_image_restore',bool(positive) and all(r.get('restored') is True and
   r['engine']['reused']>0 for r in positive),
   [dict(step=r['record']['_step'],restored=r.get('restored'),reused=r['engine']['reused']) for r in positive])
 geometry_answers=[a for a in answers if a['step'].startswith('qualified_rewrite/')]
 add('qualified_geometry_prompt_matches',len(geometry_answers)==3 and all(
  (a.get('usage') or {}).get('prompt_tokens')==a.get('expected_prompt') for a in geometry_answers),
  [dict(step=a['step'],expected=a.get('expected_prompt'),actual=(a.get('usage') or {}).get('prompt_tokens')) for a in geometry_answers])
 # This bounded HTTP qualification is deliberately serial. Concurrent admissions
 # can yield and resume, producing multiple native summaries for ONE HTTP request.
 # Their attribution needs a separate scheduler test, not relaxed count matching.
 add('serial_scope_complete',len(answers)==10 and all(
  not a['step'].startswith('interleave_four/') for a in answers),
  dict(requests=len(answers),concurrency='NOT_TESTED'))
 add('serial_spans_only',not groups and all(r['record'].get('_step') for r in rows),
  dict(groups=len(groups),attributed_requests=len(rows)))
 contaminated=[a['step'] for a in answers if any(m in a['content'] for m in a['markers'])]
 add('no_cross_conversation_contamination',not contaminated,contaminated)
 text=[a for a in answers if a['step'].startswith('text_only_after_images/')]
 add('text_only_after_images_ok',len(text)==2 and all(a['content'] or a['reasoning_chars'] for a in text),
  [a['step'] for a in text])
 reuse=[dict(step=a['step'],conv=a['conv'],expected=expectations(a['step']),
   attributed=a['step'] in by_step,
   group=next((g['group'] for g in groups if a['step'] in g['steps']),None),
   reused=(by_step.get(a['step']) or {}).get('engine',{}).get('reused'),
   read=(by_step.get(a['step']) or {}).get('engine',{}).get('read'),
   session=(by_step.get(a['step']) or {}).get('engine',{}).get('session'),
   result=(by_step.get(a['step']) or {}).get('result'),
   prompt_total=(by_step.get(a['step']) or {}).get('record',{}).get('prompt_total'),
   server_duration_s=(by_step.get(a['step']) or {}).get('record',{}).get('duration_s'),
   client_elapsed_s=a['client_elapsed_s'],finish=a['finish'],answer_chars=len(a['content']),
   reasoning_chars=a['reasoning_chars'],tool_calls=a['tool_calls']) for a in answers]
 return checks,reuse

def divergence(left,right):
 """Rendered-token and answer divergence between two arms, REPORTED not required: the
 arms may take different prefill schedules, so byte equality is not the goal."""
 steps={}
 for row in left:steps[row['step']]=row
 out=[]
 for row in right:
  other=steps.get(row['step'])
  if other is None:continue
  # a concurrent group cannot be attributed per step in either arm, so its rows are
  # compared at group level (see four_clients_used_distinct_slots), not here
  if not row.get('attributed',True) or not other.get('attributed',True):continue
  if row.get('prompt_total')!=other.get('prompt_total'):
   out.append(dict(step=row['step'],kind='rendered_tokens',left=other.get('prompt_total'),right=row.get('prompt_total')))
  if (row.get('answer_chars') or 0)>0 and (other.get('answer_chars') or 0)>0 and row.get('finish')!=other.get('finish'):
   out.append(dict(step=row['step'],kind='finish',left=other.get('finish'),right=row.get('finish')))
 return out

def verdict(killed,late,exit_code,group_empty,report_status,arms_done,preflight=False,selftest='',interrupted='',
 worker_reaped=True,cleanup_complete=True,required_failures=(),server_groups_empty=True,plan_only=False):
 """Pure, mode-bound decision. A killed worker, a completion at/after the window
 deadline, a non-empty owned group, an unreaped worker, an incomplete cleanup, an
 interrupted supervisor, any REQUIRED failed check or a report whose status does not
 match the mode can never produce a PASS."""
 reasons=[]
 if interrupted:reasons.append('supervisor interrupted by '+str(interrupted))
 if killed and not interrupted:reasons.append('worker killed at the window deadline')
 if killed and interrupted:reasons.append('worker ended by the teardown of an interrupted run')
 if late:reasons.append('worker finished at or after the window deadline')
 if exit_code!=0:reasons.append('worker exit '+str(exit_code))
 if not worker_reaped:reasons.append('worker still unowned after the cleanup allowance')
 if not group_empty:reasons.append('owned process group still present after the cleanup allowance')
 if not cleanup_complete:reasons.append('cleanup did not finish inside the allowance')
 if not server_groups_empty:reasons.append('a temporary server group this run started was still present')
 accepted=('PASS_IMAGE_HTTP_PLAN_ONLY' if plan_only else 'PASS_IMAGE_HTTP_SMOKE_PREFLIGHT_ONLY' if preflight else
  'PASS_IMAGE_HTTP_SMOKE_SELFTEST' if selftest else 'PASS_BOUNDED_SERIAL_IMAGE_HTTP_SMOKE')
 if report_status!=accepted:reasons.append('worker report status '+str(report_status)+' not accepted for mode '
  +('preflight' if preflight else ('selftest:'+selftest if selftest else 'real')))
 if not arms_done:reasons.append('not every arm completed')
 if required_failures:reasons.append('required checks failed: '+','.join(required_failures))
 return (not reasons),reasons

def plan_checks(report,a):
 """What --plan-only proves about the run it did NOT start: loopback only, no production
 port, exactly the image flag differing between arms, no live credential copied, the two
 engines kept apart, and the request table. Nothing here touches the model."""
 checks=[];planned=report.get('plan') or []
 def add(name,ok,value=None):checks.append(dict(name=name,ok=bool(ok),required=True,value=value))
 add('every_arm_planned',len(planned)==len(a.arms),[x['arm'] for x in planned])
 add('listener_is_loopback_only',all(x['host']=='127.0.0.1' for x in planned))
 add('production_port_not_used',all(x['port']!=PRODUCTION_PORT for x in planned),[x['port'] for x in planned])
 # arms run one at a time, so a port may legitimately repeat; what matters is that every
 # port is a temporary loopback one and never the production listener
 add('ports_in_temporary_range',all(11500<=x['port']<=11599 for x in planned),[x['port'] for x in planned])
 add('image_flag_only_on_candidate_on',[x['image_flag'] for x in planned]==[x['arm']=='candidate_on' for x in planned],
    {x['arm']:x['image_flag'] for x in planned})
 add('baseline_uses_deployed_engine',all(x['engine']==str(a.baseline_engine) for x in planned
    if x['arm']=='baseline'))
 add('candidate_uses_qualified_engine',all(x['engine']==str(a.engine) for x in planned if x['arm']!='baseline'))
 add('vision_section_copied_from_live',bool(planned) and all(x['vision'] for x in planned),
    {x['arm']:x['vision'] for x in planned})
 add('live_credential_not_copied','api_key' not in json.dumps(planned))
 add('request_table_complete',len(report.get('steps') or [])==len(plan('smoke-x'))+3,len(report.get('steps') or []))
 report['checks']=checks
 return all(c['ok'] for c in checks if c['required'])

def geometry_selftest_steps():
 return [dict(step='qualified_rewrite/'+str(i+1),conv='qualified',op='geometry',expected_prompt=1400,
  body=dict(messages=[dict(role='user',content='stand-in geometry only')])) for i in range(3)]

def selftest_plan(scenario,arm,run_marker):
 steps=[]
 fake_steps=geometry_selftest_steps()+plan(run_marker)
 for index,step in enumerate(fake_steps):
  reused=0 if step['step'].endswith('/1') or '/1c' in step['step'] else (
   0 if scenario=='zero_stable_reuse' and step['step']=='stable_image/2' else 512)
  result='new' if reused==0 else ('checkpoint' if arm=='candidate_on' and scenario!='no_restore' else 'hit')
  steps.append(dict(step=step['step'],conv=step['conv'],nth=sum(1 for s in fake_steps[:index+1]
   if s['conv']==step['conv']),reused=reused,result=result,
   prompt_delta=(-3 if scenario=='divergence' and arm=='candidate_off' else 0),
   engine_error=scenario=='engine_error' and step['step']=='changed_pixels/1',
   missing_record=scenario=='missing_record' and step['step']=='system_rewrite/1',
   cross_talk=scenario=='cross_talk' and step['step']=='text_only_after_images/1',
   finish='stop'))
 return dict(arm=arm,scenario=scenario,marker=run_marker,steps=steps,health_images=scenario!='no_encoder',
  hang=scenario in ('hung_server','stuck_group'),descendant=scenario=='stuck_group',
  slow_start_s=25 if scenario=='slow_server' else 0)

def fake_server(port,plan_path,log_path):
 """Selftest only: a loopback stand-in for serve/server.py that replays the endpoint and
 engine-log SHAPES the real server produces (health, metrics records, completion,
 KVCACHE / strata serve / KVREPLAY lines). It proves the harness drives requests,
 correlates logs, refuses bad states and cleans up. It is NOT the real server and
 proves nothing about models, encoders, GPUs or checkpoint internals."""
 import threading as th
 plan_data=json.loads(Path(plan_path).read_text());log=Path(log_path);log.write_text('')
 state=dict(lock=th.Lock(),index=0,records=[],arrival=0)
 def tokens(messages):
  n=0;images=0
  for m in messages:
   content=m.get('content')
   if isinstance(content,str):n+=len(content.split())
   elif isinstance(content,list):
    for part in content:
     if part.get('type')=='text':n+=len(str(part.get('text','')).split())
     elif part.get('type') in ('image_url','image'):images+=1
  return n+8*len(messages)+1024*images
 class Handler(BaseHTTPRequestHandler):
  def log_message(self,*a):pass
  def _send(self,payload):
   body=json.dumps(payload).encode()
   self.send_response(200);self.send_header('Content-Type','application/json')
   self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
  def do_GET(self):
   with state['lock']:
    if self.path=='/health':
     self._send(dict(status='ok',max_context=262144,model='Qwen3.6-35B-A3B',images=plan_data['health_images'],
      api_key=False,image_generation=False,loaded=True,service='strata'));return
    self._send(dict(engine=dict(model='Qwen3.6-35B-A3B',max_context=262144,images=plan_data['health_images'],
     unified_kv='1',kv_protocol='1',batch_slots=4),live=dict(state='idle',queued=0),
     requests=list(reversed(state['records'])),requests_kept=len(state['records']),
     totals=dict(requests=len(state['records']))));return
  def do_POST(self):
   body=json.loads(self.rfile.read(int(self.headers.get('Content-Length','0'))))
   messages=body.get('messages') or []
   with state['lock']:
    # The plan is consumed in ARRIVAL order. Serial steps arrive in plan order; the four
    # concurrent ones may permute, but every value the fake reports inside such a group is
    # identical, so only the injected cross-talk differs - and it names a conversation
    # OUTSIDE the group, so the contamination check fires whichever step receives it.
    if state['index']>=len(plan_data['steps']):
     self.send_error(500,'plan exhausted');return
    s=plan_data['steps'][state['index']];conv=s['conv'];state['index']+=1
    session=state['arrival']%4;state['arrival']+=1
    total=1400 if s['step'].startswith('qualified_rewrite/') else tokens(messages)+int(s.get('prompt_delta') or 0)
    reused=int(s.get('reused') or 0);read=max(0,total-reused);prompt_ms=max(1.0,read*1.3)
    epoch=state['arrival']
    lines=['KVCACHE session=%d generation=1 epoch=%d state=active tokens=%d bytes=580911104 '
     'reserved_tokens=%d result=%s replay=%d'%(session,epoch,total,total,s['result'],
      0 if s['result'] in ('hit','checkpoint') else 1)]
    if s['result']=='checkpoint':
     lines.append('Qwen3.6 prefix checkpoint: source=0 session=%d restored_tokens=%d'%(session,reused))
    lines.append('strata serve: session %d: prompt %d tokens = %d reused + %d read in %.1f ms, 3 generated in '
     '40.0 ms (75.0 tok/s), drafts accepted 0 of 0 (%s)'%(session,total,reused,read,prompt_ms,
      s.get('finish','stop')))
    if s.get('engine_error'):lines.append('ERR simulated engine fault')
    with Path(log_path).open('a') as stream:stream.write('\n'.join(lines)+'\n')
    content='the answer is short' if not s.get('cross_talk') else 'about '+plan_data['marker']+'-stable'
    if not s.get('missing_record'):
     state['records'].append(dict(time=time.time(),duration_s=0.2,finish=s.get('finish','stop'),
      prompt_tokens=total-reused,reused=reused,output_tokens=3,prompt_total=total,prompt_read=read,
      prompt_ms=prompt_ms,decode_ms=40.0))
   self._send(dict(choices=[dict(message=dict(role='assistant',content=content,reasoning_content='thinking',
    tool_calls=[]),
    finish_reason=s.get('finish','stop'))],usage=dict(prompt_tokens=total,completion_tokens=3)))
 if plan_data.get('slow_start_s'):time.sleep(plan_data['slow_start_s'])
 server=HTTPServer(('127.0.0.1',port),Handler)
 if plan_data.get('hang'):
  import signal as sg
  for name in ('SIGTERM','SIGHUP','SIGINT'):
   try:sg.signal(getattr(sg,name),lambda *_:None)
   except (ValueError,AttributeError):pass
  with state['lock']:
   if plan_data.get('descendant'):
    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(300)','#Q36OWNEDPROBE'])
    state['descendant']=child.pid
 server.serve_forever()

def worker(a,budget):
 out=a.out;report=dict(status='STARTED',mode=('plan-only' if a.plan_only else 'preflight' if a.preflight
  else 'selftest:'+a.selftest if a.selftest else 'real'),source_commit=os.environ.get('STRATA_IMAGE_HTTP_SOURCE_COMMIT',''),
  scope='serial_image_http_10_requests_per_arm',concurrency='NOT_TESTED',
  window_seconds=a.window_seconds,cleanup_allowance_seconds=a.cleanup_seconds,
  server_root=str(a.server_root),engine_sha256=digest(a.engine) if a.engine.exists() else None,
  baseline_engine_sha256=digest(a.baseline_engine) if a.baseline_engine and a.baseline_engine.exists() else None,
  qualified_args=QUALIFIED,native_oracle='NOT_REPEARED here (completed separately: IMAGE_CHECKPOINT_ORIN_GATE PASS)',
  tensor_exactness='NOT_CHECKED by this HTTP smoke')
 try:
  images={'A':out/'image-A.png','B':out/'image-B.png'}
  if a.image_a:images['A']=a.image_a
  if a.image_b:images['B']=a.image_b
  for path in images.values():
   if not path.is_file():png(path,(200,40,40) if path.name.endswith('A.png') else (40,60,200))
  report['image_sha256']={k:digest(v) for k,v in images.items()}
  source=json.loads(a.source_config.read_text())
  run_marker='smoke-'+os.environ.get('STRATA_IMAGE_HTTP_STAMP',str(os.getpid()))
  steps=plan(run_marker)
  if not a.selftest:
   from q36_image_http_geometry import load as load_geometry
   geometry,bodies=load_geometry(a.geometry,a.model,a.engine,a.source_config,a.server_root,a.baseline_engine)
   report['geometry']=geometry
   additions=[dict(step='qualified_rewrite/'+str(i+1),conv='qualified',op='geometry',body=body,
    expected_prompt=geometry['request_prompt_tokens'][i]) for i,body in enumerate(bodies)]
   steps=additions+steps
  else:
   steps=geometry_selftest_steps()+steps
  if len(steps)!=10 or any(s['step'].startswith('interleave_four/') for s in steps):
   raise RuntimeError('serial HTTP scope requires exactly ten non-burst requests')
  if a.plan_only:
   planned=[]
   for arm in a.arms:
    engine=a.engine if arm!='baseline' else a.baseline_engine
    if engine is None:raise RuntimeError('arm %s needs --baseline-engine'%arm)
    port=a.port or free_port()
    cfg=plan_config(source,engine,port,a.out/('config-%s.json'%arm),a.out/('%s-engine.log'%arm),a.out,arm,a.model)
    planned.append(dict(arm=arm,engine=str(engine),port=port,host=cfg['host'],config=str(a.out/('config-%s.json'%arm)),
     image_flag=IMAGE_FLAG in cfg['args'],args=cfg['args'],vision=sorted(cfg['vision'])))
   report['plan']=planned;report['steps']=[s['step'] for s in steps];report['arms_done']=True
   ok=plan_checks(report,a)
   report['marker']=PLAN_MARKER if ok else None;report['status']='PASS_IMAGE_HTTP_PLAN_ONLY' if ok else 'FAIL_plan_checks'
   return 0
  if a.preflight:
   plan_arms=[]
   for arm in a.arms:
    engine=a.engine if arm!='baseline' else a.baseline_engine
    port=a.port or free_port()
    cfg=plan_config(source,engine,port,out/('config-%s.json'%arm),out/('%s-engine.log'%arm),out,arm,a.model)
    plan_arms.append(dict(arm=arm,engine=str(engine),image_flag=IMAGE_FLAG in cfg['args'],args=cfg['args'],
     vision=sorted(cfg['vision']),port=port,host=cfg['host']))
   report['plan']=plan_arms;report['steps']=[s['step'] for s in steps];report['arms_done']=True
   if not plan_checks(report,a):raise RuntimeError('preflight config/flag/port checks failed')
   report['marker']=PREFLIGHT_MARKER;report['status']='PASS_IMAGE_HTTP_SMOKE_PREFLIGHT_ONLY'
   return 0
  reports={}
  for arm in a.arms:
   budget.check('arm '+arm)
   if time.monotonic()>=budget.deadline:raise TimeoutError('window exhausted before arm '+arm)
   if not a.selftest:no_models()
   engine=a.engine if arm!='baseline' else a.baseline_engine
   if engine is None:raise RuntimeError('arm %s needs --baseline-engine'%arm)
   port=free_port()
   cfg=plan_config(source,engine,port,out/('config-%s.json'%arm),out/('%s-engine.log'%arm),out,arm,a.model)
   arm_out=out
   handle=Arm(arm_out,arm,cfg,sys.executable,a.server_root,port)
   fake=None
   if a.selftest:
    fake=out/('fake-plan-%s.json'%arm);fake.write_text(json.dumps(selftest_plan(a.selftest,arm,run_marker),indent=2)+'\n')
   child=handle.start(fake_plan=fake)
   client=Client(port,deadline=budget.deadline)
   deadline=budget.deadline
   started=time.monotonic()
   health=None;snapshot=None;answers=[];rows=[];correlation={};engine_text=''
   try:
    health=wait_healthy(client,deadline,out,arm,child,arm_out/(arm+'-server-stdout.txt'))
    log_path=out/('%s-engine.log'%arm)
    steps_out,snapshot,windows=run_steps(client,health.get('model') or 'strata-q36',steps,a.max_tokens,images,
     run_marker,budget.deadline,arm,log_path,out=out)
    log_bytes=log_path.read_bytes() if log_path.is_file() else b''
    engine_text=log_bytes.decode('utf-8','replace')
    answer_by_step={a['step']:a for a in steps_out}
    rows,correlation,groups,span_audits=attribute_spans(windows,steps_out,log_path.read_bytes(),out,arm)
    from collections import Counter
    records=snapshot.get('requests') or []
    if len(records)!=len(steps_out) or Counter(r.get('prompt_total') for r in records)!=Counter(
     (answer.get('usage') or {}).get('prompt_tokens') for answer in steps_out):
     raise RuntimeError('complete metrics request counts/token multiset differ from HTTP responses')
    events=parse_engine_log(engine_text)
    checks,reuse=arm_checks(arm,steps_out,rows,correlation,health,snapshot,engine_text,cfg['args'],groups)
    if not a.selftest and arm=='candidate_on':
     thresholds={str(r['turn']):r for r in geometry['audit']['rewrites']}
     for row in rows:
      step=row['record'].get('_step','')
      if step.startswith('qualified_rewrite/') and step.rsplit('/',1)[1] in thresholds:
       bound=thresholds[step.rsplit('/',1)[1]]
       if not bound['image_end']<=row['engine']['reused']<=bound['lcp']:
        raise RuntimeError('reuse did not land after complete image and before audited divergence')
    report['steps']=[s['step'] for s in steps]
    reports[arm]=dict(health=health,engine_info=snapshot.get('engine'),checks=checks,reuse=reuse,
     correlation=correlation,span_audits=span_audits,log_lines=len(engine_text.splitlines()),
     engine_events=len(events),sessions=sorted({e['session'] for e in events if e['kind'] in ('request','slot')}))
    print('HTTP_SMOKE_PROGRESS arm=%s stage=arm_done elapsed_s=%d requests=%d sessions=%d'%(
     arm,int(time.monotonic()-started),len(steps_out),len(reports[arm]['sessions'])),flush=True)
   finally:
    cleanup_deadline=budget.deadline if budget.left()>0 else time.monotonic()+a.cleanup_seconds
    handle.stop(cleanup_deadline)
    stop=handle.stop_result
    if not stop['owned_group_empty'] or not stop['worker_reaped']:raise RuntimeError('temporary server group did not drain; operator cgroup cleanup required')
    if not a.selftest:no_models()  # includes final arm; detect any escaped model/encoder child
    report.setdefault('arm_cleanup',{})[arm]=stop
    (out/(arm+'-report.json')).write_text(json.dumps(reports.get(arm,{}),indent=2)+'\n')
  required=[]
  for arm,rep in reports.items():
   for check in rep['checks']:
    if check['required'] and not check['ok']:required.append(arm+':'+check['name'])
  report['required_failures']=required
  report['divergence']=dict(baseline_vs_candidate_off=divergence(
    reports.get('baseline',{}).get('reuse',[]),reports.get('candidate_off',{}).get('reuse',[])),
   candidate_off_vs_candidate_on=divergence(reports.get('candidate_off',{}).get('reuse',[]),
    reports.get('candidate_on',{}).get('reuse',[])))
  if len(reports)!=len(a.arms):raise RuntimeError('an arm produced no report')
  # the worker records what it measured; the pass marker is the SUPERVISOR's to print, and
  # a run with failed required checks carries no marker at all
  print('HTTP_SMOKE_PROGRESS stage=arms action=done arms=%d required_failures=%d'%(len(reports),len(required)),
   flush=True)
  if required:
   report['status']='FAIL_required_checks';report['marker']=None;return 1
  report['arms_done']=True
  report['marker']=(('IMAGE_HTTP_SMOKE_SELFTEST PASS scenario='+a.selftest) if a.selftest else PASS_MARKER)
  report['status']=('PASS_IMAGE_HTTP_SMOKE_SELFTEST' if a.selftest else 'PASS_BOUNDED_SERIAL_IMAGE_HTTP_SMOKE')
  return 0
 except BaseException as exc:
  report['status']='FAIL';report['error']=str(exc)[:400];raise
 finally:
  (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')

def supervise(a,budget):
 """The only entity that may print a verdict marker. One window deadline, ONE cleanup
 allowance, recorded signals, persistent cleanup state, and a record-failure path that
 never restarts cleanup or prints a pass."""
 out=a.out;log=open(out/'worker-stdout.txt','wb')
 label='IMAGE_HTTP_SMOKE_SELFTEST' if a.selftest else 'IMAGE_HTTP_SMOKE'
 recorded=[False]
 def write_record(**fields):
  fields.setdefault('operator_cgroup_required',True)
  target=out/'watchdog.json'
  if getattr(a,'fault',None)=='record':
   (out/'record-fault').write_text('fault injection marker\n');target=out/'record-fault'/'watchdog.json'
  target.write_text(json.dumps(fields,indent=2)+'\n');recorded[0]=True
 def record_failed(exc):
  for line in ('HTTP_SMOKE_PROGRESS stage=record action=failed error='+type(exc).__name__+': '+str(exc)[:200],
   'HTTP_SMOKE_PROGRESS stage=record action=no_restart cleanup_state_final=1 operator_cgroup_required=1'):
   try:print(line,flush=True,file=sys.stderr)
   except Exception:pass
  try:print(label+' FAIL result_record_failed_'+type(exc).__name__,flush=True)
  except Exception:pass
 interrupted=[]
 def note(signum,frame):interrupted.append(signum)
 previous={}
 SIGNAL_NAMES={signal.SIGINT:'SIGINT',signal.SIGTERM:'SIGTERM',signal.SIGHUP:'SIGHUP'}
 for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):
  try:previous[sig]=signal.signal(sig,note)
  except (ValueError,OSError):pass
 try:
  child=None;cleanup=None
  child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--worker','--deadline',
   repr(budget.deadline)],stdin=subprocess.DEVNULL,stdout=log,stderr=sys.stderr,start_new_session=True)
  cleanup=CleanupState(child,a.cleanup_seconds)
  killed=False;deadline_seen=False;exit_code=None
  next_progress=time.monotonic()+PROGRESS_S
  while exit_code is None:
   exit_code=worker_status(child)
   if exit_code is not None:break
   if time.monotonic()>=next_progress:
    print('HTTP_SMOKE_PROGRESS stage=window elapsed_s=%d remaining_s=%d window_s=%d'%(
     int(a.window_seconds-budget.left()),max(0,int(budget.left())),a.window_seconds),flush=True)
    next_progress+=PROGRESS_S
   if budget.expired():
    deadline_seen=True
    if not killed:
     print('HTTP_SMOKE_PROGRESS stage=watchdog action=killpg elapsed_s=%d'%(a.window_seconds-budget.left()),
      flush=True,file=sys.stderr)
     killed=killed or group_kill(child)=='present'
    break
   if interrupted:
    deadline_seen=True
    print('HTTP_SMOKE_PROGRESS stage=watchdog action=teardown signal='+
     ','.join(SIGNAL_NAMES.get(s,'SIG'+str(s)) for s in interrupted)+' cleanup_allowance_s=%d'%a.cleanup_seconds,
     flush=True,file=sys.stderr)
    break
   time.sleep(min(0.25,max(0.0,budget.left())))
  cleanup.exit_code=exit_code
  exit_code,group_status,group_empty,reaped=teardown(cleanup)
  # The worker starts each temporary server in its OWN session (a model/encoder must not
  # share the worker's group), so a killed worker can leave one behind: reap exactly the
  # groups THIS run registered, under the same cleanup deadline.
  leftovers=reap_recorded_groups(out,cleanup.deadline if cleanup.deadline else time.monotonic()+a.cleanup_seconds)
  servers_empty=all(x['group_empty'] for x in leftovers) and 'pid_no_longer_ours_not_signalled' not in [x['kind'] for x in leftovers]
  killed=killed or deadline_seen or bool(interrupted) or (group_status=='present' and exit_code is None)
  signals=','.join(sorted({SIGNAL_NAMES.get(s,'SIG'+str(s)) for s in interrupted}))
  late=(deadline_seen and not interrupted) or (exit_code==0 and budget.expired())
  report={}
  try:report=json.loads((out/'result.json').read_text())
  except (OSError,ValueError):report={'status':'MISSING'}
  failures=report.get('required_failures') or []
  passed,reasons=verdict(killed,late,exit_code if exit_code is not None else -1,group_empty,report.get('status'),
   bool(report.get('arms_done')),a.preflight,a.selftest,signals,reaped,group_empty and reaped,failures,servers_empty,
    a.plan_only)
  if getattr(a,'fault',None)=='crash':
   raise RuntimeError('fault injection: post-teardown smoke failure')
  try:
   write_record(window_seconds=a.window_seconds,cleanup_allowance_seconds=a.cleanup_seconds,killed=killed,
    deadline_crossed=late,worker_exit=exit_code,worker_reaped=reaped,owned_group_empty=group_empty,
    cleanup_complete=group_empty and reaped,worker_report_status=report.get('status'),interrupted_signals=signals,
    required_failures=failures,owned_server_groups=leftovers,server_groups_empty=servers_empty,
    passed=passed,reasons=reasons)
  except OSError as record_exc:
   record_failed(record_exc);return 1
 except BaseException as exc:
  if child is not None:
   try:teardown(cleanup)
   except Exception:pass
   exit_code,_,group_empty,reaped=cleanup.result()
  else:exit_code=None;group_empty=False;reaped=False
  if not recorded[0]:
   try:
    write_record(window_seconds=a.window_seconds,cleanup_allowance_seconds=a.cleanup_seconds,
     killed=killed or (cleanup is not None and cleanup.group_status=='present'),deadline_crossed=False,
     worker_exit=exit_code,
     worker_reaped=reaped,owned_group_empty=group_empty,cleanup_complete=group_empty and reaped,
     worker_report_status='MISSING',interrupted_signals='',required_failures=[],
     owned_server_groups=reap_recorded_groups(out,cleanup.deadline if cleanup and cleanup.deadline else time.monotonic()),
     passed=False,
     reasons=['supervisor error '+type(exc).__name__+': '+str(exc)[:200]])
   except OSError as record_exc:
    record_failed(record_exc);raise
  try:print(label+' FAIL supervisor_error_'+type(exc).__name__,flush=True)
  except Exception:pass
  raise
 finally:
  log.close()
  for sig,handler in previous.items():
   try:signal.signal(sig,handler)
   except (ValueError,OSError):pass
 if passed:
  # the verdict marker comes from the verdict-checked report, so a plan-only or preflight
  # pass can never print the real smoke marker (and vice versa)
  if report.get('marker'):print(report['marker'],flush=True)
  print(label+'_SUPERVISOR PASS window_s=%d cleanup_allowance_s=%d owned_group_empty=1'%(
   a.window_seconds,a.cleanup_seconds),flush=True)
  return 0
 print(label+' FAIL '+' '.join(r.replace(' ','_') for r in reasons),flush=True)
 return 1

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('engine','model','source-config','server-root','out'):
  p.add_argument('--'+name,type=Path)
 p.add_argument('--baseline-engine',type=Path)
 p.add_argument('--geometry',type=Path)
 p.add_argument('--arms',default=None)
 p.add_argument('--window-seconds',type=int,default=WINDOW_DEFAULT)
 p.add_argument('--cleanup-seconds',type=int,default=CLEANUP_ALLOWANCE_DEFAULT)
 p.add_argument('--port',type=int,default=0)
 p.add_argument('--max-tokens',type=int,default=24)
 p.add_argument('--image-a',type=Path);p.add_argument('--image-b',type=Path)
 p.add_argument('--expect-engine-sha256');p.add_argument('--preflight',action='store_true')
 p.add_argument('--plan-only',action='store_true',help='write the exact commands and temporary configs and check them without starting anything; needs no window')
 p.add_argument('--stage',type=Path,help=argparse.SUPPRESS)
 p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
 p.add_argument('--selftest',choices=SELFTESTS,help=argparse.SUPPRESS)
 p.add_argument('--fault',choices=('record','crash'),help=argparse.SUPPRESS)
 p.add_argument('--fake-server',nargs=3,metavar=('PORT','PLAN','LOG'),help=argparse.SUPPRESS)
 p.add_argument('--deadline',type=float,help=argparse.SUPPRESS)
 a=p.parse_args()
 global STAGE
 if a.fake_server:
  fake_server(int(a.fake_server[0]),Path(a.fake_server[1]),Path(a.fake_server[2]));return 0
 if a.arms is None:a.arms=','.join(ARMS)
 arms=tuple(x.strip() for x in a.arms.split(',') if x.strip())
 a.arms=arms
 for name in ('engine','model','source_config','server_root','out'):
  if getattr(a,name) is None:p.error('--%s is required'%name.replace('_','-'))
 if not arms or len(set(arms))!=len(arms):p.error('arms must be nonempty and unique')
 if any(x not in ARMS for x in arms):p.error('arms must be a subset of '+','.join(ARMS))
 if not a.selftest and not a.plan_only and os.environ.get('STRATA_IMAGE_HTTP_REVIEW_WINDOW')!='1':
  p.error('the HTTP smoke starts a model and encoder, so it runs only in the approved Orin window: set '
   'STRATA_IMAGE_HTTP_REVIEW_WINDOW=1 and bring the live services down through the manager first')
 if not a.selftest and not a.plan_only and (a.server_root.resolve()!=ROOT or ROOT!=Path('/mnt/ssd/Strata-Orin/worktrees/multimodal-cache')):
  p.error('only the approved window, with STRATA_IMAGE_HTTP_ROOT set to the read-only server checkout')
 if 'baseline' in arms and a.baseline_engine is None:p.error('--baseline-engine required for the baseline arm')
 if getattr(a,'fault',None) and not a.selftest:p.error('fault injection requires --selftest')
 if a.selftest:STAGE=a.out.resolve().parent
 if a.plan_only and a.stage is not None:STAGE=a.stage.resolve()
 if a.plan_only and (a.engine is None or a.model is None or a.baseline_engine is None):
  p.error('--plan-only still needs --engine, --baseline-engine and --model to show the real commands')
 if not 1<=a.window_seconds<=WINDOW_MAX:p.error('window must be 1..%ds'%WINDOW_MAX)
 if not 1<=a.cleanup_seconds<=CLEANUP_ALLOWANCE_MAX:p.error('cleanup allowance must be 1..%ds'%CLEANUP_ALLOWANCE_MAX)
 if not a.out.resolve().is_relative_to(STAGE) or a.out.resolve()==STAGE:
  p.error('all smoke results must live in a NEW directory under '+str(STAGE))
 if a.port and (a.port==PRODUCTION_PORT or not 11500<=a.port<=11599):p.error('refusing the production port')
 for path in (a.model,a.source_config,a.server_root):
  if not path.exists():p.error('missing '+str(path))
 if a.server_root.resolve()==STAGE or STAGE in a.server_root.resolve().parents:
  p.error('the server checkout must be read-only code, never the staging directory')
 if a.expect_engine_sha256 and digest(a.engine)!=a.expect_engine_sha256:
  p.error('candidate engine hash differs from the qualified build')
 for path in (a.engine,)+(a.baseline_engine,) if a.baseline_engine else ():
  if not path.is_file():p.error('missing engine '+str(path))
 if not a.selftest and not a.geometry:p.error('--geometry from the CPU-only actual-GGUF preflight is required')
 if a.worker:return worker(a,Budget(a.window_seconds,'worker',a.deadline))
 if a.out.exists():p.error('results directory already exists; preserve old artifacts')
 a.out.mkdir(parents=True)
 if not a.preflight and not a.selftest:
  print('HTTP_SMOKE_PROGRESS stage=window stage_root=%s out=%s window_s=%d cleanup_s=%d production_health=%s'%(
   STAGE,a.out,a.window_seconds,a.cleanup_seconds,PRODUCTION_HEALTH),flush=True)
 return supervise(a,Budget(a.window_seconds))

if __name__=='__main__':sys.exit(main())
