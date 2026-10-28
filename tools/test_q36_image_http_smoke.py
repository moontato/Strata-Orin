"""CPU/mock tests for the bounded HTTP image-checkpoint smoke harness.
No model, no encoder, no GPU, no service: the temporary listener is a loopback stand-in
that produces the same server records and engine log shapes as serve/server.py.
"""
import json,os,signal,subprocess,sys,time
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import q36_image_http_smoke as smoke
import itertools
counter=itertools.count()
SMOKE=ROOT/'tools/q36_image_http_smoke.py'

def pngs(tmp_path):
 return {name:smoke.png(tmp_path/('image-%s.png'%name),colour) for name,colour in
  (('A',(200,40,40)),('B',(40,60,200)))}

def source_config(tmp_path):
 """A stand-in for the live config: it carries an api_key, a non-loopback host and the
 production port, none of which may reach a temporary config."""
 path=tmp_path/'config.json'
 path.write_text(json.dumps({'model_name':'q36','tokenizer':'/models/q36.gguf','api_key':'secret-value',
  'host':'100.100.10.3','port':11436,'replicas':2,'vision':{'model':'/models/vit.fp16.gguf',
  'mmproj':'/models/mmproj.bin','gpu':True,'elastic':True,'exe':'/opt/vision'}}))
 return path

def config_for(tmp_path,arm,port=11501):
 source=json.loads(source_config(tmp_path).read_text())
 engine=tmp_path/'engine'
 if not engine.exists():engine.write_text('fake binary\n')
 return smoke.plan_config(source,engine,port,tmp_path/('config-%s.json'%arm),tmp_path/'engine.log',
  tmp_path,arm,tmp_path/'model.gguf')

# ---- the request sequence ------------------------------------------------------------
def test_plan_covers_every_requested_exercise():
 steps={s['step'] for s in smoke.plan('m')}
 for name in ('stable_image/1','stable_image/2','earlier_user_rewrite/1','system_rewrite/1','changed_pixels/1',
  'interleave_four/1c1','interleave_four/1c2','interleave_four/1c3','interleave_four/1c4',
  'interleave_four/2c1','interleave_four/2c2','interleave_four/2c3','interleave_four/2c4',
  'text_only_after_images/1','text_only_after_images/2'):
  assert name in steps,name
 inter=[s for s in smoke.plan('m') if s['step'].startswith('interleave_four')]
 assert len(inter)==8 and len({s['conv'] for s in inter})==4
 text=[s for s in smoke.plan('m') if s['step'].startswith('text_only_after_images')]
 assert text and all('image' not in s for s in text)
 # every requested exercise happens AFTER an image has entered the history
 order=[s['step'] for s in smoke.plan('m')]
 for name in order[order.index('stable_image/2')+1:]:
  assert name in ('earlier_user_rewrite/1','system_rewrite/1','changed_pixels/1') or \
   name.startswith(('interleave_four','text_only_after_images')),name

def build_conversations(tmp_path,marker='mark'):
 """Replay the plan exactly as run_steps does. Returns the rendered messages of every
 conversation at the moment its last step was sent, plus the assistant replies that
 conversation's OWN requests produced (a copied conversation legitimately carries the
 source's replies - that is the point of the exercise)."""
 convs={};snapshots={};own={};imgs=pngs(tmp_path)
 for step in smoke.plan(marker):
  if step['conv'] not in convs:
   source=convs.get(step.get('source'))
   convs[step['conv']]=source.copy() if source is not None else smoke.Conversation('%s-%s'%(marker,step['conv']))
  conv=convs[step['conv']];op=step['op']
  if op=='user':conv.user(step['text'],step.get('image'))
  elif op=='rewrite_first_user':conv.rewrite_first_user(step['text'])
  elif op=='rewrite_system':
   conv.system=step['system']
   if step.get('text'):conv.user(step['text'])
  elif op=='replace_image':
   conv.replace_image(step['image'])
   if step.get('text'):conv.user(step['text'])
  else:raise AssertionError('unknown op '+op)
  content='reply-%s-%d'%(step['conv'],len(conv.render(imgs)))
  conv.reply({'role':'assistant','content':content,'reasoning_content':'thought-%s'%step['conv'],
   'tool_calls':[]})
  own.setdefault(step['conv'],[]).append(content)
  snapshots[step['conv']]=(conv.render(imgs),own[step['conv']])
 return snapshots

def test_replies_only_travel_inside_their_own_conversation(tmp_path):
 for name,(messages,own_replies) in build_conversations(tmp_path).items():
  roles=[m['role'] for m in messages]
  for index,role in enumerate(roles):
   # a reply is always immediately preceded, in this same conversation, by its request
   if role=='assistant':assert index and roles[index-1]=='user',(name,roles)
  assistants=[m['content'] for m in messages if m['role']=='assistant']
  # every reply this conversation carries was produced by a request of THIS conversation
  # or of the conversation it was copied from; nothing else can appear
  assert own_replies==assistants[-len(own_replies):],(name,assistants,own_replies)
  assert not [m for m in messages if m['role']=='user' and 'reply-' in str(m.get('content') or '')]
  assert all((m.get('reasoning_content') or '').startswith('thought-') for m in messages
   if m['role']=='assistant')

def test_copy_steps_change_only_what_they_claim(tmp_path):
 shots={k:v[0] for k,v in build_conversations(tmp_path).items()}
 stable,rewrite,system,pixels=shots['stable'],shots['rewrite'],shots['system'],shots['pixels']
 for name,messages in shots.items():
  roles=[m['role'] for m in messages]
  assert roles==['system']+ [r for pair in zip(['user']*len(roles),['assistant']*len(roles)) for r in pair][:len(roles)-1],   (name,roles)
 text=lambda ms,role:[str(m.get('content') or '') if isinstance(m.get('content'),str)
  else ' '.join(p.get('text','') for p in m['content'] if p.get('type')=='text') for m in ms if m['role']==role]
 # an edited earlier question keeps the system prompt and the image turn, and drops the
 # turns after the edit (they would be regenerated), so the request ends on a user message
 assert len(text(rewrite,'user'))==1 and len(text(stable,'user'))==2
 assert text(rewrite,'system')==text(stable,'system')
 assert text(rewrite,'user')[0]!=text(stable,'user')[0]
 # an early system rewrite keeps every user turn and changes only the system prompt
 assert text(system,'user')[:2]==text(stable,'user') and len(text(system,'user'))==3
 assert text(system,'system')!=text(stable,'system')
 def image_parts(ms):return [p['image_url']['url'] for m in ms for p in
  (m['content'] if isinstance(m['content'],list) else []) if isinstance(p,dict) and p.get('type')=='image_url']
 assert image_parts(pixels) and image_parts(pixels)!=image_parts(stable)
 assert text(pixels,'user')[:2]==text(stable,'user') and len(text(pixels,'user'))==3
 assert text(pixels,'system')==text(stable,'system')

# ---- the temporary server configuration ---------------------------------------------
@pytest.mark.parametrize('arm',('baseline','candidate_off','candidate_on'))
def test_temporary_config_keeps_the_live_service_out(tmp_path,arm):
 cfg=config_for(tmp_path,arm)
 assert cfg['host']=='127.0.0.1' and cfg['port']==11501 and cfg['port']!=smoke.PRODUCTION_PORT
 text=json.dumps(cfg)
 assert 'secret-value' not in text and '100.100.10.3' not in text and '"replicas"' not in text
 assert '--kv-preallocate' in cfg['args'] and '--kv-checkpoint-interval' in cfg['args']
 assert '--chunk' in cfg['args'] and '--kv-vision-max-tokens' in cfg['args']

def test_only_the_candidate_arm_carries_the_image_flag(tmp_path):
 arms={a:config_for(tmp_path,a,11501+i)['args'] for i,a in enumerate(smoke.ARMS)}
 assert [smoke.IMAGE_FLAG in arms[a] for a in smoke.ARMS]==[False,False,True]
 assert [x for x in arms['candidate_on'] if x!=smoke.IMAGE_FLAG]==arms['candidate_off']==arms['baseline']

def test_vision_section_is_copied_from_the_live_config(tmp_path):
 cfg=config_for(tmp_path,'candidate_on')
 assert cfg['vision']['model']=='/models/vit.fp16.gguf' and cfg['vision']['gpu'] is True
 assert cfg['vision']['exe']=='/opt/vision' and 'api_key' not in cfg['vision']

# ---- engine log reading and correlation ---------------------------------------------
ENGINE_LOG="""strata serve: session 1: prompt 1054 tokens = 0 reused + 1054 read in 4904.1 ms (stop)
KVCACHE session=1 generation=1 epoch=1 state=active tokens=1024 bytes=1048576 reserved_tokens=8192 result=checkpoint replay=0
strata batch: slot 1 takes a prompt of 1054 tokens = 0 reused + 1054 read in 4904.0 ms
strata serve: session 2: prompt 1079 tokens = 512 reused + 567 read in 341.2 ms (stop)
KVCACHE session=2 generation=2 epoch=1 state=active tokens=1024 bytes=1048576 reserved_tokens=8192 result=checkpoint replay=1
KVREPLAY session=2 tokens=512 reason=prefix_checkpoint
"""
def test_engine_log_yields_slots_reuse_and_restores():
 events=smoke.parse_engine_log(ENGINE_LOG)
 requests=[e for e in events if e['kind']=='request']
 assert [e['session'] for e in requests]==[1,2] and [e['prompt'] for e in requests]==[1054,1079]
 slots=[e for e in events if e['kind']=='slot']
 assert len(slots)==1 and slots[0]['session']==1 and slots[0]['reused']==0
 assert requests[1]['reused']==512 and requests[1]['read']==567 and requests[1]['prompt_ms']==pytest.approx(341.2)
 assert [e['kind'] for e in events if e['kind'] in ('kvcache','restored')]==['kvcache','kvcache','restored']
 assert [e['session'] for e in events if e['kind']=='restored']==[2]

def test_correlation_marks_restores_and_reports_strays():
 events=smoke.parse_engine_log(ENGINE_LOG)
 records=[dict(_step='a',prompt_total=1054,time=1.0),dict(_step='b',prompt_total=1079,time=2.0)]
 rows,extra=smoke.correlate(records,events)
 assert extra=={'unmatched_requests':0,'unmatched_engine_lines':0}
 assert [r['engine']['session'] for r in rows]==[1,2]
 assert rows[0]['restored'] is False and rows[1]['restored'] is True and rows[1]['result']=='checkpoint'
 rows,extra=smoke.correlate(records[:1],events)
 assert extra['unmatched_engine_lines']==1
 rows,extra=smoke.correlate(records+[dict(_step='c',prompt_total=999,time=3.0)],events)
 assert extra['unmatched_requests']==1

# ---- the per-arm verdicts -----------------------------------------------------------
def arm_inputs(tmp_path,arm='candidate_on',**change):
 """Synthesise the three inputs a real arm produces: the answers the API returned, the
 server's /metrics records, and the engine's own log text."""
 steps=smoke.plan('mark')
 answers=[dict(step=s['step'],conv=s['conv'],op=s['op'],user_turns=1+(i%2),content='ok',reasoning_chars=12,
   tool_calls=0,finish='stop',client_elapsed_s=0.4,started=float(i),started_wall=float(i),
   markers=['mark-%s'%c for c in sorted({x['conv'] for x in steps}) if c!=s['conv']])
  for i,s in enumerate(steps)]
 prompt=1051 if change.get('tokens_differ') else 1054
 lines=[]
 for i,s in enumerate(steps):
  session=i%4+1;reused=0 if s['step'].endswith('/1') or '/1c' in s['step'] else 512
  if change.get('zero_reuse') and s['step']=='stable_image/2':reused=0
  if change.get('second_group_zero') and s['step'].startswith('interleave_four/2'):reused=0
  restore=reused>0 and not change.get('no_restore')
  lines.append('strata serve: session %d: prompt %d tokens = %d reused + %d read in 400.0 ms (stop)'%
   (session,prompt,reused,prompt-reused))
  lines.append('KVCACHE session=%d generation=1 epoch=1 state=active tokens=1024 bytes=1048576 '
   'reserved_tokens=8192 result=%s replay=%d'%(session,'checkpoint' if restore else 'new',1 if restore else 0))
  if restore:lines.append('KVREPLAY session=%d tokens=%d reason=prefix_checkpoint'%(session,reused))
 engine_text='\n'.join(lines)
 if change.get('fault'):engine_text+='\nERR session 9: KV restore failed\n'
 if change.get('off_checkpoint_lines'):engine_text+='\nIMAGE_CHECKPOINT_MEASURE slot=1\n'
 events=smoke.parse_engine_log(engine_text)
 # the eight interleave steps ran as two bursts of four clients: their records are
 # attributed to the BURST, not to a guessed client order
 concurrent=[a['step'] for a in answers if a['step'].startswith('interleave_four/')]
 records=[]
 for i,a in enumerate(answers):
  record=dict(time=float(i),prompt_total=prompt)
  if a['step'] in concurrent:record['_group']='1clients' if concurrent.index(a['step'])<4 else '2clients'
  else:record['_step']=a['step']
  records.append(record)
 groups=[dict(group=g,steps=[s for s in concurrent if (concurrent.index(s)<4)==(g=='1clients')],records=[],
   exact=True) for g in ('1clients','2clients')]
 if change.get('group_short'):groups[1]['steps']=groups[1]['steps'][:3];groups[1]['exact']=False
 if change.get('same_slot'):
  for e in events:
   if e['kind']=='request' and e['session'] in (2,3,4) and e['prompt']==prompt:e['session']=1
 rows,correlation=smoke.correlate(records,events)
 if change.get('contaminate'):answers[6]['content']='about mark-stable'
 cfg_args=list(smoke.QUALIFIED)+([smoke.IMAGE_FLAG] if (arm=='candidate_on')!=bool(change.get('flip_flag')) else [])
 checks,reuse=smoke.arm_checks(arm,answers,rows,correlation,
  {'images':not change.get('no_encoder'),'loaded':True},{'live':{'state':'idle'},'requests':records},
  engine_text,cfg_args,groups)
 return dict(arm=arm,checks=checks,reuse=reuse,correlation=correlation)

def check(report,name):return next(c for c in report['checks'] if c['name']==name)

def test_a_clean_arm_passes_every_required_check(tmp_path):
 report=arm_inputs(tmp_path)
 assert check(report,'stable_image_history_reuse_positive')['value']==512
 assert check(report,'checkpoint_restore_observed')['ok']
 assert report['correlation']=={'unmatched_requests':0,'unmatched_engine_lines':0}
 assert not [c['name'] for c in report['checks'] if c['required'] and not c['ok']]

def test_zero_reuse_fails_a_required_check(tmp_path):
 failed=check(arm_inputs(tmp_path,zero_reuse=True),'stable_image_history_reuse_positive')
 assert not failed['ok'] and failed['required'] and failed['value']==0

def test_missing_restore_fails_a_required_check(tmp_path):
 assert not check(arm_inputs(tmp_path,no_restore=True),'checkpoint_restore_observed')['ok']

def test_engine_fault_fails_a_required_check(tmp_path):
 assert not check(arm_inputs(tmp_path,fault=True),'engine_log_without_fault')['ok']

def test_foreign_reply_text_fails_a_required_check(tmp_path):
 failed=check(arm_inputs(tmp_path,contaminate=True),'no_cross_conversation_contamination')
 assert not failed['ok'] and failed['required'] and any('interleave_four' in x for x in failed['value'])

def test_checkpoint_lines_never_appear_with_the_flag_off(tmp_path):
 failed=check(arm_inputs(tmp_path,arm='candidate_off',off_checkpoint_lines=True),'no_image_checkpoint_lines_in_log')
 assert not failed['ok'] and failed['required']
 assert not any(c['name']=='no_image_checkpoint_lines_in_log' for c in arm_inputs(tmp_path)['checks'])

def test_flag_mismatch_between_config_and_arm_is_caught(tmp_path):
 for arm in ('baseline','candidate_on'):
  assert check(arm_inputs(tmp_path,arm=arm),'image_flag_matches_the_running_config')['ok']
  assert not check(arm_inputs(tmp_path,arm=arm,flip_flag=True),'image_flag_matches_the_running_config')['ok'],arm

def test_concurrent_group_is_checked_as_a_group(tmp_path):
 for change,name in ((dict(group_short=True),'concurrent_groups_attributed'),
  (dict(same_slot=True),'four_clients_used_distinct_slots')):
  failed=check(arm_inputs(tmp_path,**change),name)
  assert not failed['ok'] and failed['required'],(change,name)

def test_interleave_second_burst_reuse_is_required(tmp_path):
 report=arm_inputs(tmp_path)
 for row in report['reuse']:
  if row['group']=='2clients':row['reused']=0
 for r in report['checks']:
  pass
 broken=arm_inputs(tmp_path,second_group_zero=True)
 assert not check(broken,'interleave_second_group_reuse_positive')['ok']

def test_missing_encoder_fails_a_required_check(tmp_path):
 assert not check(arm_inputs(tmp_path,no_encoder=True),'encoder_resident')['ok']

def test_rendered_token_divergence_is_reported_not_required(tmp_path):
 left=arm_inputs(tmp_path,arm='baseline')
 right=arm_inputs(tmp_path,arm='candidate_off',tokens_differ=True)
 gaps=smoke.divergence(left['reuse'],right['reuse'])
 assert gaps and all(g['kind']=='rendered_tokens' for g in gaps)
 assert {g['left'] for g in gaps}=={1054} and {g['right'] for g in gaps}=={1051}
 assert not [c['name'] for c in right['checks'] if c['required'] and not c['ok']]

# ---- supervisor verdict -------------------------------------------------------------
def verdict(**kw):
 base=dict(killed=False,late=False,exit_code=0,group_empty=True,report_status='PASS_BOUNDED_IMAGE_HTTP_SMOKE',
  arms_done=True)
 base.update(kw)
 return smoke.verdict(**base)

def test_pass_marker_is_bound_to_the_mode_that_ran():
 assert verdict()[0]
 assert verdict(report_status='PASS_IMAGE_HTTP_SMOKE_PREFLIGHT_ONLY',preflight=True)[0]
 assert verdict(report_status='PASS_IMAGE_HTTP_SMOKE_SELFTEST',selftest='normal')[0]
 assert verdict(report_status='PASS_IMAGE_HTTP_PLAN_ONLY',plan_only=True)[0]
 for status,kw in (('PASS_IMAGE_HTTP_SMOKE_PREFLIGHT_ONLY',{}),('PASS_IMAGE_HTTP_PLAN_ONLY',{}),
  ('PASS_IMAGE_HTTP_SMOKE_SELFTEST',{}),('PASS_BOUNDED_IMAGE_HTTP_SMOKE',{'preflight':True}),
  ('PASS_BOUNDED_IMAGE_HTTP_SMOKE',{'plan_only':True}),('PASS_IMAGE_HTTP_SMOKE_PREFLIGHT_ONLY',{'selftest':'x'})):
  assert not verdict(report_status=status,**kw)[0],(status,kw)

def test_verdict_refuses_unclean_or_late_runs():
 for kw in (dict(killed=True),dict(late=True),dict(exit_code=1),dict(group_empty=False),
  dict(server_groups_empty=False),dict(cleanup_complete=False),dict(arms_done=False),
  dict(required_failures=['x']),dict(interrupted='SIGINT'),dict(worker_reaped=False)):
  assert not verdict(**kw)[0],kw

def test_plan_checks_required_set(tmp_path):
 report=dict(plan=[dict(arm=a,host='127.0.0.1',port=11500+i,engine=str(tmp_path/('b' if a=='baseline' else 'e')),
   image_flag=(a=='candidate_on'),args=[],vision=['gpu']) for i,a in enumerate(smoke.ARMS)],
  steps=[s['step'] for s in smoke.plan('m')])
 class A:arms=smoke.ARMS;baseline_engine=tmp_path/'b';engine=tmp_path/'e'
 assert smoke.plan_checks(report,A())
 for break_,value in (('image_flag',lambda:report['plan'][0].update(image_flag=True)),
  ('production_port',lambda:report['plan'][1].update(port=11436)),
  ('non_loopback',lambda:report['plan'][2].update(host='100.100.10.3')),
  ('wrong_engine',lambda:report['plan'][0].update(engine=str(tmp_path/'e'))),
  ('short_plan',lambda:report.update(steps=report['steps'][:3]))):
  value();assert not smoke.plan_checks(report,A()),break_
  report['plan'][0]['image_flag']=False;report['plan'][1]['port']=11501
  report['plan'][2]['host']='127.0.0.1';report['plan'][0]['engine']=str(tmp_path/'b')
  report['steps']=[s['step'] for s in smoke.plan('m')]

# ---- process-group ownership --------------------------------------------------------
def test_groups_are_registered_and_reaped_once(tmp_path):
 child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(300)'],start_new_session=True,
  stdout=subprocess.DEVNULL)
 pid=child.pid
 try:
  smoke.register_group(tmp_path,child,'baseline')
  entries=[json.loads(x) for x in (tmp_path/'owned-groups.jsonl').read_text().splitlines()]
  assert entries[0]['pid']==child.pid and entries[0]['pgid']==child.pid and entries[0]['starttime']
  report=smoke.reap_recorded_groups(tmp_path,time.monotonic()+10)
  assert report[0]['kind']=='killed_owned_group' and report[0]['group_empty']
  assert not Path('/proc/%d'%pid).exists()
 finally:
  try:os.killpg(pid,signal.SIGKILL)
  except ProcessLookupError:pass
  try:child.wait(timeout=10)
  except subprocess.TimeoutExpired:pass

def test_recycled_pid_is_never_signalled(tmp_path):
 """The recorded start time must still match: otherwise the pid is somebody else's and
 the harness leaves it alone."""
 innocent=subprocess.Popen([sys.executable,'-c','import time;time.sleep(300)'],start_new_session=True)
 try:
  (tmp_path/'owned-groups.jsonl').write_text(json.dumps(dict(arm='baseline',pid=innocent.pid,
   pgid=innocent.pid,starttime='1',registered=time.time()))+'\n')
  report=smoke.reap_recorded_groups(tmp_path,time.monotonic()+2)
  assert report[0]['kind']=='pid_no_longer_ours_not_signalled' and not report[0]['group_empty']
  assert Path('/proc/%d'%innocent.pid).exists()
 finally:
  try:os.killpg(innocent.pid,signal.SIGKILL)
  except ProcessLookupError:pass
  innocent.wait(timeout=10)

def test_gone_group_counts_as_empty(tmp_path):
 (tmp_path/'owned-groups.jsonl').write_text(json.dumps(dict(arm='baseline',pid=999999,pgid=999999,
  starttime='1',registered=time.time()))+'\n')
 assert smoke.reap_recorded_groups(tmp_path,time.monotonic()+1)[0]['group_empty']

def test_temporary_listener_never_uses_the_production_port():
 for _ in range(3):
  assert smoke.free_port()!=smoke.PRODUCTION_PORT

# ---- the harness end to end against a loopback stand-in -----------------------------
def harness_dirs(tmp_path):
 stage=tmp_path/'stage';stage.mkdir(exist_ok=True)
 (stage/'engine').write_text('candidate binary\n')
 (stage/'engine-baseline').write_text('deployed binary\n')
 (stage/'model.gguf').write_bytes(b'not a model')
 (tmp_path/'server-root').mkdir(exist_ok=True)
 return stage

def run_smoke(tmp_path,scenario,window=120,cleanup=10,extra=(),preflight=False,plan_only=False,env=None):
 stage=harness_dirs(tmp_path)
 out=stage/('run-%s-%d-%d'%(scenario,os.getpid(),next(counter)))
 command=[sys.executable,str(SMOKE),'--engine',str(stage/'engine'),
  '--baseline-engine',str(stage/'engine-baseline'),'--model',str(stage/'model.gguf'),
  '--source-config',str(source_config(tmp_path)),'--server-root',str(tmp_path/'server-root'),
  '--out',str(out),'--window-seconds',str(window),'--cleanup-seconds',str(cleanup),'--max-tokens','8']
 if scenario:command+=['--selftest',scenario]
 if preflight:command+=['--preflight']
 if plan_only:command+=['--plan-only','--stage',str(stage)]
 run=subprocess.run(command+list(extra),capture_output=True,text=True,timeout=window+180,
  env=dict(os.environ,PYTHONPATH=str(ROOT),**(env or {})))
 watch=json.loads((out/'watchdog.json').read_text()) if (out/'watchdog.json').exists() else {}
 report=json.loads((out/'result.json').read_text()) if (out/'result.json').exists() else {}
 arms={a:json.loads((out/('%s-report.json'%a)).read_text()) for a in smoke.ARMS if (out/('%s-report.json'%a)).exists()}
 return run,watch,report,arms,out

def test_selftest_normal_passes_with_every_arm(tmp_path):
 run,watch,report,arms,out=run_smoke(tmp_path,'normal')
 assert run.returncode==0,run.stdout+run.stderr
 assert watch['passed'] and watch['worker_report_status']=='PASS_IMAGE_HTTP_SMOKE_SELFTEST'
 assert watch['owned_group_empty'] and watch['server_groups_empty'] and not watch['interrupted_signals']
 assert report['mode']=='selftest:normal' and report['required_failures']==[]
 assert set(arms)==set(smoke.ARMS)
 for arm,rep in arms.items():
  failed=[c['name'] for c in rep['checks'] if c['required'] and not c['ok']]
  assert not failed,(arm,failed)
  assert rep['sessions']==[0,1,2,3],(arm,rep['sessions'])
  named={c['name']:c for c in rep['checks']}
  assert named['concurrent_groups_attributed']['ok'] and named['four_clients_used_distinct_slots']['ok']
  assert named['interleave_second_group_reuse_positive']['ok'],(arm,named['interleave_second_group_reuse_positive'])
  # the eight concurrent steps are NOT attributed per step: the report says so
  assert all(not r['attributed'] for r in rep['reuse'] if r['step'].startswith('interleave_four/')),rep['reuse']
  assert all(r['attributed'] for r in rep['reuse'] if not r['step'].startswith('interleave_four/')),rep['reuse']
 for arm,cleanup in report['arm_cleanup'].items():
  assert cleanup['owned_group_empty'] and cleanup['worker_reaped'],(arm,cleanup)
 assert 'IMAGE_HTTP_SMOKE_SELFTEST PASS scenario=normal' in run.stdout
 text=(out/'worker-stdout.txt').read_text()
 assert 'stage=arms action=done arms=3 required_failures=0' in text
 # the deployed baseline and the candidate with the flag off must render the same prompts
 assert report['divergence']['baseline_vs_candidate_off']==[] and \
  report['divergence']['candidate_off_vs_candidate_on']==[],report['divergence']
 # slot/session choice is recorded from the engine's own lines, not guessed
 assert arms['candidate_on']['correlation']=={'unmatched_requests':0,'unmatched_engine_lines':0}
 restores={c['name']:c for c in arms['candidate_on']['checks']}
 assert restores['checkpoint_restore_observed']['value']>0,restores['checkpoint_restore_observed']
 assert restores['interleave_group_restores_reported']['ok'],restores['interleave_group_restores_reported']
 # the two bursts each landed on four distinct engine sessions
 assert restores['four_clients_used_distinct_slots']['ok'],restores['four_clients_used_distinct_slots']

def test_semantic_failures_are_reported_as_failures(tmp_path):
 for scenario,broken in (('zero_stable_reuse','stable_image_history_reuse_positive'),
  ('no_restore','checkpoint_restore_observed'),('engine_error','engine_log_without_fault'),
  ('cross_talk','no_cross_conversation_contamination'),('missing_record','correlation_complete'),
  # an encoder that is not resident stops the arm before any request: the refusal is
  # recorded as the run error, which is equally loud
  ('no_encoder','the image encoder is not resident')):
  run,watch,report,arms,out=run_smoke(tmp_path,scenario)
  assert run.returncode==1,scenario
  assert not watch['passed'] and 'PASS' not in run.stdout,scenario
  assert broken in json.dumps(watch)+json.dumps(report),(scenario,watch['required_failures'],
   report.get('error'))
  # a failed required check ends the run without a marker; a refused arm (no encoder)
  # raises before any request is sent, which is reported as an error
  assert report['status'] in ('FAIL_required_checks','FAIL') and report.get('marker') is None,scenario

def test_divergence_is_reported_and_the_run_still_passes(tmp_path):
 run,watch,report,arms,out=run_smoke(tmp_path,'divergence')
 assert run.returncode==0,run.stdout+run.stderr
 assert watch['passed']
 gaps=report['divergence']['baseline_vs_candidate_off']
 assert gaps and all(g['kind']=='rendered_tokens' for g in gaps)
 assert report['divergence']['candidate_off_vs_candidate_on'],report['divergence']

def test_hung_and_stuck_servers_are_killed_inside_the_cleanup_allowance(tmp_path):
 for scenario in ('hung_server','stuck_group'):
  run,watch,report,arms,out=run_smoke(tmp_path,scenario,cleanup=15)
  assert run.returncode==0,run.stdout+run.stderr
  assert watch['server_groups_empty'] and watch['cleanup_complete'],scenario
  for arm,cleanup in report['arm_cleanup'].items():
   assert cleanup['owned_group_empty'],(scenario,arm,cleanup)
   assert cleanup['worker_reaped'],(scenario,arm,cleanup)
   assert not cleanup['graceful'],(scenario,arm,cleanup)
  kinds=[x['kind'] for x in watch['owned_server_groups']]
  assert kinds==['already_reaped']*3,(scenario,kinds)

def test_window_deadline_kills_a_slow_run_and_leaves_nothing_behind(tmp_path):
 run,watch,report,arms,out=run_smoke(tmp_path,'slow_server',window=20,cleanup=10)
 assert run.returncode==1
 assert watch['killed'] and watch['worker_reaped'] and watch['owned_group_empty']
 assert watch['server_groups_empty'] and watch['worker_report_status']=='MISSING'
 assert any('worker killed at the window deadline' in x for x in watch['reasons'])
 assert 'PASS' not in run.stdout
 assert [x['kind'] for x in watch['owned_server_groups']]==['killed_owned_group']

def test_signal_teardown_reaps_the_worker_and_its_server_group(tmp_path):
 stage=harness_dirs(tmp_path)
 out=stage/('sig-%d-%d'%(os.getpid(),next(counter)))
 command=[sys.executable,str(SMOKE),'--engine',str(stage/'engine'),
  '--baseline-engine',str(stage/'engine-baseline'),'--model',str(stage/'model.gguf'),
  '--source-config',str(source_config(tmp_path)),'--server-root',str(tmp_path/'server-root'),
  '--out',str(out),'--selftest','slow_server','--window-seconds','300','--cleanup-seconds','15','--max-tokens','8']
 child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True,
  env=dict(os.environ,PYTHONPATH=str(ROOT)))
 time.sleep(6);child.send_signal(signal.SIGINT)
 text,_=child.communicate(timeout=120)
 assert child.returncode==1,text
 assert 'stage=watchdog action=teardown signal=SIGINT cleanup_allowance_s=15' in text
 assert 'PASS' not in text
 watch=json.loads((out/'watchdog.json').read_text())
 assert watch['interrupted_signals']=='SIGINT' and watch['killed'] and watch['worker_reaped']
 assert watch['owned_group_empty'] and watch['cleanup_complete'] and watch['server_groups_empty']
 assert 'killed_owned_group' in [x['kind'] for x in watch['owned_server_groups']],watch['owned_server_groups']

def test_record_failure_never_restarts_cleanup_or_prints_pass(tmp_path):
 run,watch,report,arms,out=run_smoke(tmp_path,'normal',extra=('--fault','record'))
 assert run.returncode==1
 assert 'stage=record action=failed' in run.stderr and 'action=no_restart cleanup_state_final=1' in run.stderr
 assert 'result_record_failed' in run.stdout and 'PASS' not in run.stdout
 assert not (out/'watchdog.json').exists()

def test_supervisor_crash_reaps_the_server_group_it_left_behind(tmp_path):
 run,watch,report,arms,out=run_smoke(tmp_path,'normal',extra=('--fault','crash'))
 assert run.returncode==1 and 'supervisor_error' in run.stdout and 'PASS' not in run.stdout
 # the crash happens after cleanup: whatever the run started must already be gone
 assert not [x for x in Path('/proc').iterdir() if x.name.isdigit()
  and b'q36_image_http_smoke.py' in (x/'cmdline').read_bytes() and b'--fake-server' in (x/'cmdline').read_bytes()]

def test_plan_only_and_preflight_are_bounded_modes(tmp_path):
 run,watch,report,arms,out=run_smoke(tmp_path,'normal',plan_only=True)
 assert run.returncode==0,run.stdout+run.stderr
 assert 'IMAGE_HTTP_PLAN_ONLY PASS nothing_started=1' in run.stdout
 assert 'PASS_BOUNDED_IMAGE_HTTP_SMOKE' not in run.stdout and 'PREFLIGHT' not in run.stdout
 assert report['mode']=='plan-only' and report['status']=='PASS_IMAGE_HTTP_PLAN_ONLY'
 assert [x['image_flag'] for x in report['plan']]==[False,False,True]
 assert all(x['port']!=smoke.PRODUCTION_PORT and x['host']=='127.0.0.1' for x in report['plan'])
 assert smoke.IMAGE_FLAG in next(x['args'] for x in report['plan'] if x['arm']=='candidate_on')
 assert smoke.IMAGE_FLAG not in next(x['args'] for x in report['plan'] if x['arm']=='baseline')
 assert not [p for p in out.iterdir() if p.name.endswith('-engine.log')],'plan-only must not start a server'
 run,watch,report,arms,out=run_smoke(tmp_path,'normal',preflight=True)
 assert run.returncode==0,run.stdout+run.stderr
 assert 'IMAGE_HTTP_SMOKE_PREFLIGHT PASS model_not_started=1' in run.stdout
 assert 'PASS_BOUNDED_IMAGE_HTTP_SMOKE' not in run.stdout
 assert report['status']=='PASS_IMAGE_HTTP_SMOKE_PREFLIGHT_ONLY' and len(report['plan'])==3
 assert [x['image_flag'] for x in report['plan']]==[False,False,True]

def test_guards_refuse_the_wrong_window_and_the_production_port(tmp_path):
 stage=harness_dirs(tmp_path)
 base=[sys.executable,str(SMOKE),'--engine',str(stage/'engine'),'--baseline-engine',str(stage/'engine-baseline'),
  '--model',str(stage/'model.gguf'),'--source-config',str(source_config(tmp_path)),
  '--server-root',str(tmp_path/'server-root')]
 real=subprocess.run(base+['--out',str(stage/'guarded')],capture_output=True,text=True,timeout=60,
  env=dict(os.environ,PYTHONPATH=str(ROOT)))
 assert real.returncode==2 and 'approved Orin window' in real.stderr
 window=subprocess.run(base+['--out',str(smoke.STAGE)+('/guarded-%d'%os.getpid())],
  capture_output=True,text=True,
  timeout=60,env=dict(os.environ,PYTHONPATH=str(ROOT),STRATA_IMAGE_HTTP_ROOT=str(smoke.STAGE),
  STRATA_IMAGE_HTTP_REVIEW_WINDOW='1'))
 assert window.returncode==2 and 'STRATA_IMAGE_HTTP_ROOT' in window.stderr
 pre=subprocess.run(base+['--out',str(stage/'p1'),'--selftest','normal','--preflight','--port','11436'],
  capture_output=True,text=True,timeout=60,env=dict(os.environ,PYTHONPATH=str(ROOT)))
 assert pre.returncode==2 and 'production port' in pre.stderr
 fault=subprocess.run(base+['--out',str(stage/'p3'),'--plan-only','--stage',str(stage),'--fault','record'],
  capture_output=True,text=True,timeout=60,env=dict(os.environ,PYTHONPATH=str(ROOT)))
 assert fault.returncode==2 and 'fault injection requires --selftest' in fault.stderr
 sha=subprocess.run(base+['--out',str(stage/'p2'),'--selftest','normal','--preflight',
  '--expect-engine-sha256','0'*64],capture_output=True,text=True,timeout=60,env=dict(os.environ,PYTHONPATH=str(ROOT)))
 assert sha.returncode==2 and 'hash differs' in sha.stderr
 outside=subprocess.run(base+['--out',str(tmp_path/'outside'),'--plan-only','--stage',str(stage)],
  capture_output=True,text=True,timeout=60,env=dict(os.environ,PYTHONPATH=str(ROOT)))
 assert outside.returncode==2 and 'NEW directory' in outside.stderr
 same=subprocess.run(base+['--out',str(stage/'planned'),'--plan-only','--stage',str(stage)],
  capture_output=True,text=True,timeout=60,env=dict(os.environ,PYTHONPATH=str(ROOT)))
 assert same.returncode==0,same.stdout+same.stderr

def test_harness_never_names_a_service_operation():
 """The harness may MENTION the live services in its prose, but no string it can execute
 may name a service manager, the OWUI stack or an API key, and no listener may be
 non-loopback."""
 import ast
 tree=ast.parse(SMOKE.read_text())
 doc=tree.body and isinstance(tree.body[0],ast.Expr) and getattr(tree.body[0].value,'value',None)
 strings=[n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)
  and n.value!=doc]
 for banned in ('systemctl','open-webui','docker','service start','service stop','--api-key','0.0.0.0'):
  assert not [s for s in strings if banned in s],banned
 # the live front end is named ONLY to record that its savings were not measured here
 owui=[s for s in strings if 'OWUI' in s]
 assert all('UNMEASURED' in s for s in owui),owui
 hosts=[n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)
  and n.value.count('.')==3 and n.value[0].isdigit()]
 assert set(hosts)<={smoke.PRODUCTION_HOST,'127.0.0.1'}|set(),hosts
 assert 'start_new_session=True' in ast.unparse(tree)
