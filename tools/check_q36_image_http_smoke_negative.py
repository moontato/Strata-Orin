#!/usr/bin/env python3
"""CPU-only mutation controls for the bounded HTTP image-checkpoint smoke harness.
Each mutation removes ONE part of the harness's own bound; the harness must then behave
like the defect that part was written to prevent. Everything runs against the loopback
stand-in (a fake /health, /metrics and engine log), never a model, encoder, GPU, engine
binary or service, and no real staging directory or worktree is touched.

Every control is run twice: once with the mutation and once with the harness as it is
committed. The control only passes when the mutated harness behaves WRONG and the
committed harness behaves RIGHT.
"""
import json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SMOKE=ROOT/'tools/q36_image_http_smoke.py'
sys.path.insert(0,str(ROOT))
from tools import q36_image_http_smoke as smoke

def make_stage(tmp):
 """A throwaway staging root: fake binaries, a fake config, a fake server checkout."""
 stage=tmp/'stage';stage.mkdir(parents=True,exist_ok=True)
 (stage/'engine').write_text('candidate binary\n');(stage/'engine-baseline').write_text('deployed binary\n')
 (stage/'model.gguf').write_bytes(b'not a model')
 (tmp/'server-root').mkdir(exist_ok=True)
 config=tmp/'config.json'
 config.write_text(json.dumps({'model_name':'q36','tokenizer':'/models/q36.gguf','api_key':'secret',
  'host':'100.100.10.3','port':11436,'vision':{'model':'/models/vit.gguf','mmproj':'/models/mm.bin',
  'gpu':True,'exe':'/opt/vision'}}))
 return stage,config,tmp/'server-root'

def base_command(stage,config,server_root,out):
 return [sys.executable,str(SMOKE),'--engine',str(stage/'engine'),'--baseline-engine',
  str(stage/'engine-baseline'),'--model',str(stage/'model.gguf'),'--source-config',str(config),
  '--server-root',str(server_root),'--out',str(out),'--window-seconds','90','--cleanup-seconds','10',
  '--max-tokens','8']

def live_listeners(needle=''):
 """Every fake server still running for a given run: the stand-in names its own port, plan
 file and engine log, all of which live under that run's directory."""
 found=[]
 for entry in Path('/proc').iterdir():
  if not entry.name.isdigit():continue
  try:raw=(entry/'cmdline').read_bytes()
  except OSError:continue
  parts=raw.split(b'\0')
  if len(parts)>2 and b'--fake-server' in raw and (not needle or needle.encode() in raw):
   found.append(int(entry.name))
 return found

def execute(script,stage,config,server_root,out,extra=(),window=None,env=None):
 command=[sys.executable,str(script)]+base_command(stage,config,server_root,out)[2:]
 if window:command[command.index('--window-seconds')+1]=str(window)
 environment=dict(os.environ,STRATA_IMAGE_HTTP_ROOT=str(ROOT),PYTHONPATH=str(ROOT),**(env or {}))
 return subprocess.run(command+list(extra),capture_output=True,text=True,timeout=400,env=environment)

# ----------------------------------------------------------------------------------------------
# control 1: the image flag reaches every arm
FLAG_ON_ALL=(" if arm=='candidate_on':args.append(IMAGE_FLAG)"," args.append(IMAGE_FLAG)")
def check_flag_on_all(tmp,out_text,returncode,extra_state):
 return returncode!=0 and 'image_flag_matches_the_running_config' in out_text

# control 2: a concurrent burst is attributed per client by guessed arrival order
GROUP_LABEL="""     if len(window['steps'])==1 and len(take)==1:take[0]['_step']=window['steps'][0]
     else:
      name=window['steps'][0].split('/')[1][0]+'clients' if '/' in window['steps'][0] else window['steps'][0]"""
GROUP_LABEL_MUT="""     if len(window['steps'])==1:
      # MUTATION: name each client of a concurrent burst by the order its record arrived,
      # which is the server's order, not the client's
      for step,r in zip(window['steps'],take):r['_step']=step
     else:
      name='-'"""
def check_group_label(tmp,out_text,returncode,extra_state):
 return returncode!=0 and 'concurrent_groups_attributed' in out_text

# control 3: the stable-history reuse requirement is dropped
REUSE=" add('stable_image_history_reuse_positive',stable_reuse is not None and stable_reuse>0,stable_reuse)"
REUSE_MUT=" add('stable_image_history_reuse_positive',True,stable_reuse,required=False)  # MUTATION"
def check_reuse_dropped(tmp,out_text,returncode,extra_state):
 # the mutated harness accepts an arm whose second image turn reused nothing
 return returncode==0 and 'stable_image_history_reuse_positive' in json.dumps(extra_state.get('checks',{}))

# control 4: the supervisor stops reaping the server groups it started
REAP_BODY=" report=[]\n try:lines="
REAP_MUT=" report=[]\n return report   # MUTATION: nothing this run started behind the worker is reaped\n try:lines="
def check_no_reap(tmp,out_text,returncode,extra_state):
 return bool(extra_state.get('listeners_left'))

# control 5: the pass marker is not bound to the mode that ran
MARKER="  if report.get('marker'):print(report['marker'],flush=True)"
MARKER_MUT="  print(PASS_MARKER,flush=True)   # MUTATION: every passing mode prints the real marker"
def check_marker(tmp,out_text,returncode,extra_state):
 return smoke.PASS_MARKER in out_text

# control 6: the approved-window guard is removed
WINDOW_GUARD=""" if not a.selftest and not a.plan_only and os.environ.get('STRATA_IMAGE_HTTP_REVIEW_WINDOW')!='1':
  p.error('the HTTP smoke starts a model and encoder, so it runs only in the approved Orin window: set '
   'STRATA_IMAGE_HTTP_REVIEW_WINDOW=1 and bring the live services down through the manager first')"""
WINDOW_MUT=""" # MUTATION: the window guard is gone"""
def check_window_guard(tmp,out_text,returncode,extra_state):
 # the mutated build never refuses an off-window run: whatever it complains about next,
 # the window refusal is gone
 return 'approved Orin window' not in out_text

CONTROLS=(
 dict(name='flag_on_every_arm',pair=FLAG_ON_ALL,scenario='normal',check=check_flag_on_all,
  kept=lambda text,code,state:'PASS' in text),
 dict(name='attribute_concurrent_burst_by_order',pair=(GROUP_LABEL,GROUP_LABEL_MUT),scenario='normal',
  check=check_group_label,kept=lambda text,code,state:'PASS' in text),
 dict(name='stable_reuse_not_required',pair=(REUSE,REUSE_MUT),scenario='zero_stable_reuse',
  check=check_reuse_dropped,want_checks=True,
  kept=lambda text,code,state:'PASS' not in text),
 dict(name='server_groups_not_reaped',pair=(REAP_BODY,REAP_MUT),scenario='slow_server',window=25,
  check=check_no_reap,want_listeners=True,
  kept=lambda text,code,state:'PASS' not in text),
 dict(name='marker_not_mode_bound',pair=(MARKER,MARKER_MUT),scenario=None,plan_only=True,
  check=check_marker,kept=lambda text,code,state:smoke.PASS_MARKER not in text),
 dict(name='window_guard_removed',pair=(WINDOW_GUARD,WINDOW_MUT),scenario=None,real=True,
  check=check_window_guard,kept=lambda text,code,state:'approved Orin window' in text),
)

def main():
 results=[];failures=[]
 for control in CONTROLS:
  with __import__('tempfile').TemporaryDirectory(prefix='q36-smoke-mut-') as raw:
   tmp=Path(raw);stage,config,server_root=make_stage(tmp)
   source=SMOKE.read_text()
   before,after=control['pair']
   if source.count(before)!=1:
    failures.append((control['name'],'anchor appears %d times'%source.count(before)));continue
   mutated=tmp/'harness-mutated.py';mutated.write_text(source.replace(before,after))
   extra=['--plan-only','--stage',str(stage)] if control.get('plan_only') else []
   real=control.get('real')
   scenario=control.get('scenario_override',(control['scenario'],None)) if control['scenario'] else (None,None)
   if scenario[0]:extra=['--selftest',scenario[0]]+extra
   if scenario[1]:control=dict(control,window=scenario[1])
   outs={};states={}
   for label in ('mutated','committed'):
    state={};script=mutated if label=='mutated' else SMOKE
    out=stage/('%s-%s'%(control['name'],label))
    if out.exists():subprocess.run(['rm','-rf',str(out)])
    before_listeners=set(live_listeners(str(tmp)))
    run=execute(script,stage,config,server_root,out,extra=extra,window=control.get('window'))
    if control.get('want_listeners'):
     time.sleep(1.5)
     state['listeners_left']=sorted(set(live_listeners(str(tmp)))-before_listeners)
     for pid in state['listeners_left']:
      try:os.killpg(pid,signal.SIGKILL)
      except ProcessLookupError:pass
    if control.get('want_checks') and (out/'baseline-report.json').exists():
     checks=json.loads((out/'baseline-report.json').read_text()).get('checks',[])
     state['checks']={c['name']:c for c in checks}
    outs[label]=(run.stdout+run.stderr,run.returncode);states[label]=state
   mutated_text,mutated_code=outs['mutated'];committed_text,committed_code=outs['committed']
   bit=control['check'](tmp,mutated_text,mutated_code,states['mutated'])
   committed_ok=control['kept'](committed_text,committed_code,states['committed'])
   results.append(dict(control=control['name'],mutation_detected=bool(bit),
    committed_behaviour_kept=bool(committed_ok),mutated_exit=mutated_code,committed_exit=committed_code,
    evidence=(str(states['mutated'].get('listeners_left')) if control.get('want_listeners')
     else (mutated_text.strip().splitlines()[-1] if mutated_text.strip() else ''))[:200]))
   if not bit or not committed_ok:failures.append((control['name'],'mutation_detected=%s committed_kept=%s'%(bit,committed_ok)))
 print(json.dumps(dict(script='q36_image_http_smoke',controls=results,status='FAIL' if failures else 'PASS',
  failures=[f[0] for f in failures]),indent=2))
 return 1 if failures else 0

if __name__=='__main__':sys.exit(main())
