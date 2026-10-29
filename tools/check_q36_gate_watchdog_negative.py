#!/usr/bin/env python3
"""CPU-only mutation controls for the gate supervisor's execution budget and teardown.
Each mutation removes one part of the bound; the deterministic selftest scenario, the
pure verdict, the bounded teardown or the real signal path MUST then behave like the
documented defect. Fake children only: no engine, model, encoder or service is ever
started, and no real staging/worktree is touched.
"""
from pathlib import Path
import importlib.util,importlib.machinery,json,os,resource,signal,subprocess,sys,tempfile,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import q36_image_checkpoint_gate as gate

GROUP_KILL='os.killpg(child.pid,signal.SIGKILL)'
LATE_REASON=" if late:reasons.append('worker finished at or after the execution deadline')\n"
SUPERVISE=' return supervise(a,Budget(a.budget_seconds))'
HANDLER='try:previous[sig]=signal.signal(sig,record)'
PROBE_LOOP=" state.exit_code=exit_code\n while True:                               # 3. prove the owned group empty (probe only)\n"
RE_ARM=" deadline=time.monotonic()+1.0   # MUTATION: the allowance is re-armed\n"
RE_ARM_STATE='  if self.deadline is None:self.deadline=time.monotonic()+self.allowance'
POST_REAP='  if state.reaped:return state.group_status'

class StuckChild:
    """Fake worker: blocked reaping, blocked group cleanup. pid is not a usable group,
    and every OS hook is injected, so no real process is ever signalled."""
    pid=-1
    def wait(self,timeout=None):raise subprocess.TimeoutExpired(cmd='fake-worker',timeout=timeout)

class FinishedChild:
    """Fake worker that already exited and was reaped normally."""
    pid=-1
    def wait(self,timeout=None):return 0

def cleanup_seconds(module,allowance=1.0):
 """Elapsed teardown time for a worker that never exits and a group that never empties."""
 started=time.monotonic()
 module.teardown(module.CleanupState(StuckChild(),allowance),
     waiter=lambda:None,killer=lambda:'present',prober=lambda:'present')
 return time.monotonic()-started

def reentry_seconds(module,allowance=1.0):
 """Teardown twice over the SAME state (the error path that used to restart cleanup):
 the second call must spend no new allowance."""
 state=module.CleanupState(StuckChild(),allowance)
 module.teardown(state,waiter=lambda:None,killer=lambda:'present',prober=lambda:'present')
 started=time.monotonic()
 module.teardown(state,waiter=lambda:None,killer=lambda:'present',prober=lambda:'present')
 return time.monotonic()-started

def group_signals(module):
 """Teardown over a worker whose leader was ALREADY reaped by the earlier normal path
 (state.reaped set, cleanup not finished). The ownership guard must stop every signal:
 the former PGID may already belong to an unrelated process."""
 state=module.CleanupState(FinishedChild(),1.0);kills=[];probes=[]
 state.reaped=True
 first=module.teardown(state,waiter=lambda:0,killer=lambda:(kills.append(1),'present')[1],
     prober=lambda:(probes.append(1),'empty')[1])
 second=module.teardown(state,killer=lambda:(kills.append(2),'present')[1],
     prober=lambda:(probes.append(2),'empty')[1])
 return len(kills),len(probes),first==second

def settle(seconds=0.4):
 """Give an un-reaped owned group a moment to become visible/orphan before judging."""
 time.sleep(seconds)

def probes():
 found=[]
 for entry in Path('/proc').iterdir():
  if not entry.name.isdigit():continue
  try:cmd=Path(entry/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  except OSError:continue
  # ONLY the gate's own fake children: a real python -c child, never a shell that
  # merely mentions the marker in its command line.
  if gate.OWNED_PROBE in cmd and cmd.startswith((sys.executable,'/usr/bin/python'))and 'check_q36_gate_watchdog_negative' not in cmd:found.append(int(entry.name))
 return found

def staging(temp):
 stage=Path(temp)/('stage-'+str(len(os.listdir(temp))));(stage/'build').mkdir(parents=True);(stage/'fixtures').mkdir()
 engine=stage/'build'/'strata-q36';engine.write_text('# fake binary marker --kv-image-checkpoint-test\n')
 (stage/'build'/'CMakeCache.txt').write_text('CMAKE_HOME_DIRECTORY:INTERNAL='+str(ROOT)+'\n')
 model=stage/'model.gguf';model.write_bytes(b'not a model')
 (stage/'fixtures'/'manifest.json').write_text(json.dumps({'kind':'q36-image-checkpoints-v1',
  'model':{'path':str(model),'size':model.stat().st_size,'mtime_ns':model.stat().st_mtime_ns}})+'\n')
 return stage,engine,model,stage/'fixtures'

def gate_args(script,engine,model,fixtures,out,scenario,budget,cleanup):
 # The MUTATED file is what runs: pointing at the shipped gate would silently test
 # the real code instead of the mutation.
 return [sys.executable,str(script),'--engine',str(engine),
  '--model',str(model),'--fixtures',str(fixtures),'--out',str(out),'--selftest',scenario,
  '--budget-seconds',str(budget),'--cleanup-seconds',str(cleanup)]

def run_mutated(source,name,temp,scenario,budget=1,cleanup=1):
 gate_path=Path(temp)/name;gate_path.write_text(source)
 stage,engine,model,fixtures=staging(temp)
 out=stage/('run-'+scenario)
 env=dict(os.environ,STRATA_IMAGE_CHECKPOINT_REVIEW_ROOT=str(ROOT),PYTHONPATH=str(ROOT))
 env.pop('STRATA_IMAGE_CHECKPOINT_REVIEW_WINDOW',None)
 run=subprocess.run(gate_args(gate_path,engine,model,fixtures,out,scenario,budget,cleanup),
  capture_output=True,text=True,env=env,timeout=120)
 watch={}
 try:watch=json.loads((out/'watchdog.json').read_text())
 except OSError:pass
 return run,watch

def load_mutated(source,name,temp):
 path=Path(temp)/name;path.write_text(source)
 loader=importlib.machinery.SourceFileLoader(path.stem,str(path))
 spec=importlib.util.spec_from_loader(path.stem,loader)
 module=importlib.util.module_from_spec(spec);loader.exec_module(module)
 return module

def kill_probes():
 # By PID only: a marked fake child can share THIS script's process group (the direct
 # worker mutation), so a group kill here would destroy the control and the shell.
 for pid in probes():
  try:os.kill(pid,signal.SIGKILL)
  except (ProcessLookupError,PermissionError,OSError):pass

def run():
 resource.setrlimit(resource.RLIMIT_CORE,(0,0))
 source=(ROOT/'tools/q36_image_checkpoint_gate.py').read_text()
 phases=dict(arms_completed=True)
 clean=dict(killed=False,late=False,exit_code=0,group_empty=True,
  report_status='PASS_BOUNDED_INDEPENDENT_IMAGE_CHECKPOINT_GATE',phases=phases)
 if not gate.verdict(**clean)[0]:raise RuntimeError('real verdict rejects a clean run')
 if gate.verdict(**{**clean,'late':True})[0]:raise RuntimeError('real verdict accepts a late success')
 if gate.verdict(**{**clean,'group_empty':False})[0]:raise RuntimeError('real verdict accepts a live owned group')
 if gate.verdict(**{**clean,'interrupted':'SIGTERM'})[0]:raise RuntimeError('real verdict accepts an interrupted supervisor')
 if gate.verdict(**{**clean,'worker_reaped':False})[0]:raise RuntimeError('real verdict accepts an unreaped worker')
 with tempfile.TemporaryDirectory(prefix='q36-gate-watchdog-') as temp:
  if source.count(GROUP_KILL)!=1 or source.count(LATE_REASON)!=1 or source.count(SUPERVISE)!=1:
   raise RuntimeError('watchdog mutation anchors changed')
  if source.count(HANDLER)!=1 or source.count(PROBE_LOOP)!=1:raise RuntimeError('teardown mutation anchors changed')
  if source.count(POST_REAP)!=1 or source.count(RE_ARM_STATE)!=1:raise RuntimeError('cleanup-state mutation anchors changed')
  if source.count('if state.complete:return state.result()')!=1:raise RuntimeError('idempotence anchor changed')
  # Baseline: one cleanup allowance really bounds waiting + reaping + verification, a
  # re-entered teardown costs nothing more, and nothing is signalled after the reap.
  real=cleanup_seconds(gate)
  if not 1.0<=real<1.6:raise RuntimeError('baseline teardown spent %.2fs for a 1.0s allowance'%real)
  real_reentry=reentry_seconds(gate)
  if real_reentry>=0.3:raise RuntimeError('baseline re-entry spent %.2fs of a new allowance'%real_reentry)
  real_kills,real_probes,real_same=group_signals(gate)
  if real_kills!=0 or real_probes==0 or not real_same:raise RuntimeError('baseline signals a group after reaping')
  if real_reentry>=0.3:raise RuntimeError('baseline re-entry spent %.2fs of a new allowance'%real_reentry)
  print('GATE_WATCHDOG BASELINE PASS one_cleanup_allowance_s=%.2f (limit 1.60) reentry_s=%.2f '
        'signals_after_reap=%d probes_after_reap=%d result_unchanged=%s'%(real,real_reentry,real_kills,real_probes,real_same))
  # Mutation 1: the supervisor never group-kills -> a hung descendant leaks and the
  # owned group is reported non-empty.
  run,watch=run_mutated(source.replace(GROUP_KILL,'pass  # MUTATION: no group kill'),
   'mutated_gate.py',temp,'hung_child')
  settle();leaked=probes()
  if not leaked:raise RuntimeError('MUTATION ESCAPED: no-group-kill still reaped the owned descendant')
  if run.returncode==0 or watch.get('owned_group_empty'):raise RuntimeError('MUTATION ESCAPED: live group accepted')
  print('GATE_WATCHDOG_NEGATIVE PASS no_group_kill leaked_owned_pids=%d exit=%d group_empty=%s'
        %(len(leaked),run.returncode,watch.get('owned_group_empty')))
  kill_probes()
  # Mutation 2: the verdict ignores lateness -> a past-deadline completion would pass.
  module=load_mutated(source.replace(LATE_REASON,''),'mutated_verdict.py',temp)
  if not module.verdict(**{**clean,'late':True})[0]:raise RuntimeError('MUTATION ESCAPED: late success still refused')
  print('GATE_WATCHDOG_NEGATIVE PASS late_success_accepted')
  # Mutation 3: no supervisor at all -> only the worker's cooperative checks remain,
  # and a blocking descendant survives.
  run,watch=run_mutated(source.replace(SUPERVISE,' return worker(a,Budget(a.budget_seconds))'),
   'mutated_direct.py',temp,'hung_child')
  settle();leaked=probes()
  if not leaked:raise RuntimeError('MUTATION ESCAPED: direct worker reaped the hung descendant')
  if run.returncode==0 or watch.get('killed'):raise RuntimeError('MUTATION ESCAPED: direct worker looked supervised')
  print('GATE_WATCHDOG_NEGATIVE PASS supervisor_bypass leaked_owned_pids=%d exit=%d'%(len(leaked),run.returncode))
  kill_probes()
  # Mutation 4: the cleanup allowance is re-armed for the verification phase -> waiting
  # plus verification costs about TWICE the allowance (the reported 2.11s defect shape).
  module=load_mutated(source.replace(PROBE_LOOP,RE_ARM+PROBE_LOOP),'mutated_rearm.py',temp)
  slow=cleanup_seconds(module)
  if slow<1.8:raise RuntimeError('MUTATION ESCAPED: re-armed cleanup still took only %.2fs'%slow)
  print('GATE_WATCHDOG_NEGATIVE PASS cleanup_rearmed teardown_s=%.2f (baseline %.2f, one allowance 1.00)'%(slow,real))
  # Mutation 4b: the persistent state is bypassed (the old error path: re-entered
  # teardown with a FRESH deadline) -> a second cleanup round spends a second allowance.
  module=load_mutated(source.replace(RE_ARM_STATE,'  self.deadline=time.monotonic()+self.allowance')
     .replace('if state.complete:return state.result()','pass  # MUTATION: no idempotence guard'),
     'mutated_state_rearm.py',temp)
  rearm_reentry=reentry_seconds(module)
  if rearm_reentry<0.8:raise RuntimeError('MUTATION ESCAPED: re-entry still spent only %.2fs'%rearm_reentry)
  print('GATE_WATCHDOG_NEGATIVE PASS cleanup_state_rearmed reentry_s=%.2f (baseline %.2f, one allowance 1.00)'%(rearm_reentry,real_reentry))
  # Mutation 4c: remove the post-reap ownership guard -> a second teardown signals the
  # former worker group again although its leader was already reaped (the PGID may by
  # then belong to an unrelated process).
  module=load_mutated(source.replace(POST_REAP,'  if False:pass  # MUTATION: no ownership guard'),'mutated_post_reap.py',temp)
  mut_kills,mut_probes,mut_same=group_signals(module)
  if mut_kills==0:raise RuntimeError('MUTATION ESCAPED: no group signal after reaping')
  print('GATE_WATCHDOG_NEGATIVE PASS post_reap_signal signals_after_reap=%d probes=%d (real supervisor: %d signals)'
        %(mut_kills,mut_probes,real_kills))
  # Mutation 5: the supervisor installs no signal teardown -> the worker group survives
  # an interruption and no failure record is written.
  gate_path=Path(temp)/'mutated_signal.py';gate_path.write_text(source.replace(HANDLER,'try:previous[sig]=signal.signal(sig,signal.SIG_DFL)'))
  stage,engine,model,fixtures=staging(temp)
  out=stage/'run-signalled'
  env=dict(os.environ,STRATA_IMAGE_CHECKPOINT_REVIEW_ROOT=str(ROOT),PYTHONPATH=str(ROOT))
  env.pop('STRATA_IMAGE_CHECKPOINT_REVIEW_WINDOW',None)
  proc=subprocess.Popen(gate_args(gate_path,engine,model,fixtures,out,'slow_preflight',20,3),
   stdout=open(stage/'ctl-stdout.txt','w'),stderr=open(stage/'ctl-stderr.txt','w'),text=True,env=env)
  workers=[];waited=0.0
  while not workers and waited<15:
   time.sleep(0.05);waited+=0.05
   for entry in Path('/proc').iterdir():
    if not entry.name.isdigit():continue
    try:
     cmd=Path(entry/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
     ppid=Path(entry/'stat').read_text().split(')')[-1].split()[1]
    except (OSError,IndexError):continue
    if '--worker' in cmd and '--selftest' in cmd and ppid==str(proc.pid):workers.append(int(entry.name))
  survived=[];recorded=False
  try:
   os.kill(proc.pid,signal.SIGTERM)
   # The worker holds the inherited stderr, so a pipe read would wait for IT to finish.
   # Watch the supervisor itself: its death must be immediate, then snapshot the group.
   waited=0.0
   while proc.poll() is None and waited<15:time.sleep(0.05);waited+=0.05
   survived=[pid for pid in workers if Path('/proc/%d'%pid).exists()]
   recorded=(out/'watchdog.json').is_file()
  finally:
   for pid in survived:
    try:os.killpg(pid,signal.SIGKILL)
    except (ProcessLookupError,PermissionError):pass
   try:proc.wait(timeout=10)
   except subprocess.TimeoutExpired:proc.kill()
   try:proc.communicate(timeout=5)
   except Exception:pass
  try:
   if not survived:raise RuntimeError('MUTATION ESCAPED: the worker group still went down')
   if recorded:raise RuntimeError('MUTATION ESCAPED: a failure record was still written')
   print('GATE_WATCHDOG_NEGATIVE PASS signal_teardown_removed supervisor_exit=%s surviving_worker_pids=%d watchdog_record=%s'
         %(proc.returncode,len(survived),recorded))
  finally:
   for pid in survived:
    try:os.killpg(pid,signal.SIGKILL)
    except (ProcessLookupError,PermissionError):pass
  kill_probes()
 print('GATE_WATCHDOG_NEGATIVE PASS supervisor bounds, cleanup state and teardown only; fake children; NOT an Orin gate result')

def main():
 try:run()
 finally:
  kill_probes()
  if probes():time.sleep(0.5);kill_probes()
  if probes():raise RuntimeError('control could not clean its own probe processes')

if __name__=='__main__':main()
