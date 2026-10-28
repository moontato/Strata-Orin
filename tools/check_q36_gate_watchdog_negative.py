#!/usr/bin/env python3
"""CPU-only mutation controls for the gate supervisor's execution budget.
Each mutation removes one part of the bound; the deterministic selftest scenario or the
pure verdict MUST then behave like the documented defect. Fake children only: no engine,
model, encoder or service is ever started, and no real staging/worktree is touched.
"""
from pathlib import Path
import importlib.util,importlib.machinery,json,os,resource,signal,subprocess,sys,tempfile,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import q36_image_checkpoint_gate as gate

GROUP_KILL='os.killpg(child.pid,signal.SIGKILL)'
LATE_REASON=" if late:reasons.append('worker finished at or after the execution deadline')\n"
SUPERVISE=' return supervise(a,Budget(a.budget_seconds))'

def probes():
 found=[]
 for entry in Path('/proc').iterdir():
  if not entry.name.isdigit():continue
  try:cmd=Path(entry/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  except OSError:continue
  if gate.OWNED_PROBE in cmd and 'check_q36_gate_watchdog_negative' not in cmd:found.append(int(entry.name))
 return found

def staging(temp):
 stage=Path(temp)/('stage-'+str(len(os.listdir(temp))));(stage/'build').mkdir(parents=True);(stage/'fixtures').mkdir()
 engine=stage/'build'/'strata-q36';engine.write_text('# fake binary marker --kv-image-checkpoint-test\n')
 (stage/'build'/'CMakeCache.txt').write_text('CMAKE_HOME_DIRECTORY:INTERNAL='+str(ROOT)+'\n')
 model=stage/'model.gguf';model.write_bytes(b'not a model')
 (stage/'fixtures'/'manifest.json').write_text(json.dumps({'kind':'q36-image-checkpoints-v1'})+'\n')
 return stage,engine,model,stage/'fixtures'

def run_mutated(source,name,temp,scenario,budget=1,cleanup=1):
 gate_path=Path(temp)/name;gate_path.write_text(source)
 stage,engine,model,fixtures=staging(temp)
 out=stage/('run-'+scenario)
 env=dict(os.environ,STRATA_IMAGE_CHECKPOINT_REVIEW_ROOT=str(ROOT),PYTHONPATH=str(ROOT))
 env.pop('STRATA_IMAGE_CHECKPOINT_REVIEW_WINDOW',None)
 run=subprocess.run([sys.executable,str(gate_path),'--engine',str(engine),'--model',str(model),
  '--fixtures',str(fixtures),'--out',str(out),'--selftest',scenario,'--budget-seconds',str(budget),
  '--cleanup-seconds',str(cleanup)],capture_output=True,text=True,env=env,timeout=120)
 watch={}
 try:watch=json.loads((out/'watchdog.json').read_text())
 except OSError:pass
 return run,watch

def kill_probes():
 for pid in probes():
  try:os.kill(pid,signal.SIGKILL)
  except ProcessLookupError:pass

def main():
 resource.setrlimit(resource.RLIMIT_CORE,(0,0))
 source=(ROOT/'tools/q36_image_checkpoint_gate.py').read_text()
 phases=dict(arms_completed=True)
 clean=dict(killed=False,late=False,exit_code=0,group_empty=True,
  report_status='PASS_BOUNDED_INDEPENDENT_IMAGE_CHECKPOINT_GATE',phases=phases)
 if not gate.verdict(**clean)[0]:raise RuntimeError('real verdict rejects a clean run')
 if gate.verdict(**{**clean,'late':True})[0]:raise RuntimeError('real verdict accepts a late success')
 if gate.verdict(**{**clean,'group_empty':False})[0]:raise RuntimeError('real verdict accepts a live owned group')
 try:
  with tempfile.TemporaryDirectory(prefix='q36-gate-watchdog-') as temp:
   if source.count(GROUP_KILL)!=3 or source.count(LATE_REASON)!=1 or source.count(SUPERVISE)!=1:
    raise RuntimeError('watchdog mutation anchors changed')
   # Mutation 1: the supervisor never group-kills -> a hung descendant leaks and the
   # owned group is reported non-empty.
   run,watch=run_mutated(source.replace(GROUP_KILL,'pass  # MUTATION: no group kill'),
    'mutated_gate.py',temp,'hung_child')
   leaked=probes()
   if not leaked:raise RuntimeError('MUTATION ESCAPED: no-group-kill still reaped the owned descendant')
   if run.returncode==0 or watch.get('owned_group_empty'):raise RuntimeError('MUTATION ESCAPED: live group accepted')
   print('GATE_WATCHDOG_NEGATIVE PASS no_group_kill leaked_owned_pids=%d exit=%d group_empty=%s'
         %(len(leaked),run.returncode,watch.get('owned_group_empty')))
   kill_probes()
   # Mutation 2: the verdict ignores lateness -> a past-deadline completion would pass.
   module_path=Path(temp)/'mutated_verdict.py';module_path.write_text(source.replace(LATE_REASON,''))
   loader=importlib.machinery.SourceFileLoader('mutated_verdict',str(module_path))
   spec=importlib.util.spec_from_loader('mutated_verdict',loader)
   module=importlib.util.module_from_spec(spec);loader.exec_module(module)
   if not module.verdict(**{**clean,'late':True})[0]:raise RuntimeError('MUTATION ESCAPED: late success still refused')
   print('GATE_WATCHDOG_NEGATIVE PASS late_success_accepted')
   # Mutation 3: no supervisor at all -> only the worker's cooperative checks remain,
   # and a blocking descendant survives.
   run,watch=run_mutated(source.replace(SUPERVISE,' return worker(a,Budget(a.budget_seconds))'),
    'mutated_direct.py',temp,'hung_child')
   leaked=probes()
   if not leaked:raise RuntimeError('MUTATION ESCAPED: direct worker reaped the hung descendant')
   if run.returncode==0 or watch.get('killed'):raise RuntimeError('MUTATION ESCAPED: direct worker looked supervised')
   print('GATE_WATCHDOG_NEGATIVE PASS supervisor_bypass leaked_owned_pids=%d exit=%d'%(len(leaked),run.returncode))
   kill_probes()
 finally:
  kill_probes()
  if probes():time.sleep(0.5);kill_probes()
  if probes():raise RuntimeError('control could not clean its own probe processes')
 print('GATE_WATCHDOG_NEGATIVE PASS supervisor bounds only; fake children; NOT an Orin gate result')
if __name__=='__main__':main()
