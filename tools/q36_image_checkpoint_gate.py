#!/usr/bin/env python3
"""Future APPROVED Orin cgroup worker, run under a hard whole-run supervisor.
NO service/config/encoder operations. The supervisor bounds EVERY phase of the worker
(synchronous preflight/hashing, native execution, log reading, comparisons, final
reporting) with one execution budget (default300s, explicit maximum420s) and may
SIGKILL the worker's whole process group during a blocking phase. Waiting for the
worker, reaping it and proving the owned group empty share ONE cleanup allowance
(default20s, maximum60s); the allowance is never re-armed. SIGINT/SIGTERM/SIGHUP are
recorded and routed through that same teardown, which always writes the failure record
and exits nonzero. The supervisor only ever signals the group IT created (its own
session), so an enclosing OPERATOR-OWNED CGROUP is still required to catch anything
else. A timeout, a killed worker or a completion at/after the deadline is always
nonzero: a late success, an uninterruptible-teardown run or an incomplete cleanup is
never accepted. Independent correctness is native empty-state recomputation, not
OFF/ON equality. All staging/results under strata-checkpoints.
"""
import argparse,json,os,re,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(os.environ.get('STRATA_IMAGE_CHECKPOINT_REVIEW_ROOT',Path(__file__).resolve().parents[1])).resolve();sys.path.insert(0,str(ROOT))
from tools.q36_multimodal_cache_gate import no_models,digest
STAGE=Path('/mnt/ssd/tmp/strata-checkpoints');REVIEW=Path('/mnt/ssd/Strata-Orin/worktrees/multimodal-cache')
EXECUTION_BUDGET_DEFAULT=300;EXECUTION_BUDGET_MAX=420;CLEANUP_ALLOWANCE_DEFAULT=20;CLEANUP_ALLOWANCE_MAX=60
SELFTESTS=('normal','slow_preflight','hung_child','slow_postprocess','late_success')
OWNED_PROBE='#Q36OWNEDPROBE'   # marks the gate's OWN fake hung child/descendant for reaping checks
PASS_MARKER='IMAGE_CHECKPOINT_ORIN_GATE PASS independent=1 exact=1 synthetic_only=1 actual_OWUI_savings=UNMEASURED'

class Budget:
    """One wall-clock limit for the ENTIRE run, checked between and after phases.
    Cooperative checks cannot interrupt a blocking call, so the supervisor's group
    kill is the authoritative bound; these checks bound the in-process work.
    The deadline is an absolute CLOCK_MONOTONIC value, so supervisor and worker
    measure the SAME end of the execution budget."""
    def __init__(self,seconds,label='supervisor',deadline=None):
        self.seconds=seconds;self.label=label
        self.deadline=time.monotonic()+seconds if deadline is None else deadline
    def left(self):return self.deadline-time.monotonic()
    def expired(self):return time.monotonic()>=self.deadline
    def check(self,phase):
        if self.expired():raise TimeoutError(f'{self.label} execution budget {self.seconds}s exhausted in phase {phase}')

def measures(text):
 rows=[]
 for line in text.splitlines():
  if line.startswith('IMAGE_CHECKPOINT_MEASURE '):
   values=dict(x.split('=',1) for x in line.split()[1:]);rows.append(values)
 if len(rows)<8 or any(r.get('independent')!='1' or r.get('exact')!='1' for r in rows):raise ValueError('missing independent exact measurement coverage')
 return rows

def validate(directory,manifest,budget):
 from tools.q36_multimodal_cache_fixture import metadata,classify
 directory=Path(directory).resolve()
 if manifest.get('kind')!='q36-image-checkpoints-v1' or not manifest.get('real_tokenizer_template'):raise ValueError('wrong fixture type')
 for name,sha in manifest['files'].items():
  budget.check('fixture hash '+name)
  path=(directory/name).resolve()
  if not path.is_relative_to(directory) or digest(path)!=sha:raise ValueError('fixture hash/path changed')
 rows=manifest['rows'];actual=[];case=None;old=None;last=None;positive=False
 for line in (directory/'image-checkpoints.txt').read_text().splitlines():
  if line.startswith('# case '):case=line[7:]
  elif line:actual.append((case,line.split('\t')))
 if not 8<=len(rows)<=24 or len(actual)!=len(rows):raise ValueError('bounded fixture row count')
 for index,(r,(case,f)) in enumerate(zip(rows,actual)):
  budget.check('fixture audit row '+str(index+1))
  if r['case']!=last:old=None;last=r['case']
  path=Path(r['emb']).resolve()
  if case!=r['case'] or not path.is_relative_to(directory) or f!=[r['slot'],r['phase'],r['expect'],r['key'],r['emb'],','.join(map(str,r['ids']))]:raise ValueError('native input differs from audit')
  if not 2<len(r['ids'])<=6144 or not re.fullmatch('[0-9a-f]{16}',r['key']):raise ValueError('input bounds/key')
  images=[metadata(r['ids'],path,r['key'])]
  if images!=r['images'] or classify(old,r['ids'],images)!=r['audit']:raise ValueError('actual token/image audit differs')
  if r['case']=='real_earlier_user_rewrite' and r['phase']=='steady' and r['expect']=='after' and r['audit']['classification']=='token_rewrite':positive=True
  old=(r['ids'],images)
 if not positive:raise ValueError('no demonstrated steady real rendered image-history rewrite')

def run(command,out,arm,budget,negative=False):
 """One native child. It stays in the worker's process group so the supervisor's
 group kill reaches it AND every descendant, including during a blocking phase."""
 no_models();budget.check('arm '+arm+' start');started=time.monotonic();next_progress=started
 (out/(arm+'.command.json')).write_text(json.dumps(command,indent=2)+'\n')
 with (out/(arm+'.log')).open('wb') as log:
  child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=False);error=None
  try:
   while child.poll() is None:
    now=time.monotonic()
    if now>=budget.deadline:raise TimeoutError('execution budget exhausted while arm '+arm+' was running')
    if now>=next_progress:
     print(f'IMAGE_GATE_PROGRESS arm={arm} elapsed_s={int(now-started)} remaining_s={max(0,int(budget.left()))}',flush=True,file=sys.stderr);next_progress=now+20
    time.sleep(min(0.1,max(0,budget.left())))
   rc=child.returncode
  except BaseException as exc:
   error=str(exc)
   try:os.kill(child.pid,signal.SIGKILL)
   except ProcessLookupError:pass
   # Descendants are the supervisor's job (same process group); do not stall here.
   raise
  finally:
   try:rc=child.wait(timeout=20)
   except subprocess.TimeoutExpired:rc=child.returncode;error=(error or '')+' child reaping exceeded20s'
   (out/(arm+'.exit.json')).write_text(json.dumps(dict(exit=rc,error=error,elapsed_s=time.monotonic()-started))+'\n')
 budget.check('arm '+arm+' reaped')
 text=(out/(arm+'.log')).read_text(errors='replace')
 budget.check('arm '+arm+' log read')
 if negative:
  if not rc or 'IMAGE_CHECKPOINT_CORRUPT category=conv' not in text or 'IMAGE_CHECKPOINT_FAILURE comparison=restored' not in text:raise RuntimeError('corruption did not fail the independent RESTORED-state oracle; stop')
 elif rc or 'IMAGE_CHECKPOINT_TEST PASS' not in text or 'KV_PREALLOCATION_MODEL PASS' not in text or re.search(r'^(?:ERR|BERR) ',text,re.M):raise RuntimeError(arm+' native gate failed; retain logs, no tolerance waiver')
 return text

def selftest_arm(name,out,arm,budget):
 """Deterministic fake children for supervisor bounds only. No engine/model/encoder."""
 if name=='hung_child':
  # A fake child AND its own descendant, both marked so owned-group reaping is checkable.
  child_code='import time;time.sleep(600) '+OWNED_PROBE
  parent_code=('import subprocess,sys,time;subprocess.Popen([sys.executable,'+repr('-c')+','+repr(child_code)+']);'
               'time.sleep(600) '+OWNED_PROBE)
  return run([sys.executable,'-c',parent_code],out,arm,budget)
 if name=='late_success':
  time.sleep(min(4.0,max(0.2,budget.left()+0.7)))   # deliberately cross the deadline
  return run([sys.executable,'-c','print("IMAGE_CHECKPOINT_TEST PASS");print("KV_PREALLOCATION_MODEL PASS")'],out,arm,budget)
 return run([sys.executable,'-c','print("IMAGE_CHECKPOINT_TEST PASS");print("KV_PREALLOCATION_MODEL PASS")'],out,arm,budget)

def worker(a,budget):
 """All real work, inside the supervisor's bound. Prints progress to stderr and the
 verdict marker to stdout, which the supervisor relays ONLY if it verifies the run."""
 report=dict(status='PENDING',source_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD']).decode().strip(),
  execution_budget_seconds=a.budget_seconds,selftest=a.selftest or '',
  actual_OWUI='token divergence and savings UNMEASURED',
  note='public synthetic fixture timing, not actual OWUI; OFF/ON schedules may differ, each must pass independent exact recomputation')
 try:
  budget.check('binary hash')
  report['binary_sha256']=digest(a.engine)
  budget.check('binary provenance')
  cache=a.engine.resolve().parent/'CMakeCache.txt'
  if not cache.is_file() or ('CMAKE_HOME_DIRECTORY:INTERNAL='+str(REVIEW)) not in cache.read_text():raise RuntimeError('build provenance must name ONLY the existing multimodal-cache worktree')
  if b'--kv-image-checkpoint-test' not in a.engine.read_bytes():raise RuntimeError('new diagnostic binary required')
  manifest=json.loads((a.fixtures/'manifest.json').read_text())
  if not a.selftest:
   budget.check('preflight')
   validate(a.fixtures,manifest,budget)
   st=a.model.stat()
   if manifest['model']!=dict(path=str(a.model.resolve()),size=st.st_size,mtime_ns=st.st_mtime_ns):raise RuntimeError('model provenance changed')
  if a.preflight:
   report['status']='PASS_PREFLIGHT_ONLY';print('IMAGE_CHECKPOINT_PREFLIGHT PASS model_not_started=1',flush=True);return 0
  report['inputs']=manifest
  if a.selftest:
   budget.check('selftest '+a.selftest)
   if a.selftest=='slow_preflight':time.sleep(a.budget_seconds+2)   # blocking synchronous phase
   for arm in ('negative-conv','disabled','enabled'):selftest_arm(a.selftest,out=a.out,arm=arm,budget=budget)
   if a.selftest=='slow_postprocess':time.sleep(a.budget_seconds+2)  # blocking post-run phase
   report['status']='PASS_BOUNDED_INDEPENDENT_IMAGE_CHECKPOINT_GATE';report['selftest_measure']=True
   print(PASS_MARKER,flush=True);return 0
  no_models();budget.check('model window start')
  common=[str(a.engine.resolve()),'--model',str(a.model.resolve()),'--serve','--vision','--max-context','262144','--batch','4',
   '--kv-pool-tokens','524288','--kv','int8','--mtp','2','--batch-mtp','--mtp-sampled','--kv-checkpoint-mib','1024',
   '--kv-checkpoint-interval','1024','--kv-checkpoint-policy','adaptive','--kv-preallocate','--kv-vision-max-tokens','1024',
   '--chunk','512','--kv-image-checkpoint-test',str((a.fixtures/'image-checkpoints.txt').resolve())]
  run(common+['--kv-image-checkpoints','--kv-image-checkpoint-corrupt','conv'],a.out,'negative-conv',budget,True)
  off=run(common,a.out,'disabled',budget);on=run(common+['--kv-image-checkpoints'],a.out,'enabled',budget)
  budget.check('measurement parse')
  report['measurements']={'disabled':measures(off),'enabled':measures(on)}
  positive=[r for r in report['measurements']['enabled'] if r['case']=='real_earlier_user_rewrite' and r['phase']=='steady' and int(r['checkpoint'])>0]
  if not positive:raise RuntimeError('no positive checkpoint reuse in steady real rendered image-history rewrite')
  for name,sha in manifest['files'].items():
   budget.check('post-run hash '+name)
   if digest(a.fixtures/name)!=sha:raise RuntimeError('input mutated during gate')
  budget.check('verdict')
  report.update(status='PASS_BOUNDED_INDEPENDENT_IMAGE_CHECKPOINT_GATE',synthetic_only=True,steady_rewrite_reuse=positive)
  print(PASS_MARKER,flush=True);return 0
 except BaseException as exc:report.update(status='FAIL',error=str(exc));raise
 finally:
  (a.out/'result.json').write_text(json.dumps(report,indent=2)+'\n')

def verdict(killed,late,exit_code,group_empty,report_status,phases,preflight=False,interrupted='',worker_reaped=True):
 """Pure decision: a killed worker, a late completion, a non-empty owned group, a
 worker that was never reaped, an interrupted supervisor or a non-PASS report can
 never produce a PASS. The accepted report status also depends on the mode, so a
 preflight-only report can never be counted as a numerical gate pass (nor a numerical
 gate report as preflight). Deterministic and unit-testable."""
 reasons=[]
 if interrupted:reasons.append('supervisor interrupted by '+str(interrupted))
 if killed and not interrupted:reasons.append('worker killed at the execution deadline')
 if killed and interrupted:reasons.append('worker ended by the teardown of an interrupted run')
 if late:reasons.append('worker finished at or after the execution deadline')
 if exit_code!=0:reasons.append('worker exit '+str(exit_code))
 if not worker_reaped:reasons.append('worker still unowned after the cleanup allowance')
 if not group_empty:reasons.append('owned process group still present after cleanup allowance')
 accepted='PASS_PREFLIGHT_ONLY' if preflight else 'PASS_BOUNDED_INDEPENDENT_IMAGE_CHECKPOINT_GATE'
 if report_status!=accepted:reasons.append('worker report status '+str(report_status)+' not accepted for preflight='+str(preflight))
 if not phases.get('arms_completed'):reasons.append('not every gate arm completed')
 return (not reasons),reasons

CLEANUP_POLL=0.05

def group_kill(child):
 """SIGKILL ONLY the process group this supervisor created (start_new_session=True).
 Never a process name, never a foreign pid, never a group we did not make."""
 try:
  os.killpg(child.pid,signal.SIGKILL);return 'present'
 except ProcessLookupError:return 'empty'
 except PermissionError:return 'denied'

def group_probe(child):
 """Presence probe of the owned group only (signal 0)."""
 try:
  os.killpg(child.pid,0);return 'present'
 except ProcessLookupError:return 'empty'
 except PermissionError:return 'denied'

def worker_status(child):
 """Worker exit status WITHOUT consuming it, so the leader stays unreaped and its PGID
 cannot be recycled and signalled by accident while the group is being cleaned.
 Returns None while the worker is still running (siginfo-style waitid result)."""
 try:info=os.waitid(os.P_PID,child.pid,os.WEXITED|os.WNOHANG|os.WNOWAIT)
 except (ChildProcessError,NotImplementedError,AttributeError,OSError):return None
 if info is None:return None
 code=getattr(info,'si_code',getattr(info,'code',1));status=getattr(info,'si_status',getattr(info,'status',0))
 if code==1:return status                                  # CLD_EXITED
 if code in (2,3):return -status                           # CLD_KILLED / CLD_DUMPED
 return None

class CleanupState:
 """Persistent cleanup progress for ONE run.
 The absolute cleanup deadline is created once and reused by every later call (normal
 path, signal path or an exceptional path), so no path can re-arm the allowance. It
 also remembers whether the leader was reaped: after that, the former group is NEVER
 signalled again, because its PGID may already belong to another process."""
 def __init__(self,child,allowance):
  self.child=child;self.allowance=allowance;self.deadline=None
  self.exit_code=None;self.group_status='empty';self.group_empty=False
  self.reaped=False;self.complete=False
 def begin(self):
  if self.deadline is None:self.deadline=time.monotonic()+self.allowance
  return self.deadline
 def result(self):return self.exit_code,self.group_status,self.group_empty,self.reaped

def teardown(state,poll=CLEANUP_POLL,waiter=None,killer=None,prober=None):
 """ONE absolute deadline covers ALL cleanup: killing the owned group, waiting for the
 worker, killing stragglers, reaping the leader and proving the group empty. The
 allowance is never re-armed, so a run cannot spend twice the cleanup allowance.
 Idempotent: calling it again after cleanup finished (for example from an error path
 while writing the result) changes nothing, spends no new allowance and, once the
 leader is reaped, sends NO further signal to the former group.
 Ordering avoids two hazards: the group is only ever SIGNALED while the leader is still
 unreaped (so its PGID cannot have been recycled), and the leader is reaped before the
 final emptiness proof (a reaped leader is no longer a group member, so it cannot make
 the group look alive). Only probing happens after the reap. Returns
 (exit_code_or_None, group_kill_status, group_empty, reaped); success is impossible
 unless the worker was reaped AND the owned group is empty. The kill STATUS is
 returned, not a bool: a group that still holds only the already-exited, not-yet-reaped
 leader was never killed by us."""
 child=state.child
 if state.complete:return state.result()
 deadline=state.begin()
 waiter=waiter or (lambda:worker_status(child))
 killer=killer or (lambda:group_kill(child))
 prober=prober or (lambda:group_probe(child))
 def left():return deadline-time.monotonic()
 def nap():time.sleep(min(poll,max(0.0,left())))
 def signal_group():
  # Ownership guard: after the leader is reaped we can no longer prove the PGID is ours.
  if state.reaped:return state.group_status
  status=killer()
  if status=='present':state.group_status='present'
  return status
 signal_group();exit_code=state.exit_code
 if exit_code is None:
  while True:                              # 1. wait for the worker (same deadline)
   code=waiter()
   if code is not None:exit_code=code;break
   if left()<=0:break
   nap()
  state.exit_code=exit_code
 signal_group()                            # 2. last group kill while the PGID is reserved
 if left()>0 or exit_code is not None:
  try:
   code=child.wait(timeout=max(0.0,left()));state.reaped=True
   if exit_code is None:exit_code=code
  except (subprocess.TimeoutExpired,OSError,ValueError):state.reaped=False
 state.exit_code=exit_code
 while True:                               # 3. prove the owned group empty (probe only)
  status=prober()
  if status=='empty':state.group_empty=True;break
  if status=='denied' or left()<=0:break
  nap()
 state.complete=True
 return state.result()

def supervise(a,budget):
 """Spawn the worker in its OWN process group and hard-bound the whole run, including
 any blocking phase, with a group kill. Waiting, reaping and group verification share
 ONE cleanup allowance. SIGINT/SIGTERM/SIGHUP are recorded (never raised): an exception
 used to escape before reaping and before the failure record. Whatever happens, the
 owned group is killed, reaped and verified, the failure is recorded and the exit is
 nonzero. The supervisor signals only its own group, so the enclosing operator-owned
 cgroup remains a requirement, not something this code replaces."""
 out=a.out
 log=open(out/'worker-stdout.txt','wb')
 label='IMAGE_CHECKPOINT_GATE_SELFTEST' if a.selftest else 'IMAGE_CHECKPOINT_ORIN_GATE'
 recorded=[False]
 def write_record(**fields):
  fields.setdefault('operator_cgroup_required',True)   # the operator still owns the enclosing cgroup
  target=out/'watchdog.json'
  if getattr(a,'fault',None)=='record':
   # Fault injection (selftest only): writing under a regular file reproduces the
   # "the result could not be recorded" (ENOSPC/EIO class) failure deterministically.
   (out/'record-fault').write_text('fault injection marker\n')
   target=out/'record-fault'/'watchdog.json'
  target.write_text(json.dumps(fields,indent=2)+'\n');recorded[0]=True
 def record_failed(exc):
  # A run whose result cannot be written is NOT a pass and must not restart cleanup:
  # the cleanup state is already final, so this only reports and forces a nonzero exit.
  for line in ('IMAGE_GATE_PROGRESS stage=record action=failed error='+type(exc).__name__+': '+str(exc)[:200],
   'IMAGE_GATE_PROGRESS stage=record action=no_restart cleanup_state_final=1 operator_cgroup_required=1'):
   try:print(line,flush=True,file=sys.stderr)
   except Exception:pass
  try:print(label+' FAIL result_record_failed_'+type(exc).__name__,flush=True)
  except Exception:pass
 interrupted=[]
 def record(signum,frame):interrupted.append(signum)   # recorded, never raised
 previous={}
 for sig,name in ((signal.SIGINT,'SIGINT'),(signal.SIGTERM,'SIGTERM'),(signal.SIGHUP,'SIGHUP')):
  try:previous[sig]=signal.signal(sig,record)
  except (ValueError,OSError):pass   # not the main thread: keep the old disposition
 try:
  child=None
  # The worker shares this absolute deadline: its own cooperative checks and the
  # supervisor's group kill then end at the same instant.
  child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--worker','--deadline',repr(budget.deadline)],
   stdin=subprocess.DEVNULL,stdout=log,stderr=sys.stderr,start_new_session=True)
  killed=False;deadline_seen=False;exit_code=None
  cleanup=CleanupState(child,a.cleanup_seconds)   # ONE cleanup state and deadline for this run
  SIGNAL_NAMES={signal.SIGINT:'SIGINT',signal.SIGTERM:'SIGTERM',signal.SIGHUP:'SIGHUP'}
  while exit_code is None:
   exit_code=worker_status(child)     # non-consuming: the PGID stays reserved
   if exit_code is not None:break
   if budget.expired():
    deadline_seen=True
    if not killed:
     print(f'IMAGE_GATE_PROGRESS stage=watchdog action=killpg elapsed_s={int(a.budget_seconds-budget.left())}',flush=True,file=sys.stderr)
     killed=killed or group_kill(child)=='present'
    break
   if interrupted:
    deadline_seen=True
    print('IMAGE_GATE_PROGRESS stage=watchdog action=teardown signal='+
     ','.join(SIGNAL_NAMES.get(s,'SIG'+str(s)) for s in interrupted)+
     ' cleanup_allowance_s='+str(a.cleanup_seconds),flush=True,file=sys.stderr)
    break
   time.sleep(min(0.25,max(0.0,budget.left())))
  # ONE cleanup deadline for waiting + reaping + group proof: no second allowance, and
  # the SAME state is reused if an error path needs cleanup again.
  cleanup.exit_code=exit_code            # status already seen without consuming it
  exit_code,group_status,group_empty,reaped=teardown(cleanup)
  # 'killed' means the supervisor had to end a run that did NOT finish on its own; a
  # group that only held the already-exited leader was never killed by us.
  killed=killed or deadline_seen or bool(interrupted) or (group_status=='present' and exit_code is None)
  signals=','.join(sorted({SIGNAL_NAMES.get(s,'SIG'+str(s)) for s in interrupted}))
  late=(deadline_seen and not interrupted) or (exit_code==0 and budget.expired())
  report={}
  try:report=json.loads((out/'result.json').read_text())
  except (OSError,ValueError):report={'status':'MISSING'}
  phases=dict(arms_completed=all((out/(arm+'.exit.json')).is_file() for arm in ('negative-conv','disabled','enabled')) and not a.preflight and not a.selftest,
   preflight=bool(a.preflight),selftest=bool(a.selftest))
  if a.preflight or a.selftest:phases['arms_completed']=True
  passed,reasons=verdict(killed,late,exit_code if exit_code is not None else -1,group_empty,report.get('status'),phases,
   a.preflight,signals,reaped)
  if getattr(a,'fault',None)=='crash':
   # Fault injection (selftest only): an unexpected failure AFTER cleanup finished.
   raise RuntimeError('fault injection: post-teardown supervisor failure')
  try:
   write_record(execution_budget_seconds=a.budget_seconds,cleanup_allowance_seconds=a.cleanup_seconds,
    killed=killed,deadline_crossed=late,worker_exit=exit_code,worker_reaped=reaped,owned_group_empty=group_empty,
    cleanup_complete=group_empty and reaped,worker_report_status=report.get('status'),interrupted_signals=signals,
    passed=passed,reasons=reasons)
  except OSError as record_exc:
   # The owned group is already cleaned. A failure while RECORDING the result must never
   # look like a pass, must not re-run cleanup and must not re-signal a reaped group.
   record_failed(record_exc)
   return 1
 except BaseException as exc:
  # A supervisor fault must still leave a failure record. The cleanup state is shared
  # and teardown is idempotent, so this can never spend a new allowance or signal a
  # group whose leader was already reaped.
  if child is not None:
   try:teardown(cleanup)
   except Exception:pass
   exit_code,_,group_empty,reaped=cleanup.result()
  else:exit_code=None;group_empty=False;reaped=False
  if not recorded[0]:
   try:
    write_record(execution_budget_seconds=a.budget_seconds,cleanup_allowance_seconds=a.cleanup_seconds,
     killed=killed or cleanup.group_status=='present',deadline_crossed=False,worker_exit=exit_code,
     worker_reaped=reaped,owned_group_empty=group_empty,cleanup_complete=group_empty and reaped,
     worker_report_status='MISSING',interrupted_signals='',passed=False,
     reasons=['supervisor error '+type(exc).__name__+': '+str(exc)[:200]])
   except OSError as record_exc:
    record_failed(record_exc)   # labels the run as failed and notes cleanup is final
    raise
  # Label the failure on stdout as well, so an operator never sees a silent run; the
  # exception is re-raised afterwards for the log.
  try:print(label+' FAIL supervisor_error_'+type(exc).__name__,flush=True)
  except Exception:pass
  raise
 finally:
  log.close()
  for sig,handler in previous.items():
   try:signal.signal(sig,handler)
   except (ValueError,OSError):pass
 if passed:
  # A selftest never relays a production marker: only the supervisor's labelled
  # verdict is printed, so a fake arm can never look like a real gate pass.
  if not a.selftest:
   for line in (out/'worker-stdout.txt').read_text(errors='replace').splitlines():
    if line.startswith('IMAGE_CHECKPOINT_'):print(line,flush=True)
  if a.preflight:print('IMAGE_CHECKPOINT_PREFLIGHT PASS model_not_started=1',flush=True)
  else:print(PASS_MARKER if not a.selftest else label+' verdict=PASS selftest='+str(a.selftest),flush=True)
  print(label+'_SUPERVISOR PASS budget_s=%d cleanup_allowance_s=%d owned_group_empty=1'%(a.budget_seconds,a.cleanup_seconds),flush=True)
  return 0
 print(label+' FAIL '+' '.join(f'{r.replace(" ","_")}' for r in reasons),flush=True)
 return 1

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for x in ('engine','model','fixtures','out'):p.add_argument('--'+x,required=True,type=Path)
 p.add_argument('--preflight',action='store_true')
 p.add_argument('--budget-seconds',type=int,default=EXECUTION_BUDGET_DEFAULT)
 p.add_argument('--cleanup-seconds',type=int,default=CLEANUP_ALLOWANCE_DEFAULT)
 p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
 p.add_argument('--selftest',choices=SELFTESTS,help=argparse.SUPPRESS)
 p.add_argument('--fault',choices=('record','crash'),help=argparse.SUPPRESS)
 p.add_argument('--deadline',type=float,help=argparse.SUPPRESS)
 a=p.parse_args()
 global STAGE,REVIEW
 if a.selftest:
  # Bounds checks ONLY: selftest arms run fake children, never an engine, model,
  # encoder or service, so they may use a throwaway staging directory.
  REVIEW=ROOT;STAGE=a.out.resolve().parent
 if getattr(a,'fault',None) and not a.selftest:p.error('fault injection requires --selftest')
 if ROOT!=REVIEW or (not a.preflight and not a.selftest and os.environ.get('STRATA_IMAGE_CHECKPOINT_REVIEW_WINDOW')!='1'):p.error('only approved existing multimodal-cache checkout/window')
 if not 1<=a.budget_seconds<=EXECUTION_BUDGET_MAX:p.error(f'execution budget must be 1..{EXECUTION_BUDGET_MAX}s')
 if not 1<=a.cleanup_seconds<=CLEANUP_ALLOWANCE_MAX:p.error(f'cleanup allowance must be 1..{CLEANUP_ALLOWANCE_MAX}s')
 if not a.engine.resolve().is_relative_to(STAGE) or not a.fixtures.resolve().is_relative_to(STAGE) or not a.out.resolve().is_relative_to(STAGE) or a.out.resolve()==STAGE:p.error('NEW staging results and isolated test binary required; never live checkpoint-policy')
 if a.worker:return worker(a,Budget(a.budget_seconds,'worker',a.deadline))
 # Supervisor: the only entity that may print a verdict marker.
 if a.out.exists():p.error('results directory already exists; preserve old artifacts')
 a.out.mkdir(parents=True)
 if not a.preflight and not a.selftest:no_models()
 return supervise(a,Budget(a.budget_seconds))

if __name__=='__main__':sys.exit(main())
