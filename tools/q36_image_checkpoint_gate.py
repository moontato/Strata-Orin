#!/usr/bin/env python3
"""Future APPROVED Orin cgroup worker. NO service/config/encoder operations.
At most300s total (operator may choose <=420s), three sequential LLM children, progress
20s, fail-fast. Independent correctness is native empty-state recomputation,
not OFF/ON equality. All staging/results under strata-checkpoints.
"""
import argparse,json,os,re,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(os.environ.get('STRATA_IMAGE_CHECKPOINT_REVIEW_ROOT',Path(__file__).resolve().parents[1])).resolve();sys.path.insert(0,str(ROOT))
from tools.q36_multimodal_cache_gate import no_models,digest
STAGE=Path('/mnt/ssd/tmp/strata-checkpoints');REVIEW=Path('/mnt/ssd/Strata-Orin/worktrees/multimodal-cache')

def measures(text):
 rows=[]
 for line in text.splitlines():
  if line.startswith('IMAGE_CHECKPOINT_MEASURE '):
   values=dict(x.split('=',1) for x in line.split()[1:]);rows.append(values)
 if len(rows)<8 or any(r.get('independent')!='1' or r.get('exact')!='1' for r in rows):raise ValueError('missing independent exact measurement coverage')
 return rows

def validate(directory,manifest,deadline=None):
 from tools.q36_multimodal_cache_fixture import metadata,classify
 directory=Path(directory).resolve()
 if manifest.get('kind')!='q36-image-checkpoints-v1' or not manifest.get('real_tokenizer_template'):raise ValueError('wrong fixture type')
 for name,sha in manifest['files'].items():
  if deadline is not None and time.monotonic()>=deadline:raise TimeoutError('preflight budget exhausted')
  path=(directory/name).resolve()
  if not path.is_relative_to(directory) or digest(path)!=sha:raise ValueError('fixture hash/path changed')
 rows=manifest['rows'];actual=[];case=None;old=None;last=None;positive=False
 for line in (directory/'image-checkpoints.txt').read_text().splitlines():
  if line.startswith('# case '):case=line[7:]
  elif line:actual.append((case,line.split('\t')))
 if not 8<=len(rows)<=24 or len(actual)!=len(rows):raise ValueError('bounded fixture row count')
 for index,(r,(case,f)) in enumerate(zip(rows,actual)):
  if deadline is not None:
   if time.monotonic()>=deadline:raise TimeoutError('preflight budget exhausted')
   print(f'IMAGE_GATE_PROGRESS stage=preflight row={index+1}/{len(rows)}',flush=True)
  if r['case']!=last:old=None;last=r['case']
  path=Path(r['emb']).resolve()
  if case!=r['case'] or not path.is_relative_to(directory) or f!=[r['slot'],r['phase'],r['expect'],r['key'],r['emb'],','.join(map(str,r['ids']))]:raise ValueError('native input differs from audit')
  if not 2<len(r['ids'])<=6144 or not re.fullmatch('[0-9a-f]{16}',r['key']):raise ValueError('input bounds/key')
  images=[metadata(r['ids'],path,r['key'])]
  if images!=r['images'] or classify(old,r['ids'],images)!=r['audit']:raise ValueError('actual token/image audit differs')
  if r['case']=='real_earlier_user_rewrite' and r['phase']=='steady' and r['expect']=='after' and r['audit']['classification']=='token_rewrite':positive=True
  old=(r['ids'],images)
 if not positive:raise ValueError('no demonstrated steady real rendered image-history rewrite')


def run(command,out,arm,deadline,negative=False):
 no_models();started=time.monotonic();next_progress=started
 (out/(arm+'.command.json')).write_text(json.dumps(command,indent=2)+'\n')
 if started>=deadline:raise TimeoutError('bounded gate budget exhausted before child')
 with (out/(arm+'.log')).open('wb') as log:
  child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);error=None
  try:
   while os.waitid(os.P_PID,child.pid,os.WEXITED|os.WNOHANG|os.WNOWAIT) is None:
    now=time.monotonic()
    if now>=deadline:raise TimeoutError('bounded image checkpoint gate timeout')
    if now>=next_progress:
     print(f'IMAGE_GATE_PROGRESS arm={arm} elapsed_s={int(now-started)} remaining_s={max(0,int(deadline-now))}',flush=True);next_progress=now+20
    time.sleep(min(0.1,max(0,deadline-now)))
  except BaseException as exc:error=str(exc);raise
  finally:
   handlers={s:signal.signal(s,signal.SIG_IGN) for s in (signal.SIGINT,signal.SIGTERM)}
   try:
    try:os.killpg(child.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    rc=child.wait(timeout=20)
   finally:
    for s,h in handlers.items():signal.signal(s,h)
   (out/(arm+'.exit.json')).write_text(json.dumps(dict(exit=rc,error=error,elapsed_s=time.monotonic()-started))+'\n')
 text=(out/(arm+'.log')).read_text(errors='replace')
 if negative:
  if not rc or 'IMAGE_CHECKPOINT_CORRUPT category=conv' not in text or 'IMAGE_CHECKPOINT_FAILURE comparison=restored' not in text:raise RuntimeError('corruption did not fail the independent RESTORED-state oracle; stop')
 elif rc or 'IMAGE_CHECKPOINT_TEST PASS' not in text or 'KV_PREALLOCATION_MODEL PASS' not in text or re.search(r'^(?:ERR|BERR) ',text,re.M):raise RuntimeError(arm+' native gate failed; retain logs, no tolerance waiver')
 return text


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for x in ('engine','model','fixtures','out'):p.add_argument('--'+x,required=True,type=Path)
 p.add_argument('--preflight',action='store_true');p.add_argument('--budget-seconds',type=int,default=300);a=p.parse_args()
 if ROOT!=REVIEW or (not a.preflight and os.environ.get('STRATA_IMAGE_CHECKPOINT_REVIEW_WINDOW')!='1'):p.error('only approved existing multimodal-cache checkout/window')
 if not a.engine.resolve().is_relative_to(STAGE) or not a.fixtures.resolve().is_relative_to(STAGE) or not a.out.resolve().is_relative_to(STAGE) or a.out.resolve()==STAGE or a.out.exists():p.error('NEW staging results and isolated test binary required; never live checkpoint-policy')
 if not 1<=a.budget_seconds<=420:p.error('maximum total model budget420s')
 cache=a.engine.resolve().parent/'CMakeCache.txt'
 if not cache.is_file() or ('CMAKE_HOME_DIRECTORY:INTERNAL='+str(REVIEW)) not in cache.read_text():p.error('build provenance must name ONLY the existing multimodal-cache worktree')
 if b'--kv-image-checkpoint-test' not in a.engine.read_bytes():p.error('new diagnostic binary required')
 deadline=time.monotonic()+a.budget_seconds
 print('IMAGE_GATE_PROGRESS stage=preflight_start',flush=True)
 manifest=json.loads((a.fixtures/'manifest.json').read_text());validate(a.fixtures,manifest,deadline)
 st=a.model.stat()
 if manifest['model']!=dict(path=str(a.model.resolve()),size=st.st_size,mtime_ns=st.st_mtime_ns):p.error('model provenance changed')
 if a.preflight:print('IMAGE_CHECKPOINT_PREFLIGHT PASS model_not_started=1');return
 no_models();a.out.mkdir(parents=True)
 def interrupted(s,_):raise KeyboardInterrupt('signal '+str(s))
 signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
 common=[str(a.engine.resolve()),'--model',str(a.model.resolve()),'--serve','--vision','--max-context','262144','--batch','4',
  '--kv-pool-tokens','524288','--kv','int8','--mtp','2','--batch-mtp','--mtp-sampled','--chunk','512','--kv-checkpoint-mib','1024',
  '--kv-checkpoint-interval','1024','--kv-checkpoint-policy','adaptive','--kv-preallocate','--kv-vision-max-tokens','1024',
  '--kv-image-checkpoint-test',str((a.fixtures/'image-checkpoints.txt').resolve())]
 report=dict(status='PENDING',source_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD']).decode().strip(),binary_sha256=digest(a.engine),inputs=manifest,budget_seconds=a.budget_seconds,actual_OWUI='token divergence and savings UNMEASURED',
  note='public synthetic fixture timing, not actual OWUI; OFF/ON schedules may differ, each must pass independent exact recomputation')
 try:
  run(common+['--kv-image-checkpoints','--kv-image-checkpoint-corrupt','conv'],a.out,'negative-conv',deadline,True)
  off=run(common,a.out,'disabled',deadline);on=run(common+['--kv-image-checkpoints'],a.out,'enabled',deadline)
  report['measurements']={'disabled':measures(off),'enabled':measures(on)}
  positive=[r for r in report['measurements']['enabled'] if r['case']=='real_earlier_user_rewrite' and r['phase']=='steady' and int(r['checkpoint'])>0]
  if not positive:raise RuntimeError('no positive checkpoint reuse in steady real rendered image-history rewrite')
  for name,sha in manifest['files'].items():
   if digest(a.fixtures/name)!=sha:raise RuntimeError('input mutated during gate')
  report.update(status='PASS_BOUNDED_INDEPENDENT_IMAGE_CHECKPOINT_GATE',synthetic_only=True,steady_rewrite_reuse=positive)
  print('IMAGE_CHECKPOINT_ORIN_GATE PASS independent=1 exact=1 synthetic_only=1 actual_OWUI_savings=UNMEASURED',flush=True)
 except BaseException as exc:report.update(status='FAIL',error=str(exc));raise
 finally:(a.out/'result.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
