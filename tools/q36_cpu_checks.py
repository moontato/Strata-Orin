#!/usr/bin/env python3
"""Portable focused CPU checks. Never runs setup.py, GPU/model/encoder/services.
Use a fresh venv installed from requirements-q36-dev.txt; no hidden PYTHONPATH.
"""
import argparse
import datetime
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
TARGETS=['unified_kv_test','kv_preallocation_test','unified_kv_host_capability_test',
 'q36_cache_diagnostic_test','q36_checkpoint_schedule_native_test','q36_checkpoint_schedule_test',
 'q36_adaptive_checkpoint_test','q36_prefix_checkpoint_test','q36_sampled_mtp_test','q36_sampled_request_test',
 'q36_vision_preallocation_test','q36_vision_staging_test','q36_image_checkpoint_test','q36_image_checkpoint_helper_test']
CASES=TARGETS+['q36_schedule_corrupt_'+x for x in ('conv','gdn','hlast')]

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--build-dir',type=Path,default=ROOT/'build-q36-dev-cpu')
 p.add_argument('--log-dir',type=Path,default=ROOT/'logs/q36-dev-cpu')
 p.add_argument('--jobs',type=int,default=2);p.add_argument('--sanitizers',action='store_true')
 p.add_argument('--negative-controls',action='store_true');a=p.parse_args()
 if not 1<=a.jobs<=32:p.error('jobs must be1..32')
 for tool in ('g++','cmake','ctest','ninja'):
  if not shutil.which(tool):p.error('missing '+tool+'; follow docs/Q36_MULTIMODAL_CACHE_HANDOFF.md')
 build=a.build_dir.resolve();stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+str(os.getpid())
 logs=a.log_dir.resolve()/stamp;logs.mkdir(parents=True,exist_ok=False)
 scratch=logs/'scratch';scratch.mkdir()
 env=dict(os.environ,TMPDIR=str(scratch),PYTHONNOUSERSITE='1',PYTHONPATH=str(ROOT),STRATA_IMAGE_CHECKPOINT_REVIEW_ROOT=str(ROOT))
 for key in ('STRATA_IMAGE_CHECKPOINT_REVIEW_WINDOW','STRATA_MM_CACHE_REVIEW_WINDOW','STRATA_CHECKPOINT_REVIEW_WINDOW'):env.pop(key,None)
 record=dict(source=str(ROOT),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
  source_dirty=bool(subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],cwd=ROOT).strip()),
  checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),python=sys.version,targets=TARGETS,cases=CASES,status='RUNNING',sanitizers=a.sanitizers,negative_controls=a.negative_controls,
  packages={x:importlib.metadata.version(x) for x in ('cmake','ninja','pytest','numpy','Pillow','regex','Jinja2','MarkupSafe')})
 (logs/'environment.json').write_text(json.dumps(record,indent=2)+'\n')
 def run(name,cmd,extra=None):
  print('Q36_CPU_CHECK',name,flush=True)
  with (logs/(name+'.txt')).open('w') as f:
   f.write('$ '+repr(list(map(str,cmd)))+'\n');f.flush()
   subprocess.run(list(map(str,cmd)),cwd=ROOT,env=dict(env,**(extra or {})),stdout=f,stderr=subprocess.STDOUT,check=True)
 def suite(directory,sanitized):
  cache=directory/'CMakeCache.txt'
  if cache.exists() and ('CMAKE_HOME_DIRECTORY:INTERNAL='+str(ROOT)) not in cache.read_text():raise RuntimeError('foreign build cache; choose a NEW directory')
  args=['cmake','-S',ROOT,'-B',directory,'-G','Ninja','-DCMAKE_CXX_COMPILER='+shutil.which('g++'),
   '-DSTRATA_ENABLE_CUDA=OFF','-DSTRATA_ENABLE_HIP=OFF','-DSTRATA_ENABLE_SYCL=OFF','-DSTRATA_NATIVE_EXPERTS=OFF',
   '-DSTRATA_BUILD_TESTS=ON','-DSTRATA_BUILD_CONVERSATION_TESTS=OFF','-DCMAKE_BUILD_TYPE=Release']
  label='sanitized' if sanitized else 'release'
  if sanitized:args+=['-DCMAKE_CXX_FLAGS=-O1 -g -Wall -Wextra -Werror -pedantic-errors -fsanitize=address,undefined -fno-omit-frame-pointer -no-pie',
    '-DCMAKE_CXX_FLAGS_RELEASE=-O1 -g','-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined -no-pie']
  run(label+'-configure',args)
  run(label+'-build',['cmake','--build',directory,'--target',*TARGETS,'-j',a.jobs])
  run(label+'-ctest',['ctest','--test-dir',directory,'--output-on-failure','-R','^('+'|'.join(CASES)+')$'],
   dict(ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1') if sanitized else None)
 try:
  suite(build,False)
  tests=sorted(str(x.relative_to(ROOT)) for x in (ROOT/'tools').glob('test_q36_*.py'))
  run('python',[sys.executable,'-m','pytest','-q',*tests,'serve/test_q36_config.py','serve/test_unified_kv.py','tools/test_unified_kv_validate.py'])
  run('source-preservation',[sys.executable,'tools/check_q36_image_preservation.py'])
  if a.negative_controls:
   for name in ('image_checkpoint','vision','restore_observation','gate_watchdog'):
    run('negative-'+name,[sys.executable,'tools/check_q36_'+name+'_negative.py'])
   run('negative-bank',[sys.executable,'tools/check_kv_preallocation_negative.py'])
  if a.sanitizers:suite(build.with_name(build.name+'-sanitized'),True)
  record['status']='PASS_CPU_ONLY';print('Q36_CPU_CHECKS PASS_CPU_ONLY logs='+str(logs),flush=True)
 except BaseException as error:record.update(status='FAIL',error=str(error));print('Q36_CPU_CHECKS FAIL logs='+str(logs),flush=True);raise
 finally:(logs/'result.json').write_text(json.dumps(record,indent=2)+'\n')

if __name__=='__main__':main()
