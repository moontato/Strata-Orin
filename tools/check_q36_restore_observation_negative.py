#!/usr/bin/env python3
"""CPU-only negative controls for the restored-state OBSERVATION BOUNDARY.
Each mutation reproduces one documented defect; the host harness or the placement
invariants MUST fail. No CUDA/GPU/model/service run.
"""
from pathlib import Path
import resource,shutil,sys,tempfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.q36_restore_observation_harness import (COPY_START,COPY_END,DEFER,LEGACY_OBSERVE,GUARD,
    build,copy_session_body,placement_errors)

def copy_at_copy_time(source):
    """Pre-fix placement: observe the copy while the destination is not current."""
    body=copy_session_body(source)
    if body.count(DEFER)!=1:raise RuntimeError('deferral anchor changed')
    return source.replace(body,body.replace(DEFER,LEGACY_OBSERVE))

def fire_before_select(source):
    """Pre-fix placement: fire the deferred observation before the caller selects dst."""
    fire='        E.fire_restore_snapshot();\n'
    if source.count(fire)!=1:raise RuntimeError('fire anchor changed')
    select='        E.select(s);\n'
    moved=source.replace(fire,'').replace(select,fire+select,1)
    if moved==source:raise RuntimeError('fire-before-select mutation did not apply')
    return moved

def guard_removed(primitives):
    """Pre-fix primitive: observe a destination that is not the current slot."""
    if primitives.count(GUARD)!=2:raise RuntimeError('guard anchor changed')
    return primitives.replace(GUARD,'if(false) { ++restore_snapshot_refused;')

def main():
 resource.setrlimit(resource.RLIMIT_CORE,(0,0))
 if placement_errors(ROOT):raise RuntimeError('current tree violates the observation contract')
 code,stdout,stderr=build(ROOT)
 if code:raise RuntimeError('current tree fails its own observation harness: '+(stderr or stdout))
 print('RESTORE_OBSERVATION BASELINE PASS placement=1 harness=0 exit=0')
 with tempfile.TemporaryDirectory(prefix='q36-restore-negative-') as temp:
  for name,mutate,kind in (('observe_at_copy_time',copy_at_copy_time,'harness'),
                           ('fire_before_select',fire_before_select,'placement'),
                           ('guard_removed',guard_removed,'primitives')):
   if kind=='primitives':
    code,stdout,stderr=build(ROOT,primitives_overlay=mutate((ROOT/'src/qwen35/q36_restore_observation.inc').read_text()))
    if not code:raise RuntimeError('MUTATION ESCAPED: '+name)
    print('RESTORE_OBSERVATION_NEGATIVE PASS',name,'exit='+str(code))
    continue
   stage=Path(temp)/name;(stage/'src/qwen35').mkdir(parents=True,exist_ok=True)
   shutil.copytree(ROOT/'include',stage/'include')
   (stage/'src/qwen35/q36_engine.cu').write_text(mutate((ROOT/'src/qwen35/q36_engine.cu').read_text()))
   shutil.copy(ROOT/'src/qwen35/q36_restore_observation.inc',stage/'src/qwen35/q36_restore_observation.inc')
   errors=placement_errors(stage)
   if not errors:raise RuntimeError('MUTATION ESCAPED (placement): '+name)
   if kind=='harness':
    code,_,_=build(stage)
    if not code:raise RuntimeError('MUTATION ESCAPED (harness): '+name)
    print('RESTORE_OBSERVATION_NEGATIVE PASS',name,'placement_exit=1 harness_exit='+str(code))
   else:print('RESTORE_OBSERVATION_NEGATIVE PASS',name,'placement_exit=1')
 print('RESTORE_OBSERVATION_NEGATIVE PASS host-only boundary controls; NOT GPU qualification')
if __name__=='__main__':main()
