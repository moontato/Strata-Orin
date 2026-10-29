#!/usr/bin/env python3
"""Deliberate production-header mutations MUST fail the CPU regression.
All overlays/binaries/data stay in session temporary storage; no source edits.
"""
import os
from pathlib import Path
import resource
import subprocess
import tempfile
ROOT=Path(__file__).resolve().parents[1];HEADER='strata/core/q36_vision_preallocation.hpp'
MUTATIONS={
 'unbounded-rows':('if(total>max_rows)','if(false && total>max_rows)'),
 'overwrite-active-staging':('if(slot<0 || busy())','if(slot<0)'),
 'ignore-stale-epoch':('if(generation!=gen || epoch!=ep)','if(generation!=gen)'),
 'zero-retained-rope-offset':('return (int32_t)((int64_t)std::max({mrope[at],mrope[at+1],mrope[at+2]})+1-(int64_t)length);','(void)at;return 0;'),
 'allow-nan-embeddings':('if(!std::isfinite(v))','if(false && !std::isfinite(v))'),
 'trust-key-ignore-actual-rows':('hash=q36_image_hash(out.rows.data()+row*width,(size_t)r.n*width*4,hash);','hash=q36_image_hash(out.rows.data()+row*width,0,hash);'),
}
def main():
 resource.setrlimit(resource.RLIMIT_CORE,(0,0));original=(ROOT/'include'/HEADER).read_text()
 for name,(old,new) in MUTATIONS.items():
  if original.count(old)!=1:raise RuntimeError('mutation anchor changed: '+name)
  with tempfile.TemporaryDirectory(prefix='q36-vision-negative-') as temp:
   root=Path(temp);header=root/HEADER;header.parent.mkdir(parents=True);header.write_text(original.replace(old,new));exe=root/'test'
   subprocess.run(['g++','-std=c++20','-O0','-UNDEBUG','-I'+str(root),'-I'+str(ROOT/'include'),str(ROOT/'tests/core/q36_vision_preallocation_test.cpp'),'-o',str(exe)],check=True,capture_output=True)
   run=subprocess.run([str(exe),str(root)],capture_output=True)
   if run.returncode==0:raise RuntimeError('MUTATION ESCAPED: '+name)
   print('VISION_NEGATIVE_CONTROL PASS',name,'exit='+str(run.returncode))
if __name__=='__main__':main()
