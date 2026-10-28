#!/usr/bin/env python3
"""CPU deliberate mutations must FAIL; overlays only in session temp, no GPU."""
from pathlib import Path
import resource,subprocess,tempfile
ROOT=Path(__file__).resolve().parents[1]
MUTATIONS=[
 ('trust_keys_not_bytes','header','!std::memcmp(a.rows.data()+at,b.rows.data()+bt,(size_t)n*4)','true','core'),
 ('omit_grid','header',' || x.nx!=y.nx || x.ny!=y.ny','','core'),
 ('omit_placement','header',' || x.cell!=y.cell','','core'),
 ('permit_incomplete_image','header','if(x.cell<end && end<x.cell+x.n) return false;','if(false && x.cell<end && end<x.cell+x.n) return false;','core'),
 ('ignore_point_readiness','header','return ready && end==pos','return end==pos','core'),
 ('ignore_point_delta','header',' && delta==q36_image_source(source).delta(pos)','','core'),
 ('drop_early_legal_boundary','header','it==safe.begin()?safe.front():*--it','it==safe.begin()?uint64_t{0}:*--it','core'),
 ('capture_unseeded_MTP','helper',' || ss.image_ready_end!=(uint64_t)pos','','helper'),
]
def main():
 resource.setrlimit(resource.RLIMIT_CORE,(0,0))
 for name,kind,old,new,test in MUTATIONS:
  path=ROOT/('include/strata/core/q36_image_checkpoint.hpp' if kind=='header' else 'src/qwen35/q36_image_checkpoint.inc');source=path.read_text()
  if source.count(old)!=1:raise RuntimeError('mutation anchor changed: '+name)
  with tempfile.TemporaryDirectory(prefix='image-checkpoint-negative-') as temp:
   root=Path(temp);target=root/('strata/core/q36_image_checkpoint.hpp' if kind=='header' else 'q36_image_checkpoint.inc');target.parent.mkdir(parents=True,exist_ok=True);target.write_text(source.replace(old,new))
   code='q36_image_checkpoint_test.cpp' if test=='core' else 'q36_image_checkpoint_helper_test.cpp';exe=root/'test'
   subprocess.run(['g++','-std=c++20','-O0','-UNDEBUG','-pthread','-I'+str(root),'-I'+str(ROOT/'include'),'-I'+str(ROOT/'src/qwen35'),str(ROOT/'tests/core'/code),'-o',str(exe)],check=True,capture_output=True)
   run=subprocess.run([str(exe)],capture_output=True)
   if not run.returncode:raise RuntimeError('MUTATION ESCAPED: '+name)
   print('IMAGE_CHECKPOINT_NEGATIVE PASS',name,'exit='+str(run.returncode))
if __name__=='__main__':main()
