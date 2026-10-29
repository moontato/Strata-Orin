#!/usr/bin/env python3
"""Read-only source invariants; NOT CUDA compilation/SASS/model qualification."""
from pathlib import Path
import hashlib,os,re,shutil,subprocess,sys
ROOT=Path(os.environ.get('STRATA_IMAGE_CHECKPOINT_REVIEW_ROOT',Path(__file__).resolve().parents[1])).resolve();sys.path.insert(0,str(ROOT))
from tools.check_q36_integration_preservation import body
BASE='765068db267f8cd8f846f8ef45c9cc94f90075d5'
def original(path,ref=BASE):return subprocess.check_output(['git','show',ref+':'+path],cwd=ROOT)
def main():
 old=original('src/qwen35/q36_engine.cu').decode();new=(ROOT/'src/qwen35/q36_engine.cu').read_text();count=0
 for match in re.finditer(r'(?:__global__|__device__)[^\n{]*\([^;]*?\)\s*\{',old):
  assert body(new,match[0])==body(old,match[0]);count+=1
 print('PASS',count,'CUDA global/device definitions byte-identical (SOURCE ONLY)')
 for marker in ['void f32mm(','void mm(','void layer(','void forward(','void forward_mtp(','int64_t process(','void run_rows(','void commit_rows(','void mtp_rows(','int spec_window(','void spec_commit(','int spec_step(','void checkpoint_transfer(','void preallocate_kv(','void set_images(','void clear_images(','void chunk_pictures(']:
  assert body(new,marker)==body(old,marker),marker;print('PASS numerical/lifetime method identical',marker)
 for path in ['include/strata/core/unified_kv_bank.hpp','include/strata/core/unified_kv_capacity.hpp','include/strata/core/unified_kv_protocol.hpp','include/strata/core/q36_prefix_checkpoint.hpp','include/strata/core/q36_sampled_mtp.hpp','src/qwen35/q36_kernels.cuh','src/qwen35/q36_kv_int8.cuh','src/qwen35/q36_vision_staging.inc','serve/server.py','serve/frontend.py','serve/runconfig.py','src/prefill/gemm.cu','tools/vision/strata_vision.cpp']:
  content=(ROOT/path).read_bytes();assert content==original(path),path;print('PASS unchanged',path,hashlib.sha256(content).hexdigest())
 print('Backend compilers:',{x:shutil.which(x) for x in ['nvcc','hipcc','icpx']})
 print('PASS source preservation only; NOT CUDA/SASS/default-output or native image-oracle execution')
if __name__=='__main__':main()
