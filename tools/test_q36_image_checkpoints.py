"""CPU/mock tests only. No actual encoder/model/service/CUDA operation."""
import copy,json,os,re,struct,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1]
from tools.check_q36_integration_preservation import body
from tools import q36_image_checkpoint_fixture as fixture
from tools import q36_image_checkpoint_gate as gate
BASE='765068db267f8cd8f846f8ef45c9cc94f90075d5'

def base(path):return subprocess.check_output(['git','show',BASE+':'+path],cwd=ROOT)

def test_numeric_sampler_bank_staging_and_protected_files_unchanged():
 s=(ROOT/'src/qwen35/q36_engine.cu').read_text();old=base('src/qwen35/q36_engine.cu').decode()
 for marker in ('int64_t process(','void f32mm(','void layer(','void forward(','void forward_mtp(','void run_rows(','void mtp_rows(','void preallocate_kv(','void checkpoint_transfer(','int spec_window('):assert body(s,marker)==body(old,marker),marker
 for m in re.finditer(r'(?:__global__|__device__)[^\n{]*\([^;]*?\)\s*\{',old):assert body(s,m.group(0))==body(old,m.group(0))
 for path in ('include/strata/core/q36_prefix_checkpoint.hpp','include/strata/core/unified_kv_capacity.hpp','include/strata/core/unified_kv_bank.hpp','include/strata/core/unified_kv_protocol.hpp','include/strata/core/unified_kv_vmm.hpp','include/strata/core/q36_sampled_mtp.hpp','src/qwen35/q36_vision_staging.inc','serve/server.py','serve/frontend.py','tools/vision/strata_vision.cpp','tools/q36_adaptive_integration_fixture.py'):
  assert (ROOT/path).read_bytes()==base(path),path
 for path in ('tools/test_q36_integration_ordinary.py','docs/Q36_ADAPTIVE_ORIN_VALIDATION.json'):
  original=subprocess.run(['git','show',BASE+':'+path],capture_output=True,cwd=ROOT)
  if original.returncode==0:assert (ROOT/path).read_bytes()==original.stdout
  else:assert not (ROOT/path).exists() # local base lacks Orin-only f934 files: NEVER invent/overwrite
 marker='const bool sampled_spec ='
 assert s[s.index(marker):s.index(';',s.index(marker))]==old[old.index(marker):old.index(';',old.index(marker))]
 h=(ROOT/'include/strata/core/q36_vision_preallocation.hpp').read_text()
 h=h.replace('uint64_t hash=0;int32_t nx=0,ny=0;uint64_t rotary=0,row=0;','uint64_t hash=0;').replace('hash,r.nx,r.ny,position,row});','hash});')
 assert h==base('include/strata/core/q36_vision_preallocation.hpp').decode() # only returned CPU descriptors expanded
 assert 'bool image_checkpoints=false;' in s and 'const bool image_checkpoint_mode=geni && E.image_checkpoints;' in s
 assert 'if(!geni && fresh>1 && (!reuse || restored)) E.checkpoint_capture();' in s
 assert s.index('image_full_source>=0 && !cp_pin')<s.index('const auto before = E.kv_pool->session(s);')


def test_actual_new_engine_methods_and_native_oracle_host_syntax(tmp_path):
 s=(ROOT/'src/qwen35/q36_engine.cu').read_text();inc=tmp_path/'methods.inc';inc.write_text(s[s.index('    void checkpoint_transfer('):s.index('    /// Session dst takes')])
 run=subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror','-pedantic-errors','-fsyntax-only','-I'+str(ROOT/'include'),'-I'+str(ROOT/'src/qwen35'),'-DIMAGE_ENGINE_INC="'+str(inc)+'"',str(ROOT/'tests/core/q36_image_checkpoint_syntax.cpp')],capture_output=True)
 assert run.returncode==0,run.stderr.decode()


def test_independent_oracle_uses_original_files_and_empty_state_not_restore():
 s=(ROOT/'src/qwen35/q36_image_checkpoint_regression.inc').read_text()
 assert 'e.select(3);e.reset();e.clear_images();' in s
 assert 'build_images(call.image_file,call.image_ids' in s and 'call.image_keys,fresh' not in s
 assert 'call.ids.size()' in s and 'call.ends[boundary++]' in s and 'call.checkpoint_suffix' in s
 assert 'checkpoint_restore(' not in s and 'copy_session(' not in s and 'checkpoint_transfer(' not in s
 assert 'schedule_compare(restored,image_schedule_state' in s and 'schedule_logits(logits,reference' in s
 assert 'fresh.mrope[at]' in s and 'device_rotary_delta' in s and 'complete_image_metadata' in s
 assert 'different_schedule' not in s and 'image_cancel_before' in s


class Words:
 control_tokens=['<|im_start|>','<|im_end|>','<|vision_start|>','<|image_pad|>','<|vision_end|>']
 def __init__(self):self.vocab={x:248050+i for i,x in enumerate(self.control_tokens)};self.vocab['<|image_pad|>']=248056;self.next=1000
 def encode(self,text,parse_special=False,**kwargs):
  parts=re.findall(r'<\|[^>]+\|>|[^\s<>]+',text);out=[]
  for part in parts:
   if part not in self.vocab:self.vocab[part]=self.next;self.next+=1
   out.append(self.vocab[part])
  return out
TEMPLATE='''{% for m in messages %}<|im_start|> {{m.role}} {% if m.content is string %}{{m.content}}{% else %}{% for p in m.content %}{% if p.type == 'image' %}<|vision_start|> <|image_pad|> <|vision_end|>{% else %} {{p.text}} {% endif %}{% endfor %}{% endif %}<|im_end|> {% endfor %}{% if add_generation_prompt %}<|im_start|> assistant {% endif %}'''

def test_fixture_pipeline_with_fake_vocab_encoder_is_NOT_real_GGUF_evidence(tmp_path,monkeypatch):
 from tools import strata_tokenizer,gguf_reader,q36_multimodal_cache_fixture as diagnostic
 tk=Words();monkeypatch.setattr(strata_tokenizer.Tokenizer,'from_gguf',staticmethod(lambda _:tk));monkeypatch.setattr(gguf_reader,'GGUFFile',lambda _:SimpleNamespace(metadata={'tokenizer.chat_template':TEMPLATE}))
 model=tmp_path/'model';model.write_bytes(b'CPU MOCK NOT A MODEL')
 diagnostic.make_images(tmp_path/'images');a=tmp_path/'a.sve';b=tmp_path/'b.sve'
 for p,value in ((a,0.25),(b,0.5)):p.write_bytes(struct.pack('<5i',0x31455653,1024,32,32,2048)+struct.pack('<f',value)*(1024*2048))
 image_a=tmp_path/'images/red-square.png';image_b=tmp_path/'images/blue-square.png'
 diagnostic.prepare(model,image_a,image_b,a,b,tmp_path/'diagnostic')
 m=fixture.prepare(model,image_a,image_b,a,b,tmp_path/'diagnostic',tmp_path/'new')
 assert len(m['rows'])==17 and m['rows'][2]['phase']=='steady' and m['rows'][2]['audit']['classification']=='token_rewrite'
 assert m['rows'][2]['expect']=='after' and m['actual_OWUI']['savings']=='UNMEASURED'
 gate.validate(tmp_path/'new',m)
 broken=copy.deepcopy(m);broken['rows'][2]['ids'][0]+=1
 with pytest.raises(ValueError):gate.validate(tmp_path/'new',broken)
 # This uses actual Service.prepare but FAKE tokenizer/encoder IO. Not an Orin qualification.


def test_measurements_require_independent_full_gate_coverage():
 text='\n'.join('IMAGE_CHECKPOINT_MEASURE case=synthetic turn=%d phase=steady reused=2048 replayed=100 checkpoint=2048 prefill_ms=1 copy_us=2 copy_bytes=4 independent=1 exact=1'%i for i in range(8))
 assert len(gate.measures(text))==8
 with pytest.raises(ValueError):gate.measures(text.replace('exact=1','exact=0'))
 with pytest.raises(ValueError):gate.measures(text.splitlines()[0])


def test_worker_fake_python_children_fail_fast_cleanup_and_wrong_reason(tmp_path,monkeypatch):
 monkeypatch.setattr(gate,'no_models',lambda:None)
 good='print("IMAGE_CHECKPOINT_TEST PASS");print("KV_PREALLOCATION_MODEL PASS")'
 assert 'PASS' in gate.run([sys.executable,'-c',good],tmp_path,'mock-good',time.monotonic()+5)
 negative='print("IMAGE_CHECKPOINT_CORRUPT category=conv");print("IMAGE_CHECKPOINT_FAILURE comparison=restored");raise SystemExit(1)'
 gate.run([sys.executable,'-c',negative],tmp_path,'mock-negative',time.monotonic()+5,True)
 with pytest.raises(RuntimeError,match='independent RESTORED'):
  gate.run([sys.executable,'-c','raise SystemExit(1)'],tmp_path,'mock-wrong-reason',time.monotonic()+5,True)
 with pytest.raises(TimeoutError):gate.run([sys.executable,'-c','import time;time.sleep(60)'],tmp_path,'mock-timeout',time.monotonic()+0.1)
 result=json.loads((tmp_path/'mock-timeout.exit.json').read_text());assert result['exit']!=0 and result['error']
 # These children are tiny Python fakes, NEVER an encoder/engine or service.


def test_gate_has_no_service_operations_or_live_worktree_builds():
 s=(ROOT/'tools/q36_image_checkpoint_gate.py').read_text()
 assert 'systemctl' not in s and 'systemd-run' not in s and 'pgrep' not in s
 assert 'REVIEW=Path(\'/mnt/ssd/Strata-Orin/worktrees/multimodal-cache\')' in s
 assert "'--kv-vision-max-tokens','1024'" in s and '--kv-image-checkpoint-corrupt' in s
 assert 'os.waitid' in s and 'os.WNOWAIT' in s and 'os.killpg(child.pid' in s
 assert 'deadline=time.monotonic()+a.budget_seconds' in s
