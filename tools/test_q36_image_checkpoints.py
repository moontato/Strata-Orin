"""CPU/mock tests only. No actual encoder/model/service/CUDA operation."""
import copy,json,os,re,shutil,struct,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1]
import itertools
counter=itertools.count()
from tools.check_q36_integration_preservation import body
from tools import q36_image_checkpoint_fixture as fixture
from tools import q36_image_checkpoint_gate as gate
from tools import q36_restore_observation_harness as observation
BASE='765068db267f8cd8f846f8ef45c9cc94f90075d5'

def base(path):return subprocess.check_output(['git','show',BASE+':'+path],cwd=ROOT)

def staging(tmp_path):
    """Throwaway selftest staging: fake binary + cache provenance, no model involved."""
    stage=tmp_path/'stage';(stage/'build').mkdir(parents=True,exist_ok=True);(stage/'fixtures').mkdir(exist_ok=True)
    engine=stage/'build'/'strata-q36';engine.write_text('# fake binary marker --kv-image-checkpoint-test\n')
    (stage/'build'/'CMakeCache.txt').write_text('CMAKE_HOME_DIRECTORY:INTERNAL='+str(ROOT)+'\n')
    model=stage/'model.gguf';model.write_bytes(b'not a model')
    (stage/'fixtures'/'manifest.json').write_text(json.dumps({'kind':'q36-image-checkpoints-v1',
        'model':{'path':str(model),'size':model.stat().st_size,'mtime_ns':model.stat().st_mtime_ns}})+'\n')
    return stage,engine,model,stage/'fixtures'

def run_gate(tmp_path,scenario,budget=2,cleanup=2):
    stage,engine,model,fixtures=staging(tmp_path)
    out=stage/('run-'+scenario+'-'+str(os.getpid())+'-'+str(next(counter)))
    env=dict(os.environ,STRATA_IMAGE_CHECKPOINT_REVIEW_ROOT=str(ROOT),PYTHONPATH=str(ROOT))
    env.pop('STRATA_IMAGE_CHECKPOINT_REVIEW_WINDOW',None)
    run=subprocess.run([sys.executable,str(ROOT/'tools/q36_image_checkpoint_gate.py'),'--engine',str(engine),
        '--model',str(model),'--fixtures',str(fixtures),'--out',str(out),'--selftest',scenario,
        '--budget-seconds',str(budget),'--cleanup-seconds',str(cleanup)],capture_output=True,text=True,env=env,timeout=120)
    return run,json.loads((out/'watchdog.json').read_text()),out

def owned_probes():
    found=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) in {os.getpid(),os.getppid()}:continue
        try:cmd=Path(entry/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        except OSError:continue
        if gate.OWNED_PROBE in cmd and entry.name!=str(os.getpid()):found.append(int(entry.name))
    return found

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


def test_restored_state_observation_boundary_and_host_harness():
 # The real tree must satisfy the placement contract AND run the real copy_session tail.
 assert observation.placement_errors(ROOT)==[]
 code,stdout,stderr=observation.build(ROOT)
 assert code==0,(stdout or stderr)
 for marker in ('cross_slot_deferred=1','observed_cpu=-992','observed_device=-992','previous_current_device=0',
                'exactly_once=1','noncurrent_refused=1','same_slot_in_place=1','counters_balanced=1','no_hook_inert=1'):
  assert marker in stdout,marker
 assert 'IMAGE_CHECKPOINT_OBSERVATIONS total=' in (ROOT/'src/qwen35/q36_image_checkpoint_regression.inc').read_text()


def test_gate_owns_boundary_counters_and_refuses_noncurrent_observation():
 inc=(ROOT/'src/qwen35/q36_restore_observation.inc').read_text()
 engine=(ROOT/'src/qwen35/q36_engine.cu').read_text();oracle=(ROOT/'src/qwen35/q36_image_checkpoint_regression.inc').read_text()
 assert inc.count('if(cur!=dst)')==2 and '++restore_snapshot_refused' in inc
 assert 'if(schedule_restored) schedule_restored(dst);' not in engine[engine.index('    void copy_session('):]
 assert 'E.discard_restore_snapshot();' in engine
 for marker in ('require(e.cur==dst,','require(e.pos==(int64_t)end && e.sess[(size_t)dst].live.size()==end',
                'image gate restored-state observation count mismatch','for(const auto& o:observations) require(o.verified'):
  assert marker in oracle,marker
 # The text oracle keeps using the in-place hook and never gains copy_session coverage.
 text=(ROOT/'src/qwen35/q36_checkpoint_schedule.inc').read_text()
 assert 'E.schedule_restored=[&](int dst)' in text and 'copy_session(' not in text


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
 assert 'schedule_compare(o.state,image_schedule_state' in s and 'schedule_logits(logits,reference' in s
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
 gate.validate(tmp_path/'new',m,gate.Budget(120))
 broken=copy.deepcopy(m);broken['rows'][2]['ids'][0]+=1
 with pytest.raises(ValueError):gate.validate(tmp_path/'new',broken,gate.Budget(120))
 # This uses actual Service.prepare but FAKE tokenizer/encoder IO. Not an Orin qualification.


def test_measurements_require_independent_full_gate_coverage():
 text='\n'.join('IMAGE_CHECKPOINT_MEASURE case=synthetic turn=%d phase=steady reused=2048 replayed=100 checkpoint=2048 prefill_ms=1 copy_us=2 copy_bytes=4 independent=1 exact=1'%i for i in range(8))
 assert len(gate.measures(text))==8
 with pytest.raises(ValueError):gate.measures(text.replace('exact=1','exact=0'))
 with pytest.raises(ValueError):gate.measures(text.splitlines()[0])


def test_worker_fake_python_children_fail_fast_cleanup_and_wrong_reason(tmp_path,monkeypatch):
 monkeypatch.setattr(gate,'no_models',lambda:None)
 good='print("IMAGE_CHECKPOINT_TEST PASS");print("KV_PREALLOCATION_MODEL PASS")'
 assert 'PASS' in gate.run([sys.executable,'-c',good],tmp_path,'mock-good',gate.Budget(5))
 negative='print("IMAGE_CHECKPOINT_CORRUPT category=conv");print("IMAGE_CHECKPOINT_FAILURE comparison=restored");raise SystemExit(1)'
 gate.run([sys.executable,'-c',negative],tmp_path,'mock-negative',gate.Budget(5),True)
 with pytest.raises(RuntimeError,match='independent RESTORED'):
  gate.run([sys.executable,'-c','raise SystemExit(1)'],tmp_path,'mock-wrong-reason',gate.Budget(5),True)
 with pytest.raises(TimeoutError):gate.run([sys.executable,'-c','import time;time.sleep(60)'],tmp_path,'mock-timeout',gate.Budget(1))
 result=json.loads((tmp_path/'mock-timeout.exit.json').read_text());assert result['exit']!=0 and result['error']
 # These children are tiny Python fakes, NEVER an encoder/engine or service.


def test_budget_check_covers_blocking_phases_and_reads(tmp_path,monkeypatch):
 # A slow log read past the deadline is a failure, not a late acceptance.
 monkeypatch.setattr(gate,'no_models',lambda:None)
 original=Path.read_text
 def slow(path,*args,**kwargs):
  if str(path).endswith('slow.log'):time.sleep(1.2)
  return original(path,*args,**kwargs)
 monkeypatch.setattr(Path,'read_text',slow)
 with pytest.raises(TimeoutError):
  gate.run([sys.executable,'-c','print("IMAGE_CHECKPOINT_TEST PASS");print("KV_PREALLOCATION_MODEL PASS")'],tmp_path,'slow',gate.Budget(1))
 # Every phase name passes a budget check before starting.
 for phase in ('binary hash','preflight','model window start','measurement parse','verdict'):
  with pytest.raises(TimeoutError):gate.Budget(0).check(phase)


def test_supervisor_verdict_never_accepts_a_late_or_killed_success():
 phases=dict(arms_completed=True)
 clean=dict(killed=False,late=False,exit_code=0,group_empty=True,report_status='PASS_BOUNDED_INDEPENDENT_IMAGE_CHECKPOINT_GATE',phases=phases)
 assert gate.verdict(**clean)[0] is True
 for field,value in (('late',True),('killed',True),('exit_code',-9),('group_empty',False),
                     ('report_status','MISSING'),('report_status','FAIL')):
  case=dict(clean);case[field]=value
  passed,reasons=gate.verdict(**case)
  assert passed is False and reasons,(field,value)
 assert gate.verdict(**{**clean,'phases':{'arms_completed':False}})[0] is False


def test_supervisor_bounds_every_scenario(tmp_path):
 # Normal completion inside the budget: exit0, group empty, labelled PASS only.
 run,watch,out=run_gate(tmp_path,'normal',budget=6)
 assert run.returncode==0 and watch['passed'] and watch['owned_group_empty'] and not watch['killed']
 assert 'IMAGE_CHECKPOINT_GATE_SELFTEST verdict=PASS' in run.stdout and 'IMAGE_CHECKPOINT_ORIN_GATE PASS' not in run.stdout
 assert (out/'watchdog.json').is_file() and (out/'result.json').is_file()
 # Blocking preflight, hung native child + descendant, blocking postprocessing and a
 # completion past the deadline are all killed and nonzero, never accepted.
 for scenario in ('slow_preflight','hung_child','slow_postprocess','late_success'):
  run,watch,_=run_gate(tmp_path,scenario,budget=2,cleanup=2)
  assert run.returncode!=0,scenario
  assert watch['killed'] and watch['deadline_crossed'] and watch['owned_group_empty'],(scenario,watch)
  assert 'IMAGE_CHECKPOINT_ORIN_GATE PASS' not in run.stdout and 'verdict=PASS' not in run.stdout,scenario
  assert 'IMAGE_CHECKPOINT_GATE_SELFTEST FAIL' in run.stdout,scenario
 assert 'IMAGE_CHECKPOINT_GATE_SELFTEST_SUPERVISOR PASS' in run_gate(tmp_path,'normal',budget=6)[0].stdout


def test_supervisor_reaps_only_its_own_group(tmp_path):
 unowned=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'])   # started OUTSIDE the gate group
 try:
  run,watch,_=run_gate(tmp_path,'hung_child',budget=2,cleanup=2)
  assert run.returncode!=0 and watch['killed'] and watch['owned_group_empty']
  assert not owned_probes()                                        # gate's child AND its descendant are gone
  assert unowned.poll() is None                                   # a foreign process was never touched
 finally:
  unowned.kill();unowned.wait()
 assert owned_probes()==[]


def test_gate_has_no_service_operations_and_separates_cleanup_from_budget():
 s=(ROOT/'tools/q36_image_checkpoint_gate.py').read_text()
 assert 'systemctl' not in s and 'systemd-run' not in s and 'pgrep' not in s
 assert "STAGE=Path('/mnt/ssd/tmp/strata-checkpoints')" in s
 assert "REVIEW=Path('/mnt/ssd/Strata-Orin/worktrees/multimodal-cache')" in s
 assert "'--kv-vision-max-tokens','1024'" in s and '--kv-image-checkpoint-corrupt' in s
 assert 'start_new_session=True' in s and 'os.killpg(child.pid,signal.SIGKILL)' in s
 assert 'EXECUTION_BUDGET_MAX=420' in s and 'CLEANUP_ALLOWANCE_MAX=60' in s
 assert 'cleanup_allowance_seconds' in s and 'owned_group_empty' in s
 assert 'def verdict(' in s and "'--deadline'" in s
 assert 'budget.check(' in s and s.count('budget.check(')>=10
