"""CPU/mock tests only. No actual encoder/model/service/CUDA operation."""
import copy,json,os,re,shutil,signal,struct,subprocess,sys,time
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

def run_gate(tmp_path,scenario,budget=2,cleanup=2,preflight=False):
    stage,engine,model,fixtures=staging(tmp_path)
    out=stage/('run-'+scenario+'-'+str(os.getpid())+'-'+str(next(counter)))
    env=dict(os.environ,STRATA_IMAGE_CHECKPOINT_REVIEW_ROOT=str(ROOT),PYTHONPATH=str(ROOT))
    env.pop('STRATA_IMAGE_CHECKPOINT_REVIEW_WINDOW',None)
    run=subprocess.run([sys.executable,str(ROOT/'tools/q36_image_checkpoint_gate.py'),'--engine',str(engine),
        '--model',str(model),'--fixtures',str(fixtures),'--out',str(out),'--selftest',scenario,
        '--budget-seconds',str(budget),'--cleanup-seconds',str(cleanup)]
        +(['--preflight'] if preflight else []),capture_output=True,text=True,env=env,timeout=120)
    return run,json.loads((out/'watchdog.json').read_text()),out

def start_gate(tmp_path,scenario,budget,cleanup):
    """Launch the supervisor as a real process so a signal can be delivered to IT."""
    stage,engine,model,fixtures=staging(tmp_path)
    out=stage/('sig-'+scenario+'-'+str(os.getpid())+'-'+str(next(counter)))
    env=dict(os.environ,STRATA_IMAGE_CHECKPOINT_REVIEW_ROOT=str(ROOT),PYTHONPATH=str(ROOT))
    env.pop('STRATA_IMAGE_CHECKPOINT_REVIEW_WINDOW',None)
    proc=subprocess.Popen([sys.executable,str(ROOT/'tools/q36_image_checkpoint_gate.py'),'--engine',str(engine),
        '--model',str(model),'--fixtures',str(fixtures),'--out',str(out),'--selftest',scenario,
        '--budget-seconds',str(budget),'--cleanup-seconds',str(cleanup)],stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,text=True,env=env)
    return proc,out

def worker_pids(supervisor_pid):
    """Read-only observation of the supervisor's OWN worker (never used for killing)."""
    found=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            cmd=Path(entry/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            ppid=Path(entry/'stat').read_text().split(')')[-1].split()[1]   # pid (comm) STATE ppid ...
        except (OSError,IndexError):continue
        if '--worker' in cmd and '--selftest' in cmd and ppid==str(supervisor_pid):found.append(int(entry.name))
    return found

def cleanup_gate(proc):
    """Test hygiene ONLY: end the supervisor and any worker group it left behind, so a
    failing assertion can never orphan a marked fake process (the gate itself is never
    relied on for this cleanup path)."""
    for pid in worker_pids(proc.pid):
        try:os.killpg(pid,signal.SIGKILL)
        except (ProcessLookupError,PermissionError):pass
    if proc.poll() is None:
        try:proc.kill()
        except OSError:pass
    try:proc.communicate(timeout=10)
    except Exception:pass

def owned_probes():
    found=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) in {os.getpid(),os.getppid()}:continue
        try:cmd=Path(entry/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        except OSError:continue
        if gate.OWNED_PROBE in cmd and cmd.startswith(sys.executable) and entry.name!=str(os.getpid()):found.append(int(entry.name))
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
 # The accepted report status depends on the mode: a preflight-only report is never a
 # numerical gate pass, and a numerical gate report is never accepted for preflight.
 assert gate.verdict(**{**clean,'report_status':'PASS_PREFLIGHT_ONLY','preflight':True})[0] is True
 assert gate.verdict(**{**clean,'report_status':'PASS_PREFLIGHT_ONLY'})[0] is False
 assert gate.verdict(**{**clean,'preflight':True})[0] is False


def test_preflight_runs_under_the_supervisor_and_stays_cpu_only(tmp_path):
    # A real (non-selftest) preflight is Orin-only by design: ROOT must equal the
    # reviewed multimodal-cache worktree and results must live under the Orin staging
    # path. Here the supervisor's preflight VERDICT path is exercised with the labelled
    # selftest staging, which never starts an engine, model or encoder.
    run,watch,_=run_gate(tmp_path,'normal',preflight=True)
    assert run.returncode==0,run.stderr[-400:]
    assert 'IMAGE_CHECKPOINT_PREFLIGHT PASS model_not_started=1' in run.stdout
    assert 'SUPERVISOR PASS' in run.stdout
    # Preflight is labelled CPU-only and can never look like a numerical gate pass.
    assert 'IMAGE_CHECKPOINT_ORIN_GATE PASS' not in run.stdout
    assert watch['passed'] is True and watch['worker_report_status']=='PASS_PREFLIGHT_ONLY'


def test_real_preflight_is_refused_outside_the_reviewed_orin_checkout(tmp_path):
    # ROOT here is the coding checkout, not the reviewed multimodal-cache worktree, so
    # an unsupervised/real preflight must be refused instead of quietly running.
    stage,engine,model,fixtures=staging(tmp_path)
    run=subprocess.run([sys.executable,str(ROOT/'tools/q36_image_checkpoint_gate.py'),'--engine',str(engine),
        '--model',str(model),'--fixtures',str(fixtures),'--out',str(tmp_path/'real-out'),'--preflight'],
        capture_output=True,text=True,env=dict(os.environ,STRATA_IMAGE_CHECKPOINT_REVIEW_ROOT=str(ROOT),
        PYTHONPATH=str(ROOT)),timeout=60)
    assert run.returncode!=0 and 'multimodal-cache' in (run.stderr+run.stdout)


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
 # Fault injection (--fault) is selftest-only and can never appear in a real gate run.
 assert 'fault injection requires --selftest' in (ROOT/'tools/q36_image_checkpoint_gate.py').read_text()
 stage,engine,model,fixtures=staging(tmp_path)
 run=subprocess.run([sys.executable,str(ROOT/'tools/q36_image_checkpoint_gate.py'),'--engine',str(engine),
     '--model',str(model),'--fixtures',str(fixtures),'--out',str(stage/'faultrefused'),
     '--preflight','--fault','record'],capture_output=True,text=True,
     env=dict(os.environ,STRATA_IMAGE_CHECKPOINT_REVIEW_ROOT=str(ROOT)))
 assert run.returncode==2 and 'fault injection requires --selftest' in run.stderr,run.stderr[-200:]


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


class StuckChild:
    """Fake worker used to simulate BLOCKED reaping/group cleanup. Its pid is not a
    usable process group and every OS hook below is injected, so nothing real is ever
    signalled by these cases."""
    pid=-1
    def wait(self,timeout=None):raise subprocess.TimeoutExpired(cmd='fake-worker',timeout=timeout)


def test_one_cleanup_allowance_covers_waiting_reaping_and_verification():
 allowance=1.0
 # (a) the worker never exits AND the owned group never empties: waiting, group killing
 # and the emptiness proof must all finish inside ONE allowance. The previous code
 # waited for the allowance and THEN re-armed it for cleanup (about twice the bound).
 started=time.monotonic()
 stuck=gate.CleanupState(StuckChild(),allowance)
 exit_code,status,group_empty,reaped=gate.teardown(stuck,
     waiter=lambda:None,killer=lambda:'present',prober=lambda:'present')
 elapsed=time.monotonic()-started
 assert allowance<=elapsed<allowance*1.6,(elapsed,)
 assert exit_code is None and status=='present' and group_empty is False and reaped is False
 # (b) the worker exits promptly but the group never empties: the SAME absolute deadline
 # still bounds the whole teardown, so verification cannot start a fresh allowance.
 class Exited(StuckChild):
     def wait(self,timeout=None):return -9
 started=time.monotonic()
 exit_code,status,group_empty,reaped=gate.teardown(gate.CleanupState(Exited(),allowance),
     waiter=lambda:0,killer=lambda:'present',prober=lambda:'present')
 elapsed=time.monotonic()-started
 assert elapsed<allowance*1.6,(elapsed,)
 # The status already seen without consuming it (waitid) wins, and the leader is reaped.
 assert exit_code==0 and reaped is True and group_empty is False
 # An incomplete cleanup or an interrupted supervisor is never accepted, whatever the
 # worker reported.
 for group,reap in ((False,False),(False,True),(True,False)):
  assert gate.verdict(True,False,-9,group,'PASS_BOUNDED_INDEPENDENT_IMAGE_CHECKPOINT_GATE',
     {'arms_completed':True},False,'',reap)[0] is False
 assert gate.verdict(True,False,0,True,'PASS_BOUNDED_INDEPENDENT_IMAGE_CHECKPOINT_GATE',
     {'arms_completed':True},False,'SIGTERM',True)[0] is False


def test_teardown_is_idempotent_and_never_signals_a_reaped_group():
 """The old error path re-ran teardown with a FRESH deadline after cleanup had already
 reaped the leader, which signalled a PGID we could no longer prove we owned."""
 class Finished:
     pid=-1
     def wait(self,timeout=None):return 0
 state=gate.CleanupState(Finished(),1.0)
 signals=[];probes=[]
 first=gate.teardown(state,waiter=lambda:0,
     killer=lambda:(signals.append('first'),'present')[1],
     prober=lambda:(probes.append('first'),'empty')[1])
 assert first==(0,'present',True,True) and signals==['first','first']
 kills=len(signals);deadline=state.deadline
 started=time.monotonic()
 second=gate.teardown(state,waiter=lambda:0,
     killer=lambda:(signals.append('after_reap'),'present')[1],
     prober=lambda:(probes.append('after_reap'),'empty')[1])
 elapsed=time.monotonic()-started
 assert second==first                                   # the recorded result is unchanged
 assert signals==['first','first'] and 'after_reap' not in signals   # NO signal after reaping
 assert probes==['first'] and state.deadline==deadline  # no probing storm, no new deadline
 assert elapsed<0.2,(elapsed,)                          # and no renewed cleanup allowance
 # A cleanup that FAILED to reap keeps its recorded failure and still gets no new time.
 stuck=gate.CleanupState(StuckChild(),0.2)
 stuck_signals=[]
 recorded=gate.teardown(stuck,waiter=lambda:None,killer=lambda:(stuck_signals.append(1),'present')[1],
     prober=lambda:'present')
 assert recorded[2] is False and recorded[3] is False
 before=len(stuck_signals)
 started=time.monotonic()
 again=gate.teardown(stuck,killer=lambda:(stuck_signals.append(2),'present')[1],prober=lambda:'present')
 assert again==recorded and time.monotonic()-started<0.2 and len(stuck_signals)==before
 # The ownership guard alone also holds: a state whose leader is already reaped gets
 # probing only, never a signal, even when cleanup had not finished.
 pre=gate.CleanupState(Finished(),0.5);pre.reaped=True
 kills=[];probes=[]
 gate.teardown(pre,waiter=lambda:0,killer=lambda:(kills.append(1),'present')[1],
     prober=lambda:(probes.append(1),'empty')[1])
 assert kills==[] and probes==[1]              # probed, never signalled
 assert not gate.verdict(True,False,-1,False,'PASS_BOUNDED_INDEPENDENT_IMAGE_CHECKPOINT_GATE',
     {'arms_completed':True},False,'',recorded[3])[0]


def run_gate_fault(tmp_path,scenario,fault,budget=6,cleanup=2):
 """Run the supervisor with a selftest-only fault injected (hidden --fault option);
 returns the run, the results directory and the wall time."""
 stage,engine,model,fixtures=staging(tmp_path)
 out=stage/('fault-'+scenario+'-'+fault+'-'+str(os.getpid())+'-'+str(next(counter)))
 env=dict(os.environ,STRATA_IMAGE_CHECKPOINT_REVIEW_ROOT=str(ROOT),PYTHONPATH=str(ROOT))
 env.pop('STRATA_IMAGE_CHECKPOINT_REVIEW_WINDOW',None)
 started=time.monotonic()
 run=subprocess.run([sys.executable,str(ROOT/'tools/q36_image_checkpoint_gate.py'),
     '--engine',str(engine),'--model',str(model),'--fixtures',str(fixtures),'--out',str(out),
     '--selftest',scenario,'--budget-seconds',str(budget),'--cleanup-seconds',str(cleanup),
     '--fault',fault],capture_output=True,text=True,env=env,timeout=120)
 return run,out,time.monotonic()-started


def test_record_failure_after_completed_cleanup_never_passes_or_restarts(tmp_path):
 """A watchdog.json write that fails after cleanup finished (ENOSPC-class) used to fall
 into the supervisor fault path and re-run teardown with a fresh deadline, signalling a
 group whose leader had already been reaped."""
 run,out,elapsed=run_gate_fault(tmp_path,'normal','record',budget=6,cleanup=2)
 assert run.returncode!=0
 assert 'IMAGE_CHECKPOINT_GATE_SELFTEST FAIL result_record_failed_' in run.stdout,run.stdout
 assert 'stage=record action=failed error=' in run.stderr and 'action=no_restart cleanup_state_final=1' in run.stderr
 assert not (out/'watchdog.json').exists()               # the record really could not be written
 assert (out/'record-fault').is_file()                   # the fault was injected, not a silent pass
 assert json.loads((out/'result.json').read_text())['status']=='PASS_BOUNDED_INDEPENDENT_IMAGE_CHECKPOINT_GATE'
 # A worker PASS can never become a supervisor PASS when the result cannot be recorded.
 assert 'verdict=PASS' not in run.stdout and 'SUPERVISOR PASS' not in run.stdout
 assert 'IMAGE_CHECKPOINT_ORIN_GATE PASS' not in run.stdout
 assert elapsed<3.0,(elapsed,)                           # cleanup was not run a second time
 assert owned_probes()==[]


def test_exception_after_leader_reaping_records_failure_without_new_cleanup(tmp_path):
 run,out,elapsed=run_gate_fault(tmp_path,'normal','crash',budget=6,cleanup=2)
 assert run.returncode!=0
 watch=json.loads((out/'watchdog.json').read_text())     # the safety-net record still exists
 assert watch['passed'] is False and watch['reasons'][0].startswith('supervisor error RuntimeError')
 assert watch['worker_reaped'] is True and watch['owned_group_empty'] is True
 assert watch['cleanup_complete'] is True
 assert 'IMAGE_CHECKPOINT_GATE_SELFTEST FAIL supervisor_error_RuntimeError' in run.stdout
 assert 'RuntimeError: fault injection: post-teardown supervisor failure' in run.stderr
 assert 'verdict=PASS' not in run.stdout and 'IMAGE_CHECKPOINT_ORIN_GATE PASS' not in run.stdout
 assert elapsed<3.0,(elapsed,)                           # the shared state added no new allowance
 assert owned_probes()==[]


@pytest.mark.parametrize('signame',('SIGINT','SIGTERM','SIGHUP'))
def test_signal_to_the_supervisor_tears_down_and_records_failure(tmp_path,signame):
 # The worker is blocked in a synchronous phase, so ONLY the supervisor's teardown can
 # end the run. SIGTERM/SIGHUP used to leave the worker group running with no record at
 # all, and SIGINT's exception path escaped before reaping and before the record.
 proc,out=start_gate(tmp_path,'slow_preflight',budget=20,cleanup=3)
 try:
  waited=0.0
  while not worker_pids(proc.pid) and waited<15:time.sleep(0.05);waited+=0.05
  assert worker_pids(proc.pid),'the worker never started'
  os.kill(proc.pid,getattr(signal,signame))
  stdout,stderr=proc.communicate(timeout=30)
 except BaseException:
  cleanup_gate(proc);raise
 assert proc.returncode!=0,(signame,stdout[-300:],stderr[-300:])
 watch=json.loads((out/'watchdog.json').read_text())      # a failure record ALWAYS exists
 assert watch['passed'] is False and signame in watch['interrupted_signals'],watch
 assert watch['owned_group_empty'] is True and watch['worker_reaped'] is True,watch
 assert watch['cleanup_complete'] is True and watch['operator_cgroup_required'] is True
 assert 'IMAGE_CHECKPOINT_GATE_SELFTEST FAIL' in stdout and 'verdict=PASS' not in stdout
 assert 'IMAGE_CHECKPOINT_ORIN_GATE PASS' not in stdout
 assert worker_pids(proc.pid)==[]                          # no surviving owned worker
 assert 'action=teardown' in stderr and 'cleanup_allowance_s=3' in stderr


def test_interrupt_during_a_hung_child_reaps_the_whole_owned_group(tmp_path):
 unowned=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'])   # foreign process
 try:
  proc,out=start_gate(tmp_path,'hung_child',budget=30,cleanup=3)   # far from expiry
  try:
   waited=0.0
   while (not worker_pids(proc.pid) or not owned_probes()) and waited<15:time.sleep(0.05);waited+=0.05
   assert worker_pids(proc.pid) and owned_probes(),'the hung child and its descendant never started'
   os.kill(proc.pid,signal.SIGTERM)
   stdout,stderr=proc.communicate(timeout=30)
  finally:
   cleanup_gate(proc)
  assert proc.returncode!=0
  watch=json.loads((out/'watchdog.json').read_text())
  assert watch['passed'] is False and 'SIGTERM' in watch['interrupted_signals']
  assert watch['owned_group_empty'] is True and watch['worker_reaped'] is True
  assert 'verdict=PASS' not in stdout and 'IMAGE_CHECKPOINT_ORIN_GATE PASS' not in stdout
  assert owned_probes()==[] and worker_pids(proc.pid)==[]   # worker, child AND descendant
  assert unowned.poll() is None                             # a foreign process untouched
 finally:
  unowned.kill();unowned.wait()


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
 # Teardown structure: one signal path, one cleanup deadline, and the requirement that
 # an OPERATOR-OWNED CGROUP still encloses the run (the supervisor signals ONLY its own
 # group, so it can never replace that requirement).
 for sig in ('signal.SIGINT','signal.SIGTERM','signal.SIGHUP'):
  assert sig in s.split('def supervise')[1],sig
 assert 'os.waitid' in s and 'os.WNOWAIT' in s
 assert 'operator_cgroup_required' in s and 'operator-owned' in s
 assert 'def teardown(state' in s and 'def group_kill(' in s and 'def group_probe(' in s
 # Cleanup state is persistent: one deadline for the run, reused by every path, and a
 # permanent guard that a reaped leader's former group is never signalled again.
 assert 'class CleanupState' in s and 'if self.deadline is None:' in s
 assert s.count('self.deadline=time.monotonic()+self.allowance')==1
 assert 'if state.reaped:return state.group_status' in s and 'if state.complete:return state.result()' in s
 assert 'def record_failed(' in s and 'result_record_failed_' in s and 'action=no_restart' in s
