#!/usr/bin/env python3
"""Host-only harness + placement invariants for the restored-state observation boundary.
Runs the REAL production copy_session tail with the REAL boundary primitives; CUDA calls
are inert stubs. No CUDA/GPU/model/service run, and no numerical claim.
"""
from pathlib import Path
import subprocess,tempfile

COPY_START='    void copy_session(int src, int dst) {'
COPY_END='    int hybrid_swaps_after_prompt = 2048;'
DEFER='request_restore_snapshot(dst);'
LEGACY_OBSERVE='if(schedule_restored) schedule_restored(dst);'
PRIMITIVES='src/qwen35/q36_restore_observation.inc'
GUARD='if(cur!=dst)'

SCAFFOLD=r'''
// Declaration + execution scaffold. CUDA calls are inert stubs.
#include "strata/core/q36_checkpoint_schedule.hpp"
#include "strata/core/q36_prefix_checkpoint.hpp"
#include "strata/core/q36_image_checkpoint.hpp"
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <functional>
#include <stdexcept>
#include <vector>
#define CK(value) ((void)(value))
constexpr int cudaMemcpyDeviceToDevice=1;
int cudaMemcpyAsync(void*,const void*,size_t,int,void*) {return 0;}
int cudaMemcpy2DAsync(void*,size_t,const void*,size_t,size_t,size_t,int,void*) {return 0;}
int cudaStreamSynchronize(void*) {return 0;}
struct Cache {int8_t* k=nullptr;int8_t* v=nullptr;uint16_t* ks=nullptr;uint16_t* vs=nullptr;};
namespace q36kv8 {int copy_prefix(Cache&,const Cache&,size_t,size_t,size_t,void*) {return 0;}}
struct Geometry {int64_t n_embd=2048,ssm_conv_channels=0,ssm_d_conv=0,ssm_v_heads=0,ssm_state_size=0,n_head_kv=0,head_dim=0;};
struct Model {Geometry g;};
struct ImgKey {int64_t cell,n;uint64_t hash;};
struct Engine {
    struct Sess {
        std::vector<float*> conv,S,kc,vc;std::vector<Cache> kv8;float* hlast=nullptr;
        int64_t pos=0;int32_t rdelta=0;std::vector<int32_t> live;std::vector<ImgKey> imgs;
        strata::core::Q36PrefixCheckpoints checkpoints;
        strata::core::Q36ImageSourcePtr image_source;uint64_t image_ready_end=0;
        std::array<strata::core::Q36ImagePoint,4> image_points{};
        strata::core::Q36ImagePlan image_plan;
        void clear_image_points(){image_source.reset();image_points={};image_plan={};image_ready_end=0;}
    };
    Model model;Model* m=&model;std::array<Sess,4> sess;
    int ctx=262144,cur=0;int64_t pos=0;bool kv_int8=false;void* kv_pool=nullptr;void* st=nullptr;
    int32_t d_delta=0;                                   // ONE device scalar, bound to cur
    strata::core::Q36ScheduleBook book;strata::core::Q36ScheduleBook* schedule_book=&book;
    std::function<void(int)> schedule_restored;
    /// Mirrors Engine::set_delta: one write updates BOTH the session copy and the device scalar.
    void set_delta(int32_t d){sess[(size_t)cur].rdelta=d;d_delta=d;}
    /// Mirrors Engine::load_cur.
    void load_cur(){pos=sess[(size_t)cur].pos;set_delta(sess[(size_t)cur].rdelta);}
    /// Mirrors Engine::select.
    void select(int i){if(i==cur)return;sess[(size_t)cur].pos=pos;cur=i;load_cur();}
#include "q36_restore_observation.inc"
#include "copied_session.inc"
};
int main() {
    struct Observed {int dst;int32_t cpu;int32_t device;int64_t pos;};
    Engine e;std::vector<Observed> seen;e.cur=2;
    e.schedule_restored=[&](int dst){seen.push_back({dst,e.sess[(size_t)dst].rdelta,e.d_delta,e.pos});};
    // The reported geometry: donor 0 sits at the post-image delta -992, an unrelated
    // CURRENT slot 2 holds delta 0, and destination 1 is empty and not current.
    e.sess[0].pos=2048;e.sess[0].rdelta=-992;e.sess[0].live.assign(2048,17);
    e.sess[0].imgs.push_back({700,1024,99});e.sess[2].pos=10;e.set_delta(0);
    assert(e.cur==2 && e.d_delta==0 && e.sess[1].rdelta==0);
    e.copy_session(0,1);                                  // production cross-slot copy
    assert(seen.empty());                                 // the defect observed HERE, at cur==2
    assert(e.d_delta==0);                                 // device scalar still slot 2's
    assert(e.restore_snapshot_fires==0 && e.restore_snapshot_deferred==1);
    e.select(1);                                          // the caller's select(s)
    assert(e.fire_restore_snapshot());
    assert(seen.size()==1 && seen[0].dst==1 && seen[0].pos==2048);
    assert(seen[0].cpu==-992 && seen[0].device==-992);     // one slot's pair, never a mix
    assert(!e.fire_restore_snapshot());                   // exactly once per producer
    assert(e.restore_snapshot_slot==-1);
    // A destination that is not current is refused instead of being observed.
    e.select(2);e.request_restore_snapshot(1);bool refused=false;
    try{e.fire_restore_snapshot();}catch(const std::logic_error&){refused=true;}
    assert(refused && seen.size()==1 && e.restore_snapshot_refused==1);   // counted against the deferred producer
    assert(!e.fire_restore_snapshot());e.discard_restore_snapshot();
    // Same-slot restoration: the producer is already current, so it observes in place.
    e.select(1);assert(e.observe_restored_state(1));
    assert(seen.size()==2 && seen[1].dst==1 && seen[1].cpu==-992 && seen[1].device==-992);
    refused=false;e.select(2);
    try{e.observe_restored_state(1);}catch(const std::logic_error&){refused=true;}
    assert(refused && seen.size()==2 && e.restore_snapshot_refused==1);   // an in-place producer never observed late
    assert(e.restore_snapshot_fires==2 && e.restore_snapshot_deferred==2 && e.restore_snapshot_immediate==1);
    assert(e.restore_snapshot_fires+e.restore_snapshot_refused==e.restore_snapshot_deferred+e.restore_snapshot_immediate);
    // A normal engine invocation installs no hook: nothing is requested or observed.
    Engine p;p.cur=2;p.sess[0].rdelta=-992;p.sess[2].pos=10;p.set_delta(0);
    p.copy_session(0,1);p.request_restore_snapshot(1);
    assert(!p.fire_restore_snapshot() && !p.observe_restored_state(1));
    assert(p.restore_snapshot_fires==0 && p.restore_snapshot_refused==0 && p.d_delta==0 && p.sess[1].rdelta==-992);
    std::cout<<"Q36_RESTORE_OBSERVATION PASS cross_slot_deferred=1 observed_cpu=-992 observed_device=-992 "
                "previous_current_device=0 exactly_once=1 noncurrent_refused=1 same_slot_in_place=1 "
                "counters_balanced=1 no_hook_inert=1\n";
    return 0;
}
'''

def copy_session_body(source):
    start=source.index(COPY_START);end=source.index(COPY_END,start)
    return source[start:end]

def build(root,out=None,copy_overlay=None,primitives_overlay=None):
    """Compile+run the harness. Overlays reproduce a defect; the real tree must pass."""
    root=Path(root)
    source=(root/'src/qwen35/q36_engine.cu').read_text()
    body=copy_overlay if copy_overlay is not None else copy_session_body(source)
    if body.count(DEFER)+body.count(LEGACY_OBSERVE)!=1:raise ValueError('unexpected copy_session observation placement')
    primitives=primitives_overlay if primitives_overlay is not None else (root/PRIMITIVES).read_text()
    directory=(Path(out).resolve() if out else None)
    temp=tempfile.TemporaryDirectory(prefix='q36-restore-observation-') if directory is None else None
    directory=directory or Path(temp.name)
    scaffold=directory/'scaffold';scaffold.mkdir(parents=True,exist_ok=True)
    (scaffold/'copied_session.inc').write_text(body)
    (scaffold/Path(PRIMITIVES).name).write_text(primitives)
    (scaffold/'main.cpp').write_text(SCAFFOLD)
    exe=directory/'harness'
    compile=subprocess.run(['g++','-std=c++20','-O1','-g','-Wall','-Wextra','-UNDEBUG',
        '-I'+str(scaffold),'-I'+str(root/'include'),'-I'+str(root/'src/qwen35'),
        str(scaffold/'main.cpp'),'-o',str(exe)],capture_output=True,text=True)
    if compile.returncode:
        if temp:temp.cleanup()
        return compile.returncode,'',compile.stderr
    run=subprocess.run([str(exe)],capture_output=True,text=True)
    if temp is None:print('Q36_RESTORE_OBSERVATION artifact='+str(exe))
    else:temp.cleanup()
    return run.returncode,run.stdout,run.stderr

def placement_errors(root):
    """Source-order invariants a host build cannot show (Serve/CUDA is not host-buildable)."""
    root=Path(root);errors=[]
    source=(root/'src/qwen35/q36_engine.cu').read_text()
    copy=copy_session_body(source)
    if DEFER not in copy:errors.append('copy_session does not defer its observation')
    if LEGACY_OBSERVE in copy or 'schedule_restored(' in copy:errors.append('copy_session observes state directly')
    restore=source[source.index('    bool checkpoint_restore('):source.index(COPY_START)]
    if 'observe_restored_state(dst)' not in restore:errors.append('checkpoint_restore does not observe in place')
    if LEGACY_OBSERVE in restore:errors.append('checkpoint_restore observes through the raw hook')
    copy_at=source.index('E.copy_session(src, s);')
    marks={'destination selection':source.find('E.select(s);',copy_at),
           'deferred observation':source.find('E.fire_restore_snapshot();',copy_at),
           'incoming image upload':source.find('E.set_images(ir.rows',copy_at),
           'suffix processing':source.find('const int64_t got = E.process(ids.data() + reuse',copy_at)}
    if any(at<0 for at in marks.values()):
        errors.append('request path lost one of copy/select/observe/upload/process')
    else:
        # The observation must come after the destination is selected and before any
        # new input is uploaded or any suffix token is processed.
        if not copy_at<marks['destination selection']<marks['deferred observation']<marks['incoming image upload']<marks['suffix processing']:
            errors.append('request path order is not copy -> select -> observe -> upload -> process')
    primitives=(root/PRIMITIVES).read_text()
    if primitives.count(GUARD)!=2:errors.append('boundary primitives lack the current-slot guard')
    if 'discard_restore_snapshot()' not in source:errors.append('request path does not discard a stale pending observation')
    return errors

def main():
    root=Path(__file__).resolve().parents[1]
    errors=placement_errors(root)
    if errors:raise SystemExit('placement invariants: '+'; '.join(errors))
    code,stdout,stderr=build(root)
    print(stdout or stderr)
    if code:raise SystemExit('host restore-observation harness failed exit='+str(code))
    print('Q36_RESTORE_OBSERVATION CHECK PASS placement=1 host_harness=0 NOT_GPU_EVIDENCE')
if __name__=='__main__':main()
