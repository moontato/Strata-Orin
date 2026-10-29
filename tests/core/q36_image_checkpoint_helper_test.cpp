// Execute production image eligibility/learning/capture-control methods.
// CPU pool + fake payload writer; NOT Engine/CUDA/model validation.
#include "strata/core/q36_image_checkpoint.hpp"
#include "strata/core/q36_vision_preallocation.hpp"
#include <cassert>
#include <iostream>
using namespace strata::core;
struct Adapter:KvAdapter {uint64_t bytes_for(uint64_t)const override{return 0;}uint64_t committed_bytes()const override{return 0;}bool grow(uint64_t,std::string&)override{return true;}bool quiesce(std::string&)override{return true;}bool discard(std::string&)override{return true;}bool covers(uint64_t)const{return true;}};
struct Engine {
 struct Sess {Q36PrefixCheckpoints checkpoints;Q36ImageSourcePtr image_source;std::array<Q36ImagePoint,4> image_points{};Q36ImagePlan image_plan;uint64_t image_ready_end=0;bool active=false;uint64_t pos=0;std::vector<int32_t> live;};
 Q36VisionLease vision_lease;
 mutable uint64_t image_metadata_us=0;
 bool image_checkpoints=true;size_t checkpoint_count=4;Q36PrefixCheckpoints::Policy checkpoint_policy=Q36PrefixCheckpoints::Policy::adaptive;
 int cur=0,n_sess=4,tmax=512;int64_t pos=0;size_t captures=0;
 std::unique_ptr<UnifiedKvPool> kv_pool=std::make_unique<UnifiedKvPool>(8192,32768);
 std::array<Sess,4> sess;std::vector<std::unique_ptr<Adapter>> kv_adapters;std::array<KvIdentity,4> ids;
 Engine(){for(int i=0;i<4;++i){kv_adapters.push_back(std::make_unique<Adapter>());ids[i]=kv_pool->add(*kv_adapters.back());sess[i].checkpoints.points.resize(4);}}
 size_t checkpoint_common(int s,const std::vector<int32_t>& ids_)const {size_t n=0;while(n<sess[s].live.size() && n<ids_.size() && sess[s].live[n]==ids_[n]) ++n;return n;}
 bool checkpoint_available(int s,int i,const KvIdlePin* pin=nullptr,const KvIdentity* lease=nullptr,bool image=false)const {
  if(i<0) return false;
  const auto p=sess[s].checkpoints.points.at((size_t)i);
  return (!image || sess[s].image_points[(size_t)i].valid(p.pos)) && Q36PrefixCheckpoints::available(p,kv_pool->session((uint32_t)s),sess[s].live.size(),sess[s].pos,false,sess[s].active,true,kv_pool.get(),pin,lease);
 }
 void checkpoint_capture(int index,const Q36ImagePoint* proof) {
  ++captures;const auto o=kv_pool->session((uint32_t)cur);sess[cur].image_points[(size_t)index]=*proof;
  sess[cur].checkpoints.publish((size_t)index,{(uint64_t)pos,o.id.generation,o.epoch});
 }
#include "q36_image_checkpoint.inc"
};
int main() {
 Engine e;auto source=std::make_shared<Q36ImageSource>();source->width=2;source->rows.assign(2048,0.25f);source->images.push_back({700,1024,32,32,700,0,99});
 auto& s=e.sess[0];s.image_source=source;s.image_plan.make(s.checkpoints,*source,0,4000,512);
 assert(e.kv_pool->begin(e.ids[0],4000));
 const auto admission=e.kv_pool->session(0);e.vision_lease.acquire(0,admission.id.generation,admission.epoch);
 e.pos=512;s.image_ready_end=0;e.image_checkpoint_capture();assert(!e.captures); // readiness is mandatory
 s.image_ready_end=512;++e.vision_lease.epoch;bool stale=false;try{e.image_checkpoint_capture();}catch(const std::runtime_error&){stale=true;}assert(stale && !e.captures);--e.vision_lease.epoch;
 e.image_checkpoint_capture();assert(e.captures==1 && s.image_points[0].valid(512));
 e.pos=1024;s.image_ready_end=1024;e.image_checkpoint_capture();assert(e.captures==1); // no capture inside picture
 e.pos=2048;s.image_ready_end=2048;e.image_checkpoint_capture();assert(e.captures==2);
 s.pos=4000;s.live.assign(4000,17);assert(e.kv_pool->end(e.ids[0],4000));
 auto next=s.live;next[2500]=18;assert(e.image_checkpoint_match(0,next,*source)==1);
 auto changed=*source;changed.rows[3]=0.5f;assert(e.image_checkpoint_match(0,next,changed)==0); // only BEFORE image
 auto early=next;early[1]=99;assert(e.image_checkpoint_match(0,early,*source)<0); // real earlier header/user edit
 {KvIdlePin pin(e.kv_pool.get(),e.ids[0]);assert(e.image_checkpoint_match(0,next,*source)<0);assert(e.checkpoint_available(0,1,&pin,nullptr,true));}
 const size_t common[4]={2500,0,0,0};const auto learn=e.image_checkpoint_learning(common);assert(learn.rewrites==1);
 assert(e.kv_pool->reset(e.ids[0],e.ids[0]));assert(e.image_checkpoint_match(0,next,*source)<0); // stale identity cannot authorize
 e.image_checkpoints=false;assert(e.image_checkpoint_match(0,next,*source)<0);
 std::cout<<"Q36_IMAGE_CHECKPOINT_HELPER_TEST PASS actual control methods, CPU pool/fake payload writer; NOT GPU\n";
}
