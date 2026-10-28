#include "strata/core/q36_image_checkpoint.hpp"
#include <cassert>
#include <iostream>
using namespace strata::core;
struct Adapter:KvAdapter {
 Q36ImagePoint image;Q36PrefixCheckpoints cp;
 uint64_t bytes_for(uint64_t n)const override{return n;}uint64_t committed_bytes()const override{return 0;}
 bool grow(uint64_t,std::string&)override{return true;}bool quiesce(std::string&)override{return true;}bool discard(std::string&)override{image={};cp.clear();return true;}
};
Q36ImageSource image(uint64_t cell=700) {
 Q36ImageSource a;a.width=2;a.rows.assign(1024*2,0.25f);a.images.push_back({cell,1024,32,32,cell,0,123});return a;
}
int main() {
 auto a=image(),b=a;
 assert(q36_image_prefix(a,b,512));assert(q36_image_prefix(a,b,2048));
 assert(!q36_image_prefix(a,b,1024));assert(!q36_image_prefix(a,b,1536));
 b.rows[13]=0.5f; // stable Python key/hash deliberately unchanged
 assert(q36_image_prefix(a,b,512));assert(!q36_image_prefix(a,b,2048));assert(q36_image_common(a,b,3000)==700);
 b=a;b.images[0].nx=16;b.images[0].ny=64;
 assert(!q36_image_prefix(a,b,2048));assert(q36_image_prefix(a,b,512));
 auto nonsquare=a;nonsquare.images[0].nx=16;nonsquare.images[0].ny=64;
 auto transposed=nonsquare;transposed.images[0].nx=64;transposed.images[0].ny=16;
 assert(nonsquare.delta(2048)==transposed.delta(2048) && !q36_image_prefix(nonsquare,transposed,2048));
 b=a;b.images[0].cell++;assert(!q36_image_prefix(a,b,2048));
 b=a;b.images[0].rotary++;assert(!q36_image_prefix(a,b,2048));
 b=a;b.images[0].hash++;assert(q36_image_prefix(a,b,2048)); // advisory keys cannot grant/deny actual identity
 assert(q36_image_common(a,a,600)==600); // genuine earlier-message/header/reasoning token LCP
 assert(q36_image_common(a,a,1100)==700);assert(q36_image_common(a,a,2400)==2400);
 assert(a.delta(512)==0 && a.delta(2048)==-992);
 auto c=a;c.images.push_back({3000,1,1,1,2008,1024,77});c.rows.push_back(1);c.rows.push_back(2);
 assert(q36_image_prefix(a,c,2048)); // changed/new images AFTER boundary confer no extra reuse
 assert(!q36_image_prefix(a,c,3500));
 Q36PrefixCheckpoints cp;cp.points.resize(4);Q36ImagePlan plan;plan.make(cp,a,0,6000,512);
 assert(plan.anchors[0]==512 && plan.anchors[1]==2048 && plan.anchors[2]==4096);
 assert(plan.placement(cp,512,a)==0);assert(plan.placement(cp,1024,a)<0);
 Q36PrefixCheckpoints single;single.points.resize(1);Q36ImagePlan one;one.make(single,a,0,6000,512);assert(one.placement(single,512,a)==0);
 cp.points[0]={512,1,1};assert(plan.placement(cp,2048,a)==1);
 cp.policy=Q36PrefixCheckpoints::Policy::adaptive;cp.learning.observe(1300,6000,512);plan.make(cp,a,0,6000,512);
 for(auto p:plan.anchors) assert(!p || a.boundary(p)); // learned targets inside picture never poison readiness
 auto early=image(50);plan.make(cp,early,0,6000,512);assert(plan.anchors[0]==1536); // no earlier legal end: defer, don't drop
 auto source=std::make_shared<const Q36ImageSource>(a);Q36ImagePoint meta{source,2048,-992,true};
 assert(meta.valid(2048));assert(!meta.valid(1024));meta.ready=false;assert(!meta.valid(2048));meta.ready=true;meta.delta=0;assert(!meta.valid(2048));
 // REAL pool ownership lifecycle: metadata cannot replace pins/epoch/backing.
 UnifiedKvPool pool(8192,16384);Adapter ad[4];KvIdentity ids[4];for(int i=0;i<4;++i) ids[i]=pool.add(ad[i]);
 assert(pool.begin(ids[0],4096));assert(pool.end(ids[0],3000));const auto idle=pool.session(0);Q36PrefixCheckpoints::Point p{2048,idle.id.generation,idle.epoch};
 auto available=[&](const KvIdlePin* pin=nullptr,const KvIdentity* lease=nullptr,bool backed=true) {
  return Q36PrefixCheckpoints::available(p,pool.session(0),3000,3000,false,false,backed,&pool,pin,lease);
 };
 assert(available());assert(!available(nullptr,nullptr,false));
 {KvIdlePin pin(&pool,ids[0]);assert(!available());assert(available(&pin));assert(pool.begin(ids[1],4096));assert(pool.end(ids[1],0));}
 assert(available());assert(pool.begin(ids[0],4096));assert(!available());assert(available(nullptr,&ids[0]));
 assert(pool.replay_started(ids[0]));assert(!available(nullptr,&ids[0])); // stale epoch
 assert(pool.end(ids[0],3000));
 assert(pool.reset(ids[0],ids[0]));assert(!available()); // reset/generation invalidation
 {UnifiedKvPool pressure(8192,16384);Adapter owners[4];KvIdentity active[4];
  for(int i=0;i<4;++i) {active[i]=pressure.add(owners[i]);assert(pressure.begin(active[i],4096));owners[i].image={source,2048,-992,true};const auto o=pressure.session((uint32_t)i);owners[i].cp.points={{2048,o.id.generation,o.epoch}};assert(pressure.end(active[i],4096));}
  assert(pressure.begin(active[3],8192));assert(pressure.session(0).state==KvResidency::evicted);
  assert(!owners[0].image.ready && !owners[0].image.source && !owners[0].cp.points[0].pos);
  assert(pressure.end(active[3],0));
 }
 // bounded shared metadata, not a fifth device payload
 std::array<Q36ImagePoint,4> points;for(auto& x:points) x={source,2048,-992,true};
 assert(points[0].source.get()==points[3].source.get());points={};
 std::cout<<"Q36_IMAGE_CHECKPOINT_TEST PASS CPU exact bytes/grid/placement/delta/legal targets/ready/ownership; NOT GPU inference\n";
}
