// Compile/execute ACTUAL match/pick bodies and diagnostic methods (default parity)
// against CPU-only ownership/backing stand-ins. NOT actual Engine/CUDA compilation.
#include "strata/core/q36_cache_diagnostics.hpp"
#include "strata/core/q36_prefix_checkpoint.hpp"
#include "strata/core/q36_image_checkpoint.hpp"
#include "strata/core/q36_vision_preallocation.hpp"
#include <array>
#include <cassert>
#include <memory>
#include <iostream>
#include <random>
struct ImgKey {int64_t cell=0,n=0;uint64_t hash=0;bool operator==(const ImgKey& b)const{return cell==b.cell && n==b.n && hash==b.hash;}};
struct ImgReq {std::vector<ImgKey> imgs;std::vector<float> rows;std::vector<int32_t> mrope;};
struct Engine {
    static constexpr int kVerify=8;
    struct Sess {strata::core::Q36ImageSourcePtr image_source;std::vector<int32_t> live;std::vector<ImgKey> imgs;bool active=false;int64_t pos=0;int32_t rdelta=0;double used=0;
        struct Point{uint64_t pos=0;};struct Points{std::vector<Point> points;};Points checkpoints;};
    bool image_checkpoints=false;std::vector<Sess> sess=std::vector<Sess>(4);int n_sess=4;size_t checkpoint_count=0;
    struct Pool {std::array<strata::core::KvSessionInfo,4> owners{};strata::core::KvSessionInfo session(uint32_t s)const{return owners.at(s);}};
    struct Adapter {bool backed=true;bool covers(size_t)const{return backed;}};
    std::unique_ptr<Pool> kv_pool=std::make_unique<Pool>();std::vector<std::unique_ptr<Adapter>> kv_adapters;
    struct Model {struct Geometry {int n_embd=2;}g;}model,*m=&model;
    Engine(){for(int i=0;i<4;++i){auto& o=kv_pool->owners[i];o.state=strata::core::KvResidency::idle;o.id.slot=(uint32_t)i;o.id.generation=1;o.epoch=2;kv_adapters.push_back(std::make_unique<Adapter>());}}
    int checkpoint_match(int,const std::vector<int32_t>&,size_t)const{return -1;}
    bool image_prefix_equal(const strata::core::Q36ImageSource& a,const strata::core::Q36ImageSource& b,uint64_t end)const{return strata::core::q36_image_prefix(a,b,end);}
    int image_checkpoint_match(int,const std::vector<int32_t>&,const strata::core::Q36ImageSource&)const{return -1;}
};
struct Serve {
    Engine& E;explicit Serve(Engine& e):E(e){}
#include MATCH_INC
#include "../../src/qwen35/q36_cache_diagnostics.inc"
};
#ifdef BASE_MATCH_INC
struct BaseServe {Engine& E;explicit BaseServe(Engine& e):E(e){}
#include BASE_MATCH_INC
};
static void disabled_route_parity() {
    Engine e;Serve current(e);BaseServe old(e);std::mt19937 rng(73);
    for(int test=0;test<1000;++test) {
        std::vector<int32_t> ids(9);for(auto& x:ids)x=(int32_t)(rng()%5);
        std::vector<ImgKey> images;if(rng()%2) images.push_back({1,2,17});
        for(int i=0;i<4;++i) {auto& s=e.sess[(size_t)i];s.active=rng()%3==0;s.live.resize(rng()%9);for(auto& x:s.live)x=(int32_t)(rng()%5);s.imgs=(rng()%2)?images:std::vector<ImgKey>{};s.used=rng()%20;
            e.kv_pool->owners[(size_t)i].state=(rng()%4)?strata::core::KvResidency::idle:strata::core::KvResidency::active;e.kv_pool->owners[(size_t)i].last_use=rng()%20;
            assert(current.match_len(s,ids,images)==old.match_len(s,ids,images));}
        assert(current.pick_session(ids,images)==old.pick_session(ids,images));
    }
}
#endif
int main(int argc,char** argv) {
#ifdef BASE_MATCH_INC
    disabled_route_parity();
#endif
    assert(argc==2);Engine e;Serve sv(e);auto& s=e.sess[0];
    s.live={3,248056,248056,248056,248056,4};s.imgs={{1,4,17}};s.pos=6;s.rdelta=-2;
    ImgReq ir;ir.imgs=s.imgs;ir.rows={1,2,3,4,5,6,7,8};ir.mrope.resize(24,0);
    auto ids=s.live;ids.push_back(7);ids.push_back(8);
    auto held=s.live;const auto images=s.imgs;const auto pos=s.pos;
    assert(sv.pick_session(ids,ir.imgs)==0 && sv.match_len(s,ids,ir.imgs)==6);
    sv.cache_diag_before(false,true,-1,ids,ir,argv[1],{51});sv.cache_diag_selected(0);sv.cache_diag_retained(0);
    assert(sv.pick_session(ids,ir.imgs)==0 && s.live==held && s.imgs==images && s.pos==pos);
    sv.cache_diag_before(true,true,2,ids,ir,argv[1],{51});sv.cache_diag_selected(2);
    ir.rows[0]++;ir.imgs[0].hash++;
    sv.cache_diag_before(false,true,-1,ids,ir,argv[1],{51});assert(sv.match_len(s,ids,ir.imgs)==0);
    e.kv_pool->owners[0].epoch++;sv.cache_diag_before(false,true,-1,ids,ir,argv[1],{51});
    ir.imgs=s.imgs;e.kv_pool->owners[0].state=strata::core::KvResidency::active;
    sv.cache_diag_before(false,true,-1,ids,ir,argv[1],{51});assert(sv.pick_session(ids,ir.imgs)==1);
    e.kv_pool->owners[0].state=strata::core::KvResidency::idle;e.kv_adapters[0]->backed=false;
    sv.cache_diag_before(false,true,-1,ids,ir,argv[1],{51});assert(sv.pick_session(ids,ir.imgs)==0);
    assert(s.live==held && s.imgs==images && s.pos==pos);
    std::cout<<"actual native match/pick + diagnostic neutrality (CPU stand-ins) PASS\n";
}
