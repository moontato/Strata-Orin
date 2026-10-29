// Execute the production strict staging methods with ONLY CUDA IO/free-memory
// boundary replaced. This is not a CUDA/Engine/Serve build or real GPU evidence.
#include "strata/core/q36_vision_preallocation.hpp"
#include "strata/core/unified_kv_protocol.hpp"
#include <cassert>
#include <iostream>
#include <memory>
uint64_t available=1u<<20;unsigned copies=0,syncs=0;
namespace strata::core {uint64_t unified_kv_available(){return available;}}
constexpr int cudaMemcpyHostToDevice=1;
int cudaStreamSynchronize(void*){++syncs;return 0;}
int cudaMemcpyAsync(void* to,const void* from,size_t n,int,void*){++copies;std::memcpy(to,from,n);return 0;}
#define CK(value) do {if((value)!=0) throw std::runtime_error("fake CUDA IO failed");}while(0)
struct Adapter:strata::core::KvAdapter {
    uint64_t bytes_for(uint64_t)const override{return 0;}uint64_t committed_bytes()const override{return 0;}
    bool grow(uint64_t,std::string&)override{return true;}bool quiesce(std::string&)override{return true;}bool discard(std::string&)override{return true;}
};
struct Model {struct Geometry {int64_t n_embd=2;}g;};
struct Engine {
    Model* m;int cur=0,ctx=32;bool vision=true,kv_bank=true,img_active=false,map_on=false;size_t img_cap=8;
    struct Sess {int32_t rdelta=0;};std::vector<Sess> sess=std::vector<Sess>(4);
    float* d_img;int32_t *d_mtab,*d_map;const int32_t* mtab_now=nullptr;void* st=nullptr;
    int64_t img_prompt_cells=0;std::vector<int32_t> img_map;strata::core::Q36VisionLease vision_lease;
    std::unique_ptr<strata::core::UnifiedKvPool> kv_pool;
    void set_delta(int32_t d){sess[(size_t)cur].rdelta=d;}
#include "q36_vision_staging.inc"
};
template<class F> void fails(F f){bool bad=false;try{f();}catch(const std::exception&){bad=true;}assert(bad);}
int main() {
    Model model;Engine e;e.m=&model;float image[8]{};int32_t table[96]{},map[8]{};e.d_img=image;e.d_mtab=table;e.d_map=map;
    e.kv_pool=std::make_unique<strata::core::UnifiedKvPool>(32,128);Adapter adapters[4];strata::core::KvIdentity ids[4];
    for(int i=0;i<4;++i) ids[i]=e.kv_pool->add(adapters[i]);
    std::vector<float> rows(8,0.25f);std::vector<int32_t> cells={-1,0,1,2,3,-1,-1},mrope(21,7);
    fails([&]{e.strict_set_images(rows,cells,mrope,7,-2);});assert(!copies);
    assert(e.kv_pool->begin(ids[0],10));assert(e.kv_pool->begin(ids[1],10));e.sess[1].rdelta=-9;
    auto* original=e.d_img;
    e.strict_set_images(rows,cells,mrope,7,-2);
    assert(copies==2 && e.vision_lease.owner==0 && e.img_active && image[0]==0.25f && table[0]==7);
    fails([&]{e.strict_set_images(rows,cells,mrope,7,-2);});assert(copies==2);
    e.release_vision_views();assert(!e.vision_lease.busy() && !e.img_active && e.img_map.empty());
    assert(e.sess[0].rdelta==-2 && e.sess[1].rdelta==-9 && e.d_img==original);
    {Engine::VisionPrefillScope scope(e,true);e.strict_set_images(rows,cells,mrope,7,-2);}
    assert(!e.vision_lease.busy()); // includes cancellation/error scope exit: buffers remain owned/stable
    available=0;const auto before=copies;fails([&]{e.strict_set_images(rows,cells,mrope,7,-2);});assert(copies==before && !e.vision_lease.busy());
    available=1u<<20;rows.resize(10);fails([&]{e.strict_set_images(rows,cells,mrope,7,-2);});assert(copies==before);
    assert(e.kv_pool->end(ids[0],7));assert(e.kv_pool->end(ids[1],0));
    std::cout<<"Q36_VISION_STAGING_TEST PASS production method/real pool lease, fake CUDA boundary; no post-ready allocation API\n";
}
