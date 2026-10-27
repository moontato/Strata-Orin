// Runs the ACTUAL developer oracle .inc through a tiny CPU/CUDA-IO stand-in.
// This tests gate control flow/negative controls, NOT Engine/Serve/CUDA/model math.
#include "strata/core/q36_checkpoint_schedule.hpp"
#include <cmath>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <iostream>
#define CK(x) do { if((x)!=0) throw std::runtime_error("fake CUDA failure"); } while(0)
enum {cudaMemcpyDeviceToHost,cudaMemcpyHostToDevice};
static int cudaMemcpyAsync(void* a,const void* b,size_t n,int,int) {std::memcpy(a,b,n);return 0;}
static int cudaStreamSynchronize(int) {return 0;}
using Clock=std::chrono::steady_clock;
static double ms_since(Clock::time_point t) {return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
namespace strata::core {
struct KvBackingInfo {uint64_t checkpoint_copy_us=0,checkpoint_copy_bytes=0,driver_allocations=0,driver_allocation_attempts=0,allocations=0,bytes=0;};
enum class KvResidency {empty,idle,active};
struct FakePool {std::array<KvResidency,4> states{};struct Info {uint64_t mapped_bytes=0;};
    int reset(int id,int&) {states.at(id)=KvResidency::empty;return 0;}
    struct Session {KvResidency state;};Session session(uint32_t s) {return {states.at(s)};}
    Info info() {return {};}
};
struct KvIdlePin {KvIdlePin(FakePool*,int) {}};
static void kv_require(int n) {if(n) throw std::runtime_error("fake pool failure");}
struct FakeBank {KvBackingInfo backing_info(){return {};}};
}
struct GateDump {explicit GateDump(const std::string&) {} template<class T> void vector(const std::vector<T>&) {}};
static std::vector<int32_t> parse_ids(const char* text) {
    std::vector<int32_t> ids;std::istringstream in(text);std::string field;
    while(std::getline(in,field,',')) ids.push_back(std::stoi(field));
    return ids;
}
struct Engine {
    struct Geometry {int n_embd=4,ssm_conv_channels=4,ssm_d_conv=2,ssm_v_heads=1,ssm_state_size=2,n_head_kv=1,head_dim=64;} ;
    struct Model {Geometry g;std::vector<int> layers{0};bool hybrid=false;} model,*m=&model;
    struct Session {
        int64_t pos=0;bool active=false;std::vector<int32_t> live;
        std::array<float,12> recurrent{};float* hlast=recurrent.data()+8;
        std::vector<float*> conv{recurrent.data()},S{recurrent.data()+4};
        std::array<std::vector<float>,4> kv;
        std::vector<float*> kc,vc;
        struct Kv8 {uint8_t *k=nullptr,*v=nullptr;uint16_t *ks=nullptr,*vs=nullptr;};std::vector<Kv8> kv8{2};
        std::array<std::vector<uint8_t>,4> code;std::array<std::vector<uint16_t>,4> scale;
        std::array<std::array<float,12>,4> payload{};std::vector<float*> checkpoint_data;
        std::array<size_t,4> points{};
        Session() {
            for(auto& v:kv) v.resize(2048*64);
            for(auto& v:code) v.resize(2048*64);
            for(auto& v:scale) v.resize(2048);
            for(size_t i=0;i<2;++i) {kc.push_back(kv[2*i].data());vc.push_back(kv[2*i+1].data());
                kv8[i]={code[2*i].data(),code[2*i+1].data(),scale[2*i].data(),scale[2*i+1].data()};}
            for(auto& v:payload) checkpoint_data.push_back(v.data());
        }
    };
    std::array<Session,4> sess;
    strata::core::Q36ScheduleBook* schedule_book=nullptr;
    std::function<void(int)> schedule_restored;std::function<void(int,int)> schedule_before_restore;
    int n_sess=4,mtp_d=2,ctx=2048,tmax=16,cur=0,st=0;int64_t pos=0;size_t checkpoint_count=4;uint64_t checkpoint_payload=48;
    bool kv_int8=true;std::array<int,4> kv_ids{0,1,2,3};std::array<uint64_t,4> kv_required{};
    std::unique_ptr<strata::core::FakePool> kv_pool=std::make_unique<strata::core::FakePool>();
    std::unique_ptr<strata::core::FakeBank> kv_bank;
    std::vector<float> logits{0,0,0,0};
    void select(int s) {cur=s;pos=sess.at(s).pos;}
    void reset() {auto& a=sess.at(cur);a.recurrent={};a.live.clear();a.points={};pos=a.pos=0;if(schedule_book) schedule_book->reset(cur);}
    void kv_begin(int s,uint64_t,bool) {kv_pool->states.at(s)=strata::core::KvResidency::active;}
    void kv_end(int s) {kv_pool->states.at(s)=strata::core::KvResidency::idle;}
    void clear_images() {}
    void host_logits(std::vector<float>& out) {out=logits;}
    int greedy() {return 0;}
    template<class F> int64_t process(const int32_t* ids,int64_t n,F callback,bool suffix=false) {
        auto& a=sess.at(cur);const bool mtp=n!=1 || suffix;
        for(int64_t start=0;start<n;start+=tmax) {
            const auto take=std::min<int64_t>(tmax,n-start);
            for(int64_t j=0;j<take;++j) {
                const float x=ids[start+j]*0.125f+take*0.003f;
                for(auto& v:a.recurrent) v=v*0.9375f+x;
                for(size_t l=0;l<2;++l) if(!l || mtp) {
                    for(size_t d=0;d<64;++d) {const auto off=((size_t)pos+j)*64+d;
                        a.kc[l][off]=a.recurrent[0]+(float)d;a.vc[l][off]=a.recurrent[1]-(float)d;
                        a.kv8[l].k[off]=(uint8_t)(ids[start+j]+(int)d);a.kv8[l].v[off]=(uint8_t)(ids[start+j]-(int)d);}
                    a.kv8[l].ks[pos+j]=(uint16_t)(ids[start+j]+1);a.kv8[l].vs[pos+j]=(uint16_t)(ids[start+j]+2);
                }
            }
            pos+=take;a.pos=pos;
            if(start+take==n) for(auto& v:logits) v=a.recurrent[0];
            if constexpr(std::is_void_v<decltype(callback(start+take))>) callback(start+take);
            else if(!callback(start+take)) return start+take;
        }
        return n;
    }
};
struct Serve {
    Engine& e;double schedule_prefill_ms=0;explicit Serve(Engine& E):e(E){}
    void request_body(const std::string& line) {
        std::istringstream in(line);std::string cmd,csv;int slot,n;in>>cmd>>slot>>n>>csv;(void)n;
        auto ids=parse_ids(csv.c_str());auto& a=e.sess.at(slot);size_t common=0;
        while(common<a.live.size() && common<ids.size() && a.live[common]==ids[common]) ++common;
        size_t reuse=0;int cp=-1;bool restored=false;
        if(common==a.live.size() && common) reuse=common;
        else for(size_t i=0;i<4;++i) if(a.points[i]<=common && a.points[i]>reuse && a.points[i]<ids.size()) {cp=(int)i;reuse=a.points[i];}
        e.select(slot);e.kv_begin(slot,ids.size()+16,true);
        if(cp>=0) {
            if(e.schedule_before_restore) e.schedule_before_restore(slot,cp);
            a.recurrent=a.payload.at(cp);e.pos=a.pos=(int64_t)reuse;a.live.resize(reuse);
            for(auto& p:a.points) if(p>reuse) p=0;
            if(e.schedule_book) e.schedule_book->restore(slot,(size_t)cp,slot);
            if(e.schedule_restored) e.schedule_restored(slot);
            restored=true;
        } else if(!reuse) e.reset();
        if(e.schedule_book) e.schedule_book->begin(slot,ids.data()+reuse,(int64_t)(ids.size()-reuse),restored);
        e.process(ids.data()+reuse,(int64_t)(ids.size()-reuse),[&](int64_t done) {
            if(e.schedule_book) e.schedule_book->progress(slot,(uint64_t)done);
            if(ids.size()-reuse>1 && (!reuse || restored)) {
                const size_t i=(size_t)e.pos/16%4;a.points[i]=(size_t)e.pos;a.payload[i]=a.recurrent;
                if(e.schedule_book) e.schedule_book->capture(slot,i);
            }
        },restored);
        a.live=ids;e.kv_end(slot);
    }
};
#include "../../src/qwen35/q36_checkpoint_schedule.inc"
int main(int argc,char** argv) {
    if(argc<2) return 2;
    try {Engine e;return checkpoint_schedule_regression(e,argv[1],"",argc>2?argv[2]:"");}
    catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
