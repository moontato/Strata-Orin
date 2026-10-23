#include "strata/core/q36_prefix_checkpoint.hpp"
#include "strata/core/unified_kv.hpp"
#include <array>
#include <cassert>
#include <iostream>
#include <numeric>
using namespace strata::core;

// Token-level causal toy model, NOT GPU/model numerical qualification. The native
// --kv-checkpoint-test gate compares full model logits and MTP on Orin.
struct State {
    uint64_t gdn=0, hidden=0;
    std::array<uint64_t,3> conv{};
    bool operator==(const State&) const = default;
};
struct Toy {
    State state;
    std::vector<uint64_t> kv,mtp;
    void step(int token) {
        state.gdn=state.gdn*33+token+state.conv[0];
        state.conv={state.conv[1],state.conv[2],(uint64_t)token};
        kv.push_back(state.gdn^token);
        mtp.push_back(state.hidden+kv.back());
        state.hidden=state.gdn*17+mtp.back();
    }
};
struct Adapter:KvAdapter {
    Q36PrefixCheckpoints cp;
    uint64_t bytes=0;
    uint64_t bytes_for(uint64_t n) const override {return n ? n*4+4096 : 0;}
    uint64_t committed_bytes() const override {return bytes;}
    bool grow(uint64_t n,std::string&) override {bytes=bytes_for(n);return true;}
    bool quiesce(std::string&) override {return true;}
    bool discard(std::string&) override {bytes=0;cp.clear();return true;}
};
int main() {
    constexpr uint64_t gran=2u<<20,payload=123u<<20,rounded=124u<<20;
    const KvPrefix snapshot{payload,UINT64_MAX,1,1,0};
    assert(snapshot.chunks(0,gran).empty());
    assert(snapshot.chunks(1,gran).size()*gran==rounded);
    assert(snapshot.chunks(8192,gran).size()*gran==rounded);
    assert(Q36PrefixCheckpoints::capacity(0,payload,gran,4)==0);
    assert(Q36PrefixCheckpoints::capacity(4*rounded-1,payload,gran,4)==0);
    assert(Q36PrefixCheckpoints::capacity(8*rounded,payload,gran,4)==2);
    assert(Q36PrefixCheckpoints::capacity(UINT64_MAX,payload,gran,4)==4);
    for(uint64_t budget:{4*rounded,8*rounded,12*rounded,16*rounded})
        assert(Q36PrefixCheckpoints::capacity(budget,payload,gran,4)*4*rounded<=budget);

    std::vector<int> a(8192);std::iota(a.begin(),a.end(),1);
    auto b=a;b[3200]=9999; // the differing position is in TOKENS, not characters
    for(size_t cap:{1,2,3,4}) {
        Q36PrefixCheckpoints cp;cp.points.resize(cap);
        std::vector<State> packed(cap);
        Toy original;
        for(size_t i=0;i<a.size();++i) {
            original.step(a[i]);
            if((i+1)%512==0) {
                int slot=cp.placement(i+1);
                if(slot>=0) {packed[slot]=original.state;cp.points[slot]={i+1,1,7};}
            }
        }
        assert(cp.points[0].pos==1024); // late chunks never evict the first anchor
        const int selected=cp.best(3200,b.size(),1,7);assert(selected>=0);
        const auto reused=cp.points[selected].pos;
        assert(reused==(cap>=3 ? 2048u : 1024u)); // longest retained valid prefix
        assert(cp.best(3200,b.size(),2,7)<0 && cp.best(3200,b.size(),1,8)<0);
        assert(cp.best(100,b.size(),1,7)<0);
        Toy restored=original;restored.state=packed[selected];
        restored.kv.resize(reused);restored.mtp.resize(reused);
        for(size_t i=reused;i<b.size();++i)restored.step(b[i]);
        Toy fresh;for(int token:b)fresh.step(token);
        assert(restored.state==fresh.state && restored.kv==fresh.kv && restored.mtp==fresh.mtp);
        cp.rewind(reused,1,8);
        for(auto p:cp.points)assert(!p.pos || (p.pos<=reused && p.epoch==8));
        assert(cp.best(3200,b.size(),1,7)<0);
        cp.clear();assert(cp.best(3200,b.size(),1,8)<0);
    }
    // Real pool invalidation/identity lifecycle with mock backing: eviction,
    // reset and replay must destroy snapshots as well as current cache state.
    UnifiedKvPool pool(8192,16384);Adapter owners[4];KvIdentity ids[4];
    for(int s=0;s<4;++s) {
        ids[s]=pool.add(owners[s]);assert(pool.begin(ids[s],4096));
        assert(pool.grow(ids[s],4096,[]{return UINT64_MAX;}));
        auto info=pool.session(s);owners[s].cp.points={{1024,info.id.generation,info.epoch}};
        assert(pool.end(ids[s],4096));
    }
    assert(pool.begin(ids[3],8192));
    assert(pool.session(0).state==KvResidency::evicted);
    assert(!owners[0].cp.points[0].pos);
    assert(pool.end(ids[3],4096));
    const auto old=ids[1];assert(pool.reset(ids[1],ids[1]));assert(ids[1].generation!=old.generation);
    assert(!owners[1].cp.points[0].pos);
    assert(pool.begin(ids[2],4096,true));assert(!owners[2].cp.points[0].pos);
    assert(pool.end(ids[2],0));
    std::cout<<"Q36 checkpoint prefix/suffix equivalence (toy), retention, budgets, generations, replay and eviction passed\n";
}
