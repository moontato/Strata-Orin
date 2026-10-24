// Production pool + adapter + bank + VmmRange. In CPU CI only the CUDA driver
// boundary is replaced; the same test is also a native CUDA (no-model) gate.
#include "strata/core/unified_kv_vmm.hpp"
#include "strata/core/unified_kv_protocol.hpp"
#include "strata/core/q36_prefix_checkpoint.hpp"
#include <cassert>
#include <iostream>
using namespace strata::core;
static uint64_t no_fresh_memory() { return 0; }
static void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
static void zero(const void* p) {
    uint8_t bytes[32]; check(cudaMemcpy(bytes,p,sizeof bytes,cudaMemcpyDefault));
    for (auto b:bytes) assert(b==0);
}
static void capacity() {
    constexpr uint64_t ctx=262144, gran=2u<<20;
    std::vector<KvCapacityLayout> layouts;
    for (int layer=0;layer<11;++layer) for (int plane=0;plane<4;++plane) {
        const uint64_t b=plane<2 ? 256:8;
        layouts.push_back({{b,1,ctx,2,ctx*b},KvBufferKind::attention});
    }
    const auto bound=kv_capacity_bound(layouts,ctx,500000,4,gran);
    assert(bound.attention_pages==3036 && bound.bytes(gran)==6366953472ULL);
    // This skewed distribution reaches the bound: rounding and speculative
    // tails, not an equal four-way partition or 500000 * bytes/token.
    uint64_t pages=0;
    for (auto r:{253945u,245753u,1u,1u}) for(const auto& p:layouts)
        pages+=p.prefix.chunks(kv_retained_cells(r,ctx),gran).size();
    assert(pages==bound.attention_pages);
    assert(bound.attention_pages < 4*44*(32+1));
    auto full=layouts;
    for(int layer=0;layer<30;++layer) {
        full.push_back({{98304,UINT64_MAX,1,1,0},KvBufferKind::state});
        full.push_back({{2097152,UINT64_MAX,1,1,0},KvBufferKind::state});
    }
    full.push_back({{8192,UINT64_MAX,1,1,0},KvBufferKind::state});
    for(int i=0;i<4;++i) full.push_back({{65871872,UINT64_MAX,1,1,0},KvBufferKind::checkpoint});
    const auto all=kv_capacity_bound(full,ctx,500000,4,gran);
    assert(all.state_pages*gran==511705088 && all.checkpoint_pages*gran==1073741824);
    assert(all.bytes(gran)==7952400384ULL);
    // Unaligned heads, coalesced scale granules, odd context and fragmented
    // distributions: exhaustive independent enumeration of every reservation.
    const std::vector<KvCapacityLayout> small={{{3,1,17,2,53},KvBufferKind::attention},
        {{1,1,17,2,19},KvBufferKind::attention},{{7,UINT64_MAX,1,1,0},KvBufferKind::state}};
    for (uint64_t budget:{1u,13u,31u,68u}) {
        uint64_t oracle=0;
        for(uint64_t a=0;a<=17;++a) for(uint64_t b=0;b<=17;++b)
        for(uint64_t c=0;c<=17;++c) for(uint64_t d=0;d<=17;++d) {
            if(a+b+c+d>budget) continue;
            uint64_t n=0;
            for(auto r:{a,b,c,d}) for(const auto& p:small) if(p.kind==KvBufferKind::attention) {
                const auto padded=r ? std::min<uint64_t>(17,((std::min<uint64_t>(17,r+2)+3)/4)*4):0;
                n+=p.prefix.chunks(padded,16).size();
            }
            oracle=std::max(oracle,n);
        }
        const auto exact=kv_capacity_bound(small,17,budget,4,16,2,4);
        assert(exact.attention_pages==oracle && exact.state_pages==4);
    }
    for(uint64_t g:{65536u,2097152u,4194304u}) {
        const auto n=kv_capacity_bound(layouts,ctx,500000,4,g);
        uint64_t full=0; for(const auto& l:layouts) full+=l.prefix.chunks(ctx,g).size();
        assert(n.attention_pages<=4*full && n.bytes(g)>=5808000000ULL);
    }
}
static void sharing() {
    const uint64_t ctx=262144, gran=vmm_granularity();
    UnifiedKvBank bank;
    UnifiedKvVmm a(ctx,&bank), b(ctx,&bank), c(ctx,&bank), d(ctx,&bank);
    UnifiedKvVmm* adapters[]={&a,&b,&c,&d};
    uint8_t* ptr[4]{}; unsigned invalidations[4]{};
    std::vector<uint8_t*> samples[4];
    for(int s=0;s<4;++s) {
#if defined(STRATA_KV_FAKE_DRIVER)
        ptr[s]=(uint8_t*)adapters[s]->reserve({1,1,ctx,2,ctx});
        samples[s]={ptr[s],ptr[s]+ctx};
#else
        // Real INT8 Q36 planes, ten main attention layers + MTP, no model prefill.
        for(int layer=0;layer<11;++layer) for(int plane=0;plane<4;++plane) {
            const uint64_t bytes=plane<2 ? 256:8;
            auto* p=(uint8_t*)adapters[s]->reserve({bytes,1,ctx,2,ctx*bytes});
            if(!layer && !plane) ptr[s]=p;
            samples[s].push_back(p); samples[s].push_back(p+ctx*bytes);
        }
#endif
        samples[s].push_back((uint8_t*)adapters[s]->reserve({4096,UINT64_MAX,1,1,0},false,KvBufferKind::state));
        samples[s].push_back((uint8_t*)adapters[s]->reserve({4096,UINT64_MAX,1,1,0},false,KvBufferKind::checkpoint));
        adapters[s]->invalidate=[&,s](std::string&) { ++invalidations[s]; return true; };
    }
    auto zero_slot=[&](int s) { for(auto* p:samples[s]) zero(p); };
    const auto bound=kv_capacity_bound(a.capacity_layouts(),ctx,500000,4,gran);
#if defined(STRATA_KV_FAKE_DRIVER)
    bank.preallocate(bound,UINT64_MAX,UINT64_MAX);
#else
    bank.preallocate(bound,UINT64_MAX,unified_kv_available());
#endif
    UnifiedKvPool pool(ctx,500000,bound.bytes(gran),&bank);
    KvIdentity ids[4]; for(int s=0;s<4;++s) ids[s]=pool.add(*adapters[s]);
    const auto allocations=bank.backing_info().allocations;
    const auto driver=vmm_allocation_stats();
    const auto bytes=bank.backing_info().bytes;
    for(int turn=0;turn<4;++turn) {
        // Each slot individually reaches 262144; reset returns all pages but
        // neither destroys the bank nor makes a new physical allocation.
        assert(pool.begin(ids[turn],ctx,true));
        assert(pool.grow(ids[turn],ctx,no_fresh_memory)); zero_slot(turn);
        for(auto* p:samples[turn]) check(cudaMemset(p,0x79,32));
        check(cudaDeviceSynchronize());
        assert(pool.end(ids[turn],ctx));
        const auto owner=pool.session((uint32_t)turn);
        const Q36PrefixCheckpoints::Point old_point{1,owner.id.generation,owner.epoch};
        assert(Q36PrefixCheckpoints::available(old_point,owner,ctx,ctx,false,false,true,&pool));
        assert(pool.reset(ids[turn],ids[turn]));
        assert(pool.begin(ids[turn],1)); assert(pool.grow(ids[turn],1,no_fresh_memory)); zero_slot(turn);
        assert(pool.end(ids[turn],1));
        assert(!Q36PrefixCheckpoints::available(old_point,pool.session((uint32_t)turn),ctx,ctx,false,false,true,&pool));
        assert(pool.reset(ids[turn],ids[turn]));
        assert(pool.info().mapped_bytes==0 && pool.info().committed_bytes==bytes);
    }
    const uint64_t fragmented[]={8193,253945,237853,9};
    for(int s=0;s<4;++s) {
        assert(pool.begin(ids[s],fragmented[s],true));
        assert(pool.grow(ids[s],kv_retained_cells(fragmented[s],ctx),no_fresh_memory));
    }
    assert(pool.info().reserved_tokens==500000);
    if(gran==(2u<<20)) {
        assert(pool.info().bank.free_pages==0);
        assert(!bank.take()); // reaches the rounded bound, not just the raw token size
    }
    for(int s=0;s<4;++s) {
        assert(pool.end(ids[s],fragmented[s])); assert(pool.reset(ids[s],ids[s]));
    }
    const uint64_t dist[]={262144,237854,1,1};
    for(int s=0;s<4;++s) { assert(pool.begin(ids[s],dist[s],true)); assert(pool.grow(ids[s],dist[s],no_fresh_memory)); zero(ptr[s]); }
    assert(pool.info().reserved_tokens==500000);
    assert(pool.reset(ids[0],ids[0]).code==KvCode::busy); // active owners protected
    for(int s=0;s<4;++s) assert(pool.end(ids[s],dist[s]));
    {
        KvIdlePin pin(&pool,ids[0]);
        assert(pool.begin(ids[1],ctx).code==KvCode::pool_full);
        assert(pool.session(0).state==KvResidency::active && a.committed_bytes());
    }
    for(int s=0;s<4;++s) assert(pool.reset(ids[s],ids[s]));
    assert(pool.begin(ids[0],ctx)); assert(pool.grow(ids[0],ctx,no_fresh_memory));
    assert(pool.begin(ids[1],ctx).code==KvCode::pool_full); // no 524288-token promotion
    assert(pool.end(ids[0],ctx));
    assert(pool.begin(ids[1],ctx)); // idle slot 0 evicted, invalidated before reuse
    assert(pool.session(0).state==KvResidency::evicted && invalidations[0]>0);
    assert(pool.grow(ids[1],ctx,no_fresh_memory)); zero(ptr[1]);
    // Simulates warmup/reset and a checkpoint rewind: old full-context mapping
    // cannot survive with a tiny logical reservation and exhaust another owner.
    assert(pool.end(ids[1],1)); assert(b.mapped_cells()<=kv_retained_cells(1,ctx));
    assert(pool.info().committed_bytes==bytes);
    for(int s=0;s<4;++s) assert(pool.reset(ids[s],ids[s]));
    assert(pool.begin(ids[0],1,true));
    assert(pool.grow(ids[0],ctx,no_fresh_memory).code==KvCode::context_limit);
    assert(pool.grow(ids[0],1,no_fresh_memory)); zero(ptr[0]);
    assert(pool.end(ids[0],1)); assert(pool.reset(ids[0],ids[0]));
    const auto info=pool.info();
    assert(info.bank.allocations==allocations && info.bank.free_pages==bound.pages() && !info.bank.mapped_pages);
    assert(info.bank.driver_allocations==driver.allocations && info.bank.driver_allocation_attempts==driver.attempts);
    assert(info.bank.driver_releases==driver.releases);
    kv_print_status(pool,4,0);
}
static void lazy() {
    const auto gran=vmm_granularity();
    UnifiedKvVmm adapter(512); auto* p=(uint8_t*)adapter.reserve({1,1,512,1,0});
    UnifiedKvPool pool(512,512); auto id=pool.add(adapter);
    assert(pool.begin(id,512));
    const auto before=vmm_allocation_stats();
    assert(pool.grow(id,1,no_fresh_memory).code==KvCode::pool_full);
    assert(vmm_allocation_stats().allocations==before.allocations);
    assert(pool.grow(id,1,[&] { return gran; })); zero(p);
    assert(pool.info().committed_bytes==gran && pool.info().bank.granularity==0);
    assert(pool.end(id,0)); assert(adapter.committed_bytes()==gran); // lazy retains high-water backing
    assert(pool.reset(id,id)); assert(pool.info().committed_bytes==0);
    assert(vmm_allocation_stats().allocations==before.allocations+1);
    assert(vmm_allocation_stats().releases==before.releases+1);
}
static void exhaustion() {
    const auto gran=vmm_granularity();
    UnifiedKvBank bank; KvCapacityBound bound; bound.attention_pages=1;
    bank.preallocate(bound,gran,gran);
    UnifiedKvVmm a(512,&bank),b(512,&bank);
    (void)a.reserve({1,1,512,1,0}); (void)b.reserve({1,1,512,1,0});
    UnifiedKvPool pool(512,1024,gran,&bank);
    auto x=pool.add(a),y=pool.add(b);
    assert(pool.begin(x,512)); assert(pool.begin(y,512));
    assert(pool.grow(x,512,no_fresh_memory));
    const auto before=bank.backing_info().allocations;
    assert(pool.grow(y,512,no_fresh_memory).code==KvCode::pool_full);
    assert(!bank.take()); // no fallback to vmm_chunk_new
    assert(bank.backing_info().allocations==before && pool.info().mapped_bytes==gran);
    assert(pool.end(x,0)); // releases all tails
    assert(pool.grow(y,512,no_fresh_memory)); assert(pool.end(y,0));
    assert(pool.info().bank.free_pages==1);
}
#if defined(STRATA_KV_FAKE_DRIVER)
static void failures() {
    const auto gran=vmm_granularity(); const auto baseline=fake_cuda::live_handles;
    for(int mode=0;mode<3;++mode) {
        { UnifiedKvBank bank; KvCapacityBound n; n.attention_pages=3;
          if(mode==0) fake_cuda::fail_create=2;
          bool failed=false;
          try { bank.preallocate(n,mode==1 ? gran:3*gran,mode==2 ? gran:3*gran); }
          catch(const std::runtime_error&) { failed=true; }
          assert(failed); assert(bank.backing_info().allocations==(mode==0 ? 1u:0u)); }
        assert(fake_cuda::live_handles==baseline);
    }
    for(int mode=0;mode<5;++mode) {
        UnifiedKvBank bank; KvCapacityBound n; n.attention_pages=4; bank.preallocate(n,4*gran,4*gran);
        UnifiedKvVmm a(3*gran,&bank);
        auto* p=(uint8_t*)a.reserve({1,1,3*gran,1,0});
        UnifiedKvPool pool(3*gran,3*gran,4*gran,&bank); auto id=pool.add(a);
        assert(pool.begin(id,3*gran)); assert(pool.grow(id,gran,no_fresh_memory));
        check(cudaMemset(p,0x31,32));
        const auto creates=fake_cuda::creates, releases=fake_cuda::releases;
        if(mode==0) fake_cuda::fail_map=2;
        if(mode==1) fake_cuda::fail_access=2;
        if(mode==2) fake_cuda::fail_zero=2;
        if(mode==3) { fake_cuda::fail_access=1; fake_cuda::fail_unmap=-1; }
        if(mode==4) { fake_cuda::fail_map=2; fake_cuda::fail_unmap=-1; }
        const auto r=pool.grow(id,3*gran,no_fresh_memory);
        assert(!r);
        if(mode<3) { assert(a.committed_bytes()==gran && !a.faulted()); }
        else { assert(a.faulted() && pool.session(0).state==KvResidency::faulted); }
        assert(fake_cuda::creates==creates && fake_cuda::releases==releases);
        const auto i=pool.info(); assert(i.committed_bytes==4*gran);
        assert(i.mapped_bytes+i.bank.free_pages*gran==i.bank.bytes);
        uint8_t value=0; check(cudaMemcpy(&value,p,1,cudaMemcpyDefault)); assert(value==0x31);
        fake_cuda::fail_unmap=0;
        if(mode<3) { assert(pool.end(id,gran)); }
        assert(pool.reset(id,id)); assert(pool.info().mapped_bytes==0);
        assert(pool.begin(id,gran,true)); assert(pool.grow(id,gran,no_fresh_memory)); zero(p);
        assert(pool.end(id,0)); assert(pool.reset(id,id));
    }
    assert(fake_cuda::live_handles==baseline);
    {
        UnifiedKvBank bank; KvCapacityBound n; n.attention_pages=3; bank.preallocate(n,3*gran,3*gran);
        UnifiedKvVmm a(3*gran,&bank); (void)a.reserve({1,1,3*gran,1,0});
        UnifiedKvPool pool(3*gran,3*gran,3*gran,&bank); auto id=pool.add(a);
        bool valid=true; a.invalidate=[&](std::string&) { valid=false; return true; };
        assert(pool.begin(id,3*gran)); assert(pool.grow(id,3*gran,no_fresh_memory)); assert(pool.end(id,3*gran));
        fake_cuda::fail_unmap=2; assert(!pool.reset(id,id));
        assert(!valid && pool.session(0).state==KvResidency::faulted);
        assert(pool.begin(id,1).code==KvCode::backend);
        const auto i=pool.info(); assert(i.mapped_bytes==2*gran && i.bank.free_pages==1 && i.committed_bytes==3*gran);
        assert(pool.reset(id,id)); assert(pool.info().mapped_bytes==0 && pool.info().bank.free_pages==3);
    }
    assert(fake_cuda::live_handles==baseline);
    // Legacy owning mode still releases physical backing on discard, and
    // retains a failed release charge until the next successful attempt.
    { UnifiedKvVmm lazy(gran); (void)lazy.reserve({1,1,gran,1,0}); std::string error;
      assert(lazy.grow(1,error)); fake_cuda::fail_release=1;
      assert(!lazy.discard(error) && lazy.committed_bytes()==gran);
      assert(lazy.discard(error) && !lazy.committed_bytes()); }
    assert(fake_cuda::live_handles==baseline);
}
#endif
int main() {
    int devices=0; if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices) { std::cout<<"SKIP: CUDA required\n"; return 77; }
    check(cudaSetDevice(0)); if(!vmm_available()) return 77;
    capacity(); sharing(); exhaustion(); lazy();
#if defined(STRATA_KV_FAKE_DRIVER)
    failures(); assert(fake_cuda::live_handles==0 && fake_cuda::creates==fake_cuda::releases);
    assert(vmm_allocation_stats().allocations==fake_cuda::creates && vmm_allocation_stats().releases==fake_cuda::releases);
#endif
    std::cout<<"KV_PREALLOCATION_TEST PASS\n";
}
