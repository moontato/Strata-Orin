#include "strata/core/q36_vision_preallocation.hpp"
#include "strata/core/unified_kv_capacity.hpp"
#include <cassert>
#include <iostream>
using namespace strata::core;
template<class F> void fails(F f) {bool bad=false;try{f();}catch(const std::exception&){bad=true;}assert(bad);}
int main(int argc,char** argv) {
    assert(argc==2);const std::string path=std::string(argv[1])+"/q36-vision-test.sve";
    const auto budget=Q36VisionBudget::make(262144,2048,512);
    assert(budget.image_rows==262144 && budget.embedding_bytes==2147483648ULL && budget.mrope_bytes==3145728);
    assert(budget.map_bytes==2048 && budget.pinned_map_bytes==2048);
    assert(Q36VisionBudget::make(262144,2048,512,1024).embedding_bytes==8388608);
    fails([]{Q36VisionBudget::make(10,2,4,11);});fails([]{Q36VisionBudget::make(0,2,4);});
    auto write=[&](bool nan=false,bool truncate=false) {
        std::ofstream out(path,std::ios::binary|std::ios::trunc);
        for(const auto& grid:std::vector<std::vector<int32_t>>{{0x31455653,4,2,2,2},{0x31455653,6,3,2,2}}) {
            out.write((const char*)grid.data(),20);std::vector<float> data((size_t)grid[1]*2,0.5f);
            if(nan) data[0]=std::numeric_limits<float>::quiet_NaN();
            out.write((const char*)data.data(),(std::streamsize)(data.size()*4-(truncate?1:0)));
        }
    };
    std::vector<int32_t> ids={42};ids.insert(ids.end(),4,q36_image_pad);ids.push_back(43);ids.push_back(44);
    ids.insert(ids.end(),6,q36_image_pad);ids.push_back(45);
    write();auto input=q36_read_vision(path,ids,2,32,16,{17,19});
    assert(input.images.size()==2 && input.rows.size()==20 && input.cell_row.size()==14 && input.delta==-5);
    assert(input.mrope[5*3]==3 && input.mrope[7*3]==5 && input.mrope[13*3]==8);
    assert(input.cut_image(3) && input.cut_image(8) && !input.cut_image(5) && !input.cut_image(7));
    assert(input.prefix_delta(5)==-2 && input.prefix_delta(7)==-2 && input.prefix_delta(14)==-5);
    fails([&]{input.prefix_delta(3);}); // never retain/copy a partial-image prefix
    assert(input.cell_row[0]==-1 && input.cell_row[1]==0 && input.cell_row[7]==4);
    fails([&]{q36_read_vision(path,ids,2,32,8);});fails([&]{q36_read_vision(path,ids,3,32,16);});
    fails([&]{q36_read_vision(path,ids,2,32,16,{1});});
    auto bad=ids;bad[2]=10;fails([&]{q36_read_vision(path,bad,2,32,16);});
    bad=ids;bad.back()=q36_image_pad;fails([&]{q36_read_vision(path,bad,2,32,16);});
    write(true);fails([&]{q36_read_vision(path,ids,2,32,16);});write(false,true);fails([&]{q36_read_vision(path,ids,2,32,16);});
    write();{std::fstream out(path,std::ios::in|std::ios::out|std::ios::binary);out.seekp(20);float v=0.25f;out.write((char*)&v,4);}
    assert(q36_read_vision(path,ids,2,32,16,{17,19}).images[0].hash!=input.images[0].hash);
    // One shared prefill lease; no second slot can overwrite its buffer. Reset
    // another slot leaves the owner intact; generation/epoch changes are stale.
    Q36VisionLease lease;lease.acquire(0,3,7);lease.check(0,3,7);lease.invalidate(2);assert(lease.busy());
    fails([&]{lease.check(1);});fails([&]{lease.acquire(1,3,7);});fails([&]{lease.check(0,3,8);});
    lease.invalidate(0);assert(!lease.busy());
    int32_t deltas[4]={-5,0,-2,0};
    for(int slot=0;slot<4;++slot) {lease.acquire(slot,1,slot);lease.check(slot,1,slot);lease.release();}
    assert(deltas[0]==-5 && deltas[1]==0 && deltas[2]==-2 && deltas[3]==0); // staging release is NOT positional reset
    // Existing production calculator and exact layouts, real requested 524288
    // logical ceiling. M-RoPE changes coordinates, not physical cache indices.
    constexpr uint64_t ctx=262144,gran=2u<<20;std::vector<KvCapacityLayout> layouts;
    for(int l=0;l<11;++l) for(int p=0;p<4;++p) {const uint64_t bytes=p<2?256:8;layouts.push_back({{bytes,1,ctx,2,ctx*bytes},KvBufferKind::attention});}
    for(int l=0;l<30;++l) {layouts.push_back({{98304,UINT64_MAX,1,1,0},KvBufferKind::state});layouts.push_back({{2097152,UINT64_MAX,1,1,0},KvBufferKind::state});}
    layouts.push_back({{8192,UINT64_MAX,1,1,0},KvBufferKind::state});
    for(int i=0;i<4;++i) layouts.push_back({{65871872,UINT64_MAX,1,1,0},KvBufferKind::checkpoint});
    const auto bound=kv_capacity_bound(layouts,ctx,524288,4,gran);
    assert(bound.attention_pages==3168 && bound.state_pages==244 && bound.checkpoint_pages==512);
    assert(bound.bytes(gran)==8229224448ULL);
    uint64_t witness=0;for(auto r:{253945u,253945u,8185u,8185u}) for(const auto& l:layouts) if(l.kind==KvBufferKind::attention)
        witness+=l.prefix.chunks(kv_retained_cells(r,ctx),gran).size();
    assert(witness==bound.attention_pages);
    std::cout<<"Q36_VISION_PREALLOCATION_TEST PASS CPU reader/positions/lease/budget; bank=8229224448 at 2MiB granularity; NOT GPU inference\n";
}
