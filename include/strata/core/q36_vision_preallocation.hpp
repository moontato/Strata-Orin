#pragma once
// Strict-mode ancillary vision staging. Not a KV layout, checkpoint or encoder allocator.
#include "strata/core/unified_kv_layout.hpp"
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <vector>
namespace strata::core {
constexpr int32_t q36_image_pad=248056;
struct Q36VisionBudget {
    uint64_t image_rows=0,embedding_bytes=0,mrope_bytes=0,map_bytes=0,pinned_map_bytes=0;
    static Q36VisionBudget make(uint64_t context,uint64_t width,uint64_t chunk,uint64_t rows=0) {
        if(!context || context>INT32_MAX || !width || !chunk || rows>context) throw std::invalid_argument("invalid strict vision geometry");
        if(!rows) rows=context;
        return {rows,kv_mul(kv_mul(rows,width),4),kv_mul(context,12),kv_mul(chunk,4),kv_mul(chunk,4)};
    }
};
// ONE serialized prefill owns the shared staging, not four resident image copies.
// Decode uses per-session rdelta / per-row delta tables, never these image views.
struct Q36VisionLease {
    int owner=-1;uint64_t generation=0,epoch=0;
    bool busy() const {return owner>=0;}
    void acquire(int slot,uint64_t gen,uint64_t ep) {
        if(slot<0 || busy()) throw std::runtime_error("vision staging already owned");
        owner=slot;generation=gen;epoch=ep;
    }
    void check(int slot) const {if(owner!=slot) throw std::runtime_error("vision prefill owner changed");}
    void check(int slot,uint64_t gen,uint64_t ep) const {
        check(slot);if(generation!=gen || epoch!=ep) throw std::runtime_error("stale vision prefill lease");
    }
    void release() {owner=-1;generation=epoch=0;}
    void invalidate(int slot) {if(owner==slot) release();}
};
inline int32_t q36_vision_prefix_delta(const std::vector<int32_t>& mrope,uint64_t length) {
    if(length>mrope.size()/3) throw std::invalid_argument("retained prefix exceeds M-RoPE table");
    if(!length) return 0;
    const auto at=(length-1)*3;
    return (int32_t)((int64_t)std::max({mrope[at],mrope[at+1],mrope[at+2]})+1-(int64_t)length);
}
struct Q36VisionImage {int64_t cell=0,n=0;uint64_t hash=0;int32_t nx=0,ny=0;uint64_t rotary=0,row=0;};
struct Q36VisionInput {
    std::vector<float> rows;std::vector<int32_t> cell_row,mrope;
    std::vector<Q36VisionImage> images;int32_t delta=0;
    bool cut_image(uint64_t length) const {
        for(const auto& image:images) if((uint64_t)image.cell<length && length<(uint64_t)(image.cell+image.n)) return true;
        return false;
    }
    int32_t prefix_delta(uint64_t length) const {
        if(length>cell_row.size() || cut_image(length)) throw std::invalid_argument("invalid retained image prefix");
        return q36_vision_prefix_delta(mrope,length);
    }
};
inline uint64_t q36_image_hash(const void* ptr,size_t bytes,uint64_t value=1469598103934665603ull) {
    const auto* p=(const uint8_t*)ptr;for(size_t i=0;i<bytes;++i) value=(value^p[i])*1099511628211ull;return value;
}
// Bounded two-pass SVE1 reader. Validate lengths BEFORE allocating float payload.
// CPU-only: malformed input cannot modify a lease, GPU state or old cache.
inline Q36VisionInput q36_read_vision(const std::string& path,const std::vector<int32_t>& ids,
        uint64_t width,uint64_t context,uint64_t max_rows,const std::vector<uint64_t>& keys={}) {
    if(ids.empty() || ids.size()>context || !max_rows || max_rows>context || !width)
        throw std::invalid_argument("strict vision prompt/row limit invalid");
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    if(!file) throw std::invalid_argument("cannot open embeddings file");
    const auto end=file.tellg();
    const auto max_file=kv_mul(max_rows,kv_add(kv_mul(width,4),20));
    if(end<=0 || (uint64_t)end>max_file) throw std::invalid_argument("embeddings file exceeds strict vision bound or is empty");
    struct Record {int32_t n,nx,ny;uint64_t offset;};std::vector<Record> records;
    uint64_t at=0,total=0;
    while(at<(uint64_t)end) {
        int32_t h[5];file.seekg((std::streamoff)at);file.read((char*)h,sizeof h);
        if(!file || h[0]!=0x31455653 || h[1]<1 || h[2]<1 || h[3]<1 || (int64_t)h[2]*h[3]!=h[1] || h[4]!=(int64_t)width)
            throw std::invalid_argument("bad/short SVE1 header or embedding width/grid");
        total=kv_add(total,(uint64_t)h[1]);if(total>max_rows) throw std::invalid_argument("image rows exceed --kv-vision-max-tokens");
        const uint64_t offset=kv_add(at,20),bytes=kv_mul(kv_mul((uint64_t)h[1],width),4);
        if(offset>(uint64_t)end || bytes>(uint64_t)end-offset) throw std::invalid_argument("short SVE1 payload");
        records.push_back({h[1],h[2],h[3],offset});at=kv_add(offset,bytes);
    }
    if(!keys.empty() && keys.size()!=records.size()) throw std::invalid_argument("image key count differs from embeddings");
    Q36VisionInput out;out.rows.resize((size_t)kv_mul(total,width));out.cell_row.assign(ids.size(),-1);out.mrope.resize(ids.size()*3);
    uint64_t offset=0;
    for(const auto& r:records) {
        const uint64_t count=kv_mul((uint64_t)r.n,width);file.seekg((std::streamoff)r.offset);
        file.read((char*)(out.rows.data()+offset),(std::streamsize)kv_mul(count,4));if(!file) throw std::invalid_argument("short SVE1 payload read");offset+=count;
    }
    for(float v:out.rows) if(!std::isfinite(v)) throw std::invalid_argument("nonfinite image embedding");
    uint64_t cell=0,row=0,index=0,position=0;
    auto put=[&](uint64_t c,uint64_t t,uint64_t y,uint64_t x) {
        if(std::max({t,y,x})>INT32_MAX) throw std::invalid_argument("M-RoPE position overflow");
        out.mrope[c*3]=(int32_t)t;out.mrope[c*3+1]=(int32_t)y;out.mrope[c*3+2]=(int32_t)x;
    };
    while(cell<ids.size()) {
        if(ids[cell]!=q36_image_pad) {put(cell,position,position,position);++cell;++position;continue;}
        if(index>=records.size()) throw std::invalid_argument("prompt has more images than embeddings");
        const auto& r=records[index];const int64_t grid[3]={r.n,r.nx,r.ny};
        uint64_t hash=q36_image_hash(grid,sizeof grid);
        // Even supplied keys cannot authorize reuse of DIFFERENT actual rows.
        hash=q36_image_hash(out.rows.data()+row*width,(size_t)r.n*width*4,hash);
        if(!keys.empty()) hash=q36_image_hash(&keys[index],8,hash);
        out.images.push_back({(int64_t)cell,r.n,hash,r.nx,r.ny,position,row});
        for(uint64_t j=0;j<(uint64_t)r.n;++j) {
            if(cell+j>=ids.size() || ids[cell+j]!=q36_image_pad) throw std::invalid_argument("image pad run does not match embedding rows");
            put(cell+j,position,position+j/(uint64_t)r.nx,position+j%(uint64_t)r.nx);out.cell_row[cell+j]=(int32_t)(row+j);
        }
        cell+=(uint64_t)r.n;row+=(uint64_t)r.n;position+=(uint64_t)std::max(r.nx,r.ny);++index;
    }
    if(index!=records.size() || ids.back()==q36_image_pad) throw std::invalid_argument("extra embeddings or prompt ends inside/at image");
    out.delta=(int32_t)position-(int32_t)ids.size();return out;
}
} // namespace strata::core
