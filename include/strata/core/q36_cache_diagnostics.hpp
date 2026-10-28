#pragma once
// Read-only host diagnostics. NEVER an authorization/matching/routing function.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
namespace strata::core {
struct Q36CacheImageDetail {
    int64_t cell=0,n=0;int32_t nx=0,ny=0;
    uint64_t identity=0,rows=0,key=0;bool keyed=false;
};
struct Q36CacheReason {
    size_t retained=0,incoming=0,lcp=0,retained_images=0,prefix_images=0;
    bool proper=false,tokens=false,complete=true,images=true;
    bool placement=false,identity=false,grid=false,rows=false,key=false,details=false;
    const char* reason() const {
        if(!retained) return "empty";
        if(!proper) return "not_proper_extension";
        if(!tokens) return "token_divergence";
        if(!complete) return "incomplete_image_boundary";
        if(!images) return "image_mismatch";
        return "exact_prefix";
    }
};
template<class Image> Q36CacheReason q36_cache_reason(const std::vector<int32_t>& held,
        const std::vector<int32_t>& ids,const std::vector<Image>& old,const std::vector<Image>& incoming,
        const std::vector<Q36CacheImageDetail>* a=nullptr,const std::vector<Q36CacheImageDetail>* b=nullptr) {
    Q36CacheReason d;d.retained=held.size();d.incoming=ids.size();d.retained_images=old.size();
    while(d.lcp<std::min(held.size(),ids.size()) && held[d.lcp]==ids[d.lcp]) ++d.lcp;
    d.proper=!held.empty() && held.size()<ids.size();d.tokens=d.lcp==held.size();
    std::vector<size_t> prefix;
    for(size_t i=0;i<incoming.size();++i) {
        const auto& k=incoming[i];
        if(k.cell+k.n<=(int64_t)held.size()) prefix.push_back(i);
        else if(k.cell<(int64_t)held.size()) d.complete=false;
    }
    d.prefix_images=prefix.size();d.images=d.complete && prefix.size()==old.size();
    for(size_t i=0;i<std::min(prefix.size(),old.size());++i) {
        const auto& x=old[i];const auto& y=incoming[prefix[i]];
        d.placement|=x.cell!=y.cell || x.n!=y.n;d.identity|=x.hash!=y.hash;
        d.images&=x==y;
    }
    if(a && b && a->size()==old.size() && b->size()==incoming.size()) {
        d.details=true;
        for(size_t i=0;i<std::min(a->size(),prefix.size());++i) {
            const auto& x=(*a)[i];const auto& y=(*b)[prefix[i]];
            d.grid|=x.nx!=y.nx || x.ny!=y.ny;
            d.rows|=x.rows!=y.rows;d.key|=x.keyed!=y.keyed || (x.keyed && x.key!=y.key);
        }
    }
    return d;
}
// Re-read headers ONLY, never another embedding payload. Actual validated host
// rows are fingerprinted for equality flags, never logged. Failures are diagnostic
// unavailable, not admission failures. No grid/row/key detail authorizes reuse.
template<class Image> bool q36_cache_image_details(const std::string& path,const std::vector<Image>& images,
        const std::vector<float>& rows,uint64_t width,const std::vector<uint64_t>& keys,
        std::vector<Q36CacheImageDetail>& out) {
    out.clear();if(images.size()>1024 || !width || rows.size()>1024*width) return false;
    std::ifstream file(path,std::ios::binary);if(!file) return false;
    size_t row=0;uint64_t offset=0;
    auto fingerprint=[](const void* p,size_t n) {
        uint64_t v=1469598103934665603ull;const auto* b=(const uint8_t*)p;
        for(size_t i=0;i<n;++i) v=(v^b[i])*1099511628211ull;
        return v;
    };
    for(size_t i=0;i<images.size();++i) {
        int32_t h[5];file.seekg((std::streamoff)offset);file.read((char*)h,sizeof h);
        if(!file || h[0]!=0x31455653 || h[1]!=images[i].n || h[2]<1 || h[3]<1 ||
           (int64_t)h[2]*h[3]!=h[1] || h[4]!=(int64_t)width || row+(size_t)h[1]>rows.size()/width ||
           (!keys.empty() && keys.size()!=images.size())) {out.clear();return false;}
        out.push_back({images[i].cell,images[i].n,h[2],h[3],images[i].hash,
            fingerprint(rows.data()+row*width,(size_t)h[1]*width*4),keys.empty()?0:keys[i],!keys.empty()});
        row+=(size_t)h[1];offset+=20+(uint64_t)h[1]*width*4;
    }
    return row==rows.size()/width;
}
inline std::string q36_cache_line(uint64_t request,int slot,const Q36CacheReason& d,size_t actual,
        const char* state,uint64_t generation,uint64_t epoch,int64_t pos,bool decoding,bool backed,
        bool pick_eligible,bool idle_backed,int32_t delta,int32_t incoming_delta,bool delta_known) {
    // Fixed format, counts/flags/scalars only. No tokens, hashes, paths or content.
    char line[1200];
    std::snprintf(line,sizeof line,"Q36CACHE candidate request=%llu slot=%d state=%s generation=%llu epoch=%llu position=%lld decoding=%d backed=%d pick_eligible=%d idle_backed=%d retained=%zu incoming=%zu lcp=%zu proper=%d tokens_prefix=%d complete_images=%d retained_images=%zu prefix_images=%zu image_equal=%d placement_diff=%d identity_diff=%d details=%d grid_diff=%d rows_diff=%d key_diff=%d retained_delta=%d incoming_prefix_delta=%d prefix_delta_known=%d actual_match=%zu reason=%s\n",
        (unsigned long long)request,slot,state,(unsigned long long)generation,(unsigned long long)epoch,(long long)pos,
        decoding?1:0,backed?1:0,pick_eligible?1:0,idle_backed?1:0,d.retained,d.incoming,d.lcp,d.proper?1:0,d.tokens?1:0,
        d.complete?1:0,d.retained_images,d.prefix_images,d.images?1:0,d.placement?1:0,d.identity?1:0,d.details?1:0,
        d.grid?1:0,d.rows?1:0,d.key?1:0,delta,incoming_delta,delta_known?1:0,actual,d.reason());
    return line;
}
} // namespace strata::core
