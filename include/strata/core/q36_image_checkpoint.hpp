#pragma once
// CPU-only authorization/planning. No GPU payload, token rewriting or key trust.
#include "strata/core/q36_prefix_checkpoint.hpp"
#include <cstring>
#include <chrono>
#include <memory>
namespace strata::core {
struct Q36ImageHostTimer {
    uint64_t& total;std::chrono::steady_clock::time_point start=std::chrono::steady_clock::now();
    explicit Q36ImageHostTimer(uint64_t& value):total(value) {}
    ~Q36ImageHostTimer(){total+=(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-start).count();}
};
struct Q36ImageRecord {
    uint64_t cell=0,n=0,nx=0,ny=0,rotary=0,row=0,hash=0;
};
struct Q36ImageSource {
    static constexpr uint64_t max_rows=1024;
    uint64_t width=0;
    std::vector<float> rows; // immutable ACTUAL uploaded FP32 bytes, shared by live/points
    std::vector<Q36ImageRecord> images;
    bool boundary(uint64_t end) const {
        for(const auto& x:images) if(x.cell<end && end<x.cell+x.n) return false;
        return true;
    }
    int32_t delta(uint64_t end) const {
        if(!boundary(end)) throw std::invalid_argument("image checkpoint cuts an image");
        int64_t value=0;for(const auto& x:images) if(x.cell+x.n<=end) value+=(int64_t)std::max(x.nx,x.ny)-(int64_t)x.n;
        return (int32_t)value;
    }
    static const Q36ImageSource& empty() {static const Q36ImageSource value;return value;}
};
using Q36ImageSourcePtr=std::shared_ptr<const Q36ImageSource>;
inline const Q36ImageSource& q36_image_source(const Q36ImageSourcePtr& p) {return p?*p:Q36ImageSource::empty();}
inline bool q36_image_equal(const Q36ImageSource& a,const Q36ImageRecord& x,const Q36ImageSource& b,const Q36ImageRecord& y) {
    if(a.width!=b.width || x.cell!=y.cell || x.n!=y.n || x.nx!=y.nx || x.ny!=y.ny || x.rotary!=y.rotary) return false;
    const auto n=x.n*a.width,at=x.row*a.width,bt=y.row*b.width;
    return at<=a.rows.size() && n<=a.rows.size()-at && bt<=b.rows.size() && n<=b.rows.size()-bt &&
           !std::memcmp(a.rows.data()+at,b.rows.data()+bt,(size_t)n*4);
}
inline bool q36_image_prefix(const Q36ImageSource& a,const Q36ImageSource& b,uint64_t end) {
    if(!a.boundary(end) || !b.boundary(end)) return false;
    size_t i=0,j=0;
    while(i<a.images.size() && a.images[i].cell+a.images[i].n<=end) {
        if(j>=b.images.size() || b.images[j].cell+b.images[j].n>end || !q36_image_equal(a,a.images[i],b,b.images[j])) return false;
        ++i;++j;
    }
    return (j==b.images.size() || b.images[j].cell>=end) && a.delta(end)==b.delta(end);
}
// Token LCP remains mandatory. Identity can only REDUCE it, never extend it.
inline uint64_t q36_image_common(const Q36ImageSource& a,const Q36ImageSource& b,uint64_t tokens) {
    size_t i=0;
    while(i<a.images.size() || i<b.images.size()) {
        const auto ac=i<a.images.size()?a.images[i].cell:UINT64_MAX,bc=i<b.images.size()?b.images[i].cell:UINT64_MAX;
        const auto begin=std::min(ac,bc);if(begin>=tokens) break;
        if(i>=a.images.size() || i>=b.images.size() || !q36_image_equal(a,a.images[i],b,b.images[i])) return begin;
        if(tokens<a.images[i].cell+a.images[i].n) return begin;
        ++i;
    }
    return tokens;
}
struct Q36ImagePoint {
    Q36ImageSourcePtr source;uint64_t end=0;int32_t delta=0;bool ready=false;
    bool valid(uint64_t pos) const {return ready && end==pos && q36_image_source(source).boundary(pos) && delta==q36_image_source(source).delta(pos);}
};
// Plans on existing process callbacks only. Floor preferred; if every earlier
// callback cuts an image, defer to the first complete boundary. No chunk splits.
struct Q36ImagePlan {
    std::array<uint64_t,3> anchors{};
    void make(const Q36PrefixCheckpoints& cp,const Q36ImageSource& source,uint64_t start,uint64_t incoming,uint64_t chunk) {
        anchors={};if(!chunk || incoming<=start) return;
        std::vector<uint64_t> safe;
        for(uint64_t p=start+std::min(chunk,incoming-start);p<=incoming;) {
            if(p<incoming && source.boundary(p)) safe.push_back(p);
            if(p==incoming) break;
            p+=std::min(chunk,incoming-p);
        }
        auto nearest=[&](uint64_t wanted) {
            if(safe.empty()) return uint64_t{0};
            auto it=std::upper_bound(safe.begin(),safe.end(),wanted);
            return it==safe.begin()?safe.front():*--it;
        };
        anchors[0]=nearest(cp.interval);
        auto hints=cp.learning.hints;
        std::sort(hints.begin(),hints.end(),[](auto a,auto b){return a.hits!=b.hits?a.hits>b.hits:a.seen>b.seen;});
        size_t used=1;
        if(cp.policy==Q36PrefixCheckpoints::Policy::adaptive) for(auto h:hints) if(h.hits && h.common>start && used<3) {
            const auto p=nearest(std::min(h.common,incoming-1));
            if(p && std::find(anchors.begin(),anchors.end(),p)==anchors.end()) anchors[used++]=p;
        }
        for(size_t i=1;i<3;++i) if(!anchors[i]) anchors[i]=nearest(kv_mul(cp.interval,uint64_t{1}<<i));
    }
    int placement(const Q36PrefixCheckpoints& cp,uint64_t pos,const Q36ImageSource& source) const {
        if(!source.boundary(pos) || cp.points.empty() || !pos) return -1;
        for(auto p:cp.points) if(p.pos==pos) return -1;
        if(cp.points.size()==1) return !cp.points[0].pos && anchors[0]==pos ? 0:-1;
        for(size_t i=0;i+1<cp.points.size() && i<anchors.size();++i)
            if(anchors[i]==pos && (!cp.points[i].pos || (i && cp.policy==Q36PrefixCheckpoints::Policy::adaptive))) return (int)i;
        uint64_t last=0;for(auto p:cp.points) last=std::max(last,p.pos);
        return cp.points.size()>1 && pos>=cp.interval && pos>=last && pos-last>=cp.interval ? (int)cp.points.size()-1:-1;
    }
};
} // namespace strata::core
