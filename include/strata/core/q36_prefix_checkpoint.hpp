#pragma once
#include "strata/core/unified_kv_layout.hpp"
#include <algorithm>
#include <vector>

namespace strata::core {
// Metadata only. Payloads and attention prefixes are owned by the SAME session
// VMM adapter. The engine must also check residency/backing before using a record.
class Q36PrefixCheckpoints {
public:
    struct Point { uint64_t pos=0, generation=0, epoch=0; };
    std::vector<Point> points;
    uint64_t interval=1024;
    static size_t capacity(uint64_t budget, uint64_t payload, uint64_t gran, unsigned sessions) {
        if (!budget || !payload || !sessions) return 0;
        const auto rounded=kv_mul(kv_ceil(payload,gran),gran);
        return (size_t) std::min<uint64_t>(4,budget/sessions/rounded);
    }
    void clear() { for (auto& p:points) p={}; }
    // First point is pinned. With 3/4 records, pin 2x/4x interval too;
    // only the final record rolls. Thus late prompt chunks cannot erase every
    // useful early state. Positions are actual chunk ends, never characters.
    int placement(uint64_t pos) const {
        if (points.empty() || pos<interval) return -1;
        for (auto p:points) if (p.pos==pos) return -1;
        if (!points[0].pos) return 0;
        for (size_t i=1;i+1<points.size();++i)
            if (!points[i].pos && pos>=kv_mul(interval,uint64_t{1}<<i)) return (int)i;
        uint64_t last=0;for (auto p:points) last=std::max(last,p.pos);
        return points.size()>1 && pos>=last && pos-last>=interval ? (int)points.size()-1 : -1;
    }
    int best(uint64_t common, uint64_t incoming, uint64_t generation, uint64_t epoch) const {
        int result=-1;uint64_t longest=0;
        for(size_t i=0;i<points.size();++i) {
            const auto p=points[i];
            if(p.pos>longest && p.pos<=common && p.pos<incoming && p.generation==generation && p.epoch==epoch)
                { result=(int)i;longest=p.pos; }
        }
        return result;
    }
    void rewind(uint64_t pos,uint64_t generation,uint64_t epoch) {
        for(auto& p:points) {
            if(p.pos>pos) p={};
            else if(p.pos) {p.generation=generation;p.epoch=epoch;}
        }
    }
};
} // namespace strata::core
