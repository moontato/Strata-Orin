#pragma once
#include "strata/core/unified_kv_layout.hpp"
#include "strata/core/unified_kv_protocol.hpp"
#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::core {
// Metadata only. Payloads and attention prefixes are owned by the SAME session
// VMM adapter. The engine supplies live metadata/backing to available() below.
class Q36PrefixCheckpoints {
public:
    enum class Policy { fixed, adaptive };
    static Policy parse_policy(const std::string& value) {
        if (value=="fixed") return Policy::fixed;
        if (value=="adaptive") return Policy::adaptive;
        throw std::invalid_argument("--kv-checkpoint-policy must be fixed or adaptive");
    }
    static const char* name(Policy p) { return p==Policy::adaptive ? "adaptive" : "fixed"; }
    // Advisory only: no token content, state, identity or permission lives here.
    // Eight bounded candidates; frequency saturates at eight; hints expire after
    // eight OTHER rewrite observations. Append-only requests do not age them.
    struct Learning {
        struct Hint { uint64_t common=0, seen=0; unsigned hits=0; };
        std::array<Hint,8> hints{};
        uint64_t rewrites=0;
        bool observe(uint64_t common,uint64_t previous,uint64_t chunk) {
            if (!chunk || !common || common>=previous) return false;
            ++rewrites;
            for (auto& h:hints) if (h.hits && rewrites-h.seen>8) h={};
            for (auto& h:hints) if (h.hits && h.common/chunk==common/chunk) {
                h.common=std::min(h.common,common);h.seen=rewrites;h.hits=std::min(8u,h.hits+1);
                return true;
            }
            auto* victim=&hints[0];
            for (auto& h:hints)
                if (h.hits<victim->hits || (h.hits==victim->hits && h.seen<victim->seen)) victim=&h;
            *victim={common,rewrites,1};
            return true;
        }
    };
    struct Point { uint64_t pos=0, generation=0, epoch=0; };
    Policy policy=Policy::fixed;
    Learning learning;
    std::array<uint64_t,2> targets{}; // per-prefill, actual future process() chunk ends
    std::vector<Point> points;
    uint64_t interval=1024;
    // Shared by the engine and CPU lifecycle regression. Discovery has no lease
    // witness and accepts idle owners only. Restoration may use its own admitted
    // destination, or the exact idle pin this operation still holds for a donor.
    static bool available(Point p, const KvSessionInfo& owner, uint64_t live, uint64_t position,
                          bool has_images, bool decoding, bool backed, const UnifiedKvPool* pool,
                          const KvIdlePin* donor_pin=nullptr, const KvIdentity* destination_lease=nullptr) {
        const bool own_destination = destination_lease && destination_lease->slot==owner.id.slot &&
                                     destination_lease->generation==owner.id.generation;
        const bool own_donor = donor_pin && donor_pin->holds(pool,owner.id);
        return pool && !has_images && !decoding && p.pos && p.pos<=live && p.pos<=position &&
               p.generation==owner.id.generation && p.epoch==owner.epoch && backed &&
               (owner.state==KvResidency::idle ||
                (owner.state==KvResidency::active && (own_destination || own_donor)));
    }
    static size_t capacity(uint64_t budget, uint64_t payload, uint64_t gran, unsigned sessions) {
        if (!budget || !payload || !sessions) return 0;
        const auto rounded=kv_mul(kv_ceil(payload,gran),gran);
        return (size_t) std::min<uint64_t>(4,budget/sessions/rounded);
    }
    void clear() { for (auto& p:points) p={}; learning={};targets={}; }
    // Plan on the UNCHANGED process() grid for this suffix. A hint cannot create
    // a missing past state. No split chunks, fifth payload or GPU scratch.
    void plan(uint64_t start,uint64_t incoming,uint64_t chunk) {
        targets={};
        if (policy!=Policy::adaptive || !chunk || incoming<=start || points.size()<3) return;
        auto ranked=learning.hints;
        std::sort(ranked.begin(),ranked.end(),[](auto a,auto b) {
            return a.hits!=b.hits ? a.hits>b.hits : a.seen>b.seen;
        });
        size_t n=0, limit=points.size()-2;
        for (auto h:ranked) if (h.hits && n<limit && h.common>start) {
            const auto bound=std::min(h.common,incoming-1); // restore always needs a nonempty suffix
            if (bound<=start) continue;
            const auto end=start+(bound-start)/chunk*chunk;
            if (end>=interval && end>start && std::find(targets.begin(),targets.end(),end)==targets.end())
                targets[n++]=end;
        }
        // Unlearned middle records retain the old 2x/4x safety anchors.
        std::sort(targets.begin(),targets.begin()+n);
    }
    const char* capture_reason(int index) const {
        if (!index) return "safety";
        if ((size_t)index+1==points.size()) return "recent";
        return targets[(size_t)index-1] ? "observed_divergence" : "fallback_anchor";
    }
    // Two-phase payload publication: retire before a device write, publish only
    // after successful copy+sync. Failure discards the victim, not its identity.
    Point retire(size_t index) { const auto old=points.at(index);points[index]={};return old; }
    void publish(size_t index,Point p) { points.at(index)=p; }
    // Fixed policy: first point is pinned; with 3/4 records pin 2x/4x
    // interval too. Adaptive replaces middle banks at planned targets only.
    // Only the final record rolls. Thus late prompt chunks cannot erase every
    // useful early state. Positions are actual chunk ends, never characters.
    int placement(uint64_t pos) const {
        if (points.empty() || pos<interval) return -1;
        if (policy==Policy::adaptive && points[0].pos) {
            // If a learned point currently occupies the rolling bank, copy it
            // into its learned bank before the recent bank rolls again.
            for(size_t i=1;i+1<points.size();++i)
                if(targets[i-1]==pos && points[i].pos!=pos) return (int)i;
        }
        for (auto p:points) if (p.pos==pos) return -1;
        if (!points[0].pos) return 0;
        for (size_t i=1;i+1<points.size();++i) {
            if (policy==Policy::adaptive && targets[i-1]) {
                if (pos==targets[i-1]) return (int)i;
            } else if (!points[i].pos && pos>=kv_mul(interval,uint64_t{1}<<i)) return (int)i;
        }
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
    template<class Available>
    int best_available(uint64_t common,uint64_t incoming,uint64_t generation,uint64_t epoch,Available valid) const {
        int i;
        while((i=best(common,incoming,generation,epoch))>=0) {
            if(valid(i)) return i;
            common=points[(size_t)i].pos-1;
        }
        return -1;
    }
    void rewind(uint64_t pos,uint64_t generation,uint64_t epoch) {
        for(auto& p:points) {
            if(p.pos>pos) p={};
            else if(p.pos) {p.generation=generation;p.epoch=epoch;}
        }
    }
};
} // namespace strata::core
