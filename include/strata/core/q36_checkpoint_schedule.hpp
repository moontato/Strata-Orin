#pragma once
// Developer gate metadata ONLY. No tensor pointers, checkpoint payloads or CUDA.
#include <algorithm>
#include <array>
#include <utility>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::core {
struct Q36ScheduleCall {
    std::vector<int32_t> ids; // ORIGINAL declared input, including unconsumed tail
    uint64_t start=0, take=0;
    bool checkpoint_suffix=false;
    std::vector<uint64_t> ends; // absolute callback positions actually consumed
    // Developer image oracle: ORIGINAL inputs only, never restored tensors,
    // candidate M-RoPE tables, checkpoint identity or Python-normalized content.
    std::string image_file;
    std::vector<int32_t> image_ids;
    std::vector<uint64_t> image_keys;
};
using Q36Schedule = std::vector<Q36ScheduleCall>;
inline uint64_t q36_schedule_size(const Q36Schedule& recipe) {
    return recipe.empty() ? 0 : recipe.back().start+recipe.back().take;
}
inline void q36_schedule_check(const Q36Schedule& recipe,const std::vector<int32_t>& ids,uint64_t chunk) {
    uint64_t pos=0;
    for(const auto& call:recipe) {
        if(!chunk || call.start!=pos || !call.take || call.take>call.ids.size() || call.ends.empty())
            throw std::runtime_error("invalid schedule call/declared length");
        uint64_t done=0;
        for(auto end:call.ends) {
            const auto step=call.ids.size()==1 && !call.checkpoint_suffix ? 1 :
                std::min<uint64_t>(chunk,call.ids.size()-done);
            done+=step;
            if(end!=pos+done) throw std::runtime_error("schedule chunk boundary mismatch");
        }
        if(done!=call.take || pos+done>ids.size()) throw std::runtime_error("schedule consumed length mismatch");
        for(uint64_t i=0;i<done;++i) if(call.ids[(size_t)i]!=ids[(size_t)(pos+i)])
            throw std::runtime_error("schedule original token prefix mismatch");
        pos+=done;
    }
    if(pos!=ids.size()) throw std::runtime_error("schedule does not cover rendered IDs");
}
// Ordinary singleton decode does not seed MTP KV. Undefined cells MUST NOT be
// compared as if they were state. All main KV and every initialized MTP cell are.
inline std::vector<bool> q36_schedule_mtp(const Q36Schedule& recipe) {
    std::vector<bool> valid((size_t)q36_schedule_size(recipe),false);
    for(const auto& c:recipe) if(c.ids.size()!=1 || c.checkpoint_suffix)
        for(uint64_t i=0;i<c.take;++i) valid[(size_t)(c.start+i)]=true;
    return valid;
}
struct Q36ScheduleBook {
    std::array<Q36Schedule,4> live;
    std::array<std::vector<Q36Schedule>,4> saved;
    void reset(int slot) {live.at(slot).clear();saved.at(slot).clear();}
    void begin(int slot,const int32_t* ids,int64_t n,bool suffix) {
        if(n<=0) throw std::runtime_error("schedule gate requires a nonempty suffix");
        auto& r=live.at(slot);Q36ScheduleCall call;
        call.start=q36_schedule_size(r);call.ids.assign(ids,ids+n);call.checkpoint_suffix=suffix;
        r.push_back(std::move(call));
    }
    void progress(int slot,uint64_t done) {
        auto& c=live.at(slot).back();
        if(done<=c.take || done>c.ids.size()) throw std::runtime_error("nonmonotonic schedule callback");
        c.take=done;c.ends.push_back(c.start+done);
    }
    void capture(int slot,size_t index) {
        auto& bank=saved.at(slot);if(bank.size()<=index) bank.resize(index+1);
        bank[index]=live.at(slot);
    }
    void restore(int src,size_t index,int dst) {
        const auto recipe=saved.at(src).at(index); // metadata only, copy before same-slot rewind
        auto inherited=saved.at(src);const auto end=q36_schedule_size(recipe);
        for(auto& r:inherited) if(q36_schedule_size(r)>end) r.clear();
        live.at(dst)=recipe;saved.at(dst)=std::move(inherited);
    }
    void copy(int src,int dst) {live.at(dst)=live.at(src);saved.at(dst).clear();}
};
// Adapter must reset to EMPTY, consume original declared IDs with real process()
// semantics, and stop only at the recorded callback. Never turn chunks into calls.
template<class Adapter> void q36_schedule_replay(Adapter& fresh,const Q36Schedule& recipe) {
    fresh.empty();
    for(size_t i=0;i<recipe.size();++i) fresh.call(recipe[i],i);
}
} // namespace strata::core
