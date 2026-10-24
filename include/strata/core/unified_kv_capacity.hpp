#pragma once
#include "strata/core/unified_kv_layout.hpp"
#include <map>

namespace strata::core {
enum class KvBufferKind { attention, state, checkpoint };
struct KvCapacityLayout { KvPrefix prefix; KvBufferKind kind = KvBufferKind::attention; };
struct KvCapacityBound {
    uint64_t attention_pages = 0, state_pages = 0, checkpoint_pages = 0;
    uint64_t pages() const { return kv_add(attention_pages, kv_add(state_pages, checkpoint_pages)); }
    uint64_t bytes(uint64_t gran) const { return kv_mul(pages(), gran); }
};
// A retained speculative window can be past the published prefix. The engine
// currently writes at most kVerify=8 cells, then coarsens growth to 256 cells.
inline uint64_t kv_retained_cells(uint64_t logical, uint64_t context,
                                  uint64_t padding = 8, uint64_t quantum = 256) {
    if (!logical) return 0;
    return std::min(context, kv_mul(kv_ceil(std::min(context, kv_add(logical, padding)), quantum), quantum));
}
// Exact discrete maximum for identical slot layouts. For each possible physical
// footprint retain the MINIMUM logical reservation that can require it. Min-plus
// convolution across slots then finds the largest footprint whose total minimum
// reservation fits the unchanged logical pool. No equal partition assumption.
inline KvCapacityBound kv_capacity_bound(const std::vector<KvCapacityLayout>& layouts,
        uint64_t context, uint64_t tokens, unsigned slots, uint64_t gran,
        uint64_t padding = 8, uint64_t quantum = 256) {
    if (!context || !tokens || !slots || !gran || !quantum)
        throw std::invalid_argument("invalid KV preallocation geometry");
    KvCapacityBound out;
    std::vector<KvPrefix> attention;
    for (const auto& l : layouts) {
        if (l.kind == KvBufferKind::attention) attention.push_back(l.prefix);
        else {
            const auto pages = kv_mul(l.prefix.chunks(1, gran).size(), slots);
            auto& n = l.kind == KvBufferKind::state ? out.state_pages : out.checkpoint_pages;
            n = kv_add(n, pages);
        }
    }
    std::map<uint64_t, uint64_t> options{{0, 0}};
    auto option = [&](uint64_t logical) {
        if (!logical || logical > context || logical > tokens) return;
        const auto cells = kv_retained_cells(logical, context, padding, quantum);
        uint64_t pages = 0;
        for (const auto& p : attention) pages = kv_add(pages, p.chunks(cells, gran).size());
        auto it = options.find(pages);
        if (it == options.end() || logical < it->second) options[pages] = logical;
    };
    option(1);
    // The rounded write extent changes only at these logical breakpoints.
    for (uint64_t edge = quantum; edge < kv_add(context, padding); edge = kv_add(edge, quantum))
        if (edge >= padding) option(edge - padding + 1);
    std::map<uint64_t, uint64_t> dp{{0, 0}};
    for (unsigned s = 0; s < slots; ++s) {
        std::map<uint64_t, uint64_t> next;
        for (const auto& a : dp) for (const auto& b : options) {
            const auto cost = kv_add(a.second, b.second);
            if (cost > tokens) continue;
            const auto pages = kv_add(a.first, b.first);
            auto it = next.find(pages);
            if (it == next.end() || cost < it->second) next[pages] = cost;
        }
        dp.swap(next);
    }
    out.attention_pages = dp.rbegin()->first;
    return out;
}
} // namespace strata::core
