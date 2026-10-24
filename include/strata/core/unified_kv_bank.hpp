#pragma once
#include "strata/core/unified_kv.hpp"
#include "strata/core/unified_kv_capacity.hpp"
#include "strata/core/vmm.hpp"
#include <unordered_map>

namespace strata::core {
// Engine-owned, synchronous (pool lock/engine worker). Owns ALL handles, including
// those lent to ranges. Ranges never release borrowed handles. Failed unmaps keep
// the loan charged and unavailable; destruction leaks such handles rather than
// releasing physical memory still referenced by a device VA.
class UnifiedKvBank final : public KvBacking {
    struct Page { VmmChunk handle; bool lent = false; };
    std::vector<Page> pages_;
    std::vector<size_t> free_;
    std::unordered_map<VmmChunk, size_t> index_;
    KvBackingInfo info_;
    bool initialized_ = false, ready_ = false;
    uint64_t lent_ = 0;
    static uint64_t elapsed(std::chrono::steady_clock::time_point start) {
        return (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start).count();
    }
public:
    UnifiedKvBank() { info_.granularity = vmm_granularity(); }
    ~UnifiedKvBank() override {
        for (const auto& p : pages_) if (!p.lent) (void) vmm_chunk_release(p.handle);
    }
    UnifiedKvBank(const UnifiedKvBank&) = delete;
    UnifiedKvBank& operator=(const UnifiedKvBank&) = delete;
    // No retry/fallback after partial failure. Already-created handles remain
    // owned/accounted until destruction (also when the driver refuses release).
    void preallocate(const KvCapacityBound& bound, uint64_t byte_limit, uint64_t available) {
        if (initialized_) throw std::logic_error("KV bank startup already attempted");
        initialized_ = true;
        const auto bytes = bound.bytes(info_.granularity);
        if (!info_.granularity || !bytes || bytes > byte_limit || bytes > available)
            throw std::runtime_error("--kv-preallocate: physical bank needs " + std::to_string(bytes) +
                " bytes; exceeds --kv-pool-mib or fresh memory after headroom (no lazy fallback)");
        const auto count = (size_t) bound.pages();
        pages_.reserve(count); free_.reserve(count); index_.reserve(count);
        info_.state_bytes = kv_mul(bound.state_pages, info_.granularity);
        info_.checkpoint_bytes = kv_mul(bound.checkpoint_pages, info_.granularity);
        const auto start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < count; ++i) {
            ++info_.allocation_attempts;
            const auto h = vmm_chunk_new();
            if (!h) {
                info_.allocation_us = elapsed(start);
                throw std::runtime_error("--kv-preallocate: CUDA physical page allocation failed at " +
                    std::to_string(i) + "/" + std::to_string(count) + "; startup aborted (no fallback)");
            }
            pages_.push_back({h, false});
            ++info_.allocations; info_.bytes += info_.granularity;
            // Host bookkeeping was reserved above. If insertion itself throws,
            // pages_ still owns the physical handle for destructor cleanup.
            index_.emplace(h, i); free_.push_back(i);
        }
        info_.allocation_us = elapsed(start);
        ready_ = true;
    }
    KvBackingInfo backing_info() const override {
        auto out = info_; out.free_pages = pages_.size() - lent_; out.mapped_pages = lent_;
        const auto driver = vmm_allocation_stats();
        out.driver_allocations = driver.allocations; out.driver_allocation_attempts = driver.attempts;
        out.driver_releases = driver.releases;
        return out;
    }
    VmmChunk take() {
        if (!ready_) return 0;
        if (free_.empty()) return 0; // NEVER asks the driver for another page
        const auto i = free_.back(); free_.pop_back(); pages_[i].lent = true; ++lent_; return pages_[i].handle;
    }
    void put(VmmChunk h) {
        const auto i = index_.at(h);
        if (!pages_[i].lent) throw std::logic_error("KV bank double return");
        pages_[i].lent = false; --lent_; free_.push_back(i);
    }
    void mapped(uint64_t us) { ++info_.maps; info_.mapping_us += us; }
    void unmapped() { ++info_.unmaps; }
    void initialized(uint64_t pages, uint64_t us) { info_.initialized_pages += pages; info_.initialization_us += us; }
    void checkpoint_copy(uint64_t bytes, uint64_t us) {
        ++info_.checkpoint_operations; info_.checkpoint_copy_bytes += bytes; info_.checkpoint_copy_us += us;
    }
};
} // namespace strata::core
