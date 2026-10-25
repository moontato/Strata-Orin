#pragma once
#include "strata/core/unified_kv.hpp"
#include "strata/core/unified_kv_bank.hpp"
#include "strata/core/unified_kv_layout.hpp"
#include "strata/core/vmm.hpp"
#include "strata/core/unified_kv_host.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/core/conversation_memory.hpp"
#include <cuda_runtime.h>
#include <functional>
#include <memory>

namespace strata::core {

// Stable addresses, no change to attention kernels. One owner per inference
// session; all main/draft tensors of that session use the same adapter.
class UnifiedKvVmm final : public KvAdapter {
    struct Buffer {
        VmmRange range;
        std::unique_ptr<UnifiedKvHostRange> host;
        KvPrefix layout;
        KvBufferKind kind = KvBufferKind::attention;
        UnifiedKvBank* bank = nullptr;
        std::vector<VmmChunk> unreleased;
        ~Buffer() {
            if (bank) for (int64_t c = 0; c < chunks(); ++c) (void) unmap(c);
            for (auto h : unreleased) (void) vmm_chunk_release(h);
        }
        uint8_t* base() const { return host ? host->base() : range.base(); }
        int64_t chunks() const { return host ? host->chunks() : range.chunks(); }
        bool mapped(int64_t c) const { return host ? host->mapped(c) : range.mapped(c); }
        uint64_t count() const { return host ? host->count() : (uint64_t) range.physical_count() + unreleased.size(); }
        bool map(int64_t c) {
            if (!bank) return host ? host->map(c) : range.map_range(c, c + 1, [] { return VmmChunk{0}; });
            const auto h = bank->take();
            if (!h) return false;
            const auto start = std::chrono::steady_clock::now();
            const bool ok = range.map_borrowed(c, h);
            bank->mapped((uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - start).count());
            if (!ok && !range.mapped(c)) bank->put(h);
            return ok;
        }
        bool unmap(int64_t c) {
            if (host) return host->unmap(c);
            const auto h = range.unmap(c);
            if (bank) {
                if (h) { bank->put(h); bank->unmapped(); }
                return !range.mapped(c);
            }
            if (h && !vmm_chunk_release(h)) { unreleased.push_back(h); return false; }
            return !range.mapped(c);
        }
    };
    std::vector<std::unique_ptr<Buffer>> buffers_;
    uint64_t cells_ = 0;
    bool faulted_ = false, quarantined_ = false;
    UnifiedKvBank* bank_ = nullptr;
    KvHostMapFailure last_host_failure_ = KvHostMapFailure::none;
public:
    std::function<bool(std::string&)> synchronize;
    std::function<bool(std::string&)> invalidate;
    // Native fault-injection gate; unset in engines. Never changes layout or kernels.
    std::function<bool()> mapping_test_hook;
    uint64_t max_cells = 0;
    explicit UnifiedKvVmm(uint64_t max, UnifiedKvBank* bank = nullptr) : bank_(bank), max_cells(max) {}
    const KvBacking* backing() const override { return bank_; }
    uint64_t write_limit(uint64_t logical) const override {
        return bank_ ? kv_retained_cells(logical, max_cells) : UINT64_MAX;
    }
    std::vector<KvCapacityLayout> capacity_layouts() const {
        std::vector<KvCapacityLayout> out;
        for (const auto& b : buffers_) out.push_back({b->layout, b->kind});
        return out;
    }
    // Fatal owner shutdown failure: preserve every referenced VA/loan until
    // process teardown, even if a subsequent stream query happens to succeed.
    void quarantine() noexcept { quarantined_ = faulted_ = true; }
    ~UnifiedKvVmm() override {
        // Owners must destroy graphs and synchronize before destroying adapters.
        // A failed synchronization cannot make freeing in-flight memory safe.
        std::string error;
        if (quarantined_ || !quiesce(error)) {
            for (auto& b : buffers_) (void) b.release(); // deliberate leak on fatal device failure
        }
    }
    void* reserve(KvPrefix layout, bool host = false, KvBufferKind kind = KvBufferKind::attention) {
        if (bank_ && host) throw std::invalid_argument("--kv-preallocate does not support host KV backing");
        auto b = std::make_unique<Buffer>(); b->layout = layout; b->kind = kind; b->bank = bank_;
        if (host) b->host = std::make_unique<UnifiedKvHostRange>();
        const bool ok = host ? b->host->reserve(layout.extent(), vmm_granularity()) : b->range.reserve(layout.extent());
        if (!ok) throw std::runtime_error("unified KV: virtual address reservation failed (streamed host backing requires Linux UVA)");
        if (!host) b->unreleased.reserve((size_t) b->chunks());
        void* base = b->base(); buffers_.push_back(std::move(b)); return base;
    }
    bool covers(uint64_t cells) const { return cells <= cells_; }
    uint64_t mapped_cells() const { return cells_; }
    uint64_t bytes_for(uint64_t cells) const override {
        if (cells > max_cells) throw std::out_of_range("unified KV write exceeds virtual context");
        uint64_t chunks = 0;
        for (const auto& b : buffers_) chunks = kv_add(chunks, b->layout.chunks(cells, vmm_granularity()).size());
        return kv_mul(chunks, vmm_granularity());
    }
    void cover(uint64_t cells) override { cells_ = std::max(cells_, cells); }
    bool faulted() const override { return faulted_; }
    KvHostMapFailure last_host_failure() const {
        return faulted_ ? KvHostMapFailure::backend : last_host_failure_;
    }
    uint64_t committed_bytes() const override {
        uint64_t chunks = 0;
        for (const auto& b : buffers_) chunks += b->count();
        return kv_mul(chunks, vmm_granularity());
    }
    bool quiesce(std::string& error) override {
        if (synchronize) {
            const bool ok = synchronize(error); faulted_ = faulted_ || !ok; return ok;
        }
        const auto e = cudaDeviceSynchronize();
        if (e == cudaSuccess) return true;
        faulted_ = true; error = cudaGetErrorString(e); return false;
    }
    bool grow(uint64_t cells, std::string& error) override {
        last_host_failure_ = KvHostMapFailure::none;
        struct Fresh { Buffer* b; int64_t chunk; };
        std::vector<Fresh> fresh;
        const uint64_t gran = vmm_granularity();
        for (auto& b : buffers_)
            for (const uint64_t c : b->layout.chunks(cells, gran))
                if (!b->mapped((int64_t) c)) fresh.push_back({b.get(), (int64_t) c});
        if (fresh.empty()) { cover(cells); return true; }
        bool ok = true;
        uint64_t initialized = 0, initialization_us = 0;
        for (const auto& f : fresh) {
            if (mapping_test_hook && !mapping_test_hook()) { error = "injected physical mapping failure"; ok = false; break; }
            if (!f.b->map(f.chunk)) {
                if (f.b->host) {
                    error = f.b->host->error();
                    last_host_failure_ = f.b->host->last_failure();
                }
                ok = false; break;
            }
            const auto start = std::chrono::steady_clock::now();
            const auto e = cudaMemset(f.b->base() + (uint64_t) f.chunk * gran, 0, (size_t) gran);
            initialization_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - start).count();
            if (e != cudaSuccess) { error = cudaGetErrorString(e); ok = false; break; }
            ++initialized;
        }
        if (ok) {
            const auto start = std::chrono::steady_clock::now();
            const auto e = cudaDeviceSynchronize();
            initialization_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - start).count();
            if (e != cudaSuccess) { error = cudaGetErrorString(e); ok = false; }
        }
        if (bank_) bank_->initialized(initialized, initialization_us);
        if (!ok) {
            if (error.empty()) error = "CUDA physical allocation/map/access failed";
            // cudaMemset may have returned before filling a freshly mapped page.
            // Even rollback must drain those writes before unmapping it.
            const auto drained = cudaDeviceSynchronize();
            if (drained != cudaSuccess) {
                error += std::string("; rollback synchronization failed: ") + cudaGetErrorString(drained);
                faulted_ = true;
                return false; // retain mappings; manager faults the owner rather than recycling in-flight pages
            }
            for (const auto& f : fresh) {
                if (!f.b->unmap(f.chunk)) { error += "; rollback unmap failed"; faulted_ = true; }
            }
            return false;
        }
        cells_ = std::max(cells_, cells);
        return true;
    }
    bool trim_reservation(uint64_t logical, std::string& error) override {
        if (!bank_) return true;
        const auto keep = kv_retained_cells(logical, max_cells);
        const auto gran = vmm_granularity();
        for (auto& b : buffers_) {
            const auto wanted = b->layout.chunks(keep, gran);
            for (int64_t c = 0; c < b->chunks(); ++c)
                if (b->mapped(c) && !std::binary_search(wanted.begin(), wanted.end(), (uint64_t) c) && !b->unmap(c)) {
                    error = "CUDA unmap failed while trimming KV reservation"; faulted_ = true; return false;
                }
        }
        cells_ = std::min(cells_, keep);
        return true;
    }
    bool discard(std::string& error) override {
        // Invalidate while buffers are still mapped; callbacks may zero recurrence.
        if (invalidate && !invalidate(error)) return false;
        for (auto& b : buffers_) {
            for (auto it = b->unreleased.begin(); it != b->unreleased.end();) {
                if (!vmm_chunk_release(*it)) { error = "CUDA physical release failed"; return false; }
                it = b->unreleased.erase(it);
            }
            for (int64_t c = 0; c < b->chunks(); ++c)
                if (!b->unmap(c)) { error = "CUDA unmap/unregister/release failed; KV not reclaimed"; return false; }
            if (!b->host && !b->range.release_unmapped()) { error = "CUDA failed to release unmapped physical memory"; return false; }
        }
        cells_ = 0;
        faulted_ = false;
        return true;
    }
};

// Fresh free bytes already account for weights, resident experts, scratch,
// graphs, other CUDA allocations and (on Tegra) cgroup-limited shared RAM.
// Keep the release's reserve; never add CPU free RAM to CUDA free memory.
inline uint64_t unified_kv_available(uint64_t headroom = uint64_t{768} << 20) {
    const uint64_t free = strata::core::device_free_bytes();
    const auto host = strata::core::sample_host_memory();
    // Conservative on discrete GPUs: a mixed host/device commitment must fit
    // both readings. On Tegra device_free_bytes already subtracts UMA headroom.
    const uint64_t available = host ? std::min<uint64_t>(free, host->available) : 0;
    return available > headroom ? available - headroom : 0;
}
} // namespace strata::core
