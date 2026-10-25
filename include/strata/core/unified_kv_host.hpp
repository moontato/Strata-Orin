#pragma once
// Incrementally registered host backing for the existing flat-pointer QSA
// kernels. Independent cudaHostRegister calls do NOT guarantee adjacent device
// aliases. Use the CPU VA only with CUDA's explicit direct-host-pointer guarantee,
// or when EVERY live registration actually maps at that VA. Never extrapolate a
// device alias from an earlier registration. See docs/KV_HOST_MAPPING.md.
#include "strata/core/unified_kv_layout.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <string>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace strata::core {
inline std::string kv_host_cuda_error(cudaError_t error) {
    std::ostringstream out;
    out << cudaGetErrorName(error) << '(' << static_cast<int>(error) << "): " << cudaGetErrorString(error);
    return out.str();
}

// Query, don't infer from integrated/UVA/SM version. An unavailable attribute is
// recorded as unknown, never interpreted as a positive capability.
struct KvHostDeviceInfo {
    int device = -1;
    int can_use_host_pointer = -1;
    int uses_host_page_tables = -1;
    std::string attributes;
    bool query(std::string& error) {
#if !defined(STRATA_USE_HIP) && !defined(STRATA_GPU_METAL)
        const auto current = cudaGetDevice(&device);
        if (current != cudaSuccess) { error = "KV cudaGetDevice: " + kv_host_cuda_error(current); return false; }
        std::ostringstream out;
        out << "device=" << device;
        auto attribute = [&](const char* name, cudaDeviceAttr attr) {
            int value = -1;
            const auto result = cudaDeviceGetAttribute(&value, attr, device);
            out << ' ' << name << '=';
            if (result == cudaSuccess) out << value;
            else { out << "unknown[" << kv_host_cuda_error(result) << ']'; (void) cudaGetLastError(); value = -1; }
            return value;
        };
        (void) attribute("integrated", cudaDevAttrIntegrated);
        (void) attribute("unified_addressing", cudaDevAttrUnifiedAddressing);
        (void) attribute("can_map_host_memory", cudaDevAttrCanMapHostMemory);
        (void) attribute("host_register_supported", cudaDevAttrHostRegisterSupported);
        can_use_host_pointer = attribute("can_use_host_pointer_for_registered_mem", cudaDevAttrCanUseHostPointerForRegisteredMem);
        (void) attribute("pageable_memory_access", cudaDevAttrPageableMemoryAccess);
        uses_host_page_tables = attribute("pageable_memory_access_uses_host_page_tables", cudaDevAttrPageableMemoryAccessUsesHostPageTables);
        unsigned flags = 0;
        const auto result = cudaGetDeviceFlags(&flags);
        out << " context_flags=";
        if (result == cudaSuccess) out << flags;
        else { out << "unknown[" << kv_host_cuda_error(result) << ']'; (void) cudaGetLastError(); }
        attributes = out.str();
        return true;
#else
        error = "KV incremental registered backing requires the CUDA backend";
        return false;
#endif
    }
};

// Diagnostic classification only: never changes which pointer is accepted.
// Unknown attributes/API/cleanup failures are errors, not unsupported coverage.
enum class KvHostMapFailure { none, unsupported_flat_address, backend };

class UnifiedKvHostRange {
    // Retain the physical charge until unregister and OS reclamation both
    // succeed. A failed/partial registration can populate pages.
    enum : uint8_t { physical = 1, registered = 2 };
    uint8_t* base_ = nullptr;
    uint64_t gran_ = 0;
    std::vector<uint8_t> mapped_;
    std::vector<void*> aliases_;
    KvHostDeviceInfo info_;
    bool info_ready_ = false;
    std::string error_;
    KvHostMapFailure last_failure_ = KvHostMapFailure::none;
    std::string mapping(int64_t c, void* alias) const {
        const auto* p = base_ + (uint64_t) c * gran_;
        std::ostringstream out;
        out << info_.attributes << " cpu_base=" << static_cast<void*>(base_) << " granularity=" << gran_
            << " chunk=" << c << " host_pointer=" << static_cast<const void*>(p)
            << " returned_device_alias=" << alias << " expected_flat_pointer=" << static_cast<const void*>(p);
        if (c > 0 && (mapped_[(size_t) c - 1] & registered)) {
            const auto previous = reinterpret_cast<uintptr_t>(aliases_[(size_t) c - 1]);
            out << " previous_device_alias=" << aliases_[(size_t) c - 1]
                << " previous_alias_plus_granularity=" << reinterpret_cast<void*>(previous + gran_)
                << " adjacent_device_alias=" << (reinterpret_cast<uintptr_t>(alias) == previous + gran_);
        }
        return out.str();
    }
public:
    const std::string& error() const { return error_; }
    KvHostMapFailure last_failure() const { return last_failure_; }
    ~UnifiedKvHostRange() {
#if defined(__linux__)
        bool ok = true;
        for (size_t c = 0; c < mapped_.size(); ++c) ok = unmap(c) && ok;
        if (base_ && ok) (void) munmap(base_, mapped_.size() * gran_);
        // Fatal unregister failure: intentionally leak the VA, never alias it.
#endif
    }
    bool reserve(uint64_t bytes, uint64_t gran) {
#if defined(__linux__)
        const long page = sysconf(_SC_PAGESIZE);
        if (page <= 0 || gran < (uint64_t) page || gran % (uint64_t) page != 0) return false;
        gran_ = gran;
        const auto count = kv_ceil(bytes, gran);
        const auto extent = kv_mul(count, gran);
        void* p = mmap(nullptr, (size_t) extent, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) return false;
        base_ = (uint8_t*) p; mapped_.assign((size_t) count, 0); aliases_.assign((size_t) count, nullptr); return true;
#else
        (void) bytes; (void) gran; return false;
#endif
    }
    uint8_t* base() const { return base_; } // the ONLY address published to QSA/graphs
    int64_t chunks() const { return (int64_t) mapped_.size(); }
    bool mapped(int64_t c) const { return mapped_.at((size_t) c) != 0; }
    uint64_t count() const { uint64_t n = 0; for (auto m : mapped_) n += m != 0; return n; }
    bool map(int64_t c) {
        last_failure_ = KvHostMapFailure::backend;
#if defined(__linux__)
        if (mapped(c)) {
            const bool ok = (mapped_[(size_t) c] & registered) != 0;
            if (ok) last_failure_ = KvHostMapFailure::none;
            return ok;
        }
        if (!info_ready_) { if (!info_.query(error_)) return false; info_ready_ = true; }
        int current = -1;
        const auto current_result = cudaGetDevice(&current);
        if (current_result != cudaSuccess || current != info_.device) {
            error_ = "KV host range used on a different/invalid CUDA device: " + kv_host_cuda_error(current_result); return false;
        }
        uint8_t* p = base_ + (uint64_t) c * gran_;
        if (mprotect(p, (size_t) gran_, PROT_READ | PROT_WRITE) != 0) {
            error_ = std::string("KV host commitment: ") + std::strerror(errno); return false;
        }
        mapped_[(size_t) c] = physical;
        auto fail = [&](const std::string& reason, KvHostMapFailure failure = KvHostMapFailure::backend) {
            error_ = reason;
            last_failure_ = failure;
            if (!unmap(c)) {
                error_ += "; rollback retained physical host pages";
                last_failure_ = KvHostMapFailure::backend;
            }
            return false;
        };
        const auto result = cudaHostRegister(p, (size_t) gran_, cudaHostRegisterMapped | cudaHostRegisterPortable);
        if (result != cudaSuccess) {
            (void) cudaGetLastError();
            return fail("KV cudaHostRegister: " + kv_host_cuda_error(result) + "; " + mapping(c, nullptr));
        }
        mapped_[(size_t) c] |= registered;
        void* alias = nullptr;
        const auto aliased = cudaHostGetDevicePointer(&alias, p, 0);
        aliases_[(size_t) c] = alias;
        const auto details = mapping(c, alias);
        const char* trace = std::getenv("STRATA_KV_HOST_TRACE");
        if (trace && *trace && std::strcmp(trace, "0") != 0)
            std::fprintf(stderr, "KVHOSTREGISTER %s get_device_pointer=%s canonical=cpu_va\n",
                         details.c_str(), kv_host_cuda_error(aliased).c_str());
        if (aliased != cudaSuccess) {
            (void) cudaGetLastError();
            return fail("KV cudaHostGetDevicePointer: " + kv_host_cuda_error(aliased) + "; " + details);
        }
        // CUDA explicitly permits the CPU pointer when this attribute is
        // nonzero, even if GetDevicePointer returns a DIFFERENT alias. We never
        // publish/use that other alias. Otherwise require equality for EACH
        // registration/re-registration; adjacency is not inferred from chunk 0.
        if (!alias || (alias != p && info_.can_use_host_pointer <= 0))
            return fail("KV host backing cannot satisfy flat device addressing: " + details +
                        "; no direct-host-pointer guarantee; independent registration aliases cannot be concatenated",
                        alias && info_.can_use_host_pointer == 0 ? KvHostMapFailure::unsupported_flat_address
                                                                 : KvHostMapFailure::backend);
        last_failure_ = KvHostMapFailure::none;
        return true;
#else
        (void) c; return false;
#endif
    }
    bool unmap(int64_t c) {
#if defined(__linux__)
        if (!mapped(c)) return true;
        uint8_t* p = base_ + (uint64_t) c * gran_;
        if (mapped_[(size_t) c] & registered) {
            if (cudaHostUnregister(p) != cudaSuccess) return false;
            mapped_[(size_t) c] &= ~registered;
            aliases_[(size_t) c] = nullptr;
        }
        // If the OS refuses reclamation the adapter must report a fatal failure,
        // not lower its physical commitment counter.
        if (madvise(p, (size_t) gran_, MADV_DONTNEED) != 0 || mprotect(p, (size_t) gran_, PROT_NONE) != 0) return false;
        mapped_[(size_t) c] = 0;
        return true;
#else
        (void) c; return false;
#endif
    }
};
} // namespace strata::core
