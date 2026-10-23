#pragma once
// Dynamic pinned host backing for the existing streamed QSA kernels. On Linux
// UVA is checked for EVERY registration: an alias that changes would invalidate
// graph pointers, so such a device/driver is explicitly refused, never guessed.
#include "strata/core/unified_kv_layout.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <string>
#include <cerrno>
#include <cstring>
#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace strata::core {
class UnifiedKvHostRange {
    uint8_t* base_ = nullptr;
    uint64_t gran_ = 0;
    std::vector<uint8_t> mapped_;
    std::string error_;
public:
    const std::string& error() const { return error_; }
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
        base_ = (uint8_t*) p; mapped_.assign((size_t) count, 0); return true;
#else
        (void) bytes; (void) gran; return false;
#endif
    }
    uint8_t* base() const { return base_; }
    int64_t chunks() const { return (int64_t) mapped_.size(); }
    bool mapped(int64_t c) const { return mapped_.at((size_t) c) != 0; }
    uint64_t count() const { uint64_t n = 0; for (auto m : mapped_) n += m != 0; return n; }
    bool map(int64_t c) {
#if defined(__linux__)
        if (mapped(c)) return true;
        uint8_t* p = base_ + (uint64_t) c * gran_;
        if (mprotect(p, (size_t) gran_, PROT_READ | PROT_WRITE) != 0) {
            error_ = std::string("KV host commitment: ") + std::strerror(errno); return false;
        }
        const auto registered = cudaHostRegister(p, (size_t) gran_, cudaHostRegisterMapped | cudaHostRegisterPortable);
        if (registered != cudaSuccess) {
            error_ = std::string("KV cudaHostRegister: ") + cudaGetErrorString(registered);
            (void) cudaGetLastError();
            const bool discarded = madvise(p, (size_t) gran_, MADV_DONTNEED) == 0 &&
                                   mprotect(p, (size_t) gran_, PROT_NONE) == 0;
            if (!discarded) mapped_[(size_t) c] = 2; // unpinned, but still charged until OS reclamation succeeds
            return false;
        }
        mapped_[(size_t) c] = 1;
        void* device = nullptr;
        if (cudaHostGetDevicePointer(&device, p, 0) != cudaSuccess || device != p) {
            error_ = "KV host registration has no stable contiguous UVA alias; this driver/placement is unsupported";
            (void) unmap(c); return false;
        }
        return true;
#else
        (void) c; return false;
#endif
    }
    bool unmap(int64_t c) {
#if defined(__linux__)
        if (!mapped(c)) return true;
        uint8_t* p = base_ + (uint64_t) c * gran_;
        if (mapped_[(size_t) c] == 1) {
            if (cudaHostUnregister(p) != cudaSuccess) return false;
            mapped_[(size_t) c] = 2;
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
