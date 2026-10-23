#pragma once
#include "strata/core/unified_kv.hpp"
#include <cstdio>
#include <stdexcept>
#include <cerrno>
#include <cstdlib>
#include <sstream>
#include <functional>

namespace strata::core {
struct KvFailure : std::runtime_error {
    KvCode code;
    explicit KvFailure(const KvResult& r) : std::runtime_error(std::string(kv_code_name(r.code)) + ": " + r.message), code(r.code) {}
};
inline void kv_require(const KvResult& r) { if (!r) throw KvFailure(r); }
inline uint64_t kv_parse_limit(const std::string& s, const char* flag) {
    char* end = nullptr; errno = 0;
    const auto n = std::strtoull(s.c_str(), &end, 10);
    if (s.empty() || !std::all_of(s.begin(), s.end(), [](unsigned char c) { return c >= '0' && c <= '9'; }) || *end || errno || !n)
        throw std::invalid_argument(std::string(flag) + " requires a positive integer");
    return n;
}
inline void kv_print_events(UnifiedKvPool& pool) {
    for (const auto& e : pool.drain_events())
        std::printf("KVEVICT session=%u generation=%llu epoch=%llu tokens=%llu bytes=%llu sequence=%llu latency_us=%llu reason=pressure\n",
                    e.id.slot, (unsigned long long) e.id.generation, (unsigned long long) e.epoch,
                    (unsigned long long) e.tokens, (unsigned long long) e.reclaimed_bytes, (unsigned long long) e.sequence,
                    (unsigned long long) e.latency_us);
    std::fflush(stdout);
}
inline void kv_print_commit(UnifiedKvPool& pool, KvIdentity id, uint64_t cells, uint64_t latency_us) {
    const auto i = pool.session(id.slot);
    const auto p = pool.info();
    std::printf("KVCOMMIT session=%u generation=%llu epoch=%llu cells=%llu bytes=%llu pool_bytes=%llu latency_us=%llu\n",
                id.slot, (unsigned long long) id.generation, (unsigned long long) i.epoch,
                (unsigned long long) cells, (unsigned long long) i.bytes, (unsigned long long) p.committed_bytes,
                (unsigned long long) latency_us);
    std::fflush(stdout);
}
inline void kv_print_status(UnifiedKvPool& pool, unsigned slots, uint64_t available = UINT64_MAX) {
    const auto p = pool.info();
    std::printf("KVPOOL enabled=1 token_limit=%llu byte_limit=%llu committed_bytes=%llu resident_tokens=%llu "
                "reserved_tokens=%llu evictions=%llu reclaimed_bytes=%llu available_tokens=%llu eviction_us=%llu\n",
                (unsigned long long) p.token_limit, (unsigned long long) p.byte_limit,
                (unsigned long long) p.committed_bytes, (unsigned long long) p.resident_tokens,
                (unsigned long long) p.reserved_tokens, (unsigned long long) p.evictions,
                (unsigned long long) p.reclaimed_bytes, (unsigned long long) (p.token_limit - p.reserved_tokens),
                (unsigned long long) p.eviction_us);
    if (available != UINT64_MAX)
        std::printf("KVMEM physical_available_after_headroom=%llu admissible_bytes=%llu\n",
                    (unsigned long long) available,
                    (unsigned long long) std::min(available, p.byte_limit > p.committed_bytes ? p.byte_limit - p.committed_bytes : 0));
    for (unsigned s = 0; s < slots; ++s) {
        const auto i = pool.session(s);
        std::printf("KVSESSION session=%u generation=%llu epoch=%llu state=%s tokens=%llu reserved_tokens=%llu bytes=%llu\n",
                    i.id.slot, (unsigned long long) i.id.generation, (unsigned long long) i.epoch, kv_residency_name(i.state),
                    (unsigned long long) i.tokens, (unsigned long long) i.reserved_tokens, (unsigned long long) i.bytes);
    }
    std::printf("KVEND sessions=%u sequence=%llu\n", slots, (unsigned long long) p.evictions);
    std::fflush(stdout);
}
// Request error paths cannot leave an orphaned active workspace or reuse a
// partially processed recurrence. Normal end/handoff makes this guard a no-op.
class KvAbandon {
    UnifiedKvPool* pool_;
    KvIdentity* id_;
public:
    KvAbandon(UnifiedKvPool* pool, KvIdentity* id) : pool_(pool), id_(id) {}
    ~KvAbandon() {
        if (!pool_ || pool_->session(id_->slot).state != KvResidency::active) return;
        const auto end = pool_->end(*id_, 0);
        const auto reset = end ? pool_->reset(*id_, *id_) : end;
        if (!reset) std::fprintf(stderr, "unified KV: failed request remains faulted: %s\n", reset.message.c_str());
        else std::printf("KVINVALIDATE session=%u generation=%llu epoch=%llu reason=request_error\n", id_->slot,
                         (unsigned long long) id_->generation, (unsigned long long) pool_->session(id_->slot).epoch);
        std::fflush(stdout);
    }
    KvAbandon(const KvAbandon&) = delete;
    KvAbandon& operator=(const KvAbandon&) = delete;
};
class KvIdlePin {
    UnifiedKvPool* pool_ = nullptr;
    KvIdentity id_;
    uint64_t tokens_ = 0;
public:
    KvIdlePin(UnifiedKvPool* pool, KvIdentity id) : pool_(pool), id_(id) {
        if (pool_) {
            const auto i = pool_->session(id.slot); tokens_ = i.tokens;
            kv_require(pool_->begin(id, tokens_));
        }
    }
    ~KvIdlePin() {
        if (pool_) {
            const auto r = pool_->end(id_, tokens_);
            if (!r) std::fprintf(stderr, "unified KV: source pin release failed: %s\n", r.message.c_str());
        }
    }
    KvIdlePin(const KvIdlePin&) = delete;
    KvIdlePin& operator=(const KvIdlePin&) = delete;
};
// Only the engine worker calls controls, at a safe scheduling boundary.
inline bool kv_control(UnifiedKvPool& pool, std::vector<KvIdentity>& ids, const std::string& line,
                       const std::function<uint64_t()>& available = {}) {
    if (line == "KVSTATUS") { kv_print_status(pool, (unsigned) ids.size(), available ? available() : UINT64_MAX); return true; }
    if (line.rfind("KVRESET", 0) != 0) return false;
    std::istringstream input(line); std::string command, extra;
    uint64_t slot = UINT64_MAX, generation = 0;
    if (!(input >> command >> slot >> generation) || command != "KVRESET" || input >> extra || slot >= ids.size()) {
        std::printf("ERR kv_invalid_control: expected KVRESET <session> <generation>\n");
    } else {
        const auto r = pool.reset({(uint32_t) slot, generation}, ids[(size_t) slot]);
        if (!r) std::printf("ERR %s %s\n", kv_code_name(r.code), r.message.c_str());
        else std::printf("KVRESET session=%llu generation=%llu epoch=%llu\n", (unsigned long long) slot,
                         (unsigned long long) ids[(size_t) slot].generation, (unsigned long long) pool.session((uint32_t) slot).epoch);
    }
    std::fflush(stdout); return true;
}
} // namespace strata::core
