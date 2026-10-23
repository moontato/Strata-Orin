#pragma once
// GPU-independent ownership/admission policy. Callbacks execute under the pool lock;
// adapters must be synchronous, transactional on failure, and must not re-enter it.
#include <algorithm>
#include <cstdint>
#include <chrono>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::core {

enum class KvResidency { empty, idle, active, evicted, faulted };
inline const char* kv_residency_name(KvResidency s) {
    switch (s) {
        case KvResidency::empty: return "empty";
        case KvResidency::idle: return "resident";
        case KvResidency::active: return "active";
        case KvResidency::evicted: return "evicted";
        default: return "faulted";
    }
}
struct KvIdentity {
    uint32_t slot = 0;
    uint64_t generation = 1;
};
enum class KvCode { ok, cache_miss, stale, busy, context_limit, pool_full, allocation, backend };
struct KvResult {
    KvCode code = KvCode::ok;
    std::string message;
    explicit operator bool() const { return code == KvCode::ok; }
};
inline const char* kv_code_name(KvCode c) {
    switch (c) {
        case KvCode::ok: return "ok";
        case KvCode::cache_miss: return "kv_cache_miss";
        case KvCode::stale: return "kv_stale_generation";
        case KvCode::busy: return "kv_session_busy";
        case KvCode::context_limit: return "kv_context_limit";
        case KvCode::pool_full: return "kv_pool_full";
        case KvCode::allocation: return "kv_allocation_failed";
        default: return "kv_backend_failed";
    }
}

class KvAdapter {
public:
    virtual ~KvAdapter() = default;
    // Exact rounded physical commitment at this write high-water mark (may include
    // speculative cells past the logical end). No allocations or GPU work here.
    virtual uint64_t bytes_for(uint64_t cells) const = 0;
    virtual uint64_t committed_bytes() const = 0;
    virtual bool faulted() const { return false; }
    // Advance a write watermark within already-committed granules. No GPU work,
    // pointer changes or allocation; monotonic layouts only.
    virtual void cover(uint64_t cells) { (void) cells; }
    virtual bool grow(uint64_t cells, std::string& error) = 0;
    // All device work, graphs, draft streams and asynchronous manipulation complete.
    virtual bool quiesce(std::string& error) = 0;
    // Invalidate ALL dependent state, release physical KV; preserve virtual addresses.
    // Failure is fatal to this session: never claim successful reclamation.
    virtual bool discard(std::string& error) = 0;
};
struct KvSessionInfo {
    KvIdentity id;
    KvResidency state = KvResidency::empty;
    uint64_t epoch = 0, last_use = 0, tokens = 0, reserved_tokens = 0, bytes = 0;
};
struct KvEviction {
    uint64_t sequence = 0;
    KvIdentity id;
    uint64_t epoch = 0, tokens = 0, reclaimed_bytes = 0, latency_us = 0;
};
struct KvPoolInfo {
    uint64_t token_limit = 0, byte_limit = 0, resident_tokens = 0, reserved_tokens = 0;
    uint64_t committed_bytes = 0, evictions = 0, reclaimed_bytes = 0, eviction_us = 0;
};

class UnifiedKvPool {
    struct Entry { KvSessionInfo info; KvAdapter* adapter = nullptr; };
    mutable std::mutex mutex_;
    std::vector<Entry> sessions_;
    uint64_t token_limit_, byte_limit_, context_, clock_ = 0, evictions_ = 0, reclaimed_ = 0, eviction_us_ = 0;
    std::vector<KvEviction> events_;  // drained by the scheduler; no callbacks into user code

    Entry* find(KvIdentity id) {
        if (id.slot >= sessions_.size() || sessions_[id.slot].info.id.generation != id.generation) return nullptr;
        return &sessions_[id.slot];
    }
    uint64_t tokens() const {
        uint64_t n = 0;
        for (const auto& s : sessions_) n += s.info.reserved_tokens;
        return n;
    }
    uint64_t bytes() const {
        uint64_t n = 0;
        for (const auto& s : sessions_) {
            const uint64_t b = s.adapter->committed_bytes();
            if (b > UINT64_MAX - n) throw std::overflow_error("unified KV byte accounting overflow");
            n += b;
        }
        return n;
    }
    KvResult evict_one(uint32_t requester) {
        Entry* victim = nullptr;
        for (auto& s : sessions_) {
            if (s.info.id.slot == requester || s.info.state != KvResidency::idle) continue;
            if (!s.info.reserved_tokens && !s.adapter->committed_bytes()) continue;
            if (!victim || s.info.last_use < victim->info.last_use) victim = &s;
        }
        if (!victim) return {KvCode::pool_full, "no eligible idle session; active caches are protected"};
        const auto started = std::chrono::steady_clock::now();
        std::string error;
        if (!victim->adapter->quiesce(error)) {
            victim->info.state = KvResidency::faulted;
            return {KvCode::backend, error};
        }
        const uint64_t before = victim->adapter->committed_bytes();
        if (!victim->adapter->discard(error) || victim->adapter->committed_bytes() != 0) {
            victim->info.state = KvResidency::faulted;
            return {KvCode::backend, error.empty() ? "adapter failed to reclaim its KV" : error};
        }
        auto& i = victim->info;
        ++i.epoch;
        // Bounded, preallocated delivery queue. Sequence gaps are observable if a
        // caller neglects to drain it; totals remain authoritative.
        if (events_.size() == 256) events_.erase(events_.begin());
        const auto us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - started).count();
        events_.push_back({++evictions_, i.id, i.epoch, i.tokens, before, us});
        reclaimed_ += before; eviction_us_ += us;
        i.state = KvResidency::evicted;
        i.reserved_tokens = i.tokens = i.bytes = 0;
        return {};
    }
public:
    UnifiedKvPool(uint64_t context, uint64_t tokens, uint64_t bytes = UINT64_MAX)
        : token_limit_(tokens), byte_limit_(bytes), context_(context) {
        if (!context || !tokens || !bytes) throw std::invalid_argument("unified KV limits must be positive");
        events_.reserve(256);
    }
    UnifiedKvPool(const UnifiedKvPool&) = delete;
    UnifiedKvPool& operator=(const UnifiedKvPool&) = delete;
    KvIdentity add(KvAdapter& adapter) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (adapter.committed_bytes()) throw std::invalid_argument("register an uncommitted KV adapter");
        KvIdentity id{static_cast<uint32_t>(sessions_.size()), 1};
        Entry e; e.info.id = id; e.adapter = &adapter;
        sessions_.push_back(e);
        return id;
    }
    // A replay is explicit: it discards the old cache, but NOT the caller's conversation.
    // Callers must supply the entire history from token zero, including image metadata.
    KvResult begin(KvIdentity id, uint64_t required, bool replay = false) {
        std::lock_guard<std::mutex> lock(mutex_);
        Entry* e = find(id);
        if (!e) return {KvCode::stale, "session generation changed"};
        auto& i = e->info;
        if (i.state == KvResidency::active) return {KvCode::busy, "session already leased"};
        if (i.state == KvResidency::faulted) return {KvCode::backend, "session is faulted"};
        if (required > context_) return {KvCode::context_limit, "request exceeds per-session context"};
        if (!replay && i.state == KvResidency::evicted) return {KvCode::cache_miss, "replay the full conversation"};
        const uint64_t want = std::max(required, replay ? uint64_t{0} : i.tokens);
        if (want > token_limit_) return {KvCode::pool_full, "request exceeds shared token ceiling"};
        while (want > token_limit_ - (tokens() - i.reserved_tokens)) {
            const auto r = evict_one(id.slot); if (!r) return r;
        }
        if (replay) {
            std::string error;
            if (!e->adapter->quiesce(error) || !e->adapter->discard(error)) {
                i.state = KvResidency::faulted;
                return {KvCode::backend, error};
            }
            i.tokens = i.bytes = 0;
            ++i.epoch;
        }
        i.reserved_tokens = want;
        i.state = KvResidency::active;
        i.last_use = ++clock_;
        return {};
    }
    // Admission and physical growth are separate. Active reservations never cause
    // eager allocation of a request's output allowance. available is a fresh,
    // headroom-adjusted physical reading, NOT an additional pool to add to byte_limit.
    template<class Available> KvResult grow(KvIdentity id, uint64_t cells, Available available) {
        std::lock_guard<std::mutex> lock(mutex_);
        Entry* e = find(id);
        if (!e) return {KvCode::stale, "session generation changed"};
        if (e->info.state != KvResidency::active) return {KvCode::busy, "growth requires an active lease"};
        // cells can include backend-specific speculative padding. The scheduler
        // must bound it by its virtual extent; logical reservation remains unchanged.
        const uint64_t target = e->adapter->bytes_for(cells);
        const uint64_t old = e->adapter->committed_bytes();
        if (target <= old) { e->adapter->cover(cells); return {}; }
        const uint64_t extra = target - old;
        if (extra > byte_limit_) return {KvCode::pool_full, "KV write exceeds physical byte ceiling"};
        std::string last_allocation_error;
        for (;;) {
            const uint64_t held = bytes();
            if (held <= byte_limit_ - extra && available() >= extra) {
                std::string error;
                if (!e->adapter->quiesce(error)) { e->info.state = KvResidency::faulted; return {KvCode::backend, error}; }
                if (e->adapter->grow(cells, error)) {
                    e->info.bytes = e->adapter->committed_bytes();
                    if (e->info.bytes != target) {
                        e->info.state = KvResidency::faulted;
                        return {KvCode::backend, "adapter physical accounting mismatch"};
                    }
                    return {};
                }
                // Transactional adapter: old contents and accounting survive OOM.
                if (e->adapter->faulted() || e->adapter->committed_bytes() != old) {
                    e->info.state = KvResidency::faulted;
                    return {KvCode::backend, "adapter growth was not transactional: " + error};
                }
                last_allocation_error = error.empty() ? "physical allocator refused growth" : error;
            }
            const auto r = evict_one(id.slot);
            if (!r) {
                if (r.code == KvCode::backend) return r;
                if (!last_allocation_error.empty()) return {KvCode::allocation, last_allocation_error + "; " + r.message};
                return {KvCode::pool_full, "physical byte ceiling/free memory/headroom cannot admit " +
                                          std::to_string(extra) + " bytes; " + r.message};
            } // explicit failure, never length/truncation
        }
    }
    // The backend is about to zero/replay a different full history in-place.
    // Retain physical capacity, but change the validity epoch before writing it.
    KvResult replay_started(KvIdentity id) {
        std::lock_guard<std::mutex> lock(mutex_);
        Entry* e = find(id);
        if (!e) return {KvCode::stale, "session generation changed"};
        if (e->info.state != KvResidency::active) return {KvCode::busy, "replay requires an active lease"};
        std::string error;
        if (!e->adapter->quiesce(error)) { e->info.state = KvResidency::faulted; return {KvCode::backend, error}; }
        ++e->info.epoch; e->info.tokens = 0; e->info.last_use = ++clock_;
        return {};
    }
    KvResult progress(KvIdentity id, uint64_t live_tokens) {
        std::lock_guard<std::mutex> lock(mutex_);
        Entry* e = find(id);
        if (!e) return {KvCode::stale, "session generation changed"};
        if (e->info.state != KvResidency::active) return {KvCode::busy, "progress requires an active lease"};
        if (live_tokens > e->info.reserved_tokens) return {KvCode::context_limit, "unreserved logical KV tokens"};
        e->info.tokens = live_tokens; e->info.last_use = ++clock_;
        return {};
    }
    KvResult end(KvIdentity id, uint64_t live_tokens) {
        std::lock_guard<std::mutex> lock(mutex_);
        Entry* e = find(id);
        if (!e) return {KvCode::stale, "session generation changed"};
        if (e->info.state != KvResidency::active) return {KvCode::busy, "session is not leased"};
        if (live_tokens > e->info.reserved_tokens) return {KvCode::context_limit, "unreserved tokens"};
        std::string error;
        if (!e->adapter->quiesce(error)) { e->info.state = KvResidency::faulted; return {KvCode::backend, error}; }
        auto& i = e->info;
        i.state = live_tokens || e->adapter->committed_bytes() ? KvResidency::idle : KvResidency::empty;
        i.tokens = i.reserved_tokens = live_tokens;
        i.bytes = e->adapter->committed_bytes();
        i.last_use = ++clock_; // actual completed use, not allocation order
        return {};
    }
    // Existing Flash-Next admission copies a main workspace into a batch slot.
    // Both physical copies are charged, but the logical conversation is counted
    // once. The destination must be leased with a zero-token reservation.
    KvResult handoff(KvIdentity from, KvIdentity to, uint64_t live_tokens) {
        std::lock_guard<std::mutex> lock(mutex_);
        Entry* a = find(from); Entry* b = find(to);
        if (!a || !b) return {KvCode::stale, "handoff generation changed"};
        if (from.slot == to.slot || a->info.state != KvResidency::active || b->info.state != KvResidency::active ||
            b->info.reserved_tokens != 0) return {KvCode::busy, "handoff requires a source and empty leased destination"};
        if (live_tokens > a->info.reserved_tokens) return {KvCode::context_limit, "handoff exceeds reservation"};
        std::string error;
        if (!a->adapter->quiesce(error) || !b->adapter->quiesce(error)) return {KvCode::backend, error};
        if (!a->adapter->discard(error) || a->adapter->committed_bytes()) {
            a->info.state = KvResidency::faulted;
            return {KvCode::backend, "handoff could not release workspace: " + error};
        }
        b->info.reserved_tokens = a->info.reserved_tokens;
        b->info.tokens = live_tokens; b->info.bytes = b->adapter->committed_bytes();
        b->info.last_use = ++clock_;
        a->info.tokens = a->info.reserved_tokens = a->info.bytes = 0;
        a->info.state = KvResidency::empty; ++a->info.epoch;
        return {};
    }
    KvResult reset(KvIdentity id, KvIdentity& next) {
        std::lock_guard<std::mutex> lock(mutex_);
        Entry* e = find(id);
        if (!e) return {KvCode::stale, "session generation changed"};
        if (e->info.state == KvResidency::active) return {KvCode::busy, "cannot reset an active session"};
        std::string error;
        if (!e->adapter->quiesce(error) || !e->adapter->discard(error)) return {KvCode::backend, error};
        next = {id.slot, id.generation + 1};
        const uint64_t epoch = e->info.epoch + 1;
        e->info = {}; e->info.id = next; e->info.epoch = epoch;
        return {};
    }
    KvSessionInfo session(uint32_t slot) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto i = sessions_.at(slot).info;
        i.bytes = sessions_[slot].adapter->committed_bytes();
        return i;
    }
    KvPoolInfo info() const {
        std::lock_guard<std::mutex> lock(mutex_);
        KvPoolInfo i; i.token_limit = token_limit_; i.byte_limit = byte_limit_;
        i.committed_bytes = bytes(); i.reserved_tokens = tokens();
        for (const auto& s : sessions_) i.resident_tokens += s.info.tokens;
        i.evictions = evictions_; i.reclaimed_bytes = reclaimed_; i.eviction_us = eviction_us_;
        return i;
    }
    std::vector<KvEviction> drain_events() {
        std::lock_guard<std::mutex> lock(mutex_);
        // Keep the preallocated queue capacity: eviction must not allocate an
        // event record AFTER it has discarded a victim's physical state.
        std::vector<KvEviction> out(events_);
        events_.clear();
        return out;
    }
};

} // namespace strata::core
