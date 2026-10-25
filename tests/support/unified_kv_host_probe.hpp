#pragma once
// Shared single-owner native/CPU fixture policy. Probe the PRODUCTION growth
// path, not just a device attribute. Each registration/re-registration must
// satisfy its contract; this is not a cached device-wide capability flag.
#include "strata/core/unified_kv_vmm.hpp"
#include "strata/core/unified_kv_protocol.hpp"

namespace strata::test {
enum class HostProbeResult { supported, unsupported };
inline HostProbeResult probe_kv_growth(core::UnifiedKvPool& pool, core::KvIdentity id,
        core::UnifiedKvVmm& adapter, uint64_t cells, bool requires_host, std::string& reason) {
    const auto before = adapter.committed_bytes();
    const auto held = pool.info().committed_bytes;
    const auto result = pool.grow(id, cells, [] { return UINT64_MAX; });
    if (result) return HostProbeResult::supported;
    // An allocation/map/access/query/sync/cleanup failure must NEVER become an
    // unsupported pass. Only the explicit confirmed address-contract rejection,
    // with successful transactional rollback and an intact active owner, qualifies.
    if (!requires_host || result.code != core::KvCode::allocation || adapter.faulted() ||
        adapter.last_host_failure() != core::KvHostMapFailure::unsupported_flat_address ||
        adapter.committed_bytes() != before || pool.info().committed_bytes != held ||
        pool.session(id.slot).state != core::KvResidency::active)
        core::kv_require(result);
    reason = result.message;
    return HostProbeResult::unsupported;
}
inline void reset_host_probe(core::UnifiedKvPool& pool, core::KvIdentity& id, core::UnifiedKvVmm& adapter) {
    const auto old = id;
    const auto epoch = pool.session(id.slot).epoch;
    core::kv_require(pool.end(id, 0));
    core::kv_require(pool.reset(id, id));
    const auto info = pool.session(id.slot);
    if (adapter.committed_bytes() || pool.info().committed_bytes || adapter.mapped_cells() ||
        info.reserved_tokens || info.tokens || info.state != core::KvResidency::empty ||
        id.generation != old.generation + 1 || info.epoch != epoch + 1 ||
        pool.begin(old, 1).code != core::KvCode::stale)
        throw std::runtime_error("host capability/rejection probe did not reset accounting/identity");
}
} // namespace strata::test
