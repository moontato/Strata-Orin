// CPU driver shim, production host guard + adapter/pool + native probe policy.
#include "../support/unified_kv_host_probe.hpp"
#include <cassert>
#include <cstring>
#include <iostream>
using namespace strata::core;
using strata::test::HostProbeResult;
using strata::test::probe_kv_growth;
using strata::test::reset_host_probe;
static void setup(int attribute = 0, std::vector<bool> aliases = {}) {
    assert(!fake_cuda::live_host_registrations && !fake_cuda::live_handles);
    fake_cuda::host_pointer_attribute = attribute;
    fake_cuda::host_alias_identity = std::move(aliases);
    fake_cuda::host_alias_null = false;
    fake_cuda::fail_host_attribute = fake_cuda::fail_host_register = fake_cuda::fail_host_alias = 0;
    fake_cuda::fail_host_unregister = fake_cuda::fail_map = fake_cuda::fail_access = fake_cuda::fail_zero = 0;
    fake_cuda::fail_unmap = fake_cuda::fail_release = fake_cuda::fail_sync = 0;
    fake_cuda::host_registers = fake_cuda::host_unregisters = fake_cuda::host_alias_queries = 0;
}
template<class F> static void must_fail(F f) {
    bool failed = false;
    try { f(); } catch (const KvFailure&) { failed = true; }
    assert(failed); // A backend error must not return "unsupported".
}
static void pointer_contract() {
    const auto g = vmm_granularity();
    for (int attribute : {0, 1}) for (bool identity : {false, true}) {
        setup(attribute, {identity});
        UnifiedKvVmm adapter(g);
        (void) adapter.reserve({1, 1, g, 1, 0}, true);
        UnifiedKvPool pool(g, g); auto id = pool.add(adapter);
        kv_require(pool.begin(id, g));
        const auto zeros = fake_cuda::memsets;
        std::string reason;
        const auto result = probe_kv_growth(pool, id, adapter, 1, true, reason);
        const bool supported = attribute != 0 || identity;
        assert((result == HostProbeResult::supported) == supported);
        assert(adapter.committed_bytes() == (supported ? g : 0));
        if (!supported) {
            assert(adapter.last_host_failure() == KvHostMapFailure::unsupported_flat_address);
            assert(fake_cuda::memsets == zeros && !fake_cuda::live_host_registrations);
            assert(!reason.empty());
        }
        reset_host_probe(pool, id, adapter);
        assert(!fake_cuda::live_host_registrations);
    }
    // Unknown attribute is not a guarantee. An equal address still satisfies
    // the unchanged contract; unknown + unequal is a query/backend error, not a skip.
    for (bool identity : {false, true}) {
        setup(0, {identity}); fake_cuda::fail_host_attribute = -1;
        UnifiedKvVmm adapter(g); (void) adapter.reserve({1, 1, g, 1, 0}, true);
        UnifiedKvPool pool(g, g); auto id = pool.add(adapter); kv_require(pool.begin(id, g));
        std::string reason;
        if (identity) assert(probe_kv_growth(pool, id, adapter, 1, true, reason) == HostProbeResult::supported);
        else must_fail([&] { (void) probe_kv_growth(pool, id, adapter, 1, true, reason); });
        reset_host_probe(pool, id, adapter);
    }
}
static void partial_rejection_and_reprobe() {
    setup(0, {true, true, false, true, true});
    const auto g = vmm_granularity();
    UnifiedKvVmm adapter(3*g);
    auto* p = (uint8_t*) adapter.reserve({1, 1, 3*g, 1, 0}, true);
    UnifiedKvPool pool(3*g, 3*g); auto id = pool.add(adapter); kv_require(pool.begin(id, 3*g));
    std::string reason;
    assert(probe_kv_growth(pool, id, adapter, 1, true, reason) == HostProbeResult::supported);
    std::memset(p, 0x37, (size_t) g);
    assert(probe_kv_growth(pool, id, adapter, 3*g, true, reason) == HostProbeResult::unsupported);
    assert(adapter.committed_bytes() == g && pool.info().committed_bytes == g);
    assert(fake_cuda::live_host_registrations == 1 && !adapter.faulted());
    for (uint64_t i=0; i<g; ++i) assert(p[i] == 0x37);
    // Actual aliases, not a sticky device-wide unsupported flag, decide retry.
    assert(probe_kv_growth(pool, id, adapter, 3*g, true, reason) == HostProbeResult::supported);
    assert(adapter.last_host_failure() == KvHostMapFailure::none);
    for (uint64_t i=g; i<3*g; ++i) assert(p[i] == 0);
    reset_host_probe(pool, id, adapter);
}
static void re_registration_can_reject() {
    setup(0, {true, true, true, false});
    const auto g = vmm_granularity();
    UnifiedKvVmm adapter(3*g); (void) adapter.reserve({1, 1, 3*g, 1, 0}, true);
    UnifiedKvPool pool(3*g, 3*g); auto id = pool.add(adapter); kv_require(pool.begin(id, 3*g));
    std::string reason;
    assert(probe_kv_growth(pool, id, adapter, 3*g, true, reason) == HostProbeResult::supported);
    reset_host_probe(pool, id, adapter); kv_require(pool.begin(id, 3*g));
    assert(probe_kv_growth(pool, id, adapter, 1, true, reason) == HostProbeResult::unsupported);
    assert(!adapter.committed_bytes() && !fake_cuda::live_host_registrations);
    reset_host_probe(pool, id, adapter);
}
static void errors_are_not_capabilities() {
    const auto g = vmm_granularity();
    for (int failure=0; failure<4; ++failure) {
        setup(0, {false});
        if (failure==0) fake_cuda::fail_host_register=1;
        if (failure==1) fake_cuda::fail_host_alias=1;
        if (failure==2) fake_cuda::host_alias_null=true;
        if (failure==3) fake_cuda::fail_host_unregister=1; // even if adapter rollback retries successfully
        UnifiedKvVmm adapter(g); (void) adapter.reserve({1, 1, g, 1, 0}, true);
        UnifiedKvPool pool(g, g); auto id=pool.add(adapter); kv_require(pool.begin(id, g));
        std::string reason;
        must_fail([&] { (void) probe_kv_growth(pool, id, adapter, 1, true, reason); });
        assert(adapter.last_host_failure()==KvHostMapFailure::backend);
        assert(!adapter.committed_bytes() && !fake_cuda::live_host_registrations);
        reset_host_probe(pool, id, adapter);
    }
    setup(0, {false}); fake_cuda::fail_host_unregister=-1;
    UnifiedKvVmm adapter(g); (void) adapter.reserve({1, 1, g, 1, 0}, true);
    UnifiedKvPool pool(g, g); auto id=pool.add(adapter); kv_require(pool.begin(id, g));
    std::string reason;
    must_fail([&] { (void) probe_kv_growth(pool, id, adapter, 1, true, reason); });
    assert(adapter.faulted() && adapter.committed_bytes()==g && fake_cuda::live_host_registrations==1);
    assert(pool.session(0).state==KvResidency::faulted && pool.info().committed_bytes==g);
    fake_cuda::fail_host_unregister=0;
    kv_require(pool.reset(id, id)); assert(!adapter.committed_bytes() && !fake_cuda::live_host_registrations);
}
static void mixed_device_and_host() {
    const auto g=vmm_granularity();
    for (int failure=0; failure<4; ++failure) {
        setup(0, {false});
        if (failure==1) fake_cuda::fail_map=1;
        if (failure==2) fake_cuda::fail_access=1;
        if (failure==3) fake_cuda::fail_zero=1;
        UnifiedKvVmm adapter(g);
        (void) adapter.reserve({1,1,g,1,0}); // QSA's device working arrays precede authoritative host planes
        (void) adapter.reserve({1,1,g,1,0}, true);
        UnifiedKvPool pool(g,g); auto id=pool.add(adapter); kv_require(pool.begin(id,g));
        std::string reason;
        if (!failure) assert(probe_kv_growth(pool,id,adapter,1,true,reason)==HostProbeResult::unsupported);
        else must_fail([&] { (void)probe_kv_growth(pool,id,adapter,1,true,reason); });
        assert(!adapter.committed_bytes() && !pool.info().committed_bytes);
        assert(!fake_cuda::live_handles && !fake_cuda::live_host_registrations);
        reset_host_probe(pool,id,adapter);
    }
}
static void synchronization_failures_are_fatal() {
    const auto g=vmm_granularity();
    for (int call : {1,2}) { // quiescence, then rejection-rollback drain
        setup(0,{false}); fake_cuda::fail_sync=call;
        UnifiedKvVmm adapter(g); (void)adapter.reserve({1,1,g,1,0},true);
        UnifiedKvPool pool(g,g); auto id=pool.add(adapter); kv_require(pool.begin(id,g));
        std::string reason;
        must_fail([&] { (void)probe_kv_growth(pool,id,adapter,1,true,reason); });
        assert(adapter.faulted() && pool.session(0).state==KvResidency::faulted);
        kv_require(pool.reset(id,id)); assert(!adapter.committed_bytes() && !fake_cuda::live_host_registrations);
    }
}
static void device_rollback_failure_is_fatal() {
    setup(0,{false}); const auto g=vmm_granularity(); fake_cuda::fail_unmap=-1;
    UnifiedKvVmm adapter(g);
    (void)adapter.reserve({1,1,g,1,0}); (void)adapter.reserve({1,1,g,1,0},true);
    UnifiedKvPool pool(g,g); auto id=pool.add(adapter); kv_require(pool.begin(id,g));
    std::string reason;
    must_fail([&] { (void)probe_kv_growth(pool,id,adapter,1,true,reason); });
    assert(adapter.faulted() && adapter.last_host_failure()==KvHostMapFailure::backend);
    assert(adapter.committed_bytes()==g && pool.info().committed_bytes==g);
    assert(fake_cuda::live_handles==1 && !fake_cuda::live_host_registrations);
    fake_cuda::fail_unmap=0; kv_require(pool.reset(id,id));
    assert(!adapter.committed_bytes() && !fake_cuda::live_handles);
}
static void stale_reason_cannot_hide_other_failures() {
    setup(0,{false,false}); const auto g=vmm_granularity();
    UnifiedKvVmm adapter(3*g); (void)adapter.reserve({1,1,3*g,1,0},true);
    UnifiedKvPool pool(3*g,3*g,g); auto id=pool.add(adapter); kv_require(pool.begin(id,3*g));
    std::string reason;
    assert(probe_kv_growth(pool,id,adapter,1,true,reason)==HostProbeResult::unsupported);
    must_fail([&] { (void)probe_kv_growth(pool,id,adapter,3*g,true,reason); }); // byte ceiling, no allocator call
    must_fail([&] { (void)probe_kv_growth(pool,id,adapter,1,false,reason); }); // device-only case must not skip
    reset_host_probe(pool,id,adapter);
}
int main() {
    assert(vmm_available());
    pointer_contract(); partial_rejection_and_reprobe(); re_registration_can_reject();
    errors_are_not_capabilities(); mixed_device_and_host(); synchronization_failures_are_fatal();
    device_rollback_failure_is_fatal();
    stale_reason_cannot_hide_other_failures();
    assert(!fake_cuda::live_host_registrations && !fake_cuda::live_handles);
    std::cout << "UNIFIED_KV_HOST_CAPABILITY_TEST PASS pointer contracts, per-registration probes, rejection rollback, errors and accounting\n";
}
