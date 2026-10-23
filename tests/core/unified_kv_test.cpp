#include "strata/core/unified_kv.hpp"
#include "strata/core/unified_kv_layout.hpp"
#include <atomic>
#include <cassert>
#include <iostream>
#include <thread>
#include <set>

using namespace strata::core;
struct Adapter : KvAdapter {
    uint64_t bytes = 0, writes = 0;
    bool fail = false, sync_ok = true, valid = false;
    int discards = 0;
    uint64_t bytes_for(uint64_t cells) const override { return kv_ceil(cells * 10, 64) * 64; }
    uint64_t committed_bytes() const override { return bytes; }
    bool grow(uint64_t cells, std::string& error) override {
        if (fail) { error = "injected OOM"; return false; }
        bytes = bytes_for(cells); valid = true; ++writes; return true;
    }
    bool quiesce(std::string& error) override { if (!sync_ok) error = "in flight"; return sync_ok; }
    bool discard(std::string&) override { bytes = 0; valid = false; ++discards; return true; }
};
static uint64_t unlimited() { return UINT64_MAX; }
int main() {
    {
        UnifiedKvPool pool(250000, 500000);
        Adapter a[4]; KvIdentity id[4];
        for (int s = 0; s < 4; ++s) id[s] = pool.add(a[s]);
        const uint64_t dist[4][4] = {{250000,150000,70000,30000}, {250000,250000,0,0},
                                   {125000,125000,125000,125000}, {250000,10000,5000,0}};
        for (auto& d : dist) {
            for (int s = 0; s < 4; ++s) assert(pool.reset(id[s], id[s]));
            for (int s = 0; s < 4; ++s) {
                assert(pool.begin(id[s], d[s]));
                assert(a[s].bytes == 0); // admission reserves logical tokens only
                assert(pool.grow(id[s], d[s], unlimited));
            }
            for (int s = 0; s < 4; ++s) assert(pool.end(id[s], d[s]));
            assert(pool.info().committed_bytes == a[0].bytes + a[1].bytes + a[2].bytes + a[3].bytes);
            assert(pool.drain_events().empty());
        }
        assert(pool.begin(id[0], 250001).code == KvCode::context_limit);
        assert(pool.begin(id[0], 1));
        assert(pool.reset(id[0], id[0]).code == KvCode::busy);
        assert(pool.end(id[0], 250000));
    }
    {
        UnifiedKvPool pool(100, 100);
        Adapter a[4]; KvIdentity id[4];
        for (int s = 0; s < 4; ++s) {
            id[s] = pool.add(a[s]);
            assert(pool.begin(id[s], 25)); assert(pool.grow(id[s], 25, unlimited)); assert(pool.end(id[s], 25));
        }
        // Actual recent use, not allocation order: touch 0, making 1 oldest.
        assert(pool.begin(id[0], 25)); assert(pool.end(id[0], 25));
        assert(pool.begin(id[3], 80)); // requires three whole-session evictions
        const auto ev = pool.drain_events();
        assert(ev.size() == 3 && ev[0].id.slot == 1 && ev[1].id.slot == 2 && ev[2].id.slot == 0);
        for (const auto& e : ev) {
            assert(e.tokens == 25 && e.reclaimed_bytes == 256);
            assert(!a[e.id.slot].valid);
            assert(pool.begin(e.id, 1).code == KvCode::cache_miss);
        }
        assert(pool.end(id[3], 25));
        // Explicit history replay, no accidental empty-cache continuation.
        assert(pool.begin(id[1], 25, true)); assert(pool.grow(id[1], 25, unlimited)); assert(pool.end(id[1], 25));
        const auto old = id[1]; assert(pool.reset(id[1], id[1]));
        assert(id[1].generation == old.generation + 1);
        assert(pool.begin(old, 1).code == KvCode::stale);
        assert(pool.session(id[1].slot).state == KvResidency::empty);
    }
    {
        UnifiedKvPool pool(100, 100, 1024);
        Adapter a[4]; KvIdentity id[4];
        for (int s = 0; s < 4; ++s) { id[s] = pool.add(a[s]); assert(pool.begin(id[s], 25)); }
        assert(pool.begin(id[0], 25).code == KvCode::busy);
        for (int s = 0; s < 4; ++s) assert(pool.grow(id[s], 25, unlimited));
        // Every slot active: no eviction, no corruption on byte exhaustion.
        assert(pool.grow(id[0], 32, unlimited).code == KvCode::pool_full);
        assert(a[0].bytes == 256 && a[1].bytes == 256 && pool.info().evictions == 0);
        assert(pool.end(id[1], 25));
        assert(pool.grow(id[0], 32, unlimited)); // pressure evicts the only idle victim
        assert(pool.session(1).state == KvResidency::evicted);
        assert(pool.info().committed_bytes == 832);
        a[0].fail = true;
        const auto before = a[0].bytes;
        assert(!pool.grow(id[0], 40, unlimited)); assert(a[0].bytes == before);
        a[0].fail = false;
        assert(pool.grow(id[0], 40, unlimited));
        assert(pool.end(id[0], 25));
        assert(pool.end(id[2], 25)); assert(pool.end(id[3], 25));
    }
    {
        UnifiedKvPool pool(100, 100);
        Adapter a, b; const auto x = pool.add(a), y = pool.add(b);
        assert(pool.begin(x, 40)); assert(pool.grow(x, 40, unlimited)); assert(pool.end(x, 40));
        assert(pool.begin(y, 40));
        uint64_t free = 0;
        auto pressure = [&] { return free + (a.bytes == 0 ? 1024 : 0); };
        assert(pool.grow(y, 40, pressure)); // real free-byte pressure, below logical ceiling
        assert(pool.info().evictions == 1 && a.bytes == 0);
        b.sync_ok = false;
        assert(pool.end(y, 40).code == KvCode::backend);
        assert(pool.session(y.slot).state == KvResidency::faulted && b.bytes != 0);
    }
    {
        // Concurrent admission serialized by one owner lock. The winning lease is
        // kept active until all contenders observed exhaustion.
        UnifiedKvPool pool(100, 100); Adapter a[4]; KvIdentity id[4];
        for (int s = 0; s < 4; ++s) id[s] = pool.add(a[s]);
        std::atomic<int> ready{0}, go{0}, winners{0}, attempts{0};
        std::vector<std::thread> threads;
        for (int s = 0; s < 4; ++s) threads.emplace_back([&,s] {
            ++ready; while (!go.load()) std::this_thread::yield();
            const bool won = bool(pool.begin(id[s], 100));
            if (won) { ++winners; assert(pool.grow(id[s], 100, unlimited)); }
            ++attempts;
            while (attempts.load() != 4) std::this_thread::yield();
            if (won) assert(pool.end(id[s], 100));
        });
        while (ready.load() != 4) std::this_thread::yield();
        go = 1;
        for (auto& t : threads) t.join();
        assert(winners == 1 && pool.info().evictions == 0);
    }
    // Layout/accounting oracle: every byte in every required head prefix must be
    // covered, and no granule counted twice at an unaligned head boundary.
    for (uint64_t gran : {64u, 128u, 2048u})
        for (uint64_t cells : {0u, 1u, 7u, 63u, 250u}) {
            KvPrefix p{12, 1, 250, 3, 3000};
            std::set<uint64_t> oracle;
            for (uint64_t h = 0; h < 3; ++h)
                for (uint64_t b = 0; b < cells * 12; ++b) oracle.insert((h*3000+b)/gran);
            const auto c = p.chunks(cells, gran);
            assert(std::vector<uint64_t>(oracle.begin(), oracle.end()) == c);
        }
    bool overflow = false;
    try { (void) kv_mul(UINT64_MAX, 2); } catch (const std::overflow_error&) { overflow = true; }
    assert(overflow);
    std::cout << "unified KV ownership, LRU, failure, concurrency and layout tests passed\n";
}
