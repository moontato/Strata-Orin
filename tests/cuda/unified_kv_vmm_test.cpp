// Native CUDA gate: no model fixture. Run on Orin; a compile is not a pass.
#include "strata/core/unified_kv_vmm.hpp"
#include "strata/core/unified_kv_binding.hpp"
#include "strata/core/unified_kv_protocol.hpp"
#include "strata/core/layer.hpp"
#include "strata/kernels/mrope.hpp"
#include <cassert>
#include <iostream>
#include <vector>

using namespace strata::core;
static void cuda_check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
static void pattern(const uint8_t* p, size_t bytes, uint8_t expected) {
    std::vector<uint8_t> h(bytes);
    cuda_check(cudaMemcpy(h.data(), p, bytes, cudaMemcpyDeviceToHost));
    for (auto x : h) assert(x == expected);
}
int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices || !vmm_available()) {
        std::cout << "SKIP: native CUDA VMM device required\n"; return 77;
    }
    try {
        {
            // Inject a mid-growth failure without exhausting the machine's RAM.
            UnifiedKvVmm cache(8192);
            auto* p = (uint8_t*) cache.reserve({1024, 1, 8192, 2, 8192 * 1024 + 128});
            std::string error;
            assert(cache.grow(2048, error));
            const auto before = cache.committed_bytes();
            cuda_check(cudaMemset(p, 0x27, 2048 * 1024)); cuda_check(cudaDeviceSynchronize());
            int maps = 0;
            cache.mapping_test_hook = [&] { return ++maps != 2; };
            assert(!cache.grow(4096, error));
            assert(cache.committed_bytes() == before && !cache.faulted());
            pattern(p, 2048 * 1024, 0x27);
            cache.mapping_test_hook = {};
            assert(cache.grow(4096, error));
            pattern(p, 2048 * 1024, 0x27); pattern(p + 2048 * 1024, 1024, 0);
        }
        {
            constexpr uint64_t context = 8192, stride = context * 1024 + 128;
            UnifiedKvVmm a(context), b(context), c(context), d(context);
            UnifiedKvVmm* adapters[] = {&a, &b, &c, &d};
            std::vector<uint8_t*> heads;
            int invalidated[4] = {};
            for (int s = 0; s < 4; ++s) {
                heads.push_back((uint8_t*) adapters[s]->reserve({1024, 1, context, 2, stride}));
                (void) adapters[s]->reserve({4096, UINT64_MAX, 1, 1, 0}); // recurrent state
                adapters[s]->invalidate = [&,s](std::string&) { ++invalidated[s]; return true; };
                assert(adapters[s]->committed_bytes() == 0);
            }
            const uint64_t small = a.bytes_for(2048);
            UnifiedKvPool pool(context, 4 * context, 2 * small);
            KvIdentity ids[4];
            for (int s = 0; s < 4; ++s) ids[s] = pool.add(*adapters[s]);
            auto grow = [&](int s, uint64_t cells) { kv_require(pool.grow(ids[s], cells, [] { return UINT64_MAX; })); };
            for (int s = 0; s < 2; ++s) {
                kv_require(pool.begin(ids[s], context)); grow(s, 2048);
                cuda_check(cudaMemset(heads[s], 0x37, 2048 * 1024));
                cuda_check(cudaMemset(heads[s] + stride, 0x59, 2048 * 1024));
                kv_require(pool.end(ids[s], 2048));
            }
            // A captured pointer to head 1 survives growth, eviction, replay.
            cudaStream_t stream = nullptr; cudaGraph_t graph = nullptr; cudaGraphExec_t exec = nullptr;
            uint8_t* output = nullptr;
            cuda_check(cudaMalloc(&output, 4096)); cuda_check(cudaStreamCreate(&stream));
            cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            cuda_check(cudaMemcpyAsync(output, heads[0] + stride, 4096, cudaMemcpyDeviceToDevice, stream));
            cuda_check(cudaStreamEndCapture(stream, &graph));
            cuda_check(cudaGraphInstantiate(&exec, graph, 0)); cuda_check(cudaGraphDestroy(graph));
            // Touch 0: 1 must be the LRU, not the earliest allocated session.
            kv_require(pool.begin(ids[0], context)); kv_require(pool.end(ids[0], 2048));
            kv_require(pool.begin(ids[2], context)); grow(2, 2048); kv_require(pool.end(ids[2], 2048));
            const auto ev = pool.drain_events();
            assert(ev.size() == 1 && ev[0].id.slot == 1 && invalidated[1] == 1);
            assert(pool.begin(ids[1], context).code == KvCode::cache_miss);
            kv_require(pool.begin(ids[0], context)); grow(0, 4096);
            pattern(heads[0], 2048 * 1024, 0x37);
            pattern(heads[0] + stride, 2048 * 1024, 0x59);
            pattern(heads[0] + 2048 * 1024, 1024, 0); // fresh boundary is zero, no other session's KV
            cuda_check(cudaGraphLaunch(exec, stream)); cuda_check(cudaStreamSynchronize(stream));
            pattern(output, 4096, 0x59);
            kv_require(pool.end(ids[0], 4096));
            // Evict 0 by physical pressure: after any other idle victim it is the
            // only remaining cache. Replaying the old graph must read NEW pages.
            kv_require(pool.begin(ids[3], context)); grow(3, 4096); kv_require(pool.end(ids[3], 4096));
            assert(pool.session(0).state == KvResidency::evicted);
            kv_require(pool.begin(ids[0], context, true)); grow(0, 2048);
            pattern(heads[0] + stride, 4096, 0);
            cuda_check(cudaGraphLaunch(exec, stream)); cuda_check(cudaStreamSynchronize(stream));
            pattern(output, 4096, 0);
            kv_require(pool.end(ids[0], 2048));
            cuda_check(cudaGraphExecDestroy(exec)); cuda_check(cudaStreamDestroy(stream)); cuda_check(cudaFree(output));
        }
        // Exercise the ACTUAL QSA constructor/zero and all storage formats,
        // including authoritative streamed host memory and the draft ring.
        unified_qsa_enabled = true;
        for (int streamed : {0, 1, 2}) for (int format = 0; format < 4; ++format) {
            qsa_set_kv_int8(format == 1); qsa_set_kv_q4(format == 2); qsa_set_kv_hybrid(format == 3);
            qsa_set_kv_resident(streamed ? 20480 : 0);
            if (streamed == 2 && format == 3) { qsa_set_kv_hybrid(false); qsa_set_kv_int8(true); } // MtpDrafter's K8V4 rule
            ModelGeometry geometry;
            constexpr int64_t context = 32768;
            UnifiedKvVmm adapter(context);
            UnifiedKvPool pool(context, context);
            const auto id = pool.add(adapter);
            QsaState state;
            void* arena = nullptr;
            {
                UnifiedKvBinding binding(&adapter);
                const int64_t ring = streamed == 2 ? 512 : 0;
                const auto bytes = qsa_state_bytes(geometry, context, true, ring);
                cuda_check(cudaMalloc(&arena, bytes));
                assert(qsa_state_init(geometry, context, arena, state, nullptr, ring));
                qsa_state_zero(state, geometry, nullptr); // no dynamic pages at startup
                cuda_check(cudaDeviceSynchronize());
            }
            assert(adapter.committed_bytes() == 0);
            kv_require(pool.begin(id, context));
            kv_require(pool.grow(id, 512, [] { return UINT64_MAX; }));
            assert(adapter.committed_bytes() == adapter.bytes_for(512));
            qsa_state_zero(state, geometry, nullptr); cuda_check(cudaDeviceSynchronize());
            const auto pointer = state.kv_hybrid || state.kv_int8 ? (void*) state.k_q :
                                 state.kv_q4 ? (void*) state.k_q4 : (void*) state.k_pool;
            cuda_check(cudaMemset(pointer, 0x31, 128));
            kv_require(pool.grow(id, 8192, [] { return UINT64_MAX; }));
            pattern((const uint8_t*) pointer, 128, 0x31);
            kv_require(pool.end(id, 8192));
            KvIdentity next; kv_require(pool.reset(id, next)); assert(adapter.committed_bytes() == 0);
            assert(state.kv_unified_owner == &adapter); // the VA owner survives reset
            kv_require(pool.begin(next, context, true));
            kv_require(pool.grow(next, 512, [] { return UINT64_MAX; }));
            pattern((const uint8_t*) pointer, 128, 0);
            kv_require(pool.end(next, 512)); kv_require(pool.reset(next, next));
            strata::kernels::rope_table_release(state.cos_tab);
            cuda_check(cudaFreeHost(state.host_step)); cuda_check(cudaFreeHost(state.host_pos)); cuda_check(cudaFree(arena));
        }
        unified_qsa_enabled = false;
        std::cout << "native VMM, head boundaries, four owners, LRU, graph replay and QSA formats/stream/ring passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
