// Native CUDA gate: no model fixture. Run on Orin; a compile is not a pass.
#include "strata/core/unified_kv_vmm.hpp"
#include "strata/core/unified_kv_binding.hpp"
#include "strata/core/unified_kv_protocol.hpp"
#include "strata/core/layer.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/platform/mapped_memory.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace strata::core;
__global__ void copy_bytes(const uint8_t* source, uint8_t* destination, int n) {
    for (int i = (int) threadIdx.x; i < n; i += (int) blockDim.x) destination[i] = source[i];
}
static std::string active_phase = "device discovery";
static void test_phase(const std::string& phase) {
    active_phase = phase;
    // Keep a breadcrumb even for an assertion failure or an internal backend
    // error. No extra CUDA calls/synchronization are introduced by diagnostics.
    std::cerr << "[unified_kv_vmm_test] phase: " << active_phase << '\n';
}
static void cuda_check_at(cudaError_t error, const char* expression, const char* file, int line) {
    if (error == cudaSuccess) return;
    const char* name = cudaGetErrorName(error);
    const char* description = cudaGetErrorString(error);
    std::ostringstream message;
    message << file << ':' << line << ": CUDA API failed\n"
            << "  expression: " << expression << '\n'
            << "  error: " << (name ? name : "<unknown CUDA error>")
            << " (code " << static_cast<int>(error) << "): "
            << (description ? description : "<no CUDA error description>");
    throw std::runtime_error(message.str());
}
// Deliberately keep the existing spelling, but make it a CALL-SITE macro:
// evaluate the API once; capture its original expression/file/line in Release.
#define cuda_check(expression) cuda_check_at((expression), #expression, __FILE__, __LINE__)

static void pattern_at(const uint8_t* p, size_t bytes, uint8_t expected,
                       const char* expression, const char* file, int line) {
    std::vector<uint8_t> h(bytes);
    try {
        cuda_check(cudaMemcpy(h.data(), p, bytes, cudaMemcpyDefault));
    } catch (const std::exception& error) {
        // The failing API is inside this helper; retain the OUTER readback call
        // as well, so multiple pattern() checks in one phase are distinguishable.
        std::ostringstream message;
        message << error.what() << '\n'
                << "  readback requested at " << file << ':' << line << ": " << expression << '\n'
                << "  source=" << static_cast<const void*>(p) << " bytes=" << bytes
                << " expected=" << static_cast<unsigned>(expected);
        throw std::runtime_error(message.str());
    }
    for (auto x : h) assert(x == expected);
}
#define pattern(pointer, bytes, expected) \
    pattern_at((pointer), (bytes), (expected), "pattern(" #pointer ", " #bytes ", " #expected ")", __FILE__, __LINE__)
int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) {
        std::cout << "SKIP: native CUDA device required\n"; return 77;
    }
    try {
        test_phase("device setup / VMM capability");
        cuda_check(cudaSetDevice(0));
        if (!vmm_available()) { std::cout << "SKIP: CUDA VMM required\n"; return 77; }
        int max_pitch = 0;
        cuda_check(cudaDeviceGetAttribute(&max_pitch, cudaDevAttrMaxPitch, 0));
        {
            // Inject a mid-growth failure without exhausting the machine's RAM.
            test_phase("VMM growth / initial head-prefix commit");
            UnifiedKvVmm cache(8192);
            auto* p = (uint8_t*) cache.reserve({1024, 1, 8192, 2, 8192 * 1024 + 128});
            std::string error;
            assert(cache.grow(2048, error));
            const auto before = cache.committed_bytes();
            cuda_check(cudaMemset(p, 0x27, 2048 * 1024)); cuda_check(cudaDeviceSynchronize());
            test_phase("VMM growth / injected mapping failure and rollback");
            int maps = 0;
            cache.mapping_test_hook = [&] { return ++maps != 2; };
            assert(!cache.grow(4096, error));
            assert(cache.committed_bytes() == before && !cache.faulted());
            pattern(p, 2048 * 1024, 0x27);
            test_phase("VMM growth / retry and old-content preservation");
            cache.mapping_test_hook = {};
            assert(cache.grow(4096, error));
            pattern(p, 2048 * 1024, 0x27); pattern(p + 2048 * 1024, 1024, 0);
        }
        // Exercise the real Qwen3.6 head stride AND a head prefix starting
        // inside a physical granule (and crossing its far boundary).
        for (uint64_t padding : {uint64_t{0}, uint64_t{128}}) {
            test_phase("head mapping / four-owner reservation");
            constexpr uint64_t context = 8192, prefix_bytes = 2048 * 1024;
            const uint64_t stride = context * 1024 + padding;
            const KvPrefix head_layout{1024, 1, context, 2, stride};
            const uint64_t gran = vmm_granularity();
            std::cerr << "[head layout] stride=" << stride << " prefix_bytes=" << prefix_bytes
                      << " granularity=" << gran << " cuda_max_pitch=" << max_pitch
                      << " within_2d_pitch_limit=" << (max_pitch > 0 && stride <= (uint64_t) max_pitch) << '\n';
            UnifiedKvVmm a(context), b(context), c(context), d(context);
            UnifiedKvVmm* adapters[] = {&a, &b, &c, &d};
            std::vector<uint8_t*> heads;
            int invalidated[4] = {};
            for (int s = 0; s < 4; ++s) {
                heads.push_back((uint8_t*) adapters[s]->reserve(head_layout));
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
                test_phase("head mapping / initial prefix commit and fill / owner " + std::to_string(s));
                kv_require(pool.begin(ids[s], context)); grow(s, 2048);
                cuda_check(cudaMemset(heads[s], 0x37, 2048 * 1024));
                cuda_check(cudaMemset(heads[s] + stride, 0x59, 2048 * 1024));
                kv_require(pool.end(ids[s], 2048));
            }
            // A sparse VA reservation is not one densely backed pitched
            // allocation. CUDA 2D D2D copies can reject arbitrary pitches even
            // below cudaDevAttrMaxPitch. Copy each mapped head prefix, as the
            // pooled Qwen3.6 session-copy path does; do not map the gaps.
            test_phase("head mapping / committed row and non-overlap validation");
            assert(prefix_bytes <= stride);
            const auto source = reinterpret_cast<uintptr_t>(heads[0]);
            const auto destination = reinterpret_cast<uintptr_t>(heads[1]);
            const uint64_t extent = head_layout.extent();
            assert(source <= UINTPTR_MAX - extent && destination <= UINTPTR_MAX - extent);
            assert(source + extent <= destination || destination + extent <= source);
            const auto chunks = head_layout.chunks(2048, gran);
            const uint64_t gap = kv_ceil(prefix_bytes, gran);
            assert(gap < stride / gran); // at least one physical granule in the gap
            assert(std::find(chunks.begin(), chunks.end(), gap) == chunks.end());
            assert(small < a.bytes_for(context)); // the full context was not mapped
            assert(a.committed_bytes() == small && b.committed_bytes() == small);
            kv_require(pool.begin(ids[0], context)); kv_require(pool.begin(ids[1], context));
            cudaStream_t stream = nullptr;
            cuda_check(cudaStreamCreate(&stream));
            cuda_check(cudaMemsetAsync(heads[1], 0x11, prefix_bytes, stream));
            cuda_check(cudaMemsetAsync(heads[1] + stride, 0x11, prefix_bytes, stream));
            cuda_check(cudaStreamSynchronize(stream));
            // Full readbacks prove both rows are accessible, and writing the
            // destination did not modify/alias either source head.
            pattern(heads[0], prefix_bytes, 0x37); pattern(heads[0] + stride, prefix_bytes, 0x59);
            pattern(heads[1], prefix_bytes, 0x11); pattern(heads[1] + stride, prefix_bytes, 0x11);
            test_phase("head mapping / per-head async copy and untouched tail guards");
            for (uint64_t h = 0; h < 2; ++h)
                cuda_check(cudaMemcpyAsync(heads[1] + h * stride, heads[0] + h * stride,
                                          prefix_bytes - 256, cudaMemcpyDeviceToDevice, stream));
            cuda_check(cudaStreamSynchronize(stream));
            pattern(heads[1], prefix_bytes - 256, 0x37); pattern(heads[1] + stride, prefix_bytes - 256, 0x59);
            for (uint64_t h = 0; h < 2; ++h) pattern(heads[1] + h * stride + prefix_bytes - 256, 256, 0x11);
            test_phase("head mapping / full per-head async copy across sparse prefixes");
            for (uint64_t h = 0; h < 2; ++h)
                cuda_check(cudaMemcpyAsync(heads[1] + h * stride, heads[0] + h * stride,
                                          prefix_bytes, cudaMemcpyDeviceToDevice, stream));
            cuda_check(cudaStreamSynchronize(stream));
            pattern(heads[1], 2048 * 1024, 0x37); pattern(heads[1] + stride, 2048 * 1024, 0x59);
            pattern(heads[0], prefix_bytes, 0x37); pattern(heads[0] + stride, prefix_bytes, 0x59);
            assert(a.committed_bytes() == small && b.committed_bytes() == small);
            kv_require(pool.end(ids[1], 2048)); kv_require(pool.end(ids[0], 2048));
            // A captured pointer to head 1 survives growth, eviction, replay.
            test_phase("graph pointer reuse / capture head-1 reader");
            cudaGraph_t graph = nullptr; cudaGraphExec_t exec = nullptr;
            uint8_t* output = nullptr;
            cuda_check(cudaMalloc(&output, 4096));
            cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            copy_bytes<<<1, 128, 0, stream>>>(heads[0] + stride, output, 4096);
            cuda_check(cudaGetLastError());
            cuda_check(cudaStreamEndCapture(stream, &graph));
            cuda_check(cudaGraphInstantiate(&exec, graph, 0)); cuda_check(cudaGraphDestroy(graph));
            // Touch 0: 1 must be the LRU, not the earliest allocated session.
            test_phase("head mapping / activity-order LRU eviction");
            kv_require(pool.begin(ids[0], context)); kv_require(pool.end(ids[0], 2048));
            kv_require(pool.begin(ids[2], context)); grow(2, 2048); kv_require(pool.end(ids[2], 2048));
            const auto ev = pool.drain_events();
            assert(ev.size() == 1 && ev[0].id.slot == 1 && invalidated[1] == 1);
            assert(pool.begin(ids[1], context).code == KvCode::cache_miss);
            test_phase("graph pointer reuse / growth across head boundary and replay");
            kv_require(pool.begin(ids[0], context)); grow(0, 4096);
            pattern(heads[0], 2048 * 1024, 0x37);
            pattern(heads[0] + stride, 2048 * 1024, 0x59);
            pattern(heads[0] + 2048 * 1024, 1024, 0); // fresh boundary is zero, no other session's KV
            cuda_check(cudaGraphLaunch(exec, stream)); cuda_check(cudaStreamSynchronize(stream));
            pattern(output, 4096, 0x59);
            kv_require(pool.end(ids[0], 4096));
            // Evict 0 by physical pressure: after any other idle victim it is the
            // only remaining cache. Replaying the old graph must read NEW pages.
            test_phase("graph pointer reuse / eviction, recommit and replay");
            kv_require(pool.begin(ids[3], context)); grow(3, 4096); kv_require(pool.end(ids[3], 4096));
            assert(pool.session(0).state == KvResidency::evicted);
            kv_require(pool.begin(ids[0], context, true)); grow(0, 2048);
            pattern(heads[0] + stride, 4096, 0);
            cuda_check(cudaGraphLaunch(exec, stream)); cuda_check(cudaStreamSynchronize(stream));
            pattern(output, 4096, 0);
            kv_require(pool.end(ids[0], 2048));
            test_phase("graph pointer reuse / cleanup");
            cuda_check(cudaGraphExecDestroy(exec)); cuda_check(cudaStreamDestroy(stream)); cuda_check(cudaFree(output));
        }
#if defined(__linux__)
        {
            // Several independent pinned registrations in one contiguous VA,
            // CPU/GPU coherence, cross-registration copies and graph reuse after
            // all physical host pages were discarded and registered again.
            test_phase("host registration / incremental commit and CPU-GPU coherence");
            const uint64_t gran = vmm_granularity();
            UnifiedKvVmm adapter(3 * gran);
            auto* host = (uint8_t*) adapter.reserve({1, 1, 3 * gran, 1, 0}, true);
            UnifiedKvPool pool(3 * gran, 3 * gran);
            auto id = pool.add(adapter);
            kv_require(pool.begin(id, 3 * gran));
            kv_require(pool.grow(id, gran - 16, [] { return UINT64_MAX; }));
            assert(adapter.committed_bytes() == gran);
            std::memset(host, 0x73, (size_t) gran);
            strata::platform::mapped_memory_release();
            kv_require(pool.grow(id, 2 * gran + 16, [] { return UINT64_MAX; }));
            assert(adapter.committed_bytes() == 3 * gran);
            pattern(host, 128, 0x73); pattern(host + gran, 128, 0);
            test_phase("host registration / cross-registration copy");
            uint8_t* out = nullptr;
            cudaStream_t stream = nullptr; cudaGraph_t graph = nullptr; cudaGraphExec_t exec = nullptr;
            cuda_check(cudaMalloc(&out, 128)); cuda_check(cudaStreamCreate(&stream));
            // These helpers also copy contiguous runs across registered chunks.
            cuda_check(cudaMemcpy(out, host + gran - 64, 128, cudaMemcpyDefault));
            pattern(out, 64, 0x73); pattern(out + 64, 64, 0);
            test_phase("host registration / graph capture and replay");
            cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            copy_bytes<<<1, 128, 0, stream>>>(host + gran - 64, out, 128);
            cuda_check(cudaGetLastError()); cuda_check(cudaStreamEndCapture(stream, &graph));
            cuda_check(cudaGraphInstantiate(&exec, graph, 0)); cuda_check(cudaGraphDestroy(graph));
            cuda_check(cudaGraphLaunch(exec, stream)); cuda_check(cudaStreamSynchronize(stream));
            pattern(out, 64, 0x73); pattern(out + 64, 64, 0);
            test_phase("host registration / reclaim, re-register and graph pointer reuse");
            kv_require(pool.end(id, 2 * gran)); kv_require(pool.reset(id, id));
            assert(adapter.committed_bytes() == 0);
            kv_require(pool.begin(id, 3 * gran, true));
            kv_require(pool.grow(id, 2 * gran + 16, [] { return UINT64_MAX; }));
            cuda_check(cudaGraphLaunch(exec, stream)); cuda_check(cudaStreamSynchronize(stream));
            pattern(out, 128, 0);
            strata::platform::mapped_memory_acquire(); assert(host[gran - 1] == 0 && host[gran] == 0);
            kv_require(pool.end(id, 2 * gran)); kv_require(pool.reset(id, id));
            test_phase("host registration / cleanup");
            cuda_check(cudaGraphExecDestroy(exec)); cuda_check(cudaStreamDestroy(stream)); cuda_check(cudaFree(out));
        }
#endif
        // Exercise the ACTUAL QSA constructor/zero and all storage formats,
        // including authoritative streamed host memory and the draft ring.
        unified_qsa_enabled = true;
        const char* formats[] = {"FP16", "INT8", "Q4_0", "K8V4"};
        const char* placements[] = {"device resident", "host streamed", "draft ring"};
        for (int streamed : {0, 1, 2}) for (int format = 0; format < 4; ++format) {
            const std::string qsa_case = std::string("QSA [format=") + formats[format] + ", placement=" + placements[streamed]
                + (streamed == 2 && format == 3 ? ", effective=INT8 (MTP K8V4 rule)]" : "]");
            test_phase(qsa_case + " / constructor and initial zero");
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
            test_phase(qsa_case + " / first physical commit");
            assert(adapter.committed_bytes() == 0);
            kv_require(pool.begin(id, context));
            kv_require(pool.grow(id, 512, [] { return UINT64_MAX; }));
            assert(adapter.committed_bytes() == adapter.bytes_for(512));
            test_phase(qsa_case + " / zero committed state and fill prefix");
            qsa_state_zero(state, geometry, nullptr); cuda_check(cudaDeviceSynchronize());
            const auto pointer = state.kv_hybrid || state.kv_int8 ? (void*) state.k_q :
                                 state.kv_q4 ? (void*) state.k_q4 : (void*) state.k_pool;
            cuda_check(cudaMemset(pointer, 0x31, 128));
            test_phase(qsa_case + " / growth and prefix preservation");
            kv_require(pool.grow(id, 8192, [] { return UINT64_MAX; }));
            pattern((const uint8_t*) pointer, 128, 0x31);
            kv_require(pool.end(id, 8192));
            test_phase(qsa_case + " / reset, recommit and zeroed-page check");
            KvIdentity next; kv_require(pool.reset(id, next)); assert(adapter.committed_bytes() == 0);
            assert(state.kv_unified_owner == &adapter); // the VA owner survives reset
            kv_require(pool.begin(next, context, true));
            kv_require(pool.grow(next, 512, [] { return UINT64_MAX; }));
            pattern((const uint8_t*) pointer, 128, 0);
            kv_require(pool.end(next, 512)); kv_require(pool.reset(next, next));
            test_phase(qsa_case + " / cleanup");
            strata::kernels::rope_table_release(state.cos_tab);
            cuda_check(cudaFreeHost(state.host_step)); cuda_check(cudaFreeHost(state.host_pos)); cuda_check(cudaFree(arena));
        }
        unified_qsa_enabled = false;
        std::cout << "native VMM, head boundaries, four owners, LRU, graph replay and QSA formats/stream/ring passed\n";
    } catch (const std::exception& error) {
        std::cerr << "unified_kv_vmm_test FAILED [phase: " << active_phase << "]\n" << error.what() << '\n'; return 1;
    }
}
