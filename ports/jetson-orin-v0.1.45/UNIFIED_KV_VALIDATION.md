# Unified KV: implementation and local validation

Date: **2026-10-23**. This is a **build/CPU-test record, not Orin qualification**.
No model inference, logits, GPU memory growth or throughput was measured on Orin
in this session. The operator-provided Orin results remain comparison targets.

## Revision and scope

* Development only on `feature/unified-kv`, descending directly from preserved
  Jetson baseline `ec8149d1bfcff264f077297aa6de94b22dc6e3bd`.
* Engine hardening commit: `283e72ea`; native validation tools: `2ebb9b20`;
  pitched-copy GPU gate: `d3141435`. Later documentation does not change the engine.
* The `jetson-orin-v0.1.45` ref was not changed. No upstream main rebase, merge,
  push, force-push, proxy repository change or new dependency pin was made.
* PR [#1614](https://github.com/Niko1221/Strata/pull/1614) was inspected. Its current
  implementation is based on 0.2.0-beta; it was a design reference only. None of
  its code was imported.

Both v0.1.45 architecture families are wired into the common manager: Qwen3.6's
head-major FP32/GDN/MTP state and Flash-Next's format-specific QSA pools,
streamed host backing, draft cache/ring and workspace-to-slot copies. Common
policy is GPU independent. It uses request leases, exact rounded commitments,
whole-idle-cache LRU and explicit invalidation/replay identities. The old allocator
remains selected unless the pool token flag is supplied.

See [UNIFIED_KV.md](../../docs/UNIFIED_KV.md) for byte accounting, states, all CLI
semantics, native protocol, replay contract, known costs and native commands.
This is source integration coverage, **not runtime coverage of every model**.

## Actual local results

Host: Linux 6.8.0-147-generic, **x86_64**, GCC/G++ **12.2.0**, Python **3.11.2**.
CMake **3.28.4** and NVIDIA CUDA 12.6 components (**nvcc 12.6.85**, **cudart
12.6.77**, **cuBLAS 12.6.4.1**) were assembled in session scratch space; no toolkit or model was installed in the repository.
Device code target was **SM87**, but the binaries' CPU side is x86_64, **not ARM64**.
The CUDA runtime found no usable NVIDIA GPU.

The ggml source was exactly v0.1.45's existing pin,
`3cf03257f219afbe7334045ff7c6a06ac68c627d`, not upstream latest.

| Check | Observed result |
|---|---|
| Full CUDA targets `strata`, `strata-q36`, `strata_kq`, `unified_kv_test`, `unified_kv_vmm_test` | **Built and linked**, Release, SM87, native experts and K-quant MMQ enabled |
| Executable version | `strata 0.1.45` |
| `unified_kv_test` | **Passed**: all four requested token distributions; ownership, logical/byte limits, rounding, physical-pressure retry, rollback contract, activity-order LRU, multiple victims, generations, replay, source pins, handoff and competing concurrent admissions/growth |
| Same policy tests with AddressSanitizer + UndefinedBehaviorSanitizer | **Passed**; no sanitizer findings |
| Full native-experts-on CPU build | **Built**, using the pinned ggml source |
| Full CPU CTest | **49 passed / 51 attempted**; two tests failed because `pack/full/experts.bin` is absent: `expert_parity`, `pool_test` |
| Full `serve` unittest discovery | **856 run, 11 skipped**, all 845 non-skipped tests passed; includes 8 new telemetry/real-pipe capacity-error tests |
| Probe/installer pin tests | **41 passed**: `tools.test_unified_kv_validate`, `tools.test_setup_qwen36`, `tools.test_setup_pins` |
| SYCL and HIP VMM fallback API probes | Their **portable stub sources compiled, linked and ran with GCC**, including the new checked-release interface; **not full backend builds** |
| Native CUDA VMM/graph/QSA test | **Compiled**, then **skipped (77)** because this host has no usable CUDA device; **not a runtime pass** |
| Protected source check against Jetson baseline | **No changes** in production kernels, prefill, platform/mapped-memory/shared-budget code, artifact formats or expert-cache placement source |
| Qwen3.6 CUDA kernel machine-code comparison | **63/63 kernel `.text` sections byte-identical**, baseline versus candidate, CUDA 12.6, SM87, Release/native/MMQ definitions |
| Orin model inference / logits / performance | **Not run** |
| Full HIP / SYCL / Metal builds; Windows host registration | **Not run**; required before claiming those backends are qualified |

An initial broad CPU build with `STRATA_NATIVE_EXPERTS=OFF` encountered undefined
AVX1 symbols in the existing CPU tests. That configuration was not repaired as
part of this feature. The native-experts-on pinned build above succeeded. No
numerical test was weakened or its tolerance changed to obtain these results.

### Kernel comparison boundary

Both `q36_engine.cu` files were compiled with CUDA 12.6 using:

```text
-cubin -O3 -DNDEBUG -std=c++20 -arch=sm_87
-DSTRATA_NATIVE_EXPERTS=1 -DSTRATA_PREFILL_FUSED=1
-DSTRATA_PREFILL_MMQ=1 -DSTRATA_Q36_CPU_POOL=1 -DSTRATA_Q6K_EXPERTS=1
-DSTRATA_VERSION="0.1.45"
```

The original source/headers were extracted read-only with `git archive ec8149d1`
into scratch; the stable branch was not checked out or modified. NVCC embeds a
source-path hash in anonymous-namespace **symbol names**. The comparison normalizes
only that part of section names, then compares hashes of the **unmodified section
bytes**. It does not ignore instructions, registers or numerical differences.

Reproduce with:

```bash
python3 tools/check_unified_kv_preservation.py \
  --baseline-cubin /path/to/baseline-q36.cubin \
  --candidate-cubin /path/to/candidate-q36.cubin
```

This proves preservation of those compiled CUDA kernel bodies in that toolchain.
It does **not** prove identical host scheduling, launch dimensions, model outputs,
whole executable bytes or throughput. The off-path source/compute checks are not
a substitute for actual same-model inference comparisons.

## Reproduction of available CPU tests

Use an out-of-tree build and the existing ggml pin:

```bash
cmake -S . -B /path/to/build-cpu -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_NATIVE_EXPERTS=ON \
  -DSTRATA_GGML_DIR=/path/to/pinned/llama.cpp \
  -DSTRATA_BUILD_TESTS=ON -DSTRATA_BUILD_CONVERSATION_TESTS=ON
cmake --build /path/to/build-cpu -j4
ctest --test-dir /path/to/build-cpu --output-on-failure
python3 -m unittest discover -s serve -t .
python3 -m unittest tools.test_unified_kv_validate tools.test_setup_qwen36 tools.test_setup_pins
```

The full CPU command needs the two external expert-pack fixtures to pass all 51
tests. Install the server's Python test dependencies before running discovery;
the native policy tests themselves need neither Python packages, CUDA nor models.

## Remaining hardware gates and limitations

1. Native **aarch64 / CUDA 13.2 / GCC 13.3** build on the specified Orin. Run
   `unified_kv_vmm_test` without a skip: real mappings, cross-granule host
   registration/UVA coherence, pitched copies across sparse head gaps,
   partial-allocation rollback, graph replay after
   release/recommit and real QSA format/stream/ring construction must pass.
2. Actual Qwen3.6 and Flash-Next model text/logit parity against non-unified runs.
   Exercise MTP0, MTP2/equivalent Flash draft windows, batch-MTP, GDN/PLE/indexer
   restoration, cancellation/reset, image/M-RoPE, recurrent checkpoints and
   repeated cache-boundary/slot transitions. Keep existing numerical checks.
3. Run 4K/32K/128K/250K and all mixed distributions; collect physical commitment,
   RSS/swap/MemAvailable, allocator/eviction latency and cache-miss replay costs.
   An idle cache evicted during a workload is not evidence that its old resident
   tokens still occupy the pool. Configured 500K alone proves nothing about RAM.
4. Interleaved same-setting performance A/B runs, with identical hashes, power,
   clocks, chunk and placement. Record native prefill/generation timings, wall
   latency, aggregate batch throughput, MTP acceptance, captures and replay cost.
   The operator's 1,131.6 tok/s prefill and 40.2 tok/s MTP2 generation measurements
   were **not reproduced here**.
5. Review the documented costs: Flash-Next retains fixed auxiliary arenas and
   uses temporary workspace copies/reservations. These may require extra memory
   or cause additional idle eviction. Expert-cache reserve must leave room; the
   pool never makes room by offloading experts or enabling swap. Unified
   Flash-Next currently excludes multi-GPU/elastic-expert combinations, the
   experimental 256-cell dense-attention diagnostic, BYIELD parking, and disk KV.

The implementation is prepared for those gates. It is **not a production-ready
Orin release claim**, a claim of complete hardware/model coverage, or a new
performance result. No push, merge or public review request was made.
