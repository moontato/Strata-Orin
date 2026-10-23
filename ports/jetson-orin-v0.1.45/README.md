# Strata-Orin: Jetson AGX Orin 64 GB development port of v0.1.45

**Status: source port, not an Orin-qualified release.** No CUDA compilation, GPU execution, model loading or Orin throughput was tested here. Do not interpret the CPU/cross-build checks below as those tests. The 500–800 prefill tok/s llama.cpp baseline remains a benchmark target, not a result of this port.

Branch: **`jetson-orin-v0.1.45`**, based directly on upstream **v0.1.45** (`477ac13cbe8e582c20484871ce459d20d82df09d`). The four mail patches in [`series`](series) are already applied as four commits, with their authors, dates and messages retained. Do **not** apply them again on this branch. No upstream main or Strata 0.2 code was merged.

Start with the [native build guide](BUILD.md) for **`/mnt/ssd/Strata-Orin`**, then follow the [hardware verification gates](VERIFY.md). [Local validation](VALIDATION.md) records this integration's commits, changed files, checks and untested areas. [`check_preservation.py`](check_preservation.py) checks release-source preservation, including the additive notice in the root README. The rest of the upstream documentation is unchanged; its installer/update instructions are **not qualified for Jetson**.

Unified-KV development is separate, on **`feature/unified-kv`**: see the [configuration, allocator and replay contract](../../docs/UNIFIED_KV.md) and its [build/test record](UNIFIED_KV_VALIDATION.md). It is not part of the original four-patch port below; Orin model/parity/performance qualification is still required for this feature.

The earlier fork analysis and test figures below came with the supplied patch bundle. They are retained as background, not presented as new measurements from this integration.

## 1. Establish the correct comparison

| Object | Pinned commit |
|---|---|
| Upstream **v0.1.45**, peeled tag | `477ac13cbe8e582c20484871ce459d20d82df09d` |
| Fork `feat/jetson-orin-arm64` tip examined | `0be090c8997b24cf5d21557ced40a1b7e93e87d3` |
| **Actual base / implementation's direct parent** (upstream v0.1.41) | `fb58e0dbc8399662c0e47c76578c6e878b14f6cf` |
| Fork's implementation commit | `89489e996b79344d49165a1eb001c5ba1d28081e` |

Reproduce after fetching those refs:

```sh
git merge-base 477ac13cbe8e582c20484871ce459d20d82df09d \
               0be090c8997b24cf5d21557ced40a1b7e93e87d3
git log --oneline fb58e0dbc8399662c0e47c76578c6e878b14f6cf..0be090c8997b24cf5d21557ced40a1b7e93e87d3
git diff --stat fb58e0dbc8399662c0e47c76578c6e878b14f6cf \
                0be090c8997b24cf5d21557ced40a1b7e93e87d3
```

The fork has only three commits after that base: implementation, benchmark graphs, and backend-build evidence. Its authored delta has **479 paths, 159,647 insertions / 173 deletions**; **432 paths are `bench/` evidence**, and **no `third_party/` path changed**. A direct release-to-fork comparison has 1,593 changed paths and also removes 886 intervening upstream commits. It is not a portability patch. No dependency tree, generated output, graph, build log, binary, or old engine snapshot is imported here.

Existing discussion was checked: [upstream PR #1621](https://github.com/Niko1221/Strata/pull/1621) (closed, not merged; related #1306). The fork reports hardware validation on **32 GB Orin, JetPack 6.2 / CUDA 12.6, Coder IQ1_M**. That does not validate 64 GB Orin, Qwen3.6, or v0.1.45. [Pinned fork plan/evidence](https://github.com/calvinvette/strata-inference-orin/blob/0be090c8997b24cf5d21557ced40a1b7e93e87d3/docs/JETSON_ORIN_PLAN.md).

## 2. Change-by-change disposition against v0.1.45

“Needed” below means a confirmed source/build gap or a conservative accounting correction, **not confirmed end-to-end runtime compatibility**.

| Area / files | v0.1.45 finding and disposition |
|---|---|
| `CMakeLists.txt`; `src/kernels/cpu/{expert,expert_layout,native_expert,pool}.cpp` | **Needed.** AVX source files/flags and CPUID headers are still unconditional. Guard them by target architecture; provide the fork's portable Q2 support and BF16 router functions; use the pinned ggml ARM backend for existing native CPU expert operations. Preserve v0.1.45's added AVX1, exact-singleton IQ and NVFP4 dispatch on x86—not the fork's older copies. |
| `include/strata/platform/{cpu_arch,cpu_relax}.hpp`; `src/core/graph.cpp`, `src/program/generate.cpp` | **Needed.** Replace x86-only spin intrinsics with `yield` on ARM, and make AVX startup refusals x86-only. Some Metal-related XGETBV guards exist upstream, but are not a complete Linux ARM port. CPU support remains linked even for all-GPU Qwen3.6. |
| SM87 flags / CUDA kernels | **Already supported at source level; no kernel patch.** Explicit `CMAKE_CUDA_ARCHITECTURES=87` is required because the release's default is 120. Its architecture gate accepts 87. The fork's authored delta modifies no production CUDA kernel except adding host-side device properties in `src/core/device.cu`. Keep the release's SM80+ tensor-core/MMQ paths; do not pretend SM87 is SM120 or enable Blackwell FP4 instructions. Native compilation remains untested. |
| `src/core/{session,verify,mtp}.cpp`; `include/strata/platform/mapped_memory.hpp` | **Needed portability; candidate ordering fix.** Replace `_mm_sfence` with an architecture helper and add ARM `dmb sy` at mapped publication/consumption sites, including the release's newer verifier paths. Keep x86 `pause`/`sfence`; added ARM-only sites do nothing on x86. GPU fences/poll kernels are unchanged. Barriers alone do not prove driver coherency or a sound concurrent protocol: hardware handshake/stress tests are required. |
| `src/core/expert_cache.cpp`; `include/strata/platform/shared_memory_budget.hpp` | **Partially upstream, correction needed for Tegra.** Upstream detects integrated GPUs and uses `/proc/meminfo`, but only raises CUDA free memory and can retain a larger CUDA value even when available RAM is below headroom. On **Linux aarch64 integrated CUDA only**, use `min(CUDA total, max(cgroup-bounded MemAvailable − headroom, 0))`. Preserve other devices' release accounting. This deliberately does not count swap as resident GPU capacity. |
| `src/core/expert_source.cpp` | **Needed edge case.** A valid zero `MemAvailable` must be distinguished from a missing field. Retain v0.1.45's cgroup-aware reader rather than reintroducing raw memory parsing. |
| `src/qwen35/q36_engine.cu` | **New adaptation, absent from fork.** Qwen3.6 has a separate placement/chunk/session planner using six direct `cudaMemGetInfo` calls. Route those queries through the corrected ARM integrated budget. No compute kernel, loader, format selection, placement algorithm, chunk heuristic or MTP logic is replaced. This accounting adaptation is source-reviewed and hardware-unverified. |
| `include/strata/core/device.hpp`; `src/core/{device.cu,device_main.cpp}` | **Missing diagnostics.** Expose integrated/managed/mapping properties. These are observations, not a request to allocate managed memory. |
| `src/core/pinned.cu`, CUDA initialization | **Already present / no production patch needed.** Release already clears errors after failed registration and retains pageable-storage/copy fallback. Neither the fork nor source inspection establishes a need for unconditional `cudaSetDeviceFlags`, context-reset hacks, or wholesale `cudaMallocManaged` replacement. Probe actual capabilities; test ownership transitions. |
| CUDA 12 compatibility / `src/prefill/`, kernel headers | **Keep upstream.** v0.1.45 already has CUDA-version/architecture guards and older-toolkit fixes. Use JetPack's matching 12.6 development/runtime stack; do not import older fork headers, CUDA 13 packaging, or Strata 0.2 code. |
| `setup.py`, `tools/vision/CMakeLists.txt` | Fork ARM discovery, no-x86-download, JetPack-library preservation and ARM vision-default changes are relevant to **installer support**, but **excluded from this native-CMake port**. Unmodified setup is not qualified for Orin. Never use its x86 prebuilts/desktop CUDA wheel path here. Optional vision can be compiled explicitly with `STRATA_PORTABLE=OFF`; not validated. |
| Fork `src/program/generate.cpp` auto-mmap policy, 3072 MiB extra reserve, 32K starter context | **Not transplanted.** Those are 32 GB Coder/Flash-Next policies, not measured 64 GB Qwen3.6 defaults. Automatic file-backed/hybrid placement would invalidate the benchmark goal. The release's own Qwen3.6 placement must report all experts resident or the benchmark is rejected. |
| Fork single-token `spec_pos` sentinel and `tools/mtp_pack.py` memory reduction | **Still absent in the release, but excluded as unrelated.** They change the old Flash-Next driver / offline MTP packer, not ARM compilation or Qwen3.6's separate engine. The Flash-Next position-zero issue remains outside this port. |
| Tests / probes | Add small, authored memory/quantization tests, ARM guards around **direct AVX-only subtests**, and 64 GiB budget cases. GPU parity sections remain enabled. The MMQ NVFP4 test gets ARM half conversion instead of being removed. No benchmark results are imported. |

Source references: [release build/architecture configuration](https://github.com/Niko1221/Strata/blob/477ac13cbe8e582c20484871ce459d20d82df09d/CMakeLists.txt), [upstream memory accounting](https://github.com/Niko1221/Strata/blob/477ac13cbe8e582c20484871ce459d20d82df09d/src/core/expert_cache.cpp#L27-L75), [Qwen3.6 engine](https://github.com/Niko1221/Strata/blob/477ac13cbe8e582c20484871ce459d20d82df09d/src/qwen35/q36_engine.cu), [release Qwen3.6 formats/optimizations](https://github.com/Niko1221/Strata/blob/477ac13cbe8e582c20484871ce459d20d82df09d/docs/QWEN36.md), [pinned allocation fallback](https://github.com/Niko1221/Strata/blob/477ac13cbe8e582c20484871ce459d20d82df09d/src/core/pinned.cu). NVIDIA's [CUDA 12.6 Tegra guide](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-for-tegra-appnote/index.html) distinguishes shared DRAM, pinned/mapped memory, managed allocations and coherence. Shared DRAM is not a second 64 GB GPU pool.

## 3. Patch series and preservation

1. `0001-arm64-build.patch`: architecture selection, CPU support symbols, spin helper, startup guard, portable **test** FP16 conversion.
2. `0002-mapped-memory-ordering.patch`: mapped barriers and current verifier/graph/MTP integration.
3. `0003-tegra-memory-budget.patch`: ARM integrated budgeting, properties, zero-available case, Qwen3.6 planner adaptation.
4. `0004-portability-tests.patch`: deterministic budget/CPU tests, CUDA probe/handshake/cache/registration tests; no model fixtures or result files.

Total: **38 source/config/test paths, 917 insertions / 73 deletions** against the clean tag. No new optimization-disable flag, CPU substitution for a GPU operation, quantization change, or model conversion is introduced. All `src/prefill/` production files, `src/kernels/` production CUDA files, `src/qwen35/q36_kernels.cuh`, artifact/load formats, quantization tools, server, dependencies and setup remain release-identical. Qwen3.6's `.cu` changes only include a memory helper and replace host memory queries. Run `check_preservation.py` to verify this mechanically.

Supported **upstream formats retained**, not newly qualified on Orin: Qwen3.6 UD-Q4_K_XL, UD-Q6_K_XL and Q8_0. Start with UD-Q4_K_XL. The native GGUF path requires no Flash-Next `iq_pack.py`/`mtp_pack.py` conversion. MTP, graphs, MMQ, K-quant kernels and auto chunk selection stay available.

## 4. Earlier patch-preparation checks and remaining risks

**Provenance:** the following test figures were supplied by the previous patch-preparation agent. They were not independently reproduced as those same runs. This integration's actual laptop checks are recorded separately in [VALIDATION.md](VALIDATION.md). In particular, no ARM cross-build or QEMU run was performed during this integration.

- Fresh x86-64 Release CPU-only build, GCC 12.2: **43/43 selected CTests passed**. The two fixture-dependent tests (`expert_parity`, `pool_test`) initially failed because `pack/full/experts.bin` is absent; excluded, not counted as passing.
- Clean release versus patched x86 build: **all 13 existing `strata_kernels_cpu` object `.text` sections are byte-identical**, including AVX1/AVX2/AVX512, NVFP4 and the pool. This is not full engine output identity or a performance measurement.
- ARM64 CPU-only cross-build and QEMU cortex-a76: **31/31 selected tests passed**, including Q2 reference, ARM ggml Q8 contract, router, multi-token and pool-task checks. GCC 12 cross-check used `-O2` after a compiler ICE in an unrelated test at `-O3`; a test-only wrapper translated `-mcpu=native` to cortex-a78. Unsupported i8mm/SVE feature-probe results were explicitly set false for this emulation. None of these harness settings is in the patches or native build commands. Excluded two missing fixtures, x86-env wrapper test, and long-running probability/bit-pattern/pool-stress tests. QEMU is not Orin or CUDA validation.
- All four patches applied cleanly with `git am` to another pristine release worktree; its tree matched the tested port. Source-preservation, patch whitespace, Python/shell syntax checks passed. No generated test/build artifacts are shipped.

**Outstanding:** native CUDA/SM87 build and all GPU tests, including SM87 kernel launch/shared-memory/register resource limits; NVIDIA driver/JetPack minor-version behavior; mapped/atomic coherence under stress; actual VMM and pinned-memory limits; peak/transient loader memory and fragmentation; long-context/multiple-session budgets; correctness against llama.cpp; all 64 GB Orin throughput, thermal and power measurements. OS availability is a snapshot, not an allocation reservation. Qwen3.6 explicit `--expert-vram` still overrides normal automatic placement; do not use it to overcommit physical RAM.

The Qwen3.6 hybrid host+GPU expert-copy path is preserved but **not capacity-qualified on shared RAM**; it can duplicate weights and is outside the all-resident benchmark. Reject it instead of counting it as success. Installer/update packaging, Windows ARM, QNX, SYCL Qwen3.6, other ARM GPUs, multi-GPU and vision are not qualified by this work. Blackwell-native W4A4 is unavailable on SM87; this port neither emulates it on CPU nor disables applicable Ampere GPU paths. Build every affected backend (CUDA/HIP/SYCL) and run actual hardware gates before requesting upstream review; that gate has **not** been met here.
