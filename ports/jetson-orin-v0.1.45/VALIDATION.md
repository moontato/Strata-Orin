# Strata-Orin v0.1.45: local integration record

**Status: source-port candidate, not hardware-qualified.** These checks were run during this integration on **2026-10-22 UTC**, on a Linux x86-64 laptop (Intel Core i7-1165G7). No Jetson, ARM compiler/emulator or CUDA toolkit was available. This record is separate from the previous agent's figures retained in [README.md](README.md).

## Git base and commits

- Branch: **`jetson-orin-v0.1.45`**.
- Exact upstream base: **`477ac13cbe8e582c20484871ce459d20d82df09d`**.
- Both the local peeled `v0.1.45` tag and `git ls-remote upstream 'refs/tags/v0.1.45^{}'` returned that commit. The annotated tag object is `b39442e6088d3ee405ea979723d4d47ba321a5bf`.
- The first port commit is a direct child of that base. The history after the base is linear: no merge from main, no older engine snapshot, no Strata 0.2 source.
- `project(strata VERSION 0.1.45 LANGUAGES CXX)` and setup's `MIN_ENGINE = (0, 1, 45)` remain in place. Setup, dependencies and upstream documentation other than the additive root README notice are release-identical.

The supplied series was applied with **`git am`**, in this order:

| Applied commit | Supplied patch | Purpose |
| --- | --- | --- |
| `9b41757f1b615d43a18ddd6a09ca2f2b6bbcb767` | `0001-arm64-build.patch` | ARM64 CPU support/build guards |
| `b115279919f400d785e3ae2a032487c259a342c6` | `0002-mapped-memory-ordering.patch` | ARM mapped-memory ordering |
| `6e8ca9ebe4173f4fd35fb7337c5e57698729949d` | `0003-tegra-memory-budget.patch` | Shared physical-RAM budgeting |
| `1275234355ef979c95c152863ed1ac8af7a0d558` | `0004-portability-tests.patch` | Portability tests and CUDA probes |

Each applied commit has the same stable patch ID as its supplied mail patch. The supplied authors, author dates, subjects and messages are retained. Git am creates new commit IDs because committer metadata changes. No Git identity was configured; per-command committer settings use the bundle's placeholder identity, `Strata Orin Port <orin-port@example.invalid>`. No global Git configuration was changed.

A subsequent documentation/tooling commit integrates this directory and the root notice. It does not change the engine beyond those four patches. Use `git log --oneline 477ac13cbe8e582c20484871ce459d20d82df09d..HEAD` for the complete local history.

The original `main`, `origin/main` and `upstream/main` refs were all `011c9e60` and were left unchanged. **Nothing was pushed.**

## Passed

| Check | Result and scope |
| --- | --- |
| `check_preservation.py` | Passed before documentation integration (38 allowed paths) and with the integrated README notice. All other tracked release sources are unchanged. |
| Preservation-gate self-tests | 8 acceptance/rejection cases passed: patch-only tree and additive notice accepted; upstream README edits, production prefill edits, version bumps, feature-default changes, CUDA-config changes and Qwen3.6 compute edits rejected. |
| Patch preservation | All four stable patch IDs and author/date/message metadata match, in series order; the four supplied patch files and `series` are unchanged. A fresh `git am` replay on a pristine base checkout produced the same engine tree. |
| Whitespace and syntax | `git diff --check` passed for the applied source, and for the full integration with the mail-patch attribute described below; Python checker compiles; `bash -n` passed for the benchmark helper and all 16 shell blocks in BUILD.md/VERIFY.md. |
| CPU support build | Fresh Release build passed with GCC/G++ 12.2.0, CMake 3.28.4, Unix Makefiles, two jobs, native experts enabled and the exact ggml dependency pin. GPU backends were off for this laptop-only test build, not for the Orin build. |
| CTest | **48/48 selected tests passed**, none skipped. Includes the five standalone conversation tests, Q2 reference, Q8 contract, shared-memory budget, router, native CPU expert/pool tests, ISA guard and pool stress. |
| Python regressions | **36/36 passed**: `tools.test_setup_qwen36` and `tools.test_setup_pins`. Downloads, engine starts and package installs are mocked. This does not qualify the installer for Jetson. |
| x86 code preservation | Clean base and patched builds have byte-identical `.text` sections in all 13 existing `strata_kernels_cpu` object files. Not a full-binary, model-output or performance comparison. |
| SM87 source gate | The isolated, unchanged CMake architecture gate accepts architecture `87` with toolkit version `12.6`. This is not a CUDA configure or compilation. |

The first full-integration whitespace check flagged 53 empty context lines in the stored mail patches: each correctly contains the unified-diff single-space prefix. A directory-local `.gitattributes` disables only end-of-line whitespace checking for `patches/*.patch`, preserving their exact bytes. Applied source files keep normal whitespace checks. No source whitespace error was hidden or fixed this way.

The CPU build emitted GCC warnings in string and AVX intrinsic code. It was **not warning-clean**. `STRATA_WERROR` stayed at its upstream default (`OFF`); no warning or optimization flags were suppressed to pass the build.

### CPU build and test commands

Run from the repository root with CMake 3.28.4, GCC/G++ 12.2.0 and a scratch location. The dependency was fetched at **`3cf03257f219afbe7334045ff7c6a06ac68c627d`**, exactly as pinned by v0.1.45; no latest llama.cpp substitution was used.

```sh
# Set these to scratch paths outside this repository.
export CHECK_ROOT=/absolute/path/to/scratch
export LLAMA_SRC="$CHECK_ROOT/llama.cpp"
# LLAMA_SRC must contain the full llama.cpp checkout at the pinned commit.
test "$(git -C "$LLAMA_SRC" rev-parse HEAD)" = 3cf03257f219afbe7334045ff7c6a06ac68c627d
cmake -S . -B "$CHECK_ROOT/build-cpu" -G 'Unix Makefiles' \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DSTRATA_GGML_DIR="$LLAMA_SRC" \
  -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF -DSTRATA_ENABLE_SYCL=OFF \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_BUILD_CONVERSATION_TESTS=ON
cmake --build "$CHECK_ROOT/build-cpu" -j2
ctest --test-dir "$CHECK_ROOT/build-cpu" --output-on-failure --timeout 180 -j2 \
  -E '^(expert_parity|pool_test)$'
python3 -m unittest -v tools.test_setup_qwen36 tools.test_setup_pins
python3 ports/jetson-orin-v0.1.45/check_preservation.py --repo "$PWD"
git diff --check 477ac13cbe8e582c20484871ce459d20d82df09d
```

CTest registered 50 tests. **Not run:** `expert_parity` and `pool_test`, which require the absent real `pack/full/experts.bin` fixture. They are excluded, not counted as passes. No model fixture was fabricated.

For the object comparison, the base was exported with `git archive` into scratch and configured with the same settings/dependency/compiler. Its `strata_kernels_cpu` target was built. `objcopy --dump-section .text=...` extracted and compared each corresponding object section. The 13 objects were: `expert`, `expert_layout`, `iq_avx1`, `iq_avx2`, `iq_avx512`, `kq_avx1`, `kq_avx2`, `native_expert`, `nvfp4_avx2`, `nvfp4_avx512`, `pool`, `q2_avx1`, `q2_avx2` (all `.cpp.o`). Builds, downloads and throwaway test repositories were kept outside the project in session scratch.

## Source features retained, not hardware-tested

- **SM87:** explicit `CMAKE_CUDA_ARCHITECTURES=87` in the native guide; unchanged release CUDA architecture/toolkit gate. No SM120 masquerading or Blackwell-only FP4 instructions forced onto Ampere.
- **Qwen3.6-35B-A3B:** the full loader/compute/dispatch body is release-identical after removing only the audited host-memory-query helper/include and reversing its six call-site substitutions. `q36_kernels.cuh`, GGUF formats and model pins are unchanged.
- **K-quant MMQ:** the release CUDA MMQ sources, K-quant shared-library build/loader, and Qwen3.6 runtime enablement are unchanged. Keep `libstrata_kq.so` beside the executable; verify it actually loads on Orin.
- **Native experts:** remain enabled; GPU expert implementations are unchanged. The portable CPU support functions do not replace GPU inference operations.
- **Optimized prefill and graphs:** production `src/prefill/` files and production CUDA kernels are unchanged. The only prefill edit is the ARM FP16 conversion in a parity test. CUDA graph capture, optimized Qwen3.6 dispatch and automatic prompt-chunk selection remain available.
- **Defaults:** all upstream CMake feature-option defaults and CUDA configuration are preserved. No new optimization-disable setting or CPU fallback was introduced.

### Build-guide correction

The supplied BUILD.md incorrectly requested `STRATA_PREFILL_MMQ=ON` for CUDA. In v0.1.45 that **cache option is HIP-only**, and the top-level guard rejects it without HIP. The native guide now passes `STRATA_PREFILL_MMQ=OFF`, without changing CMake or the patch series. On CUDA, `STRATA_NATIVE_EXPERTS=ON` still builds `strata_mmq` and defines `STRATA_PREFILL_MMQ=1` for the prefill library; `STRATA_MMQ_KQUANTS=ON` builds `strata_kq`. This corrects an invalid command, **not a CUDA MMQ disablement**.

## Changed-file inventory against the exact release

The four patches change **38 source/config/test files, 917 insertions and 73 deletions**:

```text
CMakeLists.txt
include/strata/core/device.hpp
include/strata/platform/cpu_arch.hpp
include/strata/platform/cpu_relax.hpp
include/strata/platform/mapped_memory.hpp
include/strata/platform/shared_memory_budget.hpp
src/core/device.cu
src/core/device_main.cpp
src/core/expert_cache.cpp
src/core/expert_cache_memory_test.cpp
src/core/expert_source.cpp
src/core/graph.cpp
src/core/mtp.cpp
src/core/pinned_fallback_test.cpp
src/core/session.cpp
src/core/verify.cpp
src/kernels/cpu/expert.cpp
src/kernels/cpu/expert_layout.cpp
src/kernels/cpu/expert_multi_test.cpp
src/kernels/cpu/expert_parity.cpp
src/kernels/cpu/native_expert.cpp
src/kernels/cpu/pool.cpp
src/kernels/cpu/pool_stress.cpp
src/kernels/cpu/pool_tasks_test.cpp
src/kernels/cpu/pool_test.cpp
src/kernels/cpu/q2_portable_reference_test.cpp
src/kernels/cpu/q8_oracle_contract_test.cpp
src/kernels/cpu/router_dot_parity.cpp
src/kernels/cpu/router_portable.cpp
src/kernels/native_expert_parity.cpp
src/kernels/quantize_act_parity.cpp
src/platform/shared_memory_budget_test.cpp
src/prefill/mmq_nvfp4_parity.cpp
src/program/generate.cpp
src/qwen35/q36_engine.cu
tests/core/file_expert_source_test.cpp
tools/cuda_memory_selftest.cu
tools/cuda_platform_probe.cpp
```

Documentation/tooling integration adds **13 more paths** (51 total):

```text
README.md                                      (additive port notice only)
ports/jetson-orin-v0.1.45/.gitattributes
ports/jetson-orin-v0.1.45/README.md
ports/jetson-orin-v0.1.45/BUILD.md
ports/jetson-orin-v0.1.45/VERIFY.md
ports/jetson-orin-v0.1.45/VALIDATION.md
ports/jetson-orin-v0.1.45/series
ports/jetson-orin-v0.1.45/check_preservation.py
ports/jetson-orin-v0.1.45/benchmark_resident.sh
ports/jetson-orin-v0.1.45/patches/0001-arm64-build.patch
ports/jetson-orin-v0.1.45/patches/0002-mapped-memory-ordering.patch
ports/jetson-orin-v0.1.45/patches/0003-tegra-memory-budget.patch
ports/jetson-orin-v0.1.45/patches/0004-portability-tests.patch
```

## Not run and remaining risks

- Native ARM64 / CUDA 12.6 / SM87 compilation and all GPU tests. HIP and SYCL builds were also unavailable. This is not ready for upstream backend-validation review.
- GPU kernel resource limits, CUDA graph execution, mapped CPU/GPU coherency and doorbell stress, actual VMM/pinned registration limits and JetPack driver behavior.
- Model loading, full all-GPU expert residency, transient/peak RAM, fragmentation, long contexts, multiple sessions, model/logit correctness, vision, API serving and performance.
- The previous agent's ARM/QEMU checks were not rerun. They are not Orin evidence.
- A 64 GB Orin has one shared physical-RAM pool, not 64 GB host RAM plus 64 GB VRAM. The budget is a snapshot, not an allocation reservation. Larger formats, explicit `--expert-vram` overrides and hybrid copies can still overcommit RAM.
- The release's hybrid mode remains in the source. A run that selects it is **not** an accepted all-GPU result. The helper's log checks are not enough on their own: require the CUDA trace and memory/I/O evidence in [VERIFY.md](VERIFY.md).

Next: transfer the reviewed committed branch using [BUILD.md](BUILD.md), build natively at **`/mnt/ssd/Strata-Orin`**, and run [VERIFY.md](VERIFY.md) in order. Do not push, change the GitHub default branch or import upstream main as part of that step.
