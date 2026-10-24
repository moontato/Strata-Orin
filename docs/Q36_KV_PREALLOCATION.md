# Opt-in Qwen3.6 physical KV preallocation (review only)

Based on sampled-MTP commit `f121cd08b17f43b219df27625fcbdf3bbf1c1204`.
**Not GPU-qualified here. No deployment or speedup claim.** All existing defaults
remain unchanged. Without `--kv-preallocate`, unified KV still allocates lazily
and discard still releases its physical backing.

Requested future configuration, **not an edit to a deployed configuration**:

```text
--max-context 262144 --batch 4 --kv-pool-tokens 500000
--kv int8 --mtp 2 --batch-mtp --mtp-sampled --chunk 512
--kv-checkpoint-mib 1024 --kv-checkpoint-interval 1024 --kv-preallocate
```

The context is the existing engine contract: prompt, generated tokens and the
engine's write/safety allowances must fit. This is NOT a 262,144-token prompt
plus extra output. The logical shared ceiling remains exactly **500,000**, not
524,288 or 1,048,576. All four slots retain their full per-slot virtual extent;
there are no fixed 125,000-token partitions. Two simultaneously active full
262,144-token reservations are refused. Idle caches may still be evicted for
logical admission; active leases and pinned checkpoint donors are protected.

## Scope and startup order

Supported activation is deliberately narrow: CUDA SM87, text Qwen3.6, INT8 KV,
four slots, MTP2, batch-MTP, explicit chunk and warmup. Sampled MTP and checkpoints
remain separate opt-ins and retain their existing semantics. Images, other KV
formats, other models/backends/devices, automatic chunk, no-warmup and explicit
expert-placement overrides are refused with an actionable error. The no-model
allocator test itself can run on other CUDA VMM devices, but does not qualify
inference there.

1. Load the same dense/packed weights and allocate existing shared engine scratch,
   tables, cuBLAS/GEMM/MMQ buffers. Reserve per-slot virtual ranges only.
2. Derive the bank bound from those actual per-slot layouts, including all loaded
   main/MTP code and scale planes, state and configured checkpoint banks. Check
   `--kv-pool-mib` and fresh device/host memory after the existing headroom guard.
3. Allocate **all** physical VMM handles. A partial allocation failure aborts startup;
   already-created handles remain owned until cleanup. No retry as lazy KV.
4. Place every main **and MTP** routed expert on GPU beside the bank. Strict mode
   refuses hybrid placement, packed-copy giveback, score-buffer shrinking or reduced
   headroom. It does not adapt the requested runtime configuration to make it fit.
5. Warm each slot, including the full and shortened MTP verification graphs, then
   reset it. Reset returns its pages, not the bank's memory. Check fresh headroom
   again before the ready boundary. Additional batch graph shapes retain on-demand
   capture, with strict before/after headroom checks; an over-budget new graph is
   destroyed and refused rather than consuming the reserve unnoticed.

The bank covers managed KV/state/checkpoints, not CUDA driver bookkeeping. Total
startup memory must also include weights, all routed experts, packed decode
copies, shared scratch, library workspaces, captured graphs and headroom. These
are charged by fresh physical readings, not added to an invented free-memory
budget. On UMA the existing cgroup-aware device budget and
`STRATA_UMA_HEADROOM_GIB` (default 6 GiB), plus the pool's 768 MiB reserve, remain.
No swap or CPU-free-plus-CUDA-free arithmetic is used.

In particular, persistent MTP seed `hlast` is in the bank's state charge. The
shared MTP `mtp_cat`, `hmtp`, logits, tokens and position buffers are allocated
by the unchanged engine initialization before bank admission. Their requested
payload is `4*T*2*N + 4*T*N + 4*V + 4*T + 8` bytes (T=chunk, N=embedding,
V=vocabulary); they are scratch, not an additional persistent seed per token.
The shared main attention FP16 staging alone is
`2 * n_head_kv * context * head_dim * 2` bytes (512 MiB for this geometry/context),
and the original score/probability buffers are 384 MiB. Other engine scratch,
weights, CUDA/library overhead and graphs must still fit; 7.40625 GiB below is
**not** a total process-RAM estimate.

## Exact physical bound

`kv_capacity_bound()` uses the **actual reserved layouts**, not a bytes/token
constant. Each `KvPrefix::chunks()` computes the union of granules intersecting
each head's prefix, including unaligned head starts and overlapping scale granules.
No tensor shape, head stride, INT8 kernel or numerical kernel changes.

For logical reservation `r`, the conservative retained extent is:

```text
w(0) = 0
w(r) = min(context, 256 * ceil(min(context, r + 8) / 256)), r > 0
```

Eight cells bound the existing maximum verification window; 256 is the existing
physical-growth coarsening. Main/MTP writes are still checked against the engine's
admission/context contract. The extra bound also covers speculative tails retained
after a shorter published prefix. Before `end()` reduces a logical reservation,
strict mode synchronizes and unmaps every granule outside `w(live_tokens)`.
This matters after cancellation or checkpoint rewind: a high-water mapping may
not remain arbitrarily large while only a tiny logical prefix is charged.

For each distinct rounded footprint, the policy records its **minimum** logical
reservation. Min-plus convolution over four slots finds the largest sum of page
counts whose combined minimum reservations fit 500,000. Zero and every write-
rounding breakpoint are enumerated; duplicate footprints keep their least cost.
This is an exact discrete maximum for the padded envelope, not a partition or a
continuous approximation. Small odd/unaligned layouts are exhaustively checked
against independent enumeration in the test. Fixed state/checkpoint pages for
**all** slots are added separately, even if a slot has no tokens.

### Worked calculation at 2 MiB granularity

The runtime queries CUDA granularity. These numbers apply **if it is 2,097,152
bytes**, and the loaded model has the documented 40 main layers (10 attention,
30 GDN), two KV heads, head dimension 256, embedding 2048, GDN value heads 32,
state dimension 128 and convolution channels 8192 / width 4. Other granularities
are recomputed, not forced to these numbers.

* Eleven attention caches including MTP. Each has K/V INT8 code planes and K/V
  FP16 scale planes (8 scale bytes/token/head). At context 262,144, both head
  strides are aligned: codes 67,108,864 bytes, scales 2,097,152 bytes.
* One nonempty slot/cache needs `4*ceil(w/8192) + 4` pages. Across four slots,
  the maximum sum of the code-step counts is **65**, not 64: e.g. reservations
  `(8193,253945,237853,9)` total 500,000, and padded extents have counts
  `(2,32,30,1)`. A 66th step needs more than 500,000 logical tokens. Thus:
  `11 * 4 * (65+4) = 3036` attention pages = **6,366,953,472 bytes** (6072 MiB).
* Every GDN layer has `4*8192*3 = 98,304` convolution-history bytes, rounded to
  one page, and `4*32*128*128 = 2,097,152` recurrence bytes, one page.
  `hlast` is `4*2048 = 8192` bytes, one page.
  State: `4 * (30*(1+1)+1) = 244` pages = **511,705,088 bytes** (488 MiB).
* A checkpoint payload is
  `4*(2048 + 30*(8192*3 + 32*128*128)) = 65,871,872` bytes,
  rounded to 32 pages / 64 MiB. The existing checkpoint retention policy allows
  `min(4, floor(1024 MiB / 4 / 64 MiB)) = 4` snapshots/slot.
  Checkpoints: `4*4*32 = 512` pages = **1,073,741,824 bytes**.
* Total bank: **3792 pages = 7,952,400,384 bytes = 7584 MiB = 7.40625 GiB**.

For comparison only, four full-context attention caches would consume 5808 pages;
this implementation does not allocate them. `500000*11616 = 5,808,000,000` raw
attention bytes is too small a physical backing bound.

## Ownership, failures and reset

`UnifiedKvBank` is owned by the engine and outlives its adapters. It owns every
physical handle, including unmapped free pages and pages on loan to slots.
Adapters retain stable VAs and borrow those handles; their `committed_bytes()`
is the mapped charge. `UnifiedKvPool` counts the bank once, not bank plus adapter
mappings. Logical reservations, mapped bytes and held physical bytes are separate.
A successful discarded adapter must STILL report zero mapped bytes. The original
lazy zero-commitment assertions are not weakened.

`VmmRange::map_borrowed()` is deliberately separate from `map_range()`:

* It never allocates when a bank is empty, and never releases a supplied handle.
* Map/access failure returns a loan only after successful unmapping. If unmapping
  fails, the range retains the handle and the bank retains its charged loan.
* Partial-growth rollback drains initialization writes, then unmaps only fresh
  pages. Failed rollback faults the owner. Partial reset failure faults the owner
  too; it cannot be re-admitted as an idle cache. Successfully returned pages and
  still-mapped pages remain accounted exactly once. Recovery requires a successful
  quiesce/discard/reset, not a false cache hit.
* All fresh/recycled mappings are zeroed, then synchronized, before publication.
  Free bank pages may contain old bytes, but are inaccessible at the old VA and
  cannot be published in a new slot without initialization. Invalidation clears
  histories, images, positions and checkpoint metadata before returning pages.
* Generations/epochs, exact-prefix selection, donor pins, sampled history/RNG/
  publication, main/MTP state commit and checkpoint copy ordering stay in force.
* Normal shutdown drains adapters, returns loans, then releases bank handles.
  A fatal synchronization/unmap failure deliberately quarantines/leaks referenced
  pages/VA until process teardown instead of recycling or double-releasing them.
  Such a device failure is not a successful reset or reclamation.

## Diagnostics and interpretation

`KV_PREALLOCATE` reports geometry-derived page counts and bytes before expert
placement. `KVSTATUS` additionally emits `KVBANK` (also retained by Python Monitor
under `kv_status.bank`): logical ceiling, context limit, held bank bytes, mapped
bytes, free/mapped pages, ancillary state/checkpoint bytes, physical allocation
attempts/successes, map/unmap calls, initialization pages and separate timings.
Process-wide `driver_allocations`, `driver_allocation_attempts` and `driver_releases`
are counted at the actual VMM driver wrappers too, not just at bank bookkeeping;
the gates require these counters to remain constant after initialization.
`mapped_pages` includes quarantined mapped loans on a driver failure, not a claim
that access was successfully enabled. `checkpoint_operations`, `checkpoint_copy_bytes`
and `checkpoint_copy_us` count completed captures/restores, including prefix KV
and inherited snapshot copies in cross-slot restoration.

`KVPOOL.committed_bytes` remains the held physical charge; in strict mode it
stays constant after readiness. `KVSESSION.bytes` is mapped capacity.
`KVEVICT.bytes` / reclaimed totals mean capacity returned for reuse, NOT memory
returned to CUDA. `KVMEM.admissible_bytes` in strict mode is reusable bank space,
which need not be smaller than fresh CUDA free memory. New mappings from that
bank do not need a second fresh-memory admission. No further KV physical handles
can be created after successful startup. The original lazy path still checks
fresh free memory on every physical growth and releases handles on discard.

Times are host wall times around driver/API work and initialization synchronization,
not isolated GPU kernel times. They distinguish physical allocation from mapping,
zero initialization and checkpoint copies. Failed-map rollback can contribute to
mapping time. A profiler is needed to separate lazy `KVCOMMIT` allocation versus
map/access versus memset/sync; do not label its combined time as just allocation.

## Local validation and pending checks

Run locally on x86_64 / GCC 12.2 / Python 3.11, with CMake 3.28.4 installed only
in session scratch. No nvcc, CUDA headers/runtime/device, Orin or model is available.

* Focused CPU CMake Release/CTest: **5/5 passed** — unified KV, prefix checkpoints,
  sampled MTP, sampled request controller and new production preallocation test.
* All five also compiled with `-Wall -Wextra -Werror` and non-PIE ASan/UBSan;
  leak detection enabled. Production pool/bank/adapter/VmmRange are tested, with
  only CUDA's driver/runtime boundary replaced by a test-only shim.
* Tests cover each slot reaching 262,144, aggregate 500,000, refusal of two active
  full contexts, padded/fragmented/skewed distributions, exhaustive rounding,
  full-bank use, exhaustion, active/pinned protection, zeroed reuse, warmup-like
  reset, rewind tail reclamation, stale checkpoint identity, startup byte/free-
  memory/allocation failure, partial map/access/initialization/unmap failure,
  failed-release accounting, recovery and zero live handles at shutdown.
* Five compile-and-run negative controls deliberately break exhaustion, zeroing,
  fresh-free admission, tail reclamation and borrowed ownership. All must fail.
* **33/33 Python tests passed**: telemetry, config pass-through, validation-driver,
  sampling-key, HTTP-benchmark mock and new benchmark/reset invariant tests.
  Bash/Python syntax checks passed.
* Compute kernels, prefill/platform sources, INT8 layout kernels, sampled verifier/
  request-controller headers and the layout calculator are source-byte unchanged
  from `f121cd08`. CUDA machine-code comparison is still pending.
* HIP's no-VMM stub compiled with strict warnings. This is NOT a HIP engine build.

**Pending:** CUDA 13.2 SM87 compile/link, native VMM/INT8/preallocation gates,
compute-sanitizer, every real-model gate and dump comparison, model cancellation/
image legacy regressions, CUDA kernel machine-code preservation, all performance
measurements, and full HIP/SYCL/other-backend builds touched by common headers.
No GPU numerical parity, physical residency or performance result is claimed.

Reproduce CPU checks (build outside the source):

```bash
cd /path/to/feature/q36-kv-preallocation
CHECK=$(mktemp -d)
cmake -S . -B "$CHECK/cmake" -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF \
  -DSTRATA_ENABLE_SYCL=OFF -DSTRATA_NATIVE_EXPERTS=OFF -DSTRATA_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$CHECK/cmake" --target unified_kv_test kv_preallocation_test \
  q36_prefix_checkpoint_test q36_sampled_mtp_test q36_sampled_request_test -j4
ctest --test-dir "$CHECK/cmake" \
  -R '^(unified_kv_test|kv_preallocation_test|q36_prefix_checkpoint_test|q36_sampled_mtp_test|q36_sampled_request_test)$' \
  --output-on-failure
for n in unified_kv q36_prefix_checkpoint q36_sampled_mtp q36_sampled_request; do
  g++ -std=c++20 -O1 -g -Wall -Wextra -Werror -pthread -Iinclude \
    -fsanitize=address,undefined -fno-omit-frame-pointer -no-pie \
    "tests/core/${n}_test.cpp" -o "$CHECK/$n"
  ASAN_OPTIONS=detect_leaks=1 "$CHECK/$n"
done
g++ -std=c++20 -O1 -g -Wall -Wextra -Werror -pthread -Iinclude -Itests/fake_cuda \
  -DSTRATA_KV_FAKE_DRIVER -fsanitize=address,undefined -fno-omit-frame-pointer -no-pie \
  tests/core/kv_preallocation_test.cpp src/core/vmm.cpp tests/fake_cuda/driver.cpp -o "$CHECK/preallocation"
ASAN_OPTIONS=detect_leaks=1 "$CHECK/preallocation"
python3 tools/check_kv_preallocation_negative.py
python3 -m unittest serve.test_unified_kv serve.test_q36_config serve.test_server.SamplingKeys \
  tools.test_unified_kv_validate tools.test_q36_sampled_mtp_http_bench tools.test_q36_kv_preallocation_bench
```

## Export without pushing

On the development machine, choose an export directory outside the worktree:

```bash
DEV=/home/pi/Documents/GitHub/Strata-Orin-kv-preallocation
EXPORT="$HOME/strata-kv-review-exports"
mkdir -p "$EXPORT"
git -C "$DEV" status --short
git -C "$DEV" log -1 --format=fuller
git -C "$DEV" format-patch --binary --stdout \
  f121cd08b17f43b219df27625fcbdf3bbf1c1204..feature/q36-kv-preallocation \
  > "$EXPORT/q36-kv-preallocation.patch"
git -C "$DEV" bundle create "$EXPORT/q36-kv-preallocation.bundle" feature/q36-kv-preallocation
git -C "$DEV" bundle verify "$EXPORT/q36-kv-preallocation.bundle"
sha256sum "$EXPORT"/q36-kv-preallocation.{patch,bundle} > "$EXPORT/SHA256SUMS"
```

Copy these files to the Orin using your normal reviewed transfer mechanism.
The bundle is self-contained. The patch alternative requires the exact base:
create another isolated worktree at `f121cd08`, a new local review branch, then
`git am /absolute/path/q36-kv-preallocation.patch`. **Use patch OR bundle, not both.**

## Isolated Orin build (no model or service operation)

Do not touch `worktrees/current`, `worktrees/rollback`, `/mnt/ssd/strata-service`,
OWUI or their configuration. Do not recreate retired top-level unified/checkpoint/
sampled-MTP directories. Import the bundle into a new local branch/worktree:

```bash
REPO=/mnt/ssd/Strata-Orin
BUNDLE=/absolute/path/to/copied/q36-kv-preallocation.bundle
export TEST_REPO="$REPO/worktrees/kv-preallocation-review"
test ! -e "$TEST_REPO"
git -C "$REPO" bundle verify "$BUNDLE"
git -C "$REPO" fetch "$BUNDLE" \
  refs/heads/feature/q36-kv-preallocation:refs/heads/review/q36-kv-preallocation
git -C "$REPO" worktree add "$TEST_REPO" review/q36-kv-preallocation
cd "$TEST_REPO"
git merge-base --is-ancestor f121cd08b17f43b219df27625fcbdf3bbf1c1204 HEAD
test -z "$(git status --porcelain)"
export LLAMA_SRC=/mnt/ssd/llama-strata-pin
export MODEL=/mnt/ssd/llamacpp_models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
export PYTHON=/mnt/ssd/strata-service/venv/bin/python
export BUILD="$TEST_REPO/build-kv-preallocation"
test "$(git -C "$LLAMA_SRC" rev-parse HEAD)" = 3cf03257f219afbe7334045ff7c6a06ac68c627d
/usr/local/cuda/bin/nvcc --version # record CUDA 13.2; do not install/replace JetPack
cmake -S . -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DCMAKE_CUDA_ARCHITECTURES=87 -DSTRATA_GGML_DIR="$LLAMA_SRC" \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_HIP=OFF -DSTRATA_ENABLE_SYCL=OFF \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_PREFILL_MMQ=OFF \
  -DSTRATA_BUILD_TESTS=ON
cmake --build "$BUILD" --target strata strata-q36 strata_kq unified_kv_test \
  kv_preallocation_test unified_kv_vmm_test q36_kv_int8_test kv_preallocation_cuda_test \
  q36_prefix_checkpoint_test q36_sampled_mtp_test q36_sampled_request_test -j4
ctest --test-dir "$BUILD" \
  -R '^(unified_kv_test|kv_preallocation_test|q36_prefix_checkpoint_test|q36_sampled_mtp_test|q36_sampled_request_test)$' \
  --output-on-failure
# STRATA_PREFILL_MMQ is HIP-only in this release. OFF does not disable CUDA MMQ.
test -f "$BUILD/libstrata_kq.so"
```

### Operator-approved service-stop window — all model commands are inside it

Pause clients first, without modifying OWUI. Identify the **actual** unit and
whether it is user/system managed. Do not guess a service name, kill broadly,
edit unit files or change enablement. If it is not systemd-managed, arrange an
equivalent stop/reap/restore procedure before running any gate.

In the **same Bash shell**, after building above, set the unit and correct arrays.
The test itself runs in a transient cgroup: cleanup must stop/reap that whole
cgroup before the original service is restarted. The trap refuses restoration
if cleanup cannot be proved. No two model instances may overlap.

```bash
set -euo pipefail
SERVICE='REPLACE-WITH-ACTUAL-DEPLOYED-UNIT.service'
SYSTEMCTL=(systemctl --user)      # system unit instead: (sudo systemctl)
SYSTEMD_RUN=(systemd-run --user)  # system unit instead: (sudo systemd-run --uid="$(id -u)" --gid="$(id -g)")
TEST_UNIT="strata-kv-preallocation-review-$$"
export RESULTS="$(mktemp -d /mnt/ssd/strata-kv-review-results.XXXXXX)"
export PAIRS=5
export PROFILE=0  # set to 1 for an additional, separately recorded Nsight run (requires nsys)
[[ "$("${SYSTEMCTL[@]}" is-active "$SERVICE")" == active ]]
"${SYSTEMCTL[@]}" show "$SERVICE" > "$RESULTS/original-unit.txt"
git -C /mnt/ssd/Strata-Orin/worktrees/current rev-parse HEAD > "$RESULTS/original-current.txt"
sha256sum "$MODEL" "$BUILD/strata-q36" "$BUILD/libstrata_kq.so" > "$RESULTS/hashes.txt"
uname -a > "$RESULTS/platform.txt"
/usr/local/cuda/bin/nvcc --version >> "$RESULTS/platform.txt"
# Also record nvpmodel, clocks, cooling, memory users, ggml pin and git revision.
restore() {
  local state
  state="$("${SYSTEMCTL[@]}" show "$TEST_UNIT.service" -p LoadState --value 2>/dev/null || true)"
  if [[ -z "$state" ]]; then
    echo 'Cannot query test unit: inspect/reap it before manually restoring production' >&2; return 1
  fi
  if [[ "$state" != not-found ]]; then
    "${SYSTEMCTL[@]}" stop "$TEST_UNIT.service" || {
      echo 'STOP FAILED: inspect/reap test cgroup before manually restoring production' >&2; return 1;
    }
  fi
  if pgrep -x strata >/dev/null || pgrep -x strata-q36 >/dev/null; then
    echo 'Engine process survives: inspect/reap it before restoring production' >&2; return 1
  fi
  "${SYSTEMCTL[@]}" start "$SERVICE"
  "${SYSTEMCTL[@]}" is-active "$SERVICE"
}
trap restore EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"${SYSTEMCTL[@]}" stop "$SERVICE"
if "${SYSTEMCTL[@]}" is-active --quiet "$SERVICE"; then exit 1; fi
if pgrep -x strata >/dev/null || pgrep -x strata-q36 >/dev/null; then
  echo 'Another engine remains; abort and inspect its ownership' >&2; exit 1
fi
# Check cgroup/process listings and tegrastats: the old model's RAM must be gone.
"${SYSTEMD_RUN[@]}" --unit="$TEST_UNIT" --wait --pipe --collect \
  --property=KillMode=control-group --working-directory="$TEST_REPO" \
  /usr/bin/env STRATA_KV_REVIEW_WINDOW=1 MODEL="$MODEL" BUILD="$BUILD" \
  RESULTS="$RESULTS" PYTHON="$PYTHON" PAIRS="$PAIRS" PROFILE="$PROFILE" \
  /bin/bash "$TEST_REPO/tools/q36_kv_preallocation_gate.sh"
# Leaving this shell stops/reaps the transient unit, then restores the original unit.
exit
```

If a test/cleanup fails, inspect its cgroup before restoring anything manually.
After the trap, verify the original service at its existing address, its unchanged
revision/config, and only then resume clients. This procedure changes no deployed
files, executable, OWUI settings or service configuration.

### Required pass markers / measurements

* `KV_PREALLOCATION_TEST PASS`: direct CUDA test, not skip. It maps real Q36
  planes at full virtual extents, redistributes four slots, tests padded physical
  capacity, exhaustion and reset/eviction recycling with constant allocation
  counts. It needs no model or 262K-token prefill.
* Existing unified VMM and INT8 native tests must exit 0, not skip (77).
* Both arms: `SAMPLED_MTP_TEST PASS`, positive offers **and accepts**, unchanged
  1e-4 full-logit tolerance, exact request/output/history/live/RNG/continuation
  checks; `PREFIX_CHECKPOINT_TEST PASS`, positive reused tokens, byte-identical
  same-slot/cross-slot logits and MTP probes, active/reset invalidation checks.
* Strict arm: `KV_PREALLOCATION_MODEL PASS`, zero mapped bytes after all resets,
  bank bytes retained and physical allocation count unchanged from initialization.
* `KV_PREALLOCATION_AB PASS`: `cmp` of complete gate dumps, not just a PASS-line
  comparison. It compares off/on outputs, histories, RNG and all dumped full logits
  byte-for-byte, including checkpoint and MTP probes. Existing tolerances are not
  relaxed. A differing byte fails and requires investigation.
* Startup byte-limit and unsupported-format negative gates must fail before READY.
* `KV_PREALLOCATION_BENCH PASS`: five interleaved off/on pairs with identical
  seeds and sampling settings, startup-to-READY separately from native prefill,
  decode, wall latency and combined KVCOMMIT; strict bank map/init/copy deltas are
  separate. Native prefill/decode phase times can already include allocator work:
  do not sum overlapping measurements. Both logs must retain full-GPU placement
  and packed copies. Retain JSON, stdout/stderr, thermals and memory telemetry.
* `compute-sanitizer.log`: zero errors, or explicitly PENDING if the tool is absent.
  Final marker: `KV_PREALLOCATION_ORIN PASS`; review the individual logs as well.

For lazy allocation-versus-map/access timing, set `PROFILE=1` in the window above.
This collects an additional **separate** Nsight Systems CUDA API trace and CSV
API summary inside the same stopped-service transient cgroup, after the unprofiled
benchmark. The Nsight run is recorded under `profiled/`, not mixed with `bench/`. Count
`cuMemCreate`, `cuMemMap`, `cuMemSetAccess`, memset/sync and checkpoint copies;
verify zero post-READY `cuMemCreate` in the strict arm. Do not compare profiled
throughput to unprofiled runs. Kernel machine-code comparison against a same-day
build of `f121cd08` and broader legacy model regressions remain additional gates,
not inferred from the allocator tests. Report startup, request allocator costs,
prefill and decode separately; claim improvements only after measurements.

**No push, merge, deployment or production configuration change until review.**
