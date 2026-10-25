# Unified KV — Jetson v0.1.45

**Opt-in CUDA implementation. Not yet qualified on Orin.** See the
[local build/test record](../ports/jetson-orin-v0.1.45/UNIFIED_KV_VALIDATION.md).
No Strata 0.2 code, disk KV, swapping service or conversation database is included.

## Source inventory and ownership

The pinned source at `ec8149d1` recognizes exactly two GGUF families
(`include/strata/artifact/model_family.hpp`):

| Family | Cache adapter |
|---|---|
| `qwen35moe`: Qwen3.6-35B-A3B and supported weight quantizations | FP32 K/V by default, or opt-in INT8 K/V with FP16 scales per 64 values; head-major layout; GDN convolution/recurrence, last hidden and MTP state |
| `qwen4exp`: Flash-Next, Coder, Swift and compatible fine-tunes/pruned/quantized variants | Paged QSA K/V, FP16, INT8 with FP16 scales, rotated Q4_0 or K8V4; indexer, GDN and PLE state; optional streamed host backing and MTP cache/ring |

Weight quantization does not choose a different KV layout. Unknown architectures
are still refused. Both families are integrated in the CUDA source; that is **not**
a claim that every model/format has passed inference validation.

`UnifiedKvPool` owns admission, byte/token limits, deterministic idle-only LRU,
residency, generations and eviction records. `KvAdapter` owns layout, physical
commitment, synchronization and invalidation. Adapters outlive their pool use and
are destroyed only after the engine stops issuing work. Callbacks are synchronous,
execute under the pool lock and must not re-enter it.

| State | Meaning / transition |
|---|---|
| `empty` | Initialized/reset, no logical cached history |
| `active` | Leased: includes prompt processing, prefill, decode, speculative verification/commit, batch rows and copies |
| `resident` | Idle valid cache; the backend has synchronized before releasing its lease |
| `evicted` | Whole idle cache invalidated; full-history replay required |
| `faulted` | Backend synchronization/reclamation failed; never an LRU victim |

`resident` is the protocol spelling of the C++ `idle` state. Printing DONE/BDONE
is not itself permission to reclaim memory: the lease is released only after the
backend drains its work. Controls run on the engine worker, not its reader thread.
An explicit reset also requires quiescence. Failed quiescence never frees memory.

LRU uses a monotonic activity counter, updated on processing, completed requests
and pinned source-cache use. It does not use allocation time. The requesting cache
and every active/faulted cache are excluded from victim selection. A victim's full
KV is released; its dependent prefix/image/position/checkpoint/draft metadata is
invalidated. A later request cannot mount a checkpoint of an evicted cache.

## Physical commitment

The following describes the default **lazy** pool. The separate opt-in
[Qwen3.6 physical preallocation path](Q36_KV_PREALLOCATION.md) holds an engine-owned
page bank at startup, retains it across reset/eviction, and reports held versus
mapped bytes separately. It preserves the same logical admission contract.

CUDA VMM reserves addresses once and maps physical allocation granules as needed.
Granularity comes from the device, not a hard-coded 2 MiB assumption. Qwen3.6 maps
a prefix of **each head**, preserving its original head stride. QSA retains each
format's separate code/scale arrays and page layout. No attention block tables,
CPU attention or new expert-offload path are introduced.

New mappings are zeroed and synchronized before publication. A failed growth
rolls back only new mappings, after draining their zeroing operations. Driver
release/unmap failures remain charged and fault the owner. Virtual addresses stay
reserved across eviction, so graph arguments and batch pointer tables do not move.
Graphs are never launched against an evicted cache without a new lease and replay.
Growing within an already committed granule updates a host watermark, without
remapping or adding a synchronization on every token.

On Linux, streamed QSA's authoritative host arrays use reserved anonymous address
ranges with incremental `cudaHostRegister`. The CPU VA is used only when CUDA
explicitly guarantees direct registered-host-pointer access or each registration
returns that exact address; otherwise growth fails explicitly. See the bounded
[host-mapping finding](KV_HOST_MAPPING.md). Unregister plus
`MADV_DONTNEED` reclaims its RAM. GPU working slots and the MTP ring commit on first
use; authoritative host history grows by prefix. The existing streaming kernels,
QSA residency map and ring restoration remain in use. No filesystem backs KV.

Qwen3.6's managed byte count includes its recurrent/last-hidden buffers and MTP
K/V. Flash-Next retains the legacy fixed session arenas for GDN, PLE, indexer,
RoPE, page maps and scratch; `fixed_session_arena_bytes` reports their requested
size separately. Eviction invalidates those states but retains their reusable
arenas. It releases all managed K/V, including streamed host and draft backing.

Actual managed commitment is the sum of owned physical granules, including
alignment and speculative padding, not token count times an assumed constant.
For example, the Qwen3.6 geometry with 11 attention caches (including MTP), two KV
heads and 256 FP32 channels uses **45,056 raw KV bytes/token**. 500,000 such tokens
are 22,528,000,000 bytes, about 20.98 GiB, **before** recurrence/alignment/scratch.
This is arithmetic, not an Orin measurement. QSA's per-layer raw bytes/token at
its compiled geometry are FP16 2048, INT8 1056, Q4_0 576 and K8V4 816; its MTP
format/working set is priced separately by the adapter.

### Limits versus available memory

* `--max-context N`: logical per-session context, including the output allowance.
* `--batch 4`: four inference slots. Flash-Next also retains its existing main
  prompt workspace; it does **not** get a second token/byte budget.
* `--kv-pool-tokens N`: opt in; maximum aggregate logical capacity. Active requests
  reserve prompt plus requested output. Physical memory grows with writes, not
  with the output allowance. Idle sessions give unused reservations back.
* `--kv-pool-mib N`: optional ceiling on **managed physical KV** in MiB, including
  rounded/padded allocations. Fixed arenas, weights and other engine buffers are
  outside this ceiling, but not outside the physical memory admission guard.

`resident_tokens` counts consumed valid tokens; the last emitted, not-yet-fed
output token is not resident KV. `reserved_tokens` includes active output space.
These numbers and physically committed capacity need not be equal.

Every physical growth checks a fresh device/host budget after headroom. That
reading already charges weights, routed experts, fixed state, scratch, graph
allocations and other buffers; there is no fabricated per-category graph size.
Driver bookkeeping and non-KV allocation overhead are bounded by this guard, not
claimed as KV payload bytes. Unknown host telemetry refuses growth. On Tegra,
`device_free_bytes()` preserves the port's cgroup-aware shared-RAM accounting and
`STRATA_UMA_HEADROOM_GIB` (default 6 GiB); the pool also keeps 768 MiB free. CUDA RAM
is never added to system RAM. No swap allowance is used.

A 500K token ceiling does not promise 500K physical residency. On pressure the
manager releases eligible idle caches in LRU order and retries. If none can make
room, it returns `kv_pool_full` or `kv_allocation_failed`, not successful truncated
output. It does not move experts to CPU, disable MMQ/graphs or shrink context.

Flash-Next's automatic expert cache can consume the freed startup memory. Use
its existing `--vram-reserve-mib` / explicit expert-cache sizing to leave measured
room for KV growth. The pool does not take memory back from experts. Startup chunk
and expert planners may observe more free memory with the pool: pin their settings
and verify placement logs for controlled performance comparisons.

## Configuration

Qwen3.6 (its MTP argument is a number of drafts). The default remains FP32;
`--kv int8` selects the [INT8 cache path and native smoke gates](Q36_INT8_KV.md).
For changed text inside an earlier system-message prefix, the separate opt-in
[Qwen3.6 prefix checkpoint path](Q36_PREFIX_CHECKPOINTS.md) restores bounded recurrent snapshots.
The experimental [Qwen3.6 sampled-MTP path](Q36_SAMPLED_MTP.md) is an isolated A/B feature
(`--mtp-sampled`, `sampled_mtp=0|1`); it is not part of the baseline below.
Baseline configuration (checkpoints and sampled MTP off):

```bash
build-orin/strata-q36 --model "$MODEL" --serve \
  --max-context 250000 --batch 4 --kv-pool-tokens 500000 \
  --mtp 2 --batch-mtp
```

`strata --native "$MODEL"` routes to the same Qwen3.6 implementation. A server
configuration can keep `"parallel": 4` and put the pool flags in `"args"`;
`serve/server.py` already converts parallel to `--batch`.

Flash-Next keeps v0.1.45's different MTP semantics: **`--mtp DIR --spec 3`**, not
`--mtp 2`. Keep the model/pack/native-expert/prefill arguments of a working v0.1.45
configuration and add `--kv-pool-tokens 500000 --batch 4 --batch-mtp`. Existing
`--kv fp16|int8|q4_0|k8v4` and `--kv-resident` select its adapter layout.

Omit both pool flags for the original allocation path. Qwen3.6 supports pooled
serve and standalone inference/bench modes. Flash-Next's initial integration
requires `--serve`, one CUDA VMM device, and batch 0 or 2..8. It refuses combinations
with a layer split, `--kv-grow`, `--vram-elastic` or `--peer-device`. CPU-only, HIP,
SYCL and Metal do not implement the new physical allocator. Their original paths
remain available. Incremental streamed host backing currently requires Linux and
stable UVA aliases; other placements fail rather than silently changing KV format.

## Native protocol v1 and session-proxy replay contract

Require `INFO unified_kv=1 kv_protocol=1` before relying on this interface. Session
IDs are stable **within one native process**. A new process/READY invalidates all
old identities; the proxy must track its connection/process incarnation too.

`batch_session_offset=0` for Qwen3.6: BGEN slot b is session b.
`batch_session_offset=1` for Flash-Next: session 0 is its workspace, BGEN slot b is
session b+1. Its workspace owns the prompt lease until the slot handoff. An old
idle destination cache can still be evicted before it is selected for restoration;
a selected copy source is pinned until its copy completes.

| Line / command | Contract |
|---|---|
| `KVSTATUS` | Returns KVPOOL, KVMEM, one KVSESSION per owner, then KVEND |
| `KVPOOL` | Token/byte ceilings, actual committed bytes, consumed/reserved tokens, evictions, reclaimed bytes and eviction time |
| `KVMEM` | Fresh physical availability after headroom; admissible bytes also bounded by the pool byte ceiling |
| `KVSESSION` | `session`, `generation`, `epoch`, `state`, `tokens`, `reserved_tokens`, `bytes` |
| `KVEVICT` | Invalidated session/generation, **new** epoch, lost token count, released bytes, sequence and latency; reason=pressure |
| `KVCOMMIT` | Physical growth's session/generation/epoch, write capacity, bytes, pool bytes and latency |
| `KVCACHE` | Admission identity/residency. Qwen3.6 says hit/miss/new; Flash-Next says miss or resident_or_new. It is not Flash-Next's final prefix-hit count; RESUME/DONE report that |
| `KVREPLAY` | Fresh in-place full-history replay, with a new validity epoch |
| `KVMOVE` | Flash-Next workspace-to-slot handoff; source no longer owns a resident conversation |
| `KVRESET <session> <generation>` | Idle-only safe reset. Response carries the **new** generation and epoch; stale generation or active lease is rejected |
| `KVINVALIDATE` | A failed request discarded its partial state; new generation/epoch and reason=request_error |
| `ERR <code> ...` / `BERR <slot> <code> ...` | Explicit request/slot failure, never a successful length completion |

Generations change on explicit reset/error reset. Epochs change on cache
invalidation/replay/handoff. Pressure eviction retains the session generation.
The bounded eviction queue holds 256 events, with monotonic sequence numbers;
a sequence gap requires KVSTATUS/resynchronization, not an assumption of a hit.
The server records bounded telemetry and clears only obsolete scheduling hints.
Its Monitor `kv_status` is the last reported snapshot, not a conversation store.

**Replay:** GEN/GENI/BGEN/BGENI always carry the **full rendered prompt history**.
After a miss, send all intended tokens from token zero, with the original image
embeddings/position metadata for image requests. Do not send only an appended
suffix. KV is not enough to preserve a conversation; this engine does not retain
or reconstruct an evicted history on its own.

For a caller that wants refusal rather than automatic full-prompt replay, include
`kv_resume=1 kv_generation=N kv_epoch=M` among the request keys. Both identities
are required. Evicted/stale/non-matching state produces `kv_cache_miss` or
`kv_stale_generation` before it can be mistaken for an empty valid cache. Reissue
with complete history and without `kv_resume=1` to permit replay; consume the new
identity/status. The existing Python server already sends full prompts and exposes
native errors; a future session proxy can map capacity errors to its retry policy.
Ordinary completion/cancellation is not an eviction notification.

## Known costs and scope limits

* Flash-Next still copies between its main workspace and batch slot. Both physical
  copies count during migration; logical output reservations transfer on handoff.
  An idle source and a new workspace can also temporarily occupy separate token
  reservations. This can require extra headroom or evict more idle caches than
  steady-state occupancy alone suggests. There is no zero-copy VMM remapping.
* Unified Flash-Next disables whole-KV RAM parking (otherwise snapshots would
  escape pool accounting) and cooperative BYIELD parking. Existing recurrent
  checkpoints, prompt chunking, batch windows, MTP and compute kernels remain.
  SAVE/RESTORE are explicitly refused with `SERR invalid 0`; disk KV is out of scope.
* The experimental Flash-Next `--native-flash-attn-short` diagnostic is refused
  with the pool: its 256-cell contract does not include QSA's speculative padding.
  The diagnostic remains available with the original allocator; normal optimized
  attention kernels are not disabled.
* QSA fixed auxiliary arenas are retained, not physically decommitted on eviction.
  Qwen3.6's corresponding managed recurrent buffers are physically released.
* Growth/eviction is synchronous at safe engine boundaries. This trades allocator
  latency for straightforward ownership and graph-pointer safety. No measured
  short-context or Orin performance claim is made yet.

## Native build and validation

Start from [BUILD.md](../ports/jetson-orin-v0.1.45/BUILD.md), but stay on
`feature/unified-kv`. Keep the pinned ggml revision and SM87/native/K-quant options:

```bash
test "$(git branch --show-current)" = feature/unified-kv
git merge-base --is-ancestor ec8149d1 HEAD
cmake -S . -B build-orin -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON \
  -DSTRATA_PREFILL_MMQ=OFF -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_BUILD_CONVERSATION_TESTS=ON
cmake --build build-orin -j4 --target strata strata-q36 strata_kq \
  unified_kv_test unified_kv_vmm_test
ctest --test-dir build-orin -R '^unified_kv' --output-on-failure
```

The CUDA MMQ cache option naming is explained in BUILD.md: `STRATA_PREFILL_MMQ=OFF`
is **not** a CUDA MMQ disable. Keep `libstrata_kq.so` beside the engine. The native
VMM test must **exit 0, not skip** on Orin. Device VMM, head boundaries, rollback,
four owners, LRU, graph replay, resident QSA formats and device ring tables remain
mandatory. The [capability-aware review harness](Q36_KV_PREALLOCATION.md#capability-aware-native-host-coverage-correction-after-7cbe4df6)
reports host-backed positive coverage separately: each registration must satisfy
the flat-address contract, or explicitly reject it with verified rollback/accounting.
Unsupported host coherence/stream/ring-restoration positives are not reported as
passed. They do not skip device-only cases or excuse any allocator/API error.

Record model/binary hashes, git revision, CUDA/GCC/JetPack versions, nvpmodel,
clocks, cooling and competing memory users. Do not change these between arms.
Use the same day's non-pool binary/settings. Keep graphs, native experts and MMQ
on; confirm full-GPU placement in **both** logs. Use the same explicit chunk for
controlled runs, then measure automatic chunk selection separately.

```bash
# Short-context A/B, actual generation (not just a repeated prompt-cache hit).
python3 tools/unified_kv_validate.py --exe build-orin/strata-q36 --compare \
  --lengths 4096 --gen 1024 --out results/kv-short-mtp2 -- \
  --model "$MODEL" --max-context 8192 --batch 4 --chunk 1024 --mtp 2 --batch-mtp

# Repeat with --mtp 0 and without --batch-mtp. Optionally use
# --reference-exe /path/to/same-day-jetson-baseline/strata-q36.

# Long contexts and all four requested distributions; physical feasibility is a gate.
python3 tools/unified_kv_validate.py --exe build-orin/strata-q36 --mixed \
  --lengths 4096,32768,131072,249968 --gen 32 --out results/kv-long -- \
  --model "$MODEL" --max-context 250000 --batch 4 --mtp 2 --batch-mtp

# Smaller LRU gate first (pool_tokens must equal 2 * context for this workload).
python3 tools/unified_kv_validate.py --exe build-orin/strata-q36 --eviction \
  --pool-tokens 16384 --lengths 4096 --gen 32 --out results/kv-lru-small -- \
  --model "$MODEL" --max-context 8192 --batch 4 --mtp 2 --batch-mtp

# Then --pool-tokens 500000 --max-context 250000 with --eviction, if RAM permits.
python3 tools/check_unified_kv_preservation.py
```

Repeat with Flash-Next's working base model/pack/format/streaming arguments and
`--mtp DIR --spec 3 --batch-mtp`. Test each KV format separately. Its expert reserve
must leave room for the managed commitment and temporary copies; do not reduce
headroom or enable swap to make a run pass.

The probe records raw stdout/stderr, exact generated tokens, prefill timings,
end-to-end aggregate throughput, per-request first/last-token latency, growth and
eviction latency/bytes, cache hits/misses/replay and process RSS/swap/MemAvailable.
It queues four admissions, checks the first two LRU victims in activity order,
refuses a strict resume after eviction, then replays full history. Additional
copy-induced evictions are retained in the report, not hidden. A mixed workload
that completes without proving the requested residency is **not** called a pass.
Capacity errors are reported, never converted to smaller workloads. Native DONE
prefill timing can exclude Flash-Next's initial commitment: include KVCOMMIT latency
and wall latency when comparing it. MTP acceptance remains in the raw native logs.
For graph capture/reuse counts, collect a separate Nsight Systems CUDA API trace
and count `cudaGraphInstantiate*` / `cudaGraphLaunch`; an INFO graphs-enabled flag
is not proof of reuse. Do not use a profiled run as the throughput comparison.

Also run the port's [VERIFY.md](../ports/jetson-orin-v0.1.45/VERIFY.md) numerical and
kernel gates, including teacher-forced full-logit comparison. Do not loosen the
established numerical tolerances. Add image/M-RoPE, cancellation mid-prefill,
BSTOP/reset/reuse and repeated growth/eviction stress with the real model fixtures.
The probe is a text gate, not proof of those image or numerical cases. Compare
4K/32K/128K/250K separately and interleave at least five same-setting A/B pairs.
The operator's approximately 1,132 tok/s prefill and 40.2 tok/s MTP2 generation
figures are reference gates, **not measurements made by this implementation**.
