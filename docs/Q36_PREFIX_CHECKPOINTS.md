# Opt-in Qwen3.6 prompt-prefix checkpoints

For the vision-base integration, use the [independent schedule-matched oracle](Q36_CHECKPOINT_SCHEDULE.md)
and [staged adaptive handoff](Q36_ADAPTIVE_INTEGRATION.md). Fixed remains default;
`--kv-checkpoint-policy adaptive` is opt-in with the same four-payload ceiling.
The historical gate below compares against a one-call fresh prefill, which is not
necessarily the producer/suffix schedule. Do not use a different-schedule failure
to diagnose restoration, relax its comparison, or repeat the closed projection
study. Historical local compile/results below are not this integration's results.

This addresses a changed retrieved-memory suffix **inside the initial system message**, not a missing conversation ID. The existing full-live-token exact-prefix path stays first. On a miss, the engine compares actual token IDs, selects the longest available retained checkpoint at or before the first differing token, restores it and reads the remaining suffix. Checkpoints do not depend on chat-role boundaries. **13,917 characters is not a token offset.**

Enable only in a separate test process initially:

```text
--kv-checkpoint-mib 1024 --kv-checkpoint-interval 1024
```

Requires `--kv-pool-tokens` and `--serve` (or the developer regression below). Defaults allocate no checkpoint banks and create/restore no checkpoints. No OWUI, proxy, deployed configuration, expert kernels or inference kernels change.

## State and ownership

| State | Treatment |
|---|---|
| Every GDN layer's FP32 recurrence and convolution history | Copied together into a packed device snapshot at a completed prompt chunk |
| Last main hidden vector | Snapshotted; it seeds the next MTP call |
| Main and loaded MTP attention KV | Existing per-slot prefix remains in place, including INT8 codes **and FP16 scales**; no full KV duplicate per checkpoint |
| Position, token-prefix lineage, session generation and validity epoch | Validated before selection; position restored; later snapshots invalidated and retained earlier ones rebound to the new epoch |
| Image/M-RoPE metadata | Initial implementation is **text-only**. Image requests never select/create these checkpoints; existing exact image matching or fresh prefill remains |
| MTP draft/verify scratch, activations and logits | Not persistent checkpoint state: a nonempty suffix regenerates logits/hidden rows; each request starts with no pending draft/fix list. All MTP prefix KV and the hidden seed are retained. Even a one-token restored suffix uses prompt prefill so its MTP KV is seeded |

Creation is limited to fresh/restored multi-token prompt prefill, not decode, speculative windows or the existing full-state exact-prefix fast path. Checkpoints are at natural prompt chunk ends, with at least the requested token interval between rolling captures. With `--chunk 512 --kv-checkpoint-interval 1024`, the first anchor is at 1,024 tokens, even inside a long system message. A mismatch before the first anchor safely prefills from zero.

The metadata must match the slot's current generation/epoch, live token prefix, position and physical VMM coverage. Actively decoding, faulted, evicted and unrelated active owners are not donors. A restore may read the active lease held by its own exact donor pin (same pool, slot and generation), or its own admitted destination; discovery remains idle-only. Pool discard/reset/replay and ordinary slot replacement invalidate snapshots. Restoring an earlier prefix advances the validity epoch and drops later points. An explicit batch-slot admission may copy a matching idle donor's prefix and snapshots: the donor is pinned **before admission**, both physical copies are charged, and copies complete before publication. Exact whole-state copies invalidate the destination's previous checkpoint metadata.

## Explicit memory bound and retention

For model geometry `N=embedding width`, `C=GDN convolution channels`, `K=convolution width`, `H=value heads`, `D=state dimension`:

```text
payload = 4 * (N + sum_over_GDN_layers(C*(K-1) + H*D*D)) bytes
rounded = ceil(payload / VMM_granularity) * VMM_granularity
per_slot = min(4, floor(global_checkpoint_budget / slot_count / rounded))
physical_snapshot_ceiling = slot_count * per_slot * rounded <= global_budget
```

The startup log prints the actual geometry-derived payload, rounding, per-slot count and ceiling. These are calculated sizes, not hardware residency measurements. If the fair per-slot budget cannot hold one snapshot, checkpoint reuse is disabled with a warning and fresh-prefill fallback.

One record pins the earliest usable checkpoint. Two records pin that first point and roll the last. Three/four records also pin anchors at eligible chunks reaching 2x/4x the interval; only the last record rolls. Intervals do not split existing chunks. Long later prompt segments cannot evict every early anchor. The longest **retained**, validated point is used, not an unrecorded arbitrary token prefix.

The separate opt-in [physical page bank](Q36_KV_PREALLOCATION.md) preallocates this
same configured checkpoint ceiling before readiness. It does not change retention,
copying, donor pins or identity validation. With that flag, reset must leave zero
slot mappings while retaining bank-owned physical memory; without it, the original
zero-physical-commitment reset assertion still applies.

Snapshot buffers are fixed-state VMM ranges within their session's existing adapter. Virtual reservations occur at initialization; the bank's bounded physical storage is committed with that slot's first KV write, including for a short request that never reaches a checkpoint. All banks count in `KVPOOL`/`KVSESSION`, physical-byte limits and fresh UMA headroom. No maximum-context KV mirror, pageable host copy, persistence or swap backend is added. Captures/restores cost device copies and synchronization; opt-in performance has not been measured here.

## Focused validation

Actually run locally (x86_64, CUDA 12.6.85, SM87 target; **not Orin**):

* New CPU CTest: token-prefix/suffix equivalence using a causal **toy** recurrence/conv/attention/MTP model; pinned-anchor retention, longest-point selection, rounded memory budgets, generation/epoch mismatch, real unified-pool reset/replay/eviction callbacks. Passed, including ASan/UBSan with a non-PIE executable.
* CPU donor-lease regression uses the same eligibility function as the engine and real `UnifiedKvPool`/`KvIdlePin` with mock backing: idle → pin → destination admission → eligible restoration → release. Unrelated active owners, decoding, wrong pool/generation, stale epochs and missing backing remain excluded.
* Worktree Qwen3.6 CUDA source compiled and linked against the unchanged previously built native/MMQ/core libraries. This is targeted compile/link validation, not a clean full build or GPU execution.
* All 66 CUDA kernel text sections match the `feature/unified-kv` INT8 baseline, normalizing only anonymous-namespace source-path hashes. No compute kernels were added or changed.
* Fixture generation dry-run with the byte tokenizer only. The real GGUF tokenizer/model are unavailable here.

The native model gate below must still run on Orin. It compares **every vocabulary logit byte-for-byte**, then compares MTP2 drafts, verified picks and acceptance against fresh prefill of the same revised prompt. It also tests checkpoint cloning into another batch slot, active-owner exclusion and reset/generation invalidation, and requires managed physical bytes to return to zero after resets. No numerical tolerance is relaxed. No model/logit equivalence or performance result is claimed until this gate passes on hardware. HIP/SYCL/Windows and image checkpoint restoration are not qualified.

## Exact Orin build and verification (no service changes)

Use a **separate worktree/build directory**, not the deployed executable. Preserve the working CUDA 13.2 compiler and the existing pinned ggml sources. Example variables: `CUDA_HOME=/usr/local/cuda`, `LLAMA_SRC=/path/to/the/existing/pinned/llama.cpp`.

```bash
cmake -S . -B build-prefix-checkpoint -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" -DCUDAToolkit_ROOT="$CUDA_HOME" \
  -DCMAKE_CUDA_ARCHITECTURES=87 -DSTRATA_GGML_DIR="$LLAMA_SRC" \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_HIP=OFF -DSTRATA_ENABLE_SYCL=OFF \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PREFILL_MMQ=OFF \
  -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_BUILD_TESTS=ON
cmake --build build-prefix-checkpoint --target \
  strata strata-q36 strata_kq q36_prefix_checkpoint_test -j4
ctest --test-dir build-prefix-checkpoint -R '^q36_prefix_checkpoint_test$' --output-on-failure
```

`STRATA_PREFILL_MMQ=OFF` is the existing v0.1.45 **HIP-only CMake option**; CUDA native MMQ remains enabled. Keep `libstrata_kq.so` beside the binaries. Use available GPU/RAM headroom for this independent model process; nothing below connects to or restarts the deployed HTTP service.

```bash
MODEL=/path/to/the/actual-model-first-shard.gguf
python3 tools/q36_prefix_checkpoint_fixture.py --model "$MODEL" \
  --out /tmp/q36-prefix-checkpoint.csv
# If HTTP uses a different exported template, also pass --template /path/to/chat_template.jinja.

build-prefix-checkpoint/strata-q36 --model "$MODEL" \
  --max-context 250000 --batch 4 --kv-pool-tokens 500000 \
  --kv int8 --mtp 2 --batch-mtp --chunk 512 \
  --kv-checkpoint-mib 1024 --kv-checkpoint-interval 1024 \
  --kv-checkpoint-test /tmp/q36-prefix-checkpoint.csv
```

The fixture prints the actual encoded `common_tokens` separately from the 13,917-character system offset. The gate fails rather than skipping for insufficient checkpoint budget, no suitable point, non-full-GPU expert placement, missing MTP2, any full-logit difference or invalidation failure. Success ends with `PREFIX_CHECKPOINT_TEST PASS`, a positive `reused_tokens`, and `max_logit_abs=0`. It is one changed-system-suffix correctness scenario, not a throughput benchmark or a claim of 250K physical residency. Do not merge/deploy or change OWUI on the strength of CPU/mock tests alone.
