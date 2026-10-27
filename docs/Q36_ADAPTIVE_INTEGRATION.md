# Adaptive checkpoints on the vision-qualified Orin base — review only

## Lineage and invariants

Base: **3f3bc4e85e1d44b3501f0dcb8cd73471632753cf** (operator reports deployed,
vision qualified). First commit: **30340e2273ad9a0b23cf60446d4446a7d2d2129b**,
the [independent native schedule oracle](Q36_CHECKPOINT_SCHEDULE.md). The second
commit ports policy from `4fedc470b433712e3038bf7dfdec9c22a1a660d4` with the
`e486ac9bed04bb0ed256eb18e8b67fa124c0f5f1` Serve scope correction. Diagnostic tip
`4815a0368bc363cb86cd9d11964818f393d8aab8` descends from those adaptive commits,
NOT from the vision base. Their common ancestor is `a9837161`. All objects were
resolved locally; **no reset/merge to the older adaptive tree**.

Fixed stays default; adaptive requires explicit `--kv-checkpoint-policy adaptive`
and an existing checkpoint budget. It changes retention metadata, not process
calls, chunking, kernels or samplers. No production config, server, encoder or
OWUI change. No push/deployment or model/service operation occurred here.

The original policy is preserved: eight bounded LCP hints, saturating hit counts
and expiry after eight other rewrites; append-only requests do not train/age.
Hints come from idle, backed, text-only rendered-token lineages and grant no
permission to restore. Per-request plans select attainable existing chunk ends
at/before learned divergence positions, with nonempty suffix required. Record 0
is the early safety anchor; at most two middle records learn useful boundaries
(or retain fixed fallback anchors); the last record rolls. Learning cannot create
a missed past state. Discovery, operation-owned donor pinning, generation/epoch
checks, text-only eligibility and payload publication rules stay explicit.
Retire-before-write/publish-after-sync prevents partial payload publication.

**Four payloads per slot maximum**, unchanged budget/capacity calculation and
physical bank proof. No fifth bank or new GPU scratch. For context262144, four
slots, shared524288, Qwen geometry and *if* CUDA granularity is2MiB, existing
calculation remains3924 pages /8,229,224,448 bytes /7848MiB. This is a calculated
KV-bank bound, not measured total residency. Vision ancillary staging and encoder
allocations remain separate charges, unchanged from the qualified vision base.

Source checks retain the vision lease/staging implementation, image regression,
bounded row checks, strict preallocation, INT8 kernels, process/forward/row/MTP
numerics, sampler guard, bank layout/capacity and server configuration. GENI
never trains/plans/captures text checkpoints. The unchanged sampled-text-MTP guard
still excludes image, batch and concurrent requests. The old source-invariance
test now checks checkpoint *transfer* rather than capture's intentionally changed
metadata publication; it additionally checks retire/write/sync/publish ordering.
Source identity is not a CUDA compile, SASS comparison or new vision qualification.

## Evidence boundary

The operator established identical baseline/adaptive diagnostic full/split results,
exact same-schedule repeats, exact fresh-workspace and cross-build controls, and
shape-dependent accumulation in the isolated FP16 L0 ssm_out projection. Full vs
29+22 maximum difference6.973743438720703e-06; CPU FP64 errors5.291656918871013e-06
(full) and1.1679582030410529e-05(split), consistent with the conservative FP32
bound. Exact integer checking of461 outputs found zero FP64 error there.
**Do not repeat/expand that investigation without contradictory evidence.** It
does not prove end-to-end harmlessness or qualify adaptive caching. The new gate
therefore requires exact state/logits under *matching* schedules and separately
shows different-schedule differences, with no relaxed tolerance.

Temporary OWUI fetch_url insertion/removal and changing system memory are prompt
rewrites, not frontend bugs to work around. Use actual rendered token LCPs.
The old fixture's turn2 was a rewrite; only turn3 was append-only. New short public
fixtures use the actual GGUF/template and audit every row per slot. A name such as
`ordinary` never overrides the resulting classification. Missing genuine append
coverage or eligible user/system rewrites fails fixture generation, not skips.
Byte/mock fixtures are CPU plumbing evidence ONLY.

## Exact staging and clean switch — future Orin commands, NOT executed locally

Use ONLY the existing nested checkpoint-policy test worktree. Preserve its old
branch, commit, logs, fixtures, dumps, overlays and build directories. Do not
clean, reset, force-switch, overwrite builds, or touch production/current/rollback
worktrees. Stage exports/results under **/mnt/ssd/tmp/strata-checkpoints**.
These commands require no service operation and must be reviewed before execution.

```bash
set -euo pipefail
STAGE=/mnt/ssd/tmp/strata-checkpoints
REPO=/mnt/ssd/Strata-Orin
TEST="$REPO/worktrees/checkpoint-policy"
BASE=3f3bc4e85e1d44b3501f0dcb8cd73471632753cf
REFERENCE=30340e2273ad9a0b23cf60446d4446a7d2d2129b
DIAGNOSTIC=4815a0368bc363cb86cd9d11964818f393d8aab8
mkdir -p "$STAGE"
# Transfer q36-adaptive-integration-review.tar.gz AND .sha256 into STAGE.
cd "$STAGE"
sha256sum -c q36-adaptive-integration-review.tar.gz.sha256
tar -xzf q36-adaptive-integration-review.tar.gz
ART="$STAGE/q36-adaptive-integration-review"
(cd "$ART" && sha256sum -c SHA256SUMS)
BUNDLE="$ART/q36-adaptive-integration-incremental.bundle"
git -C "$REPO" bundle verify "$BUNDLE"
git -C "$REPO" fetch "$BUNDLE" refs/heads/review/q36-adaptive-integration
TIP=$(git -C "$REPO" rev-parse FETCH_HEAD)
test "$(git -C "$TEST" rev-parse --show-toplevel)" = "$TEST"
test "$(git -C "$TEST" rev-parse --path-format=absolute --git-common-dir)" = \
     "$(git -C "$REPO" rev-parse --path-format=absolute --git-common-dir)"
test -z "$(git -C "$TEST" status --porcelain=v1 --untracked-files=all)"
git -C "$REPO" merge-base --is-ancestor "$BASE" "$TIP"
test "$(git -C "$REPO" rev-list --count "$BASE..$TIP")" = 2
test "$(git -C "$REPO" rev-parse "$TIP^")" = "$REFERENCE"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)-$$
OLD=$(git -C "$TEST" rev-parse HEAD)
# Confirm the actual old branch is the expected diagnostic lineage, never guess.
git -C "$REPO" merge-base --is-ancestor "$OLD" "$DIAGNOSTIC"
printf 'old_branch=%s\nold_commit=%s\nnew_commit=%s\n' \
 "$(git -C "$TEST" symbolic-ref --short HEAD)" "$OLD" "$TIP" > "$STAGE/switch-$STAMP.txt"
KEEP="preserve/q36-checkpoint-policy-$STAMP"
git -C "$REPO" branch "$KEEP" "$OLD"
test "$(git -C "$REPO" rev-parse "$KEEP")" = "$OLD"
git -C "$REPO" bundle create "$STAGE/old-checkpoint-policy-$STAMP.bundle" "$KEEP" "^$BASE"
# Record old ignored artifacts without altering or copying multi-GB raw dumps.
(cd "$TEST" && find logs build-* -type f -printf '%p\t%s\t%T@\n' | LC_ALL=C sort) \
 > "$STAGE/artifacts-before-$STAMP.tsv"
git -C "$TEST" switch -c "review/q36-adaptive-integration-$STAMP" "$TIP"
test "$(git -C "$TEST" rev-parse HEAD)" = "$TIP"
test -z "$(git -C "$TEST" status --porcelain=v1 --untracked-files=all)"
test "$(git -C "$REPO" rev-parse "$KEEP")" = "$OLD"
(cd "$TEST" && find logs build-* -type f -printf '%p\t%s\t%T@\n' | LC_ALL=C sort) \
 > "$STAGE/artifacts-after-$STAMP.tsv"
cmp "$STAGE/artifacts-before-$STAMP.tsv" "$STAGE/artifacts-after-$STAMP.tsv"
# Any dirty tree, unknown lineage, existing new branch or inventory difference:
# STOP and inspect. No stash/reset/clean/forced checkout to make this pass.
cd "$TEST"
```

The bundle is incremental against the exact vision base. The complete mail patch
is an **alternative** applied to that exact base, never an extra patch on the
bundle tip. An unrelated older test HEAD must be investigated, not silently
accepted. Existing ignored artifacts remain at their old paths; Git preservation
ref/bundle protects their source commit. The inventory checks names/size/mtime,
not tensor hashes; retain existing artifact hashes/manifests too.

### New build subdirectories only

Use the already validated toolkit and ggml pin, no package/toolkit replacement.
No executable is installed/copied into a production directory. The first commit
is a non-adaptive vision-base/reference build, NOT the older a983 adaptive fork.

```bash
: "${CUDA_ROOT:?existing CUDA13.2 root}"
: "${LLAMA_SRC:?existing pinned llama.cpp checkout}"
PYTHON=/mnt/ssd/strata-service/venv/bin/python
MODEL=/mnt/ssd/llamacpp_models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
test "$(git -C "$LLAMA_SRC" rev-parse HEAD)" = 3cf03257f219afbe7334045ff7c6a06ac68c627d
BUILD_TAG=build-adaptive-integration-v1
for path in "$TEST/$BUILD_TAG-reference-source" "$TEST/$BUILD_TAG-reference" "$TEST/$BUILD_TAG-feature"; do
 test ! -e "$path" # choose a NEW tag for any rerun; never overwrite old builds
 done
mkdir "$TEST/$BUILD_TAG-reference-source"
git archive "$REFERENCE" | tar -x -C "$TEST/$BUILD_TAG-reference-source"
mkdir -p "$STAGE/build-logs-$STAMP"
"$CUDA_ROOT/bin/nvcc" --version | tee "$STAGE/build-logs-$STAMP/nvcc.txt"
grep -q 'release 13.2' "$STAGE/build-logs-$STAMP/nvcc.txt"
for arm in reference feature; do
 SRC="$TEST"; [ "$arm" != reference ] || SRC="$TEST/$BUILD_TAG-reference-source"
 cmake -S "$SRC" -B "$TEST/$BUILD_TAG-$arm" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" -DCUDAToolkit_ROOT="$CUDA_ROOT" \
  -DCMAKE_CUDA_ARCHITECTURES=87 -DSTRATA_GGML_DIR="$LLAMA_SRC" \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_HIP=OFF -DSTRATA_ENABLE_SYCL=OFF \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PREFILL_MMQ=OFF -DSTRATA_MMQ_KQUANTS=ON \
  -DSTRATA_BUILD_TESTS=ON 2>&1 | tee "$STAGE/build-logs-$STAMP/$arm-configure.txt"
 cmake --build "$TEST/$BUILD_TAG-$arm" --target strata-q36 strata_kq -j4 \
  2>&1 | tee "$STAGE/build-logs-$STAMP/$arm-build.txt"
done
# CPU-only real GGUF/template preparation. No engine/server/encoder is started.
FIXTURES="$STAGE/fixtures-$STAMP"
mkdir -p "$STAGE/tmp-$STAMP"; export TMPDIR="$STAGE/tmp-$STAMP"
"$PYTHON" tools/q36_adaptive_integration_fixture.py --model "$MODEL" --out "$FIXTURES" \
 > "$STAGE/fixture-$STAMP.json"
"$PYTHON" tools/q36_checkpoint_schedule.py --fixture "$FIXTURES/replay.csv" --out "$STAGE/prefix-audit-$STAMP.json"
# Inspect actual LCP/classification/template/IDs. Do not substitute byte fixtures.
```

### Staged native gate — ONLY in a separately approved isolation window

The development task does **not authorize stopping/starting production services**.
No service operation was executed. Before a future native run, the operator must
arrange an approved stop/reap/restore window, record exact prior service states,
verify no existing LLM/server/encoder processes, install guaranteed restoration,
and run tests in a bounded transient cgroup. Follow the already reviewed window
procedure for this host; do not guess manager scope or edit units. The worker
refuses existing models/encoders and never stops them or restores services itself.
It kills only its own child process groups. The operator must reap the **entire
test cgroup** before restoring prior states, then verify loaded health before
resuming clients. Failed inspection/reaping blocks restart pending manual recovery.
Do not invoke this worker on a loaded production host to see what happens.

Inside that approved isolated window, after the manager has proved old model
processes gone, run **one worker**, not concurrent arms:

```bash
# Set to the EXACT existing vision-qualified staging cap from saved test/config
# evidence, without editing that config. 0 means the existing context default.
: "${QUALIFIED_VISION_MAX_TOKENS:?copy existing qualified cap; do not guess}"
STRATA_CHECKPOINT_REVIEW_WINDOW=1 "$PYTHON" tools/q36_adaptive_integration_gate.py \
 --baseline "$TEST/$BUILD_TAG-reference/strata-q36" \
 --engine "$TEST/$BUILD_TAG-feature/strata-q36" --model "$MODEL" \
 --fixtures "$FIXTURES" --vision-max-tokens "$QUALIFIED_VISION_MAX_TOKENS" \
 --out "$STAGE/results-$STAMP" 2>&1 | tee "$STAGE/native-window-$STAMP.log"
```

The command's `--serve --vision` flags activate the existing strict vision staging
requirements; the **developer test dispatch exits before serve()**, so no protocol
server, HTTP listener or encoder is started. One native LLM process at a time.
No projection/full-split matrix investigation is run. A text-only gate cannot
requalify encoder co-residency or image answers.

1. **Baseline/reference smoke first**, then conv/GDN/hlast corrupted-payload
   controls. Stop at the first unexpected failure, including a control that fails
   for the wrong reason. Preserve logs/exit records; do not continue to adaptive
   or longer fixtures, increase context, or loosen comparisons.
2. **Same-day replay** on reference/fixed, integrated/fixed, integrated/adaptive:
   20 bounded requests per arm across temporary earlier-user insertion/removal,
   system memory changes, ordinary continuations and two interleaved idle chats.
   Each case marks first two requests warmup, later requests steady-state. Actual
   token-prefix classification is emitted for every row. Every arm checks its
   own matching-schedule recomputation, including the restored boundary. The
   reference/fixed and integrated/fixed schedule-ID/logit/pick dumps must be exact.
   Adaptive-vs-fixed raw dump differences are NOT restoration failures when their
   producer/suffix schedules differ. Different-schedule comparisons stay visible.
3. Inspect `report.json`: per-turn and warmup/steady reused/replayed tokens,
   checkpoint-copy bytes/us, prefill ms, first-token latency and startup warmup
   seconds. Reference computation is outside the timers; restored-state HOST
   snapshots inside the tested request still add diagnostic overhead. These
   are not full generated-answer latency or production throughput numbers.
   The report deliberately makes **no benefit claim**. Derive any diagnostic
   difference from same-day measured rows; reproduce a real serving benefit in a
   later explicitly approved benchmark before changing the fixed default.

`Q36_ADAPTIVE_INTEGRATION_ORIN PASS` qualifies only this bounded text fixture.
Full long-context/shared-pool pressure, concurrent active decode, sampled-MTP
and image inference/encoder memory/HTTP remain separate native gates. Re-run the
existing vision and sampled gates with the integrated build in separately
approved windows before deployment; sampled gate input must be its own
`sampled.csv`, not the old miswired prefix fixture. Preserve the already working
vision deployment until all required validation passes. No configuration edits
or automatic rollout are supplied by this change.
