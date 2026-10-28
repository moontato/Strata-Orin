# Image-aware Qwen3.6 prefix checkpoints — opt-in, not Orin-qualified yet

Local parent: **765068db267f8cd8f846f8ef45c9cc94f90075d5**, on4130bdfd.
This is a continuation of the multimodal-cache diagnostics, not the older
projection/adaptive fork. No kernel/sampler/tolerance, OWUI/frontend matching,
prompt rewriting, deployment, configuration or service operation is part of this
change. **`--kv-image-checkpoints` defaults OFF**, including after building it.

## Evidence, without an unsupported routing diagnosis

Operator reports: diagnostics CUDA build and15 CPU CTests passed; native
OFF/ON diagnostic state/logit/output/schedule neutrality passed. Constructed
image prefixes reused1060 and1072 cells, including cross-slot. Real rendered
follow-ups changed their previous token prefix and reused zero. Actual OWUI
follow-ups changed message0 content too, but precise token divergence and possible
checkpoint savings for those requests are **UNMEASURED**. None establishes an
exact-prefix routing defect or implicates fetch_url specifically. The tests here
separate public rendered histories, constructed controls and actual OWUI evidence.
The completed projection investigation is neither rerun nor used as a tolerance.

## State, eligibility and reference design

1. Exact complete-state prefix matches remain FIRST, including pinned cross-slot
   donors. With the new flag, actual image bytes/geometry authorize that path too;
   a changed advisory key alone does not invalidate identical actual inputs.
2. Only if no exact donor serves the request may GENI/BGENI discover an image-aware
   checkpoint. Actual token LCP is mandatory. Image semantics can only REDUCE it.
   Complete images before the boundary must have identical FP32 embedding BYTES,
   width, row count, grid, token-cell placement and rotary start. No hash/key alone
   grants permission. Images after the boundary are not part of its permission;
   changed/added later images cannot authorize a longer prefix. An incomplete
   image is never an eligible endpoint.
3. Each payload has a bounded host proof: immutable original image bytes/layout,
   consumed end, retained rotary delta and explicit readiness. GPU payload remains
   the existing packed GDN recurrence, convolution histories and last hidden seed;
   main+MTP KV codes/scales use the original charged session planes. Cross-slot
   restore copies EVERY prefix plane and eligible inherited payload. Same-slot
   rewind keeps unchanged earlier KV, restores recurrence/seed, and retires later
   records. Metadata is published only after transfer+stream synchronization.
4. Capture readiness is not inferred from `!geni` removal. A fresh empty prefix,
   a qualified restored prefix, or an exact prefix whose contiguous seeded end
   still equals reuse may extend readiness. The suffix must take the existing
   prompt/MTP path (`fresh>1` or restored singleton). Ordinary decode cannot extend
   this provenance. Unseeded exact continuations conservatively do not capture.
5. Image restore is NOT the old text restore's clear-and-forget behavior: it
   reconstructs complete-prefix image metadata and CPU/device rotary delta before
   publishing the restored state. Incoming image rows/table are uploaded from the
   independently validated full NEW request before suffix processing. Shared
   staging views are never copied from a donor or captured as persistent state.
6. Owner availability retains generation, epoch, backing, residency, decoding,
   exact donor-pin and destination-lease checks. Donors are pinned BEFORE admission
   can reclaim idle owners. Image full-state donors now retain that pin THROUGH
   admission/copy too. Discovery does not borrow arbitrary active owners. Reset,
   replay/rewind and eviction invalidate/retag the corresponding image proofs with
   their original checkpoint metadata. No new checkpoint authorization by learning.

### Legal boundaries without changing numerical schedules

Only existing process callbacks may capture. No chunks are split or calls
shortened. An interval/learned target inside a picture maps to the last legal
callback at/before it; if none exists, defer to the first legal complete callback.
This deliberately permits an early safety point BELOW interval, rather than
losing all pre-image reuse. At most two middle banks use learned/fallback legal
anchors; the final bank rolls. One-payload budgets retain the single safety bank.
At512-token chunks, image cells700..1723 and interval1024, safety512 and middle2048
are both useful;1024/1536 are forbidden. If the first image spans50..1073, the
first possible legal callback is1536. A divergence earlier than that may
legitimately yield zero reuse: a missing past state cannot be invented.

### Cancellation and transitions

Before-image/outside-image cancellation keeps only the actual read prefix and its
own positional offset; legal earlier checkpoints may survive. Cancellation inside
an image invalidates/reset that owner AFTER ending the admission, discarding its
proofs and partial image state. Error paths retain existing quiesce/reset rules.
Prefill views retire without clearing another slot's offset; new uploads drain
queued work before reuse. Exact prefix, image->text, four-slot interleaving and
sampled-image MTP exclusion remain supported as before. The sampled-image guard is
source-identical; this feature does not implement sampled-image speculation.

## Memory and defaults

Activation requires strict preallocation/vision/serve, a checkpoint budget,
chunk512 and an explicit combined image-row cap **<=1024**. The qualification gate
requires the deployed cap1024. Fixed staging remains the SAME allocation; rows,
M-RoPE and mapping buffers never grow after readiness. No new GPU scratch,
fifth payload, layout, capacity function or physical handle allocation is added.
The four-payload ceiling,1GiB configured budget and shared524288/context262144
bank calculation are unchanged (3924 pages /8,229,224,448 bank bytes if granularity
is2MiB). That is a calculation, not a new measurement.

Actual bytes live in shared immutable HOST objects. Per slot, at most four
checkpoint identities plus one live identity; an in-flight request can add one
more globally. Thus <=21 distinct1024-row byte arrays, each `1024*n_embd*4`, plus
<=21*1024 fixed-size image descriptors and small per-slot plans/proofs. At width2048
this bounds additional retained/in-flight proof bytes by168MiB plus about1.2MiB
of descriptors/control storage (calculated payloads, not measured allocator RSS).
Request parsing/upload vectors and host comparison snapshots are separate. The
planner's transient callback list is <=512 entries at this geometry. Proofs share
arrays, not one copy per capture; reset/eviction retires their references. Fresh
existing UMA/cgroup headroom checks still see actual host/encoder residency. The
encoder is external; this gate does not requalify its workspace/co-residency.

Flag OFF retains old image checkpoint bypass. Text checkpoint policy/defaults,
fixed image staging, numerical methods/kernels and sampler restrictions stay
unchanged. New CPU descriptors simply expose already-calculated grid/position
values; no parser position arithmetic changed. Host default routing is compared
to the pre-feature selector over1000 cases. This is NOT CUDA default-output/SASS
parity; actual old/new CUDA qualification remains pending.

## Independent correctness oracle

`--kv-image-checkpoint-test FILE` records original FULL declared token calls,
original SVE file names/keys, consumed callback ends and checkpoint-suffix/MTP/head
semantics. Original unconsumed tails remain in the recipe. A captured prefix in
the middle of a call must be replayed with that original declared length and stop
callback, not as a shorter call. No process/forward/GEMM algorithm is changed.

For every request, an isolated existing reference slot starts EMPTY. It rereads
original SVE files, rebuilds positions with the unchanged LEGACY image builder
(independent of candidate's bounded builder/proof/delta helper), uploads ORIGINAL
inputs, and executes real process calls under the recorded schedule. Strict
image hashes are independently derived from actual rows then original keys.
Retained delta is separately reconstructed from the legacy M-RoPE table. The
reference NEVER restores/copies a checkpoint or uploads candidate snapshots.
Other idle slots are pinned while its admission runs; no fifth GPU state is used.

Checks are EXACT: restored boundary AND final GDN/conv/hidden seed, main KV,
every initialized MTP KV code/scale, CPU position/token history/complete-image
metadata, CPU+device rotary delta, and full final logits (finite). Transient logits
at a headless checkpoint, token staging and dead scratch are not persistent
state. A corrupted convolution payload MUST fail the restored-state comparison,
not merely exit nonzero for another reason. Different OFF/ON chunk histories may
produce different outputs; each arm must pass its OWN independently reconstructed
matching schedule. Diagnostic neutrality alone is not an oracle.

Native cancellation checks also compare the valid pre-image cancelled prefix to
an empty-state replay, invalidate inside-image state, exercise operation-owned pin
eligibility and verify reset invalidation. Physical allocation/attempt/release
counters and the image staging pointer must remain constant; final mappings zero.

## Bounded gate and interpretation

The public fixture includes an actual-tokenizer/template/Service.prepare history
with an earlier USER message changed AFTER a complete historical image. First two
requests are warmup; the steady rewrite (including cross-slot) MUST restore a
positive after-image boundary. Preparation fails if actual rendered IDs cannot
support that claim. Native controls isolate before-image boundaries, exact
continuation/full-state donor preference, changed pixels with stable keys,
changed grid and placement. Grid manipulation is explicitly synthetic, not an
encoder result. Prior real rendered header/reasoning follow-ups are preserved as
observed controls; zero reuse is reported honestly.

One bounded worker runs convolution corruption, fallback DISABLED, then ENABLED,
one LLM child at a time. Total worker budget (including CPU preflight) defaults300s,
explicit maximum420s, plus at most20s owned-child reaping. Progress is emitted for
preflight rows, startup/every20s and each native turn. First unexpected failure stops
the sequence. It kills/reaps only its own groups; caller must reap the enclosing
cgroup too. No service control, encoder, HTTP listener, projection matrix or
sampling suite is launched. Fixtures <=24 turns, each <=6144 expanded cells.

`IMAGE_CHECKPOINT_MEASURE` reports reused/replayed cells, checkpoint boundary,
prefill_ms, request_ms, checkpoint copy_us/bytes and metadata_us. Prefill includes
capture copies; request includes input parsing, staging, identity work and first
token. copy_us measures synchronized payload/KV transfers; metadata_us measures
proof construction and selected identity searches (NOT all host overhead).
Reference work/comparison is OUTSIDE those request timings. Compare steady public
fixture measurements only as **synthetic performance**, never actual OWUI gains
or a general speedup. Actual OWUI token divergence/savings remain UNMEASURED.

## Orin handoff — commands only, NOT executed here

Use ONLY existing `/mnt/ssd/Strata-Orin/worktrees/multimodal-cache`.
Never modify/build live `worktrees/checkpoint-policy`. All build staging, copied
scripts, fixtures and results below live under `/mnt/ssd/tmp/strata-checkpoints`.
No push/deploy/config edits or service operations are authorized by this handoff.

The actual diagnostic checkout includes f934b85b plus a cherry-picked diagnostic
commit; its HEAD need NOT equal765068db. The incremental bundle requires4130bdfd
and carries the original765068db object plus the NEW feature commit, so verification
does not assume the original diagnostic object was previously fetched. Importing
that object does NOT mean replaying its commit onto the Orin branch. Preserve its source/ignored artifacts.
Apply ONLY this archive's incremental feature commit onto that actual clean HEAD,
not a reset/merge/switch to the archive tip which would lose the Orin follow-ups.

```bash
set -euo pipefail
STAGE=/mnt/ssd/tmp/strata-checkpoints
REPO=/mnt/ssd/Strata-Orin
TEST="$REPO/worktrees/multimodal-cache"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)-$$
mkdir -p "$STAGE"; cd "$STAGE"
# Transfer archive + its .sha256 here first.
sha256sum -c q36-image-checkpoints-review.tar.gz.sha256
tar -xzf q36-image-checkpoints-review.tar.gz
ART="$STAGE/q36-image-checkpoints-review"
(cd "$ART" && sha256sum -c SHA256SUMS)
test "$(git -C "$TEST" rev-parse --show-toplevel)" = "$TEST"
test "$(git -C "$TEST" rev-parse --path-format=absolute --git-common-dir)" = "$(git -C "$REPO" rev-parse --path-format=absolute --git-common-dir)"
test -z "$(git -C "$TEST" status --porcelain=v1 --untracked-files=all)"
OLD=$(git -C "$TEST" rev-parse HEAD)
git -C "$TEST" merge-base --is-ancestor f934b85b "$OLD"
git -C "$TEST" log -5 --oneline > "$STAGE/image-source-before-$STAMP.txt"
PROTECTED=(tools/q36_adaptive_integration_fixture.py tools/test_q36_integration_ordinary.py docs/Q36_ADAPTIVE_ORIN_VALIDATION.json)
(cd "$TEST" && sha256sum "${PROTECTED[@]}") > "$STAGE/image-protected-before-$STAMP.sha256"
KEEP="preserve/multimodal-diagnostic-$STAMP"
git -C "$REPO" branch "$KEEP" "$OLD"
git -C "$REPO" bundle create "$STAGE/multimodal-before-image-$STAMP.bundle" "$KEEP" ^4130bdfd
# Inventory old ignored artifacts without touching or hashing multi-GB dumps.
(cd "$TEST" && find . -maxdepth 1 -type d \( -name 'build*' -o -name logs \) \
 -exec find {} -type f -printf '%p\t%s\t%T@\n' \; | LC_ALL=C sort) > "$STAGE/image-artifacts-before-$STAMP.tsv"
BUNDLE="$ART/q36-image-checkpoints-incremental.bundle"
git -C "$REPO" bundle verify "$BUNDLE"
git -C "$REPO" fetch "$BUNDLE" refs/heads/feature/q36-image-checkpoints
FEATURE=$(git -C "$REPO" rev-parse FETCH_HEAD)
test "$(git -C "$REPO" rev-parse "$FEATURE^")" = 765068db267f8cd8f846f8ef45c9cc94f90075d5
git -C "$TEST" switch -c "review/image-checkpoints-$STAMP" "$OLD"
git -C "$TEST" cherry-pick "$FEATURE" # ONLY the new commit, not its ancestry
# Any conflict: STOP, preserve logs/ref/working tree for review. No force/reset/clean.
(cd "$TEST" && sha256sum -c "$STAGE/image-protected-before-$STAMP.sha256")
(cd "$TEST" && sha256sum "${PROTECTED[@]}") > "$STAGE/image-protected-after-$STAMP.sha256"
cmp "$STAGE/image-protected-before-$STAMP.sha256" "$STAGE/image-protected-after-$STAMP.sha256"
test "$(git -C "$REPO" rev-parse "$KEEP")" = "$OLD"
git -C "$TEST" diff --name-only "$OLD" HEAD > "$STAGE/image-changed-paths-$STAMP.txt"
for p in "${PROTECTED[@]}"; do ! grep -Fx "$p" "$STAGE/image-changed-paths-$STAMP.txt"; done
git -C "$TEST" rev-parse HEAD > "$STAGE/image-actual-tip-$STAMP.txt"
(cd "$TEST" && find . -maxdepth 1 -type d \( -name 'build*' -o -name logs \) \
 -exec find {} -type f -printf '%p\t%s\t%T@\n' \; | LC_ALL=C sort) > "$STAGE/image-artifacts-after-$STAMP.tsv"
cmp "$STAGE/image-artifacts-before-$STAMP.tsv" "$STAGE/image-artifacts-after-$STAMP.tsv"
```

Local765068db lacks two Orin-only files; their local absence was preserved, NOT
filled with fabricated contents. The hash checks above MUST succeed on the actual
Orin files. The exported complete patch is an alternative for exact local parent;
do NOT apply it after bundle/cherry-pick, or overwrite Orin files from `source/`.

### Build and CPU preflight (no service operations)

```bash
: "${CUDA_ROOT:?existing CUDA13.2 toolkit, no replacement/install}"
: "${LLAMA_SRC:?existing validated llama.cpp checkout}"
test "$(git -C "$LLAMA_SRC" rev-parse HEAD)" = 3cf03257f219afbe7334045ff7c6a06ac68c627d
PYTHON=/mnt/ssd/strata-service/venv/bin/python
MODEL=/mnt/ssd/llamacpp_models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
BUILD="$STAGE/build-image-checkpoints-$STAMP"
SCRIPTS="$STAGE/image-scripts-$STAMP"
test ! -e "$BUILD"; mkdir "$SCRIPTS"
cp "$TEST/tools/q36_image_checkpoint_fixture.py" "$TEST/tools/q36_image_checkpoint_gate.py" "$SCRIPTS/"
export STRATA_IMAGE_CHECKPOINT_REVIEW_ROOT="$TEST"
"$CUDA_ROOT/bin/nvcc" --version | tee "$STAGE/image-nvcc-$STAMP.txt"
grep -q 'release 13.2' "$STAGE/image-nvcc-$STAMP.txt"
cmake -S "$TEST" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" -DCUDAToolkit_ROOT="$CUDA_ROOT" \
 -DCMAKE_CUDA_ARCHITECTURES=87 -DSTRATA_GGML_DIR="$LLAMA_SRC" \
 -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_HIP=OFF -DSTRATA_ENABLE_SYCL=OFF \
 -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PREFILL_MMQ=OFF -DSTRATA_MMQ_KQUANTS=ON \
 -DSTRATA_BUILD_TESTS=ON 2>&1 | tee "$STAGE/image-configure-$STAMP.txt"
cmake --build "$BUILD" --target strata-q36 strata_kq -j4 2>&1 | tee "$STAGE/image-build-$STAMP.txt"
# Explicit paths to the already saved actual diagnostic inputs; no encoder launch.
: "${DIAGNOSTIC_FIXTURES:?existing q36-image-cache-v1 fixture directory under STAGE}"
: "${IMAGE_A:?saved public red-square PNG under STAGE}"
: "${IMAGE_B:?saved public blue-square PNG under STAGE}"
: "${SVE_A:?saved real1024-row red embeddings under STAGE}"
: "${SVE_B:?saved real1024-row blue embeddings under STAGE}"
FIXTURES="$STAGE/image-checkpoint-fixtures-$STAMP"
"$PYTHON" "$SCRIPTS/q36_image_checkpoint_fixture.py" --model "$MODEL" \
 --image-a "$IMAGE_A" --image-b "$IMAGE_B" --sve-a "$SVE_A" --sve-b "$SVE_B" \
 --diagnostic "$DIAGNOSTIC_FIXTURES" --out "$FIXTURES" > "$STAGE/image-prepare-$STAMP.json"
"$PYTHON" "$SCRIPTS/q36_image_checkpoint_gate.py" --preflight --engine "$BUILD/strata-q36" \
 --model "$MODEL" --fixtures "$FIXTURES" --out "$STAGE/image-results-$STAMP"
```

Only after a SEPARATELY approved existing operator isolation window: pause clients,
record both prior Strata/llama states, stop/reap them using the already reviewed
manager scope, and own a transient test cgroup. None of those operations was
performed or automated here. In that isolated cgroup run the same command without
`--preflight`, explicitly passing BOTH environment variables:

```bash
STRATA_IMAGE_CHECKPOINT_REVIEW_ROOT="$TEST" STRATA_IMAGE_CHECKPOINT_REVIEW_WINDOW=1 \
 "$PYTHON" "$SCRIPTS/q36_image_checkpoint_gate.py" --engine "$BUILD/strata-q36" \
 --model "$MODEL" --fixtures "$FIXTURES" --out "$STAGE/image-results-$STAMP" \
 2>&1 | tee "$STAGE/image-window-$STAMP.log"
```

No old/live process is killed by the worker; surviving models cause refusal.
The operator MUST reap the entire test cgroup, restore both exact prior active/
inactive states, and verify `service=strata,status=ok,loaded=true` at the existing
`http://100.100.10.3:11436/health` before resuming clients. Failed inspection/reaping
blocks restart; SIGKILL/power loss requires operator recovery. Do not use an old
window script that hard-codes the previous diagnostic worker. No production
executable, API binding, OWUI settings or configuration is changed.

Required numerical marker: `IMAGE_CHECKPOINT_ORIN_GATE PASS independent=1 exact=1`.
Keep commands, actual source/protected-file hashes, raw logs/exit records and every
per-turn measurement. Positive steady rendered rewrite reuse is REQUIRED; legit
zero cases stay zero. Broader sampling, actual OWUI, encoder peaks/co-residency,
other backends and performance distributions are separate pending qualifications.
