# Multi-turn image-cache misses — diagnostic stage only

Native baseline: **4130bdfdb43bdb273412cc3530660bd19ec14602**, resolved from the
existing local repository, not GitHub's default branch. Development worktree:
`/workspace/tmp/q36-multimodal-cache`, branch
`feature/q36-multimodal-cache-diagnostics`. The Orin-only follow-up
**f934b85b42e79cc2a7c4ec0ca9da7685674e1ebd** is not available locally. This patch
leaves its three files alone; the handoff cherry-picks onto that follow-up and
checks they remain identical. No production, deployment, config, OWUI, sampler,
numerical kernel, matching/routing policy or image-checkpoint change.

## What is known, and what is not

The reported completed prompts13378/13676/13812 each reused zero tokens. Those
counts and rotating slots do **not** identify a cause: with no complete-prefix
match, `pick_session` prefers an empty slot. We do not have the failing requests'
actual expanded IDs, retained streams or image metadata. The cause of those
OWUI misses remains **unresolved**; no local model/Orin result is asserted.

Source inspection (all behavior retained):

* Native `Serve::match_len` requires **the entire retained live stream** to be a
  proper token prefix. That stream includes committed decoded tokens, not just
  the preceding prepared prompt. Equal/shorter prompts, earlier reasoning/header/
  system rewrites and a boundary inside a new image fail this condition.
* It also compares complete images within that prefix: cell, row count and image
  identity. Strict `q36_read_vision` identity hashes grid `(n,nx,ny)`, actual FP32
  embedding bytes, then the optional supplied key. Placement is compared separately.
  Stable Python keys alone are NOT proof of identical strict native image identity.
  The older server comment about matching despite different floats describes a
  different/legacy assumption; it is not a guarantee of the strict reader.
* Python `openai_to_messages` preserves reasoning fields; `ChatTemplate.render`
  and `Service.encode_prompt` decide the actual history/header tokens. They must
  be audited, not inferred from a growing message list. `Service.prepare` expands
  template image-pad markers using actual encoder row counts, concatenates SVE1
  inputs and supplies normalized-image keys. Historical images keep GENI/BGENI.
* Python `StrataEngine.pick_slot` can score a turn-prefix heuristic as well as a
  full held-stream prefix. Q36's native image path does not restore such a partial
  turn checkpoint. An explicit BGENI destination can still copy a valid complete
  donor from another slot. GENI lets the native engine select its destination.
* GENI bypasses adaptive learning/capture/restore. This explains **missing partial
  fallback after a genuine rewrite**, not rejection of a valid complete match.
  All `!geni`, image eligibility and incomplete-image protections remain intact.

The completed projection investigation stays closed. Different process/chunk
schedules may differ numerically; no new projection investigation or tolerance
relaxation is included.

## Optional observations, not decision inputs

Native `--kv-cache-diagnostics` is OFF by default. It logs to stderr, once per
actual request: at most8 candidate rows plus bounded search/selection/resolution/
retention/donor rows (four candidates with the deployed configuration). Existing
`match_len`/`pick_session` remain authoritative and source-identical. Each candidate
reports state, generation/epoch, backing, decoding, position, retained/incoming
length, token LCP, proper-prefix flag, complete-image boundary, prefix image counts,
image identity/placement equality and the **actual** match result. `pick_eligible`
mirrors the existing selector's filters; `idle_backed` is a separate observation,
not a newly imposed filter. Search totals distinguish no idle/backed owner from
owners that reject the token/image prefix. Selected and resolved destination,
actual reused count, whole-state donor and image checkpoint bypass are explicit.

Grid/row/key equality details are available only when the diagnostic has seen that
owner's actual image input in the same generation/epoch with the same ImgKeys.
It fingerprints existing validated HOST rows and re-reads SVE headers (no extra
embedding payload read). Unknown/stale detail is `details=0`, not assumed equal.
`prefix_delta_known=0` likewise means no valid incoming prefix offset was computed.
No supplied key/fingerprint, raw token value, decoded text, image byte, URL,
base64 or file path is printed. Failures in diagnostic bookkeeping are reported
as unavailable, never a new admission/routing decision. No GPU allocation,
fifth checkpoint payload, bank capacity/accounting or lease change is introduced.

Python `STRATA_Q36_CACHE_DIAGNOSTICS=1` is a separate, OFF-by-default opt-in:

* `prepare_candidate`: LCP before/after image expansion, previous prepared length,
  image-placement/grid/key/embedding-byte equality and proper prepared-prefix flag.
  These compare at most four **prepared prompts, not native retained/generated
  streams**. Do not mistake them for a cache authorization. Up to8 image details;
  larger metadata is explicitly unavailable. Prompt storage is bounded by262144
  IDs per stream; only diagnostic host state is added.
* `prepare`: expanded/unexpanded counts, image count, reasoning-message count and
  thinking flag; no rendered text. `slot_candidate/selected`: original Python
  selector result and full-versus-turn-prefix scores; original selector executes
  exactly once. Repeated polling is capped at8 observations per20s plus one
  suppression notice. `dispatch`: actual GEN/GENI/BGEN/BGENI mode, explicit slot,
  expanded count and key presence, never the serialized command. A send is still
  executed exactly once with its unchanged argument.

These flags are provided for isolated diagnostics, not a production configuration
edit. Timing overhead is expected when enabled; there is no performance claim.

### Reading the evidence

| Observation | Next conclusion/check, not an automatic fix |
|---|---|
| Unexpanded LCP already breaks before old retained end | Inspect actual history/reasoning/system rendering. Native complete-prefix rejection may be correct; a growing message list proves nothing. |
| Unexpanded prefix stable, expanded IDs/placement/grid change | Inspect preparation/encoder geometry and expansion. Use actual native retained length, not just previous prepared length. |
| Keys/grid stable, rows differ | Strict identity currently rejects different embedding bytes. Establish encoder/identity contract and correctness before changing it; a stable key alone cannot authorize reuse. |
| Python prepared prefix looks valid, native LCP fails | Native stream may contain generated/header tokens missing/rewritten in the next prompt. Compare the complete retained stream; this is not alone a routing bug. |
| Idle/backed complete native prefix + image match exists but actual matching/selection/resolution loses it | Capture slot/position/generation/epoch and actual GENI vs BGENI/donor events; this can justify an exact-prefix, identity or routing defect repair once reproduced. |
| Genuine token/image rewrite, no complete-prefix donor, image fallback bypassed | Zero reuse is presently expected. Image-aware checkpoints are a possible later feature, NOT a guard-removal fix. |

## Short public reproduction and strict neutrality comparison

`tools/q36_multimodal_cache_fixture.py` uses the actual GGUF tokenizer/template,
real OpenAI conversion and `Service.prepare` with frozen IO from **actual stored
encoder outputs**. It starts no model, encoder or HTTP service. Preparation is
performed diagnostics off/on and requires identical IDs, keys and SVE bytes.
There is no byte-token fallback. Public same-sized red/blue squares give:

1. Native constructed exact-prefix control, then a longer continuation, then an
   explicit different-slot continuation (whole-state donor). These intentionally
   constructed ID prefixes are **NOT rendered/API conversation evidence**.
2. Three rendered public histories retaining the same image and stable system.
   Canned public assistant content/reasoning is used, not model-generated answers;
   this is real API **preparation**, not live HTTP/OWUI evidence. Actual expanded
   tokens/image metadata classify each transition; rewrites are not relabelled.
3. Separate earlier-system rewrite retaining the same image.
4. Changed-image negative control: proper token prefix and identical pad/grid
   geometry, but genuinely different image identity/embedding bytes.

Ten rows total, each <=4096 expanded tokens. `--kv-image-cache-test` sends GENI/
BGENI with one emitted (not committed) token, making the constructed retained
prefix controls unambiguous. It records actual call start/declared suffix/chunk
callbacks, all four slots' persistent GDN/conv/hlast, image identity, rotary offsets,
main/MTP KV codes/scales, positions, live IDs, owner generation/epoch/reservations,
full logits and picks. No address/timing is included in the raw comparison.
The existing strict image parser reconstructs rows/M-RoPE in **each** arm.

The new worker runs the SAME binary off/on, one LLM at a time, requiring EXACT
raw state/logit/schedule bytes and emitted protocol equality. It also requires
stable image staging/idle leases, zero reset mappings, no image text-checkpoints
and unchanged post-readiness physical allocation/attempt/release counters. It
stops immediately on a failed control or comparison. This is **diagnostic
neutrality**, NOT an independent image correctness oracle: common bugs can exist
in both arms. The text-only independent schedule reference is NOT used for images.
No state is borrowed/uploaded as a reference and no tolerance is relaxed.

## Orin handoff — CPU/fixture work BEFORE any service stop

All staging/scripts/transfers: **/mnt/ssd/tmp/strata-checkpoints**. New isolated
checkout: **/mnt/ssd/Strata-Orin/worktrees/multimodal-cache**. The live checkpoint-
policy worktree and `build-adaptive-integration-v1-feature` are NEVER switched,
reset, rebuilt, removed or otherwise treated as disposable.

```bash
set -euo pipefail
REPO=/mnt/ssd/Strata-Orin
TEST="$REPO/worktrees/multimodal-cache"
STAGE=/mnt/ssd/tmp/strata-checkpoints
BASE=4130bdfdb43bdb273412cc3530660bd19ec14602
FOLLOWUP=f934b85b42e79cc2a7c4ec0ca9da7685674e1ebd
mkdir -p "$STAGE"
cd "$STAGE"
# Transfer the review .tar.gz AND .sha256 here, not into the live worktree.
sha256sum -c q36-multimodal-cache-diagnostics-review.tar.gz.sha256
tar -xzf q36-multimodal-cache-diagnostics-review.tar.gz
ART="$STAGE/q36-multimodal-cache-diagnostics-review"
(cd "$ART" && sha256sum -c SHA256SUMS)
git -C "$REPO" cat-file -e "$BASE^{commit}"
git -C "$REPO" cat-file -e "$FOLLOWUP^{commit}"
git -C "$REPO" merge-base --is-ancestor "$BASE" "$FOLLOWUP"
git -C "$REPO" bundle verify "$ART/q36-multimodal-cache-diagnostics-incremental.bundle"
git -C "$REPO" fetch "$ART/q36-multimodal-cache-diagnostics-incremental.bundle" refs/heads/feature/q36-multimodal-cache-diagnostics
TIP=$(git -C "$REPO" rev-parse FETCH_HEAD)
test "$(git -C "$REPO" rev-list --count "$BASE..$TIP")" = 1
# Stop if this isolated checkout already exists; inspect/preserve it, never force.
test ! -e "$TEST"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)-$$
git -C "$REPO" worktree add -b "review/q36-mm-cache-$STAMP" "$TEST" "$FOLLOWUP"
git -C "$TEST" cherry-pick "$TIP" # no merge/checkout of the old native baseline
# PROVE all three Orin corrections survived byte-identically:
git -C "$TEST" diff --exit-code "$FOLLOWUP" HEAD -- tools/q36_adaptive_integration_fixture.py tools/test_q36_integration_ordinary.py docs/Q36_ADAPTIVE_ORIN_VALIDATION.json
test -z "$(git -C "$TEST" status --porcelain=v1 --untracked-files=all)"
cd "$TEST"
PYTHON=/mnt/ssd/strata-service/venv/bin/python
export TMPDIR="$STAGE/mm-tmp-$STAMP"; mkdir -p "$TMPDIR"
# CPU checks only; never start the server for this command.
"$PYTHON" -m pytest -q tools/test_q36_multimodal_cache.py tools/test_q36_multimodal_cache_gate.py tools/test_q36_multimodal_cache_window.py
```

If pytest is absent from the service venv, use an already available isolated test
Python (with the runtime imports) or staged test-only dependencies; do not install
or upgrade packages inside the production venv to run these checks.

The bundle's exact prerequisite is4130bdfd, not f934b85b. The complete mail patch
is an ALTERNATIVE to bundle import; do not apply both. Missing objects, conflicts,
dirty/existing checkouts or changes to the three protected files require inspection,
not a reset/forced checkout. The source tests account for either exact4130 or an
f934-based integration, so they do not undo the Orin correction.

### Build only in the NEW review checkout

```bash
: "${CUDA_ROOT:?existing validated CUDA13.2 root, no replacement}"
: "${LLAMA_SRC:?existing pinned llama.cpp checkout}"
: "${MODEL:?actual Qwen3.6 GGUF first shard}"
: "${MMPROJ:?actual validated mmproj-F16_qwen36-35b.gguf}"
: "${ENCODER:?already validated strata-vision executable}"
test "$(git -C "$LLAMA_SRC" rev-parse HEAD)" = 3cf03257f219afbe7334045ff7c6a06ac68c627d
BUILD="$TEST/build-mm-cache-$STAMP"; test ! -e "$BUILD"
mkdir -p "$STAGE/mm-build-$STAMP"
"$CUDA_ROOT/bin/nvcc" --version | tee "$STAGE/mm-build-$STAMP/nvcc.txt"
grep -q 'release 13.2' "$STAGE/mm-build-$STAMP/nvcc.txt"
cmake -S "$TEST" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" -DCUDAToolkit_ROOT="$CUDA_ROOT" -DCMAKE_CUDA_ARCHITECTURES=87 \
 -DSTRATA_GGML_DIR="$LLAMA_SRC" -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_HIP=OFF -DSTRATA_ENABLE_SYCL=OFF \
 -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PREFILL_MMQ=OFF -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_BUILD_TESTS=ON \
 2>&1 | tee "$STAGE/mm-build-$STAMP/configure.log"
cmake --build "$BUILD" --target strata-q36 strata_kq q36_cache_diagnostic_test -j4 \
 2>&1 | tee "$STAGE/mm-build-$STAMP/build.log"
ctest --test-dir "$BUILD" -R '^q36_cache_diagnostic_test$' --output-on-failure
```

Prepare real encoder inputs BEFORE the model window, using existing verified
public same-grid SVE1 files if available, or a separate CPU-only encoder invocation:

```bash
ASSETS="$STAGE/mm-assets-$STAMP"
"$PYTHON" tools/q36_multimodal_cache_fixture.py --make-images "$ASSETS"
# NO --gpu: this encoder explicitly hides CUDA before backend init, uses only
# the vocab from MODEL, and exits/reaps at EOF. Bound CPU threads/RAM appropriately.
printf 'ENC %s %s\nENC %s %s\n' "$ASSETS/red-square.png" "$ASSETS/red.sve" \
 "$ASSETS/blue-square.png" "$ASSETS/blue.sve" | \
 "$ENCODER" --model "$MODEL" --mmproj "$MMPROJ" --max-tokens 1024 --threads 4 \
 > "$ASSETS/encoder.stdout" 2> "$ASSETS/encoder.stderr"
# Check READY2048 / OK and successful exit; malformed/all-zero/nonfinite output
# is also rejected by the CPU preparation tool. No fabricated embeddings.
FIXTURES="$STAGE/mm-fixtures-$STAMP"
"$PYTHON" tools/q36_multimodal_cache_fixture.py --model "$MODEL" \
 --image-a "$ASSETS/red-square.png" --image-b "$ASSETS/blue-square.png" \
 --sve-a "$ASSETS/red.sve" --sve-b "$ASSETS/blue.sve" --out "$FIXTURES" \
 > "$ASSETS/prepare.json" 2> "$ASSETS/python-diagnostics.log"
# If the API uses a different exported template, pass its exact --template too.
MM_CACHE_ENGINE="$BUILD/strata-q36"
"$PYTHON" tools/q36_multimodal_cache_gate.py --preflight --engine "$MM_CACHE_ENGINE" \
 --model "$MODEL" --fixtures "$FIXTURES" --out "$STAGE/mm-preflight-results-$STAMP"
```

Inspect `manifest.json`/expanded prefix classifications first. CPU preparation
must finish before any stop; a CPU encoder here does not reproduce GPU encoder
floating-point variation. Its immutable real SVE inputs are appropriate for the
bounded off/on diagnostic comparison, not proof of production encoder identity.
No CPU encoder is left running during the model window.

### Future explicitly approved window ONLY — do not execute during development

Review `tools/q36_multimodal_cache_window.sh` before any approval. Pause clients;
confirm **the actual SAME manager scope** for both units and preapprove all
noninteractive privileges. The script requires Strata initially active, snapshots
both units and loaded **vision-enabled** health, completes CPU preflight before
stopping, installs EXIT/INT/TERM restoration before stop, stops/reaps Strata and
initially active llama, and runs the worker in an entire transient cgroup with
`RuntimeMaxSec=300s`, `KillMode=control-group` and bounded cleanup. Worker progress
is emitted every20s during model wait; two sequential off/on arms share a300s
model budget. It starts no encoder or HTTP listener. Native `--serve --vision`
flags activate existing staging requirements; developer dispatch exits before
`serve()`.

The worker never controls services itself. On failure/timeout/normal exit the
window script stops/reaps the **entire test cgroup**, checks recorded cgroup-v2
population and global model absence, then restores Strata and llama's exact prior
active/inactive state. Health requires `service=strata,status=ok,loaded=true,
images=true` before and after llama restoration. This confirms enabled vision,
not semantic image inference. Failed inspection/reaping blocks restart pending
manual recovery. SIGKILL/power loss also needs manual recovery. Do not blindly
restart or resume clients after failed cleanup. Kernel/device cleanup can exceed
the target five minutes; never trade reaping safety for the deadline.

Only after the operator explicitly approves that reviewed procedure:

```bash
# FUTURE OPERATOR ACTION; no such service operation was run during development.
# HEALTH is the confirmed EXISTING endpoint, SCOPE is confirmed system OR user.
: "${HEALTH:?existing loaded Strata health endpoint}"
: "${SCOPE:?confirmed actual manager scope for BOTH units}"
export MODEL FIXTURES MM_CACHE_ENGINE HEALTH SCOPE
MM_CACHE_STOP_APPROVED=1 bash "$ART/source/tools/q36_multimodal_cache_window.sh"
```

The worker preserves the deployed settings exactly: context262144, four slots,
shared524288, INT8 KV, MTP2/batch-MTP/sampled-text-MTP, chunk512, checkpoint1024MiB/
interval1024/**adaptive**, preallocation, vision and vision cap1024. No production
config or live executable is edited. If the first bounded gate fails, STOP;
preserve logs/dumps and restore services. Do not expand context, rerun projection
studies or loosen exact comparisons to obtain a pass.

## Stop point / later work

A matched-input, matched-schedule off/on PASS establishes only diagnostic
neutrality for this public fixture. Real reported misses remain unqualified until
paired preparation/dispatch/native candidate evidence identifies the first break.
An actual exact-prefix/identity/routing defect warrants a separate small repair
with matching-schedule image correctness tests, not a retention workaround.

Only demonstrated unavoidable earlier rewrites with useful reusable complete
image prefixes justify designing image-aware checkpoints. That future design
must validate every reused image's identity/grid/placement, rotary positions and
retained offsets, GDN/conv/hlast seed, all main/MTP KV codes/scales, owner pins,
generation/epoch/backing, reset/eviction/cancellation invalidation and staging
leases. Incomplete images remain ineligible boundaries. It also needs an
independent reference reconstructing image inputs and positional state from empty,
not the current text-only reference or a copied checkpoint. Four-payload/bank
bounds and sampled-image exclusion stay intact. None of that restoration work is
implemented here.
