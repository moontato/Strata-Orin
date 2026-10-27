# Qwen3.6 vision with strict KV preallocation — isolated, review only

Base **a9837161edcbce84be14d6993bcf428ef96b9ac1**, branch
`feature/q36-vision-preallocation`. No adaptive policy, projection diagnostics or
numerical fixes are ancestors of this change. No push/deployment/config/OWUI or
real service operation was performed during development.

## Established versus pending

Operator-established Orin evidence: tools/vision builds at pinned ggml
3cf03257f219afbe7334045ff7c6a06ac68c627d; GPU mmproj-F16_qwen36-35b.gguf loads;
warmup at1024 image tokens succeeds; produced embeddings are finite/nonzero,
width2048. **Actual image->LLM inference is not established by those facts.**
This revision's CUDA13.2 SM87 build, image/text/mixed/cancellation/model/bank gate,
semantic red/blue answers and HTTP integration remain pending. CPU mocks and
fake-CUDA-boundary tests are explicitly not that validation. No speedup claim.

## Why removing the guard alone is insufficient

The existing strict activation rejected vision. Its text-only bank bound can
cover image requests, but legacy vision lazily reallocates `d_img` AFTER READY.
Images use shared prefill embeddings/table/map, whereas decode uses each session's
`rdelta` and batch row-delta tables. Cancellation may cut inside a picture or
before a later picture, so retained positional state and cache eligibility must
be handled explicitly. An encoder is another process, not a bank allocation.

This change retains the narrow CUDA SM87/four-slot/INT8/full-GPU/MTP2/batch-MTP/
explicit-chunk/warmup requirements, and supports strict vision through
`--serve` GENI/BGENI. Standalone `--emb` remains refused in strict mode; legacy
vision without preallocation stays unchanged. No math kernel, sampling setting,
loss tolerance, checkpoint policy or image-checkpoint representation changes.

## KV capacity proof — expanded physical cells, not rotary coordinates

The server expands every image placeholder to exactly its number of embedding
rows. Each image-pad cell writes ONE ordinary main/MTP KV cell at `pos+t` (or the
batch row's cell). M-RoPE chooses rotation coordinates `(t,h,w)`; it does not
change the cache address, tensor width, head stride, granularity or write count.
Grid coordinates stay within the expanded logical prompt's extent. The same
admitted prompt+output/context contract, max8 verification padding, 256-cell
mapping coarsening, trim on end/rewind and four protected owners apply.

`kv_capacity_bound`, `KvPrefix`, allocator/bank ownership and every INT8 kernel are
unchanged. Images do not get extra cache planes or a fifth checkpoint record.
At context262144 / shared ceiling524288 / four slots, using actual Qwen geometry
and IF CUDA reports2MiB granularity, the existing calculator gives:

* 3168 attention pages (11 caches including MTP, two heads, INT8 K/V + FP16 scales);
* 244 state pages (GDN/conv/hlast for all slots), 512 checkpoint pages at1GiB;
* total3924 pages = **8,229,224,448 bytes = 7848MiB = 7.6640625GiB**.

This is a calculated bank-only bound, not total process residency or a measured
GPU result. A maximizing attention witness is reservations
`(253945,253945,8185,8185)`, sum524260<=524288; code-step counts `(32,32,2,2)`.
The older500000 example's7584MiB is NOT the requested524288 bank. Runtime always
recomputes from actual layouts/granularity. No equal131072 partitions or extra
output beyond context are introduced. Two full-context reservations fit the
logical ceiling only without exceeding prompt/output safety contracts.

## Ancillary memory — explicitly outside the KV bank

Only `--kv-preallocate --vision` preallocates a stable shared image arena before
bank admission/expert placement:

* `4 * max_image_rows * embedding_width` device bytes for FP32 image embeddings;
* `12 * context` device bytes for one M-RoPE table;
* `4 * chunk` device bytes + `4 * chunk` pinned host bytes for row mapping;
* existing batch-MTP logits/picks, now allocated before readiness on this new
  path: `kVerify*(4*vocabulary+4)` bytes. It used to initialize lazily at the first
  mixed batch. Text-only/legacy behavior is unchanged.

New optional **`--kv-vision-max-tokens N`** caps TOTAL image rows per request,
not rows per image or per slot. It is valid only with `--kv-preallocate --vision`,
positive and <=context. Omitted: context is the proven upper bound, permitting
any valid expanded prompt without runtime reallocation. At width2048/context262144
that default costs **2GiB** for embeddings alone. For initial1024-row review,
explicit cap1024 costs8MiB; M-RoPE costs3MiB, device and pinned maps2KiB each at
chunk512. If vocabulary248320, eager batch-MTP scratch is7,946,272 bytes. These
are calculated payload sizes, not measured CUDA bookkeeping/residency.

`KVVISION` reports these categories separately; none is included in
`KVBANK.bank_bytes` or `KVSESSION.bytes`. They are actual allocations BEFORE
fresh bank/expert/headroom readings, not fictitious capacity subtracted twice.
No image staging allocation/growth/free occurs after READY. Over-cap/malformed
input is rejected rather than resized. The host parser validates file size,
aggregate rows, grid/width, finite floats, exact pad runs and key count before
GPU mutation; it allocates exactly the accepted float payload once. CPU rows,
cell map, M-RoPE and image metadata are additionally bounded by cap/context; they
are not GPU KV pages. Static buffers are freed only after device quiescence and
all graphs are destroyed; failure quarantines KV rather than recycling live VA.

### Encoder weights/workspace are a separate process charge

No engine flag can reserve or certify a private mtmd/ggml allocator. The unchanged
server normally starts/warmups the resident encoder BEFORE the engine. The review
worker does so too and encodes BOTH test shapes before loading the LLM, keeping
that encoder resident across every arm. Existing fresh CUDA/cgroup-aware UMA
readings therefore already include its actual resident allocations; don't add
CPU-free and CUDA-free bytes or charge that memory to the KV bank. Existing6GiB
UMA headroom and768MiB pool reserve remain unchanged. `encoder-resident-status`
is host RSS evidence ONLY, not a GPU-workspace measurement.

Record encoder logs/cap and same-day tegrastats/device memory separately. N/A
GPU process metrics are not a measured zero. The encoder may allocate for a new
shape; a1024 warmup alone is not a universal workspace bound. This change checks
fresh headroom again before strict image upload and refuses exhaustion; it never
borrows more bank handles or silently offloads experts. Arbitrary external
allocations cannot be guaranteed by a KV bank. Elastic unload/reload, more image
rows/shapes or concurrent encoder-workspace peaks are NOT qualified by the short
gate; measure/reserve them separately before expanding use. No server/encoder
configuration is changed here.

## Four-slot transitions and lifetime

* ONE serialized prefill owns the shared image rows/table via a slot/generation/
  epoch lease. Active owners cannot be overwritten; every image chunk checks its
  lease. Decode windows can interleave, but use per-row slot/delta tables, not
  the image prefill views. Original numerical forward/process/row kernels stay
  source-identical. Existing exact image-prefix reuse remains, including copying
  full session state; checkpoint restore for images stays ineligible.
* After prefill, success/cancel/error scope retires host image views without
  zeroing any slot's decode offset. Buffers remain allocated even if a T=1 kernel
  is queued. New upload drains the stream before overwriting; error cleanup
  quiesces/reset admissions; shutdown drains before freeing. No graph captures
  a transient image pointer: image prompt cells run eagerly as before; decode
  graphs read stable state and positional scalars/tables.
* Image->text fresh replay resets that slot's offset/history; switching slots
  loads its own offset. Peers' KV/GDN/conv/hlast/live/image identity/offset cannot
  change. Pool invalidation retires image views ONLY if it owns them.
* Cancel inside an image: never reusable; invalidate/reset that owner after ending
  the lease. Cancel outside images: retain complete images and compute the decode
  offset for the actually read prefix, not for unread future images in the full
  request. The CPU regression covers a cut before a second image with a different
  final offset. Next GENI continuation reuploads the new complete table safely.
* Supplied image keys alone cannot authorize reuse of different rows: strict image
  identities hash actual embedding bytes too. No frontend/text-specific rules.
* Image requests still never select/capture text checkpoints. The existing
  sampled-MTP condition excludes GENI, image history, batch and concurrent requests;
  it is unchanged. Existing greedy/image/batch behavior is not a new sampled-image
  implementation or qualification. Ordinary text retains its flags/defaults and
  checkpoint/sampled eligibility.

## Focused validation (no service operation executed here)

CPU CTests: new bounded SVE/M-RoPE/lease/capacity test and production strict-staging
methods executed through real pool admission with ONLY CUDA IO/free-memory boundary
faked, plus existing allocator/checkpoint/sampled tests. Source checks retain all
numerical kernels/process/forward/f32mm/row functions and sampler/checkpoint guards.
A declaration-only syntax scaffold compiles new methods/model gate against stand-ins,
NOT the actual Engine/Serve/CUDA TU. Mutation controls must fail if row bounds,
lease exclusion, epochs, finite checks, retained delta or actual-row identity break.

Native `--kv-vision-test FILE` uses actual encoder/GGUF data and six short scenarios:
image then text, four idle slot state isolation, four ACTIVE mixed image/text slots
with windows during prefill, cancel inside image then text, cancel in a text tail
then image resume, sampled image (zero offers) followed by sampled text (offers>0).
Text must still capture eligible checkpoints; image cases must not capture them.
Missing cancellation/mixed-slot coverage fails, not skips. Full persistent state,
main/MTP KV prefixes, positions and logits are saved and compared EXACTLY under
matching execution schedules; captured logits are also finite checked. No newly
relaxed oracle/tolerance.

The worker compares a9837161 + diagnostic-only test entry point, feature/lazy and
feature/preallocated, one LLM at a time with one intentionally co-resident encoder.
It requires exact dumps AND emitted-token/protocol equality, fixed strict staging
pointer, unchanged physical VMM allocation/attempt/release counters and zero KV
mappings after reset. No hash-only pass. Test-only snapshots are HOST data, not
GPU checkpoint payloads. Allow at least8GiB disk for raw three-arm dumps; they are
not throughput measurements. Read the decoded red/blue answers manually: the
numerical gate does not certify semantic image understanding, HTTP or OWUI.

## Exact Orin handoff — nested repository layout only

All staging/fixtures/results: **/mnt/ssd/tmp/strata-vision**. Production/current/
rollback/preallocation and checkpoint-policy worktrees are never reset or edited.
New nested test worktree only; reuse it if already present and clean.

```bash
STAGE=/mnt/ssd/tmp/strata-vision
REPO=/mnt/ssd/Strata-Orin
TEST="$REPO/worktrees/vision-preallocation"
mkdir -p "$STAGE"
# Transfer q36-vision-preallocation-review.tar.gz AND .sha256 to STAGE.
cd "$STAGE"; sha256sum -c q36-vision-preallocation-review.tar.gz.sha256
 tar -xzf q36-vision-preallocation-review.tar.gz
ART="$STAGE/q36-vision-preallocation-review"
(cd "$ART" && sha256sum -c SHA256SUMS)
BUNDLE="$ART/q36-vision-preallocation-incremental.bundle"
git -C "$REPO" bundle verify "$BUNDLE"
git -C "$REPO" fetch "$BUNDLE" refs/heads/feature/q36-vision-preallocation
TIP=$(git -C "$REPO" rev-parse FETCH_HEAD)
if [ ! -e "$TEST" ]; then
  git -C "$REPO" worktree add -b review/q36-vision-preallocation "$TEST" "$TIP"
else
  test "$(git -C "$TEST" rev-parse --show-toplevel)" = "$TEST"
  test -z "$(git -C "$TEST" status --porcelain)"
  git -C "$TEST" merge --ff-only "$TIP"
fi
# The complete mail patch is an ALTERNATIVE from exact a9837161, not an extra patch.
cd "$TEST"; test -z "$(git status --porcelain)"
```

Build only in this test worktree; no package/toolkit replacements. The baseline
export has no .git and only the same diagnostic gate/cancellation hook; its normal
math, legacy parsing/allocations and original vision preallocation rejection stay.

```bash
set -euo pipefail
MODEL=/mnt/ssd/llamacpp_models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
: "${MMPROJ:?set actual already-validated mmproj-F16_qwen36-35b.gguf path}"
: "${LLAMA_SRC:?set existing pinned llama.cpp checkout path}"
: "${CUDA_ROOT:?set existing CUDA13.2 toolkit root; no replacement/install}"
PYTHON=/mnt/ssd/strata-service/venv/bin/python
test "$(git -C "$LLAMA_SRC" rev-parse HEAD)" = 3cf03257f219afbe7334045ff7c6a06ac68c627d
mkdir -p "$STAGE/build-logs"
"$CUDA_ROOT/bin/nvcc" --version | tee "$STAGE/build-logs/nvcc.txt"
grep -q 'release 13.2' "$STAGE/build-logs/nvcc.txt"
"$PYTHON" tools/q36_vision_baseline_export.py --out "$TEST/build-baseline-vision-source"
for arm in baseline feature; do
  SRC="$TEST"; [ "$arm" != baseline ] || SRC="$TEST/build-baseline-vision-source"
  BUILD="$TEST/build-$arm-vision"
  cmake -S "$SRC" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" -DCUDAToolkit_ROOT="$CUDA_ROOT" \
    -DCMAKE_CUDA_ARCHITECTURES=87 -DSTRATA_GGML_DIR="$LLAMA_SRC" \
    -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_HIP=OFF -DSTRATA_ENABLE_SYCL=OFF \
    -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PREFILL_MMQ=OFF -DSTRATA_MMQ_KQUANTS=ON \
    -DSTRATA_BUILD_TESTS=ON 2>&1 | tee "$STAGE/build-logs/$arm-configure.txt"
  cmake --build "$BUILD" --target strata-q36 strata_kq -j4 2>&1 | tee "$STAGE/build-logs/$arm-build.txt"
done
# Reuse the already validated encoder binary if known; otherwise same pin/build:
cmake -S tools/vision -B "$TEST/build-vision-encoder" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR="$LLAMA_SRC" -DSTRATA_PORTABLE=OFF \
  -DSTRATA_VISION_CUDA=ON -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES=87 2>&1 | tee "$STAGE/build-logs/encoder-configure.txt"
cmake --build "$TEST/build-vision-encoder" --target strata-vision -j4 2>&1 | tee "$STAGE/build-logs/encoder-build.txt"
```

### Future explicitly approved stop/reap/restore window ONLY

No worker controls services. An operator must pause clients, identify actual
manager scope for BOTH Strata/llama, record prior states, verify KillMode=control-group
and preapprove privileges. Do not guess scope or use mixed managers without a
separately reviewed procedure. These commands are a handoff, NOT executed here.

1. Snapshot Strata/llama unit states and verify current Strata health at the existing
   `http://100.100.10.3:11436/health` (`service=strata,status=ok,loaded=true`).
2. Install EXIT/INT/TERM cleanup BEFORE stopping. Stop Strata and initially active
   llama via their actual user/system manager; verify both model/encoder processes
   are gone. No broad pkill, unit/config edit, deployment or OWUI change.
3. Run the worker in a transient cgroup; one encoder + one LLM is INTENTIONAL,
   never two LLM arms together. The worker refuses surviving models, owns/reaps
   only its children and kills both groups on error. Preserve all failure logs.
4. On exit, stop/reap the ENTIRE test cgroup including encoder before restoring
   Strata and llama's exact prior active/inactive state. Failed process/cgroup
   inspection/reaping blocks restart to prevent overlap; operator recovery required.
   The bootstrap saves its actual cgroup-v2 path; cleanup requires it gone or
   cgroup.events populated=0, not just the parent process having exited.
   Do not interrupt restoration with a second INT/TERM. SIGKILL/power loss requires
   manual recovery. Recheck loaded Strata health before/after llama restoration,
   inspect both states before resuming clients. No model overlap with production.

The following COMPLETE shell block is a future handoff only. Review it first;
run in a dedicated shell with the build variables above, ONLY after operator
approval. Confirm unit names and SCOPE; this assumes both in the SAME manager.
Do not source it into a production shell or edit units to make it fit.

```bash
set -euo pipefail
: "${VISION_STOP_APPROVED:?operator must explicitly set to 1 after review}"
test "$VISION_STOP_APPROVED" = 1
SCOPE=system # confirmed actual scope: system OR user, not a guess
SERVICE=strata.service
LLAMA=llama-server.service
HEALTH=http://100.100.10.3:11436/health # confirmed existing endpoint
case "$SCOPE" in
 system) CTL=(sudo -n systemctl); RUN=(sudo -n systemd-run --uid="$(id -u)" --gid="$(id -g)") ;;
 user) CTL=(systemctl --user); RUN=(systemd-run --user) ;;
 *) exit 1 ;;
esac
for tool in pgrep curl; do command -v "$tool" >/dev/null; done
for name in strata strata-q36 strata-vision llama-server llama-cli llama-bench; do
 rc=0; pgrep -x "$name" >/dev/null || rc=$?; ((rc<=1)) || exit 1
done
for binary in "$TEST/build-feature-vision/strata-q36" "$TEST/build-baseline-vision/strata-q36"; do
 test -x "$binary"; grep -a -q -- '--kv-vision-test' "$binary"
done
test -x "$TEST/build-vision-encoder/bin/strata-vision"; test -f "$MMPROJ"
test "$("${CTL[@]}" is-active "$SERVICE")" = active
test "$("${CTL[@]}" show "$SERVICE" -p KillMode --value)" = control-group
llama_load=$("${CTL[@]}" show "$LLAMA" -p LoadState --value)
llama_state=absent
if [ "$llama_load" = loaded ]; then
 llama_state=$("${CTL[@]}" show "$LLAMA" -p ActiveState --value)
 case "$llama_state" in active|inactive|failed) ;; *) exit 1 ;; esac
elif [ "$llama_load" != not-found ]; then exit 1; fi
if [ "$llama_state" = active ]; then
 test "$("${CTL[@]}" show "$LLAMA" -p KillMode --value)" = control-group
elif pgrep -x llama-server >/dev/null; then echo 'Unmanaged llama; abort before stop'; exit 1; fi
health() {
 local attempt
 for attempt in $(seq 1 90); do
  if curl --fail --silent --show-error --max-time 5 "$HEALTH" > "$STAGE/health-$$.json" &&
     "$PYTHON" -c 'import json,sys;d=json.load(open(sys.argv[1]));sys.exit(not(d.get("service")=="strata" and d.get("status")=="ok" and d.get("loaded") is True))' "$STAGE/health-$$.json"; then return 0; fi
  sleep 2
 done
 return 1
}
health
test -f /sys/fs/cgroup/cgroup.controllers # reviewed cgroup-v2 host only
UNIT="strata-vision-review-$$"
CGRECORD="$STAGE/test-cgroup-$$.txt"; export CGRECORD
launched=0
RESULTS="$STAGE/results-$(date -u +%Y%m%dT%H%M%SZ)-$$"
printf 'strata=active\nllama=%s\n' "$llama_state" > "$STAGE/prior-state-$$.txt"
engines_alive() {
 local name code
 for name in strata strata-q36 strata-vision llama-server llama-cli llama-bench; do
  code=0; pgrep -x "$name" >/dev/null || code=$?
  [ "$code" = 1 ] || return 0 # inspection errors also block unsafe restart
 done
 return 1
}
restore() {
 local state rc=0
 state=$("${CTL[@]}" show "$UNIT.service" -p LoadState --value 2>/dev/null || true)
 if [ -z "$state" ]; then echo 'Cannot inspect cgroup; manual recovery required'; return 1; fi
 if [ "$state" != not-found ]; then "${CTL[@]}" stop "$UNIT.service" || return 1; fi
 if [ "$launched" = 1 ]; then
  "$PYTHON" - "$CGRECORD" "$UNIT.service" <<'PY' || return 1
from pathlib import Path
import sys
lines=Path(sys.argv[1]).read_text().splitlines()
assert len(lines)==1 and lines[0].startswith('0::/')
name=lines[0][3:]
assert name.endswith('/'+sys.argv[2]) and '..' not in Path(name).parts
root=Path('/sys/fs/cgroup'+name)
if root.exists():
    events=dict(line.split() for line in (root/'cgroup.events').read_text().splitlines())
    assert events['populated']=='0', 'test descendants still alive'
PY
 fi
 "${CTL[@]}" stop "$SERVICE" || return 1
 if [ "$llama_state" = active ]; then "${CTL[@]}" stop "$LLAMA" || return 1; fi
 if engines_alive; then echo 'Test/model survives; DO NOT restart production yet'; return 1; fi
 "${CTL[@]}" start "$SERVICE" || rc=1
 health || rc=1
 if [ "$llama_state" = active ]; then
  "${CTL[@]}" start "$LLAMA" || rc=1
  test "$("${CTL[@]}" is-active "$LLAMA")" = active || rc=1
 elif "${CTL[@]}" is-active --quiet "$LLAMA"; then rc=1; fi
 health || rc=1
 return "$rc"
}
trap 'rc=$?; trap - EXIT; trap "" INT TERM; restore || rc=1; exit "$rc"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"${CTL[@]}" stop "$SERVICE"
if [ "$llama_state" = active ]; then "${CTL[@]}" stop "$LLAMA"; fi
if engines_alive; then echo 'Production not reaped'; exit 1; fi
launched=1
"${RUN[@]}" --unit="$UNIT" --wait --pipe --collect --property=KillMode=control-group \
 --working-directory="$TEST" "$PYTHON" -c \
 'import os,sys;from pathlib import Path;Path(sys.argv[1]).write_text(Path("/proc/self/cgroup").read_text());os.execvp(sys.argv[2],sys.argv[2:])' \
 "$CGRECORD" /usr/bin/env STRATA_VISION_REVIEW_WINDOW=1 \
 "$PYTHON" tools/q36_vision_preallocation_gate.py \
 --engine "$TEST/build-feature-vision/strata-q36" \
 --baseline "$TEST/build-baseline-vision/strata-q36" \
 --encoder "$TEST/build-vision-encoder/bin/strata-vision" \
 --model "$MODEL" --mmproj "$MMPROJ" --out "$RESULTS"
# Normal shell exit invokes reaping/restoration too; preserve nonzero test status.
exit
```

Required numerical marker `Q36_VISION_PREALLOCATION_ORIN PASS` means only this
fixture/config's exact native checks passed. Keep each .exit, native/encoder log,
raw dump, actual template/IDs/grid/model fingerprints and KVVISION/KVBANK counters.
`semantic_image_review=PENDING_MANUAL` is not a semantic pass. Inspect red/blue
responses; record what was not tested. Existing separate prefix/sampled/native
allocator gates, HTTP/image-buffer/server cancellation behavior, broader shapes,
encoder elastic peaks, long context, other backends and SASS/default-vs-base
validation remain additional pending work. Never lower a tolerance to pass.
