# Qwen3.6 multimodal checkpoint handoff / fresh coding container

## Delivery status — origin publication and coding-server validation VERIFIED

Verified on 2026-10-28 in the actual checkout `/workspace` (container mount;
resolve paths with `git rev-parse --show-toplevel`, not an older absolute path):

- Origin: `https://github.com/moontato/Strata-Orin.git`.
- Branch: `feature/q36-image-checkpoints`; source status was clean before this
  documentation review.
- Local HEAD AND live `git ls-remote --heads origin
  refs/heads/feature/q36-image-checkpoints`:
  **72457256fa0367660a8e0849bb19e234c6fc1711**.
- The operator confirms the coding server fetched this branch from origin and ran
  `python tools/q36_cpu_checks.py --negative-controls --sanitizers` successfully.
  The available `logs/q36-dev-cpu/20261028T051337Z-15576/result.json` independently
  records that HEAD, clean source, `PASS_CPU_ONLY`, negative controls and sanitizers.
  Raw logs record 17/17 Release and sanitizer CTests, 86 Python tests plus 22
  subtests, 22 mutation controls and source-preservation checks. The recorded host
  used Python 3.13.5/GCC 14.2.0; its checkout path
  `/mnt/external/Documents/GitHub/Strata-Orin` is historical, not this mount path.

**Origin-only migration is complete.** The earlier authentication failure below
is historical, not a current publication blocker. No new push or CPU-suite rerun
was needed for this verification. The transport archive is now optional historical
insurance; no original archive/worktree/artifact was removed. These CPU results
are NOT CUDA/Orin correctness, positive GPU rewrite reuse or performance evidence.
Image checkpoints remain disabled by default and not Orin-qualified.

### Review blockers RESOLVED locally (CPU evidence); Orin qualification still pending

The 2026-10-28 review found two concrete defects: the complete-state donor hook could
observe another current slot's device rotary delta, and the gate's advertised absolute
worker budget was not enforced outside native-child polling. Both are now fixed and
regression-tested on this host; see
[Q36_IMAGE_CHECKPOINT_REVIEW.md](Q36_IMAGE_CHECKPOINT_REVIEW.md) for the harness,
scenario and mutation output.

- New `src/qwen35/q36_restore_observation.inc` holds the observation boundary
  contract; `Engine::copy_session` only requests an observation and the request path
  fires it after `E.select(s)` and before the incoming upload/suffix.
  `checkpoint_restore` observes in place. No production selection, caching, kernel,
  schedule or sampling behaviour changed; with no developer hook nothing happens.
- The image oracle now checks the observation boundary and producer/consumer counters
  at every observation and verifies every observation against the independent replay.
- `tools/q36_image_checkpoint_gate.py` became a supervisor+worker pair: one execution
  budget (default300s, max420s) covering ALL phases enforced by a process-group kill,
  plus a separate cleanup allowance (`--cleanup-seconds`, default20s, max60s) with an
  owned-group-emptiness record. Only the supervisor prints a verdict, and it refuses a
  killed or late worker. Never launch the worker without the supervisor.
- New host controls: `tools/q36_restore_observation_harness.py`,
  `tools/check_q36_restore_observation_negative.py` and
  `tools/check_q36_gate_watchdog_negative.py` (now five teardown mutations plus a
  measured one-allowance baseline); the CPU runner now executes four negative-control
  scripts. Deterministic short-budget supervisor scenarios, per-signal teardown cases,
  verdict/slow-log-read/unowned-process checks live in
  `tools/test_q36_image_checkpoints.py` (20 tests).

This is host evidence only. Nothing was built or run on Orin, no model/service was
started, and image checkpoints remain off by default and NOT Orin-qualified. The Orin
recipe still requires a separately approved operator isolation window.

Validation actually run after the corrections, from a clean tree at
f19e22941276cdc7d63f0bd84839626c22f2c7df:

    python3 tools/q36_cpu_checks.py --build-dir build-q36-dev-cpu-final \
      --negative-controls --sanitizers --jobs 3
    Q36_CPU_CHECKS PASS_CPU_ONLY logs=logs/q36-dev-cpu/20261028T064821Z-2583

14 Release CPU binaries / 17 ctest cases pass in BOTH the release and the
ASan/UBSan build (3 expected-failure text schedule-oracle controls included), 94
Python tests plus 22 subtests pass, and four negative-control scripts run: 8 image,
6 vision and 8 bank metadata mutations, 3 restored-observation mutations and 3
gate-watchdog mutations, every mutant failing as required. Raw outputs, the five
bounded-supervisor scenarios and the restored-state harness ladder are copied under
`docs/validation/q36-image-checkpoints/` with the `-correction` suffix
(see its `VALIDATION.md` addendum). The 5-patch series replays onto 765068db with an
identical tree `d67a44ef0749000155b1d4779515fe058260faab`, and the incremental
bundle (prerequisite 4130bdfd only) imports the tip with the same tree and order.

The two correction commits (3958d620, f19e2294) are LOCAL on
`feature/q36-image-checkpoints`. No push was performed in this session, so origin
publication for them is still pending; the earlier verified origin tip is 72457256.

### Historical authentication failure / recovery instructions (superseded)

The normal push of portable-support commit **ad0988b66fea6cd62996c6bee42a179c37dd68b3**
failed with `fatal: could not read Username for 'https://github.com': terminal
prompts disabled`. The origin feature ref was absent when checked afterward.
At that historical point, origin-only migration was not complete and no successful
push/origin-branch clone was claimed. Source, instructions and evidence were
committed locally and preserved in the standalone transport bundle. Archive
transfer was required until push succeeded; that condition is now resolved.

An independent full-bundle clone ofad0988b6 passed the documented fresh-venv CPU,
Python,22 mutation and sanitizer checks; it has its OWN Git objects/refs, no
alternates/external commondir/worktree registry and `git fsck --full` passed.
This is explicitly NOT the requested pushed-origin clone. Details/raw logs:
`docs/validation/q36-migration/INDEPENDENT_CLONE.json` and independent-clone-*.

Historical recovery instructions (already completed; do NOT repeat this push
for the current task): after configuring normal GitHub authentication OUTSIDE
source/Git (never copy credentials/tokens here), the intended non-force refspec was:

```bash
git -C /workspace/tmp/image-checkpoints push origin HEAD:refs/heads/feature/q36-image-checkpoints
git -C /workspace/tmp/image-checkpoints ls-remote --heads origin refs/heads/feature/q36-image-checkpoints
# Require the returned hash to equal local git rev-parse HEAD. Then perform a NEW
# network clone and the setup/checks below; bundle-clone proof is not a substitute.
```

If the old container is gone, first clone `feature-q36-image-checkpoints-full.bundle`
from the transferred archive into `/workspace/Strata-Orin`, set origin to the URL
below, then run the same explicit push refspec from that clone. The full bundle
has NO prerequisite repository/old object-store requirement. Originals were retained.

```bash
# Transfer the archive AND .sha256 into /workspace/tmp on the new container first.
cd /workspace/tmp
sha256sum -c q36-coding-migration-supplement.tar.gz.sha256
tar -xzf q36-coding-migration-supplement.tar.gz
(cd q36-migration-preservation && sha256sum -c SHA256SUMS)
git clone --branch feature/q36-image-checkpoints \
 /workspace/tmp/q36-migration-preservation/feature-q36-image-checkpoints-full.bundle \
 /workspace/Strata-Orin
git -C /workspace/Strata-Orin remote set-url origin https://github.com/moontato/Strata-Orin.git
# Then follow the venv/build/test steps below. Configure authentication and push
# only this feature branch before retiring the last copy of the transport archive.
```

## Start here: repository and exact implementation

- Project/origin: **https://github.com/moontato/Strata-Orin.git**.
- Work branch to clone/push: **feature/q36-image-checkpoints**. No default-branch
  change, merge, force-push, deployment or service operation is authorized.
- Final FUNCTIONAL implementation: **37911119e64aa97b837b2f6a920dbdc8cc3930bd**.
- Direct parent/diagnostics: **765068db267f8cd8f846f8ef45c9cc94f90075d5**.
- Original diagnostic base: **4130bdfdb43bdb273412cc3530660bd19ec14602**.
- Ancestors:4130bdfd ->30340e22 independent text schedule oracle ->3f3bc4e8
  vision preallocation ->a9837161 physical-bank base. The image implementation is
  a direct child of765068db, not the separate4815a036 projection/adaptive fork.
- Subsequent commits on this branch contain portable coding setup, this handoff,
  evidence and artifact generation, not a new numerical implementation. Determine
  the exact delivered support tip with `git rev-parse HEAD` and compare it to
  `git ls-remote origin refs/heads/feature/q36-image-checkpoints`; the delivery
  report supplies the LOCAL tip and push status/hash (now verified above). A document
  cannot embed its own containing Git hash. Initial portable-support commit is
  ad0988b66fea6cd62996c6bee42a179c37dd68b3; later delivery-status changes do not alter
  the functional implementation.

The old session's shell CWD `/workspace` was **feature/unified-kv** at
`dd543a34ae73040d1ddb74410dd177db85383884`, NOT the active task tree. The task
checkout was `/workspace/tmp/image-checkpoints`. Its Git common directory was the
real `/workspace/.git`, not a home directory, symlink, alternate object store or
external commondir. No replacement of that original repository was necessary.
Three older worktrees were in `/home/pi/Documents/GitHub`; their source was clean,
verified against workspace Git blobs and reachable from the feature ancestry.
Those original directories were NOT deleted. Their absolute worktree-registry
backlinks are historical, NOT instructions to recreate them. After successful
publication, a normal origin clone below is independent and needs none of those paths.

See `docs/validation/q36-migration/INVENTORY.json` for exact nine-worktree paths,
branches, HEADs, status, common-dir checks and outside-source verification counts.

## Current goal, actual behavior and next step

Implement image-aware checkpoint fallback and qualify it on the isolated Orin.
Implementation exists; **`--kv-image-checkpoints` defaults OFF** pending that
qualification. Exact-prefix complete-state reuse stays first. Only after no exact
match can serve the request does the opt-in image checkpoint search run.

Actual implementation:
- `include/strata/core/q36_image_checkpoint.hpp`: bounded immutable shared HOST
  copies of actual uploaded FP32 image bytes and grid/placement/rotary semantics;
  prefix authorization, legal boundary planning and readiness metadata. Python
  keys/hashes alone never grant reuse. Changed images before a boundary reject it;
  later changes cannot extend it. Token LCP is always required.
- `src/qwen35/q36_image_checkpoint.inc` and Engine integration: explicit contiguous
  main/MTP prefill readiness; legal checkpoints before/after COMPLETE images;
  generation/epoch/backing/owner/pin/destination leases; packed recurrence/conv/
  hidden seed and all main/MTP KV codes/scales restored; complete-prefix image
  metadata plus CPU/device rotary delta restored instead of clearing image state.
  Shared prefill views remain transient. Reset/eviction/partial-image cancellation
  invalidate proofs. Full-state donors stay pinned across admission/copy.
- Targets inside pictures floor to an earlier legal existing callback, or defer
  to the first complete legal callback if no earlier one exists. An early safety
  checkpoint may be below interval. No numerical chunk/call splitting. Four GPU
  payload ceiling, existing bank calculation, fixed combined1024-row image arena
  and sampled-image MTP exclusion remain. Host proof storage has an explicit bound.
- `src/qwen35/q36_image_checkpoint_regression.inc`: independent IMAGE schedule
  oracle from EMPTY in an existing reference slot, original declared token calls
  and original SVE inputs. Rebuilds positions through the separate legacy image
  builder; never uploads candidate snapshots or borrows restored checkpoint state.
  Exact restored/final state, full logits, token/image/position/rotary metadata and
  main/MTP codes/scales compared. Original mid-call declared length/head/MTP
  semantics retained. Caller-owned pins protect other slots. No fifth GPU payload.
- `tools/q36_image_checkpoint_fixture.py`: real GGUF/template/Service.prepare public
  earlier-user rewrite history plus labelled native structural/identity controls.
  Prepared from saved actual1024-row red/blue outputs, no encoder launch. Positive
  steady after-image rewrite reuse REQUIRED; genuine zero cases stay zero.
- `tools/q36_image_checkpoint_gate.py`: supervisor+worker; preflight, corruption/off/on
  sequential children, one execution budget (default300s, max420s) enforced over every
  phase by a process-group kill, and ONE absolute cleanup allowance (default20s, max60s)
  covering waiting, reaping and the owned-group emptiness proof. SIGINT/SIGTERM/SIGHUP
  to the supervisor are recorded and routed through that same teardown, which always
  writes the failure record and exits nonzero; the pure verdict also refuses an
  interrupted supervisor and an unreaped worker. Only groups this run created are ever
  signalled, so the operator-owned enclosing cgroup is still required. Each arm passes its OWN independent matching-schedule
  oracle; OFF/ON neutrality is not the correctness reference. Timings distinguish
  replay/reuse, prefill/request, synchronized copy cost and selected CPU metadata
  work. These are synthetic public-fixture measurements, not actual OWUI savings.

**Current next step:** review actual CUDA compile/readiness/oracle paths and
prepare isolated Orin build/CPU fixture preflight after resolving concrete review
blockers. The coding-server CPU suite above is complete; do not repeat it without
source changes or a specific concern. The setup/checks below remain reproduction
instructions for a new environment, not an outstanding migration requirement.
Read `docs/Q36_IMAGE_CHECKPOINTS.md` and inspect Git status/diff first. Do NOT run any model/service window without new operator
approval. No Orin/GGUF image-checkpoint qualification was executed in this container.

## Verified context: do not repeat or misinterpret

Production is Qwen3.6-35B-A3B on AGX Orin, CUDA13.2 / SM87:
262144 TOTAL tokens per slot, four slots sharing524288; INT8 KV physical
preallocation, adaptive TEXT checkpoints, MTP2, batch-MTP, sampled TEXT MTP,
vision with combined1024 image-row request cap, **fit_max_tokens enabled**.
Do not change these production settings merely to pass a test.

Operator-verified diagnostic evidence, NOT new image-checkpoint qualification:
- Diagnostic CUDA build and15 CPU CTests passed.
- Real CPU-encoded red/blue inputs and actual GGUF/template preflight passed.
- Diagnostics OFF/ON state/logits/output/schedules were identical.
- Constructed prefixes reused1060 and1072 tokens, including cross-slot restore.
- Rendered follow-ups rewrote their prior token prefixes and reused zero.
- Actual OWUI follow-ups changed message0 content. Precise real-request token
  divergence and potential savings are **UNMEASURED**.
- Current production image path bypasses checkpoint fallback. No exact-prefix
  routing defect is established; do NOT specifically blame fetch_url.
- Diagnostic neutrality is NOT an independent image correctness oracle.
- The completed projection investigation must NOT be reopened or used to relax
  tolerances. Equal argmax, summary metrics, or a fresh run with different chunk/
  head/MTP history do not replace exact matching-schedule state/logit comparison.

Rejected approaches: just remove `!geni`; reuse text restore that clears image
state; trust stable Python keys; checkpoint inside images; silently discard all
useful boundaries; frontend/OWUI normalization; change kernels, sampling settings,
GEMM selection or tolerances to pass; allocate a fifth GPU checkpoint; infer an
OWUI speedup from synthetic timing or token counts alone.

## Portable development environment (CPU only)

Existing `setup.sh` invokes `setup.py` and may install/download/start a model.
Existing root `Dockerfile` is a runtime CUDA13.0 GPU/server image, not this SM87
coding setup. **Do not use either for these coding checks.** They are not replaced.

Tested manual environment: Debian12 x86_64, GCC/G++12.2.0, CPython3.11.2,
CMake4.4.4, Ninja1.13.2. (This container's `.venv-q36-dev` had to be recreated with
`python3 -m venv --clear` before `pip install -r requirements-q36-dev.txt`; put the
venv `bin` on PATH so the runner finds its cmake/ninja. A build directory from another
mount is rejected as a foreign cache: choose a NEW `--build-dir`.) Project minimums: CMake3.24, C++20, runtime Python3.10+;
this focused dev lock is tested on Python3.11 (numpy2.4.6). Exact packages including
pytest9.1.1, Jinja2/MarkupSafe, regex, numpy and Pillow and their test dependencies
are in **requirements-q36-dev.txt**. Do not substitute an old container's venv or
`/workspace/tmp/adaptive-integration-deps`/home PYTHONPATH. No private packages.

```bash
# Reproduction ONLY, in a fresh Debian/Ubuntu coding container, NOT production Orin.
# Publication and the coding-server CPU suite are already verified above:
apt-get update
apt-get install -y --no-install-recommends git ca-certificates build-essential python3 python3-venv python3-pip
git clone --branch feature/q36-image-checkpoints https://github.com/moontato/Strata-Orin.git /workspace/Strata-Orin
cd /workspace/Strata-Orin
python3 -m venv .venv-q36-dev
.venv-q36-dev/bin/python -m pip install -r requirements-q36-dev.txt
.venv-q36-dev/bin/python -m pip check
export PATH="$PWD/.venv-q36-dev/bin:$PATH"
python tools/q36_cpu_checks.py --negative-controls --sanitizers
```

This builds ONLY14 focused binaries /17 CTest cases, runs the focused Python suite,
22 mutation controls, source invariants and a separate sanitizer build. It uses
GCC, Ninja and fresh CMake dirs, explicitly disables CUDA/HIP/SYCL/native experts,
sets TMPDIR to a per-run scratch directory, isolates config tests, drops external
PYTHONPATH and disables user-site packages. No GPU/model download/server/service.
Logs are under `logs/q36-dev-cpu/<UTC>-<PID>/`; all failures stay nonzero. Never
reuse a CMake cache from another clone/path. All tests requiring real GPU/GGUF
remain pending, regardless of stand-in syntax/fake IO tests passing.

Exact expanded commands are emitted to those logs; their target/case lists are
also in `tools/q36_cpu_checks.py`. Equivalent focused Release commands:

```bash
cmake -S . -B build-q36-dev-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release \
 -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF -DSTRATA_ENABLE_SYCL=OFF \
 -DSTRATA_NATIVE_EXPERTS=OFF -DSTRATA_BUILD_TESTS=ON -DSTRATA_BUILD_CONVERSATION_TESTS=OFF
cmake --build build-q36-dev-cpu --target unified_kv_test kv_preallocation_test \
 unified_kv_host_capability_test q36_cache_diagnostic_test q36_checkpoint_schedule_native_test \
 q36_checkpoint_schedule_test q36_adaptive_checkpoint_test q36_prefix_checkpoint_test \
 q36_sampled_mtp_test q36_sampled_request_test q36_vision_preallocation_test \
 q36_vision_staging_test q36_image_checkpoint_test q36_image_checkpoint_helper_test -j2
ctest --test-dir build-q36-dev-cpu --output-on-failure \
 -R '^(unified_kv_test|kv_preallocation_test|unified_kv_host_capability_test|q36_.*)$'
python -m pytest -q tools/test_q36_*.py serve/test_q36_config.py serve/test_unified_kv.py tools/test_unified_kv_validate.py
python tools/check_q36_image_checkpoint_negative.py
python tools/check_q36_vision_negative.py
python tools/check_kv_preallocation_negative.py
python tools/check_q36_image_preservation.py
```

Optional coding image, separate from the existing runtime Dockerfile:
`docker build -f docker/Dockerfile.q36-dev -t strata-q36-dev .`, then
`docker run --rm -it -v "$PWD:/workspace" strata-q36-dev` and run the CPU checker.
It has no service entrypoint, exposed ports, GPU/model or credentials. Docker is
unavailable in the migration container; the equivalent fresh-venv setup was run,
not an actual Docker image build. Debian tag/apt security patches are not a
bit-reproducible OS lock; tested compiler versions and exact Python lock are given.

### External sources and GPU limitations

CPU focus needs no external C++ source checkout/network after package install.
For CUDA/native expert/vision work: **https://github.com/ggml-org/llama.cpp.git**
EXACT commit **3cf03257f219afbe7334045ff7c6a06ac68c627d**. Existing setup.py,
third_party/ggml/VERSION.txt and CMake FetchContent agree. Verify `git rev-parse
HEAD` and use `-DSTRATA_GGML_DIR=...`; vision uses `-DLLAMA_DIR=...`. On a fresh
non-Orin coding host an explicit source dependency can be placed under
`/workspace/deps/llama.cpp`; this is NOT permission for new Orin top-level dirs.
Do not copy third-party build trees. CMake/installer may fetch this pin when
native experts are ON; the focused CPU setup turns them OFF to avoid that fetch.

No nvcc/hipcc/icpx, CUDA driver/device, Orin mount, real model or raw Orin tensor
artifacts exist locally. Actual CUDA13.2 SM87 compile/link, image oracle, positive
steady rewrite reuse, counters/timings and other touched backends are PENDING.

Known baseline limitation: unrelated unrestricted CPU/native-experts-OFF targets
hit missing AVX1 Q2 symbols (`q2_0_gguf_rows_multi_avx1`, `_legacy`,
`act_quant_q8_1_avx1`) on the older a9837161 baseline too. Historical raw linker log:
`docs/validation/q36-migration/historical-baseline-q2-linker.txt`. Not fixed here,
not newly attributed to image checkpoints. A past unrestricted CTest invocation
also found unbuilt unrelated targets; that was NOT a pass. Build/select the17
focused cases, not every registered target. Never call the partial build 'all
repository tests passed'.

## Evidence actually run and where it lives

`docs/validation/q36-image-checkpoints/` preserves the implementation's small raw
logs and VALIDATION.md in Git (now published on origin):17 selected CPU CTests;14 strict ASan/UBSan/leak
binaries +3 expected-failure text-oracle corruption cases;86 Python tests plus22
subtests;8 new image +6 vision +8 bank mutations. Source invariants cover36 device/
global definitions and unchanged numerical/transfer/bank/capacity/sampler/staging/
server/encoder code. Default selectors were compared over1000 CPU stand-in cases.

The NEW image native oracle/Engine methods passed declaration-only host C++17
syntax, NOT actual Engine/Serve/CUDA TU compilation. Production image control
methods ran with a CPU pool/fake payload writer. Fixture plumbing used fake vocab/
encoder IO; tiny fake Python children tested worker cleanup. These are NOT real
image inference, GPU eviction pressure, sampled-image qualification or OWUI tests.
Actual image-checkpoint CUDA/GGUF/model/cancellation/pin/reset/state qualification
is still pending. Encoder co-residency/workspace, elastic shapes, HTTP/OWUI,
long/full context and performance distributions also remain separate work.

`docs/validation/q36-migration/` records inventory, pinned fresh-venv reproduction
and migration verification. Raw per-run local logs/builds/venvs stay ignored and
are regenerable; do not import them as source or configuration. No preexisting
test failure was turned into a tolerance pass.

## Orin continuation: ONLY the isolated worktree

Live native engine (never rebuild/switch/reset/remove its worktree):
`/mnt/ssd/Strata-Orin/worktrees/checkpoint-policy/build-adaptive-integration-v1-feature/strata-q36`.
Python server and vision encoder still depend on the **preallocation worktree**;
do not clean/remove those dependencies casually.

Use ONLY `/mnt/ssd/Strata-Orin/worktrees/multimodal-cache` for feature source.
ALL Orin staging, scripts, transfers, builds and results:
`/mnt/ssd/tmp/strata-checkpoints`. No home scripts or new top-level Strata dirs.
Existing verified diagnostic results:
`/mnt/ssd/tmp/strata-checkpoints/mm-results-20261028T030530Z-2549313`.
Those actual Orin artifacts are not mounted here; preserve/read them on Orin.

The actual diagnostic checkout includes **f934b85b42e79cc2a7c4ec0ca9da7685674e1ebd**
plus a cherry-picked diagnostic, so HEAD need not equal765068db. Preserve actual
HEAD/branch/artifacts, require clean status and hash these BEFORE and AFTER:

- tools/q36_adaptive_integration_fixture.py
- tools/test_q36_integration_ordinary.py
- docs/Q36_ADAPTIVE_ORIN_VALIDATION.json

The last two are absent locally and were NOT invented. The first remains unchanged.
Apply only the reviewed post765068db range (37911119 plus portable-support commits)
onto the actual Orin diagnostic branch, never replay original diagnostics or reset
it to the Pi tip. See **docs/Q36_IMAGE_CHECKPOINTS.md** for exact source-preserving
Git/CUDA13.2 SM87 build/saved-real-input preflight commands. New builds/scripts
are staged under strata-checkpoints; CMake provenance must name only the isolated
multimodal-cache source. Keep actual source/protected hashes with results.

This migration grants NO service/model window. A future separately approved
window must record both services' previous active/inactive states, stop/reap
production and the entire test cgroup, then restore BOTH prior states. Reap/prove
the whole cgroup gone or cgroup.events populated=0 before restarting production;
failed inspection blocks unsafe overlap. **Do not use
/proc/PID/task/PID/children: unavailable on this Orin.** Use manager-owned cgroup
proof and owned process groups. Preserve bounded duration/progress, even on error.
After restoration verify existing `http://100.100.10.3:11436/health` reports
`service=strata,status=ok,loaded=true` AND attest the restored Qwen3.6 vision
configuration/encoder capability from the actual health schema/startup evidence;
loaded=true alone does not prove vision. No new binding, endpoint or config change.
SIGKILL/power loss requires operator recovery; never declare restoration from an
unchecked command exit. No OWUI-specific prompt normalization or service edits.

## Migration inventory disposition / artifacts from origin alone

At task start every linked worktree was source-clean, no nonignored untracked
source. The only external project trees were the three clean home worktrees:
9597a51a(prefix),a9837161(preallocation),f121cd08(sampled MTP), all ancestors of37911119.
2996/3019/3003 source/text files respectively were verified against Git blobs
already under `/workspace/.git`. No essential unique home script/fixture/note was
found in the targeted project/notes locations. No credentials, SSH keys, tokens,
Git config/credential stores, unrelated home files or entire caches were copied.

Essential ignored implementation logs were allowlist-copied into the committed
validation dirs. Existing tracked scripts/fixtures/setup docs were already in the
implementation ancestry. Generated scratch syntax includes, toy fixtures, CMake
builds, sanitized binaries, mock repos and Python packages are reproducible from
tracked code and were NOT copied as setup. Original artifacts/worktrees remain.
New uncommitted migration items are explicitly the dev lock, CPU runner, separate
dev Dockerfile, exporter, AGENTS link, this handoff, updated range-safe Orin handoff
and bounded evidence/inventory. They are reviewed/committed on the feature branch;
none is left as an undisclosed required untracked file.

Supplemental `/workspace/tmp/q36-coding-migration-supplement.tar.gz` (+.sha256)
preserves all pre-migration refs including the separate4815a036 historical fork,
old37911119 review archive and inventory/proof. It contains NO venv/build/home
cache or credentials. It also contains a CURRENT standalone full feature bundle.
**The former required-transfer condition is resolved:** origin publication and
coding-server validation are verified above. This archive is optional historical
insurance: all essential source/instructions/evidence are already committed for
that push. Keep existing copies; this is not an instruction to delete artifacts. Actual Orin GGUF/SVE/private traffic is external
qualification input, not secretly in that archive. No originals were deleted.

Regenerate a CURRENT review, including commits/patch/incremental bundle/source/
committed evidence/checksums, entirely from a full clean clone (origin after
publication, or the standalone bundle while authentication remains blocked):

```bash
python tools/q36_export_review.py --out /workspace/tmp/q36-export-$(date -u +%Y%m%dT%H%M%SZ)
```

This archives only tracked Git blobs, never walks homes/builds/credentials, and
verifies stored tar member hashes. Bundle requires4130bdfd and includes original
765068db objects plus all new feature/support commits. On Orin apply only the
post765068db range with the protected-file checks. Keep large generated artifacts
out of Git. To repeat old37911119-only evidence, use a separate detached checkout
of that exact implementation, not a reset of the live or isolated Orin worktree.
