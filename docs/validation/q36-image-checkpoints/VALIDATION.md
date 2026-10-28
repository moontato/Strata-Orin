# Validation: image-aware checkpoints, HOST ONLY

Parent765068db267f8cd8f846f8ef45c9cc94f90075d5 on4130bdfd.
No actual encoder/model/CUDA/service operations, pushes/deployments/production
configuration/OWUI edits. Other existing worktrees unchanged and clean.

Executed on this host:
- 17 selected Release CPU CTests (14 binaries including3 expected-failure text
  schedule-oracle controls). Assertions enabled. ctest-selected.txt,
  cmake-configure.txt, cmake-build.txt.
- All14 corresponding binaries newly compiled with C++20, assertions, strict
  warnings, ASan/UBSan/leak settings and executed; same3 text-oracle corrupted
  payload controls required restored-state failure. sanitizers-all.txt.
- 86 Python tests plus22 subtests: actual new Engine-method/native-image-oracle
  bodies compiled through DECLARATION-ONLY C++17 stand-ins; real production image
  control methods execute against a CPU pool/fake payload writer; routing default
  compared against pre-feature selectors over1000 cases. Python worker is exercised
  with tiny FAKE Python children (success, expected/incorrect failure, timeout and
  owned group cleanup), NOT Strata/services. python-all.txt.
- 8 new image metadata/readiness negative mutations,6 vision and8 bank controls
  all failed as required. image-negative.txt, vision-negative.txt, bank-negative.txt.
  Temporary overlays only. No production header/source mutated by controls.
- Bash handoff syntax and Python syntax; syntax.txt. NO handoff command executed.
-36 CUDA global/device definitions plus original numerical/process/forward/MTP/
  row/transfer/preallocation/staging methods and core bank/capacity/sampler/server/
  encoder files source-identical. source-and-scope.txt. SOURCE evidence only,
  not SASS or default CUDA-output parity.
- Protected local integration fixture byte-identical. Two Orin f934-only files
  absent from local base remain absent. They were NOT synthesized/overwritten.
  Actual Orin source/protected hashes must be checked during safe cherry-pick.
- Existing other worktree HEAD/cleanliness proof: worktree-scope.txt.
- GitHub open issues and PRs searched:0 returned; open-issues.txt.

Test limitations:
- The new CPU fixture test uses actual Service.prepare/control flow but a FAKE
  vocabulary/template and stored FAKE embeddings. This validates plumbing, not
  actual GGUF rendering, image semantics or positive Orin checkpoint reuse.
- CPU fake writer does not execute actual Engine/Serve/CUDA math or GPU payload
  transfers. Declaration-only host syntax is NOT compilation of the actual CUDA
  translation unit. Existing text toy oracle is NOT the new image model oracle.
- Initial unrestricted CTest invocation discovered unrelated unbuilt targets;
  that invocation was NOT a pass. Raw log retained as
  ctest-unrestricted-unbuilt-targets-NOT-PASS.txt. Only the17 explicitly built
  relevant cases above are claimed. No unrelated all-target build claimed.

PENDING:
- CUDA13.2 SM87 actual compile/link (and other touched backend builds: HIP/SYCL
  compilers absent). No local nvcc/hipcc/icpx/model/device.
- Actual GGUF/template fixture prep with prior real1024-row encoder outputs;
  demonstrated positive steady earlier-user rewrite after an historical image.
- New independent empty-state IMAGE native oracle for restored/final full logits,
  main/MTP codes/scales, GDN/conv/hidden seed, tokens/CPU+device rotary/image state;
  actual corruption rejection, cross-slot/cancellation/pins/reset/bank counters.
- Actual GPU eviction under pressure, broader decode/sample/HTTP behavior,
  encoder co-residency/elastic peaks, actual OWUI divergence/savings and timings.
- Any real service/cgroup/restoration/loaded-health operation: NOT executed.

No tolerances or numerical kernels changed. Completed projection investigation
not reopened. OFF/ON neutrality alone is not an independent correctness oracle.
New image flag remains disabled by default until actual Orin qualification.

Reproduce selected checks from isolated feature source (Python deps/CMake here
are existing /workspace/tmp/adaptive-integration-deps, not production packages):

    cmake -S . -B build-image-cpu -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF \
      -DSTRATA_ENABLE_SYCL=OFF -DSTRATA_NATIVE_EXPERTS=OFF -DSTRATA_BUILD_TESTS=ON \
      -DCMAKE_BUILD_TYPE=Release
    cmake --build build-image-cpu --target unified_kv_test kv_preallocation_test \
      unified_kv_host_capability_test q36_cache_diagnostic_test \
      q36_checkpoint_schedule_native_test q36_checkpoint_schedule_test \
      q36_adaptive_checkpoint_test q36_prefix_checkpoint_test q36_sampled_mtp_test \
      q36_sampled_request_test q36_vision_preallocation_test q36_vision_staging_test \
      q36_image_checkpoint_test q36_image_checkpoint_helper_test -j2
    ctest --test-dir build-image-cpu --output-on-failure \
      -R '^(unified_kv_test|kv_preallocation_test|unified_kv_host_capability_test|q36_.*)$'
    python3 -m pytest -q tools/test_q36_*.py serve/test_q36_config.py \
      serve/test_unified_kv.py tools/test_unified_kv_validate.py
    python3 tools/check_q36_image_checkpoint_negative.py
    python3 tools/check_q36_vision_negative.py
    python3 tools/check_kv_preallocation_negative.py
    python3 tools/check_q36_image_preservation.py
    python3 tools/q36_cpu_checks.py --build-dir build-q36-dev-cpu-final --negative-controls --sanitizers
    python3 tools/q36_restore_observation_harness.py
    python3 tools/check_q36_restore_observation_negative.py
    python3 tools/check_q36_gate_watchdog_negative.py

## Addendum: review corrections (same branch, tip f19e22941276cdc7d63f0bd84839626c22f2c7df)

Both review blockers (restored-state observation timing, unenforced execution
budget) were fixed and re-validated on this host from a CLEAN tree with
`tools/q36_cpu_checks.py --build-dir build-q36-dev-cpu-final --negative-controls
--sanitizers --jobs 3`:`PASS_CPU_ONLY`, logs/q36-dev-cpu/20261028T064821Z-2583.
Raw outputs are copied here with the`-correction`suffix; the2026-10-24 logs above
remain the implementation-time record and are not overwritten.

- 14 Release CPU binaries/17 ctest cases:100% passed in BOTH the release and the
  ASan/UBSan build, including the3 expected-failure text schedule-oracle controls.
  release-ctest-correction.txt, sanitized-ctest-correction.txt.
- 94 Python tests plus22 subtests (up from86+22), adding the restored-state
  placement/harness checks, the supervised preflight cases, the pure-verdict mode
  checks and the six bounded-supervisor cases. python-correction.txt.
- Negative controls are now four scripts:8 image,6 vision and8 bank metadata
  mutations, plus3 new restored-observation mutations and3 new gate-watchdog
  mutations. Every mutant failed as required and the baseline stayed clean.
  negative-*-correction.txt.
- Restored-state host harness (real copy_session tail + real boundary primitives,
  inert CUDA stubs) and its mutation ladder: restore-observation-correction.txt.
- Five bounded supervisor scenarios (execution budget2s, separate cleanup
  allowance2s): normal exits0 within budget; slow preflight, hung child plus its
  marked descendant, slow postprocessing and a deliberately late success are all
  killed at the deadline with owned_group_empty=1. gate-scenarios-correction.txt.
- All5 feature patches replayed onto765068db reproduce tree
d67a44ef0749000155b1d4779515fe058260faab exactly (patch-replay-correction.txt).
The incremental bundle (prerequisite4130bdfd only) imports tip f19e2294 with the
same tree and preserved object order (bundle-import-correction.txt).

Limitations are unchanged and this addendum adds none: declaration-only C++17
stand-ins and the inert-CUDA host harness are NOT compilation of the actual CUDA
translation unit; the supervisor selftest children are NOT gate arms; no model,
encoder, GPU, service, cgroup or Orin operation was performed. Image checkpoints
stay disabled by default and NOT Orin-qualified.

Review/handoff: docs/Q36_IMAGE_CHECKPOINTS.md. Archive import/patch replay/tree/
checksum verification logs accompany the packaged revision.

## Addendum2: supervisor teardown corrections (tip5d793938fbae4a2940a582fb9dc511dafa1b05ae)

Two independently reproduced teardown defects were fixed.
(1) SIGINT/SIGTERM/SIGHUP delivered to the gate supervisor left its worker group
running and wrote **no** `watchdog.json` (SIGTERM/SIGHUP took the default
disposition; SIGINT raised past the reaping, group verification and the record).
One recorded teardown path now handles every ending, always writes the failure
record and exits nonzero.
(2) The cleanup allowance was waited out and then **re-armed** for group cleanup
(measured2.11s with a0.1s execution budget and a1s allowance). One absolute
cleanup deadline now bounds waiting, reaping and the emptiness proof.

Re-validated from a CLEAN tree at5d793938with
`tools/q36_cpu_checks.py --build-dir build-q36-dev-cpu-final3 --negative-controls
--sanitizers --jobs 3`:`PASS_CPU_ONLY`, logs/q36-dev-cpu/20261028T082122Z-6987.
Raw outputs are copied here with the`-teardown`suffix (earlier logs untouched):

- 14 Release CPU binaries/17 ctest cases:100% passed in BOTH the release and the
  ASan/UBSan build. release-*-teardown.txt, sanitized-*-teardown.txt.
- 99 Python tests plus22 subtests (up from94+22), adding three per-signal teardown
  regressions (SIGINT/SIGTERM/SIGHUP to a real supervisor after its worker starts),
  a hung-child interrupt that must reap worker+child+marked descendant, a
  blocked-reaping case proving one allowance, and a foreign-process-untouched
  check. python-teardown.txt.
- Negative controls:8 image,6 vision,8 bank,3 restored-observation and now FIVE
  gate-teardown mutations. Every mutant fails as required:
  `signal_teardown_removed supervisor_exit=-15 surviving_worker_pids=1
  watchdog_record=False`,`cleanup_rearmed teardown_s=2.00` against a
  `one_cleanup_allowance_s=1.00` baseline, plus no_group_kill, late_success_accepted
  and supervisor_bypass. negative-gate_watchdog-teardown.txt,
  gate-signals-teardown.txt.
- gate-signals-teardown.txt records each signal case: supervisor exit1,
  `interrupted_signals` named, `owned_group_empty=true`, `worker_reaped=true`,
  `cleanup_complete=true`, no PASS marker, no surviving owned worker or marked
  descendant, and an unrelated process left running.
- All7 feature patches replayed onto765068db reproduce tree
c6e372e8e8017aa75293fa09d7113f5b582dff47 exactly (patch-replay-teardown.txt); the
incremental bundle imports the tip against4130bdfd only
(bundle-import-teardown.txt).

Checkpoint logic, numerical comparisons, tolerances and the existing execution
budget/preflight/mode protections are untouched by these two fixes. Host/CPU
evidence only: no CUDA, GPU, model, encoder, service, cgroup or Orin operation.
Image checkpoints remain disabled by default and NOT Orin-qualified.

## Addendum3: post-teardown error path (tip025e13df40cf7e923045cb06315bda7a1a1a44be)

One Python-only correction (025e13df): after normal teardown had reaped the worker
leader, a failed `watchdog.json` write fell into the supervisor fault path, which ran
`teardown()` again with a FRESH cleanup allowance and group-signalled a PGID whose leader
was already gone. Cleanup progress is now a persistent per-run state (one absolute
deadline reused by the normal, signal and fault paths, idempotent teardown, and a guard
that only probes a group whose leader was reaped); a result that cannot be recorded is
reported on stderr, exits nonzero and never restarts cleanup or prints a pass.

`git diff --quiet d9de98d5..HEAD -- src include tests CMakeLists.txt` exits0: no C++,
CUDA/HIP/SYCL, engine, native-test or build-tree file changed, and no line of the gate's
numerical/acceptance code (corruption arm, independent exact recomputation, no-tolerance
wording, engine flags) is touched. The full CPU suite was therefore **deliberately not
re-run**; the previously recorded `PASS_CPU_ONLY` run
(logs/q36-dev-cpu/20261028T082122Z-6987, tip5d793938) still covers the unchanged native
tree, and nothing in this commit can alter it. Raw outputs are copied here with the
`-postteardown`suffix:

- source-and-scope-postteardown.txt — HEAD, the three changed files, the byte-identical
  native tree check, numerical-token proof, f934 protected-file sha256
  f77b0d6ff8934ab60e9eeca1c274a2104dd157c9dbed08befc23dda254548320, and the two
  Orin-only f934 files still absent locally (never fabricated).
- python-postteardown.txt — 31 passed (23 image-checkpoint + 8 multimodal-cache),
  including new regressions: watchdog write failure after completed cleanup, an
  exception after leader reaping (safety-net record still `passed=false` with
  `worker_reaped=true owned_group_empty=true cleanup_complete=true`), teardown
  idempotence, no group signal once the leader is reaped, no renewed allowance, and
  `--fault` refused outside `--selftest`. The R3/R4 signal (SIGINT/SIGTERM/SIGHUP),
  timeout, hung-child reaping and foreign-process protections still pass.
- gate-fault-injection-postteardown.txt — real supervisor, fake children:
  `fault=record exit=1 elapsed_s=0.56 watchdog_record=False`,
  `fault=crash exit=1 elapsed_s=0.57 watchdog_record=True`,
  `hung_child fault=record exit=1 elapsed_s=3.11` (budget3s + one cleanup allowance, no
  second round), no surviving marked owned process in any case.
- negative-gate_watchdog-postteardown.txt — baseline plus SEVEN mutations, all biting:
  `one_cleanup_allowance_s=1.00 reentry_s=0.00 signals_after_reap=0 probes_after_reap=1
  result_unchanged=True`; `cleanup_rearmed teardown_s=2.00`;
  `cleanup_state_rearmed reentry_s=1.00`;
  `post_reap_signal signals_after_reap=2 (real supervisor: 0 signals)`; plus
  no_group_kill, late_success_accepted, supervisor_bypass, signal_teardown_removed.
- negative-image_checkpoint-postteardown.txt and
  negative-restore_observation-postteardown.txt — the two neighbouring controls that
  read the same gate module re-run clean (8 image mutations, 3 observation mutations).

## Addendum4: HTTP integration smoke harness (tip96d1a00b, three new Python files + docs)

The operator's bounded numerical window on Orin PASSED and its archive is kept in
`tmp/q36-image-checkpoint-orin-evidence-20261028T115523Z.tar.gz`; the numbers are recorded
in `docs/Q36_IMAGE_CHECKPOINTS.md` (**What the Orin numerical window measured**) and
`docs/Q36_MULTIMODAL_CACHE_HANDOFF.md`. What follows is the CPU/host evidence for the new
HTTP integration harness (`tools/q36_image_http_smoke.py`, documented in
`docs/Q36_IMAGE_HTTP_SMOKE.md`). It was exercised ONLY against a loopback stand-in that
replays `serve/server.py`'s endpoint and engine-log shapes: no model, encoder, GPU, engine
binary, service, cgroup or Orin operation was involved, and it re-verifies no GPU tensor
(the completed native oracle above owns that).

Evidence stored with the `-http-smoke` suffix:

- source-and-scope-http-smoke.txt — HEAD96d1a00b, origin tip72457256, the three new files
  with their sha256, `git diff --quiet HEAD -- src include tests CMakeLists.txt` exit0
  (no C++/CUDA, native-test or build-tree file changed and the reviewed numerical gate/oracle
  code has no changed line), the protected f934 local copy sha256
  f77b0d6ff8934ab60e9eeca1c274a2104dd157c9dbed08befc23dda254548320, and the two Orin-only
  f934 files still absent locally (never fabricated).
- python-http-smoke.txt — 70 passed: 39 new HTTP-smoke tests (plan and conversation
  semantics, temporary-config isolation and flag mapping, engine-log parsing and
  correlation, required-vs-reported split, mode-bound markers, burst attribution, the
  recycled-PID guard, ten stand-in scenarios, signal teardown, record-failure path,
  plan-only/preflight guards, `--fault` refused outside `--selftest`) plus the 23
  image-checkpoint and 8 multimodal-cache tests.
- smoke-selftest-matrix-http-smoke.txt — per scenario exit/elapsed/watchdog: `normal exit=0
  elapsed_ms=2108`, `divergence exit=0` (reported, not required),
  `hung_server exit=0 elapsed_ms=16863` and `stuck_group exit=0 elapsed_ms=16879` (the arm
  server ignores `SIGTERM`, so the owned group is killed and proven empty; graceful=false
  recorded), and `slow_server(window=20s) exit=1 elapsed_ms=20217 killed=True worker_exit=-9
  server_groups_empty=True kinds=['killed_owned_group']` — a worker killed mid-arm leaves no
  model stand-in behind. Semantic refusals: `no_encoder`, `zero_stable_reuse`, `no_restore`,
  `engine_error`, `cross_talk`, `missing_record` all exit1 with the named required check
  listed and no marker. `plan-only` and `preflight` print only their own markers
  (`real_marker_leaked=0`) and produce zero engine logs (nothing was started). After the
  whole matrix: `surviving stand-in listeners: []`.
- smoke-signals-faults-http-smoke.txt — real supervisor: `SIGINT mid-arm exit=1
  elapsed_ms=6210` through the single teardown (`interrupted=SIGINT killed=True
  worker_reaped=True group_empty=True server_groups_empty=True
  kinds=['killed_owned_group']`, no pass line); `fault=record exit=1 watchdog_record=False
  pass_lines=0` with `action=no_restart cleanup_state_final=1`; `fault=crash exit=1
  supervisor_error_RuntimeError` with the leftover group reaped; `--fault` outside
  `--selftest` refused.
- negative-image_http_smoke-http-smoke.txt — six mutation controls, each run against the
  mutated AND the committed harness, all biting: flag on every arm, per-client attribution of
  a concurrent burst, reuse requirement dropped, server groups not reaped (a listener
  survives only under the mutation), pass marker not mode-bound, window guard removed.
- negative-image_checkpoint/-vision/-restore_observation/-gate_watchdog/-bank-http-smoke.txt
  — the five neighbouring controls re-run clean after the harness landed.

Because the change is three Python files plus documentation, and
`git diff --quiet HEAD -- src include tests CMakeLists.txt` exits0, the full CPU suite was
deliberately NOT re-run; `tools/q36_cpu_checks.py` picks the new test file up through its
`tools/test_q36_*.py` glob and now also runs `negative-image_http_smoke`, so the next full
run covers both.

Host/CPU evidence only: no CUDA, GPU, model, encoder, service, cgroup or Orin operation in
this addendum. The numerical engine/oracle IS Orin-qualified (see the Orin sections); the
HTTP integration path on Orin, broader sampling, actual front-end savings, encoder
co-residency and performance distributions remain pending. `--kv-image-checkpoints` stays
disabled by default.
