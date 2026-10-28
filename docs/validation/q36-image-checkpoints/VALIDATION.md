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
