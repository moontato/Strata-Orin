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

Review/handoff: docs/Q36_IMAGE_CHECKPOINTS.md. Archive import/patch replay/tree/
checksum verification logs accompany the packaged revision.
