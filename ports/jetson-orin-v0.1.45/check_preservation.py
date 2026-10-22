#!/usr/bin/env python3
"""Source-preservation gate, not a compiler/runtime/performance test."""
import argparse
from pathlib import Path
import subprocess

BASE = "477ac13cbe8e582c20484871ce459d20d82df09d"
ALLOWED = set("""
CMakeLists.txt
include/strata/core/device.hpp
include/strata/platform/cpu_arch.hpp
include/strata/platform/cpu_relax.hpp
include/strata/platform/mapped_memory.hpp
include/strata/platform/shared_memory_budget.hpp
src/core/device.cu
src/core/device_main.cpp
src/core/expert_cache.cpp
src/core/expert_cache_memory_test.cpp
src/core/expert_source.cpp
src/core/graph.cpp
src/core/mtp.cpp
src/core/pinned_fallback_test.cpp
src/core/session.cpp
src/core/verify.cpp
src/kernels/cpu/expert.cpp
src/kernels/cpu/expert_layout.cpp
src/kernels/cpu/expert_multi_test.cpp
src/kernels/cpu/expert_parity.cpp
src/kernels/cpu/native_expert.cpp
src/kernels/cpu/pool.cpp
src/kernels/cpu/pool_stress.cpp
src/kernels/cpu/pool_tasks_test.cpp
src/kernels/cpu/pool_test.cpp
src/kernels/cpu/q2_portable_reference_test.cpp
src/kernels/cpu/q8_oracle_contract_test.cpp
src/kernels/cpu/router_dot_parity.cpp
src/kernels/cpu/router_portable.cpp
src/kernels/native_expert_parity.cpp
src/kernels/quantize_act_parity.cpp
src/platform/shared_memory_budget_test.cpp
src/prefill/mmq_nvfp4_parity.cpp
src/program/generate.cpp
src/qwen35/q36_engine.cu
tests/core/file_expert_source_test.cpp
tools/cuda_memory_selftest.cu
tools/cuda_platform_probe.cpp
""".split())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    args = ap.parse_args()
    repo = args.repo.resolve()

    def git(*argv):
        return subprocess.check_output(["git", "-C", str(repo), *argv])

    if subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", BASE, "HEAD"]).returncode:
        raise SystemExit("HEAD is not based on the pinned upstream v0.1.45 commit")
    if git("rev-list", "--merges", f"{BASE}..HEAD").strip():
        raise SystemExit("Merge commits found after the pinned release; keep this port linear")

    old_paths = set(git("ls-tree", "-r", "--name-only", BASE).decode().splitlines())
    tracked = set(git("ls-files").decode().splitlines())
    # Git honors the release's CRLF filters for .bat files; raw worktree byte
    # comparisons would falsely identify untouched launchers as source changes.
    changed = sorted(set(git("diff", "--name-only", BASE, "--").decode().splitlines()) | (tracked - old_paths))
    changed = [p for p in changed if not p.startswith("ports/jetson-orin-v0.1.45/")]
    source_changed = [p for p in changed if p != "README.md"]
    unexpected = set(source_changed) - ALLOWED
    if unexpected:
        raise SystemExit("Unexpected release changes:\n" + "\n".join(sorted(unexpected)))

    # Allow only the additive port notice, not arbitrary edits to upstream's README.
    readme_before = git("show", f"{BASE}:README.md").decode()
    readme_after = (repo / "README.md").read_text()
    if readme_after != readme_before:
        begin = "<!-- Strata-Orin v0.1.45:begin -->\n"
        end = "<!-- Strata-Orin v0.1.45:end -->\n\n"
        if readme_after.count(begin) != 1 or readme_after.count(end) != 1:
            raise SystemExit("Root README must have one delimited additive port notice")
        prefix, notice_and_tail = readme_after.split(begin, 1)
        if end not in notice_and_tail:
            raise SystemExit("Root README port notice delimiters are out of order")
        _, tail = notice_and_tail.split(end, 1)
        if prefix + tail != readme_before:
            raise SystemExit("Root README changed beyond the additive port notice")

    # Prove the entire Qwen3.6 translation unit is otherwise release-identical,
    # including its model loader, GPU kernels, MMQ and optimized dispatches.
    qpath = "src/qwen35/q36_engine.cu"
    before = git("show", f"{BASE}:{qpath}").decode()
    after = (repo / qpath).read_text()
    after = after.replace('#include "strata/core/expert_cache.hpp"\n', "", 1)
    marker = "// Qwen3.6 has its own placement/chunk planner:"
    if marker in after:
        start = after.index(marker)
        end = after.index("using Clock = std::chrono::steady_clock;", start)
        after = after[:start] + after[end:]
    after = after.replace("CK(memory_info(", "CK(cudaMemGetInfo(")
    if before != after:
        raise SystemExit("Qwen3.6 changed beyond the audited host memory-query adaptation")

    cm_before = git("show", f"{BASE}:CMakeLists.txt").decode()
    cm_after = (repo / "CMakeLists.txt").read_text()
    if "project(strata VERSION 0.1.45 LANGUAGES CXX)" not in cm_after:
        raise SystemExit("Engine version is no longer Strata v0.1.45")
    options = lambda s: [line.strip() for line in s.splitlines() if line.strip().startswith("option(")]
    if options(cm_before) != options(cm_after):
        raise SystemExit("Release CMake feature option defaults changed")
    def cuda_config(s):
        start = s.index("elseif(STRATA_ENABLE_CUDA)")
        return s[start:s.index("elseif(STRATA_ENABLE_HIP)", start)]
    if cuda_config(cm_before) != cuda_config(cm_after):
        raise SystemExit("Release CUDA architecture/toolkit configuration changed")
    print(f"PASS: {len(source_changed)} allowed source/config/test paths differ; other tracked release files unchanged")
    print("PASS: upstream root README preserved apart from the additive port notice")
    print("PASS: linear history based on pinned v0.1.45; engine version remains 0.1.45")
    print("PASS: Qwen3.6 loader/compute body, GPU kernel files and prefill production sources preserved")
    print("PASS: release CMake option defaults and CUDA architecture configuration preserved")
    print("This does NOT establish CUDA compilation, runtime correctness, full residency or throughput.")


if __name__ == "__main__":
    main()
