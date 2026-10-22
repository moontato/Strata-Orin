# Orin verification and benchmark acceptance

**Nothing in this file is a recorded Orin result.** Run gates in order; keep failures, skip reasons and raw logs. Never report a skipped GPU test as a pass. A successful compiler invocation does not establish model correctness or speed.

Use the variables/checkout from [BUILD.md](BUILD.md): `WORK=/mnt/ssd/Strata-Orin` and `PATCHDIR="$WORK/ports/jetson-orin-v0.1.45"`. The patches are already applied on the prepared branch. Laptop-only results are in [VALIDATION.md](VALIDATION.md), not evidence for the hardware gates below. Run only one inference engine at a time; stop any other model service only with its owner's approval. Keep cooling, power mode and clocks the same for both engines. Do not disable swap on a busy machine just to make the residency claim.

## A. Static preservation, architecture and capabilities

```sh
set -o pipefail
cd "$WORK"
mkdir -p results/orin-0145
python3 "$PATCHDIR/check_preservation.py" --repo "$WORK"
git diff --check 477ac13cbe8e582c20484871ce459d20d82df09d
file build-orin/strata build-orin/strata-q36 build-orin/libstrata_kq.so
ldd build-orin/strata-q36
ldd build-orin/libstrata_kq.so
"$CUDA_HOME/bin/cuobjdump" --list-elf build-orin/strata-q36 > results/orin-0145/engine-images.txt
"$CUDA_HOME/bin/cuobjdump" --list-elf build-orin/libstrata_kq.so > results/orin-0145/kquant-images.txt
./build-orin/strata-device | tee results/orin-0145/device.txt
./build-orin/diagnostics/strata-cuda-probe --json > results/orin-0145/probe.json
python3 - <<'PY'
import json
p = json.load(open('results/orin-0145/probe.json'))
d = p['devices'][0]
assert d['sm'] == 87 and d['integrated'] == 1
assert d['unified_addressing'] and d['can_map_host_memory']
print(p)
PY
```

Require ELF **aarch64**, real JetPack runtime/cuBLAS/driver dependencies, SM87 images in **both** engine and K-quant library, and the expected device. Inspect the compile commands for `compute_87` / `sm_87`; there must be no AVX options on ARM CPU compilation. Record actual physical/CUDA total/free bytes. They describe **one shared pool**, not host RAM plus extra VRAM. Record registration/VMM/managed capabilities rather than assuming them from the marketing name. If a required mapping/registration test fails, stop and investigate; do not route inference to CPU.

## B. Model-free memory and GPU correctness gates

```sh
./build-orin/diagnostics/strata-cuda-memory-test | tee results/orin-0145/memory.txt
ctest --test-dir build-orin -N > results/orin-0145/test-list.txt
ctest --test-dir build-orin --output-on-failure \
  -R '^(shared_memory_budget_test|q2_portable_reference_test|q8_oracle_contract_test|pool_tasks_test|expert_multi_test|router_dot_parity|pool_affinity_test|pinned_fallback_test|expert_cache_memory_test|expert_cache_memory_refusal|file_expert_source_test)$'
ctest --test-dir build-orin --output-on-failure \
  -R '^(quantize_act_parity|native_mmvq_multi|mmvq_multi_parity|gdn_parity|gdn_rec_parity|prefill_mmq_scatter_test|prefill_mmq_tab_test|prefill_mmq_kquant_test|native_expert_parity_.*)$'
```

Use `set -o pipefail` when piping test commands to `tee`. Inspect `ctest -N` first: a regex that selects zero cases proves nothing. `prefill_mmq_tab_test` is optional: it is only registered with upstream's opt-in `STRATA_MMQ_TAB=ON`, which the standard build leaves off. `native_mmvq_multi` is only registered when its benchmark source is present. Their absence is not a pass. The ordinary scatter/K-quant MMQ and native-expert parity tests must still be present and run. The memory selftest must execute CUDA increments, registered mapped access, **512 CPU/GPU doorbell exchanges**, and actual VMM map/access (or explicitly report unsupported VMM). The cache tests exercise stable addresses, data retention, shrink/grow, cudaMalloc fallback and refusal with excessive headroom. The registration-failure test injects a failure and checks error cleanup plus data-copy integrity; it is not a production CPU-inference fallback.

Run the broader eligible suite after these gates. Two legacy tests (`expert_parity`, `pool_test`) require the real Flash-Next `pack/full/experts.bin`; do not fabricate a fixture for Qwen3.6 or count them as passing. Some existing VMM tests assume dedicated-VRAM free-byte deltas; on shared RAM another process can invalidate that observation. Investigate any failure and preserve the actual mapping/data-retention checks, not just the free-byte delta.

## C. Load the exact model and check optimized-kernel correctness

Check the model SHA-256 from BUILD.md first. A minimal native Qwen3.6 run:

```sh
IDS=$(python3 -c "print(open('tools/qwen35/parity_prompts.ids').readline().strip())")
STRATA_Q36_VERIFY=1 ./build-orin/strata-q36 --model "$MODEL" \
  --max-context 8192 --chunk 512 --tokens "$IDS" --n 64 --mtp 0 --no-graph \
  > results/orin-0145/check.tokens 2> results/orin-0145/check.log
```

Require successful GGUF architecture/shape/quantization recognition, finite output, nonzero verifier comparisons and **zero mismatches**. This eager/512-chunk run is a **correctness diagnostic**, not a throughput result. Do not run timing with `STRATA_Q36_VERIFY`, `--no-graph`, sanitizer or profiler overhead and call it optimized performance.

The release's existing token gates can compare new versus old kernels at a fixed chunk, on this hardware:

```sh
python3 tools/qwen35/q36_solo_gate.py --exe "$WORK/build-orin/strata-q36" \
  --args "--model $MODEL --max-context 8192 --chunk 512 --mtp 0" \
  --out results/orin-0145/optimized.json
STRATA_Q36_OLD_KERNELS=1 python3 tools/qwen35/q36_solo_gate.py \
  --exe "$WORK/build-orin/strata-q36" \
  --args "--model $MODEL --max-context 8192 --chunk 512 --mtp 0" \
  --out results/orin-0145/old-kernels.json
python3 tools/qwen35/q36_cmp_gate.py \
  results/orin-0145/optimized.json results/orin-0145/old-kernels.json
```

Paths passed through `--args` should not contain spaces. Repeat with `--mtp 2` as a **separate** gate if testing MTP. Both models' main and draft experts must then fit. Include one-token and long-prompt cases, repeated loads/requests and recovery tests. Batch, vision, API streaming/cancellation and long contexts require additional gates; not implied by a solo text pass.

For **llama.cpp numerical validation**, use the same GGUF and identical prompt token IDs, then teacher-force the same generated IDs and compare full logits (KL, finite values, top-1 agreement), not only independently diverging greedy text. `tools/qwen35/q36_llama_dump.cpp`, `parity.py`, `kl.py` and `compare_dumps.py` provide the release's reference harness; its helper must be built against the chosen llama.cpp reference revision. `strata-q36 --force IDS --dump-full-logits FILE` supports teacher forcing. Establish a CPU-versus-CUDA llama.cpp rounding floor on this same model rather than inventing a universal KL tolerance. Different chunks can change GEMM rounding; preserve a 512-chunk correctness comparison and record the default chunk separately.

## D. Confirm all-GPU residency and actual optimized CUDA execution

“All resident” here means the weights used during inference are in CUDA device allocations, with no expert CPU execution or recurring weight streaming. Orin shares physical DRAM; this is **not** a second dedicated VRAM allocation, and small token/control/logit copies are normal.

Before accepting throughput:

- Require startup **`every routed expert in VRAM`** and **`prompt experts on MMQ (int8)`**. Reject any hybrid expert report, CPU expert count, slot replacements, streamed expert layers, or packed Q8 decode-copy give-back. Do not set `--expert-slots` or lower `--expert-vram` to force hybrid mode.
- Verify `libstrata_kq.so` loads. Missing it can select a slower GPU path. A file existing on disk is not sufficient; the MMQ log and CUDA trace must agree.
- Profile one **separate, untimed** run with JetPack's Nsight Systems, if installed:

```sh
nsys profile --trace=cuda,nvtx,cublas --sample=none \
  -o results/orin-0145/q36-cuda \
  ./build-orin/strata-q36 --model "$MODEL" --max-context 8192 \
  --bench-pp 4096 --bench-tg 128 --mtp 0
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum \
  results/orin-0145/q36-cuda.nsys-rep > results/orin-0145/cuda-summary.txt
```

Inspect the timeline, not only a summary: CUDA kernels/cuBLAS GEMMs and MMQ launches on Orin, captured graph decode where applicable, no repeated large host→device weight transfers after loading, and no CPU expert work. Nsight availability/trace options vary with the installed JetPack tool; retain its version/errors. Hardware counters are optional, but CUDA execution evidence is required.

- Monitor `tegrastats --interval 1000` and `vmstat 1` in separate terminals throughout loading and steady-state repetitions; save timestamps and logs. Watch **MemAvailable, swap-in/out, storage reads, GR3D/EMC clocks, temperature and power**. Zero new swap traffic and no recurring model reads in the timed interval are required for the resident comparison. Existing system swap usage alone does not prove or disprove weight residency. File-cache reads during initial model loading are expected.
- Retain at least the configured physical headroom (default 6 GiB) after loading and under sustained operation. This is a conservative safety policy, not a measured optimum. Budget tests cannot prove absence of fragmentation, transient peaks or other-process interference.

If the model cannot meet these conditions, the requested all-resident benchmark is **not yet available**; a lower-memory hybrid run is not a substitute.

## E. Prefill throughput against the 500–800 tok/s baseline

Run without profiling/debug knobs. The helper refuses known optimization-disable/diagnostic variables, preserves graphs/default chunk selection, requires MMQ and all-expert logs, and reports medians of the engine's three repetitions:

```sh
bash "$PATCHDIR/benchmark_resident.sh" "$WORK/build-orin/strata-q36" \
  "$MODEL" "$WORK/results/orin-0145/native-default"
```

Its shapes are pp512 / pp2048 / pp4096, tg128, context8192, MTP0. Prefill includes the engine's prompt processing and final stream synchronization, **not model loading or Python tokenization**. It is not an HTTP prompt-cache benchmark. Compare the raw logs to the source's timing boundary.

Re-run the existing llama.cpp baseline **the same day** on the same model SHA, clocks/power/cooling and prompt lengths, all layers on GPU. Example (adapt to the baseline binary's supported flags and retain the flags that produced the stated baseline):

```sh
/path/to/baseline/llama-bench --version
/path/to/baseline/llama-bench -m "$MODEL" -ngl 99 \
  -p 4096 -n 128 -b 4096 -ub 2048 -r 5
```

Confirm llama.cpp's own full-GPU offload logs; do not assume `-ngl 99` proves residency. Interleave at least five Strata/llama.cpp process pairs, one engine at a time; retain every repetition, medians/range, effective chunk/microbatch, context, thermal trend and peak memory. A chunk sweep (512/1024/2048/4096) may be useful **after** the default run; record it as an explicit experiment and recheck residency. Never change the default just because another GPU measured it faster.

If the historical 500–800 tok/s used another quantization, model file, context, microbatch, power mode or measurement boundary, show it only as historical context. A speed ratio requires a controlled same-file/same-day baseline. Report prefill separately from decode and MTP; no claimed improvement until the actual Orin measurements and correctness/residency gates are available.

Record: patch/tag/dependency SHAs; model SHA/format; JetPack/L4T/driver/nvcc/GCC versions; CMake cache and compiler flags; power mode/clocks/cooling; context/chunk/batch/MTP; MMQ/graph/residency logs; CUDA trace; RAM/swap/I/O/thermal traces; prompt and decode samples; skips/failures; what remains untested. Do not expose a later API server beyond `127.0.0.1` without `--api-key`.
