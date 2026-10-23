# Qwen3.6 INT8 KV on the v0.1.45 Orin branch

Opt in with `--kv int8`. Omitting `--kv`, or selecting `--kv fp32`, retains FP32 KV. This is for Qwen3.6-35B-A3B's 256-dimensional heads and GQA 8. It does not change weight quantization, expert placement, MMQ, or the existing FP32 kernels.

## Format and integration

K **and** V are signed INT8, with separate FP16 scales per 64 dimensions, per token and KV head. K is quantized after its existing RMS normalization and RoPE; V after projection. Codes use round-to-nearest-even against the **stored** half scale, clamped to [-127,127]; zero scales produce zero codes. Queries, softmax and accumulation stay FP32. The new decode kernel shares vector-loaded K and V across the eight GQA heads. Prompt attention dequantizes into the existing shared FP16 operands and retains the cuBLAS prompt path; expert MMQ and GPU-resident experts are not disabled.

Each session owns four head-major VMM planes for every attention layer, including the MTP block: K codes, V codes, K scales, V scales. No persistent FP32 KV mirror is allocated. The pre-existing shared prefill scratch remains. Codes/scales grow together through the same admission policy, byte accounting, idle LRU and replay identities. Copies enqueue each head's code **and scale** prefix, with the existing stream synchronization and donor lease. Batch row tables carry all four pointers; graph arguments keep stable VA across growth and eviction.

Raw bytes per token per attention cache: `2 * 2 * (256 + 4*2) = 1056`, versus 4096 in FP32. Ten main attention caches plus MTP give 11,616 bytes/token: 5,808,000,000 raw bytes for 500K tokens (about 5.41 GiB). This is arithmetic, not a residency measurement. Physical commitment also charges VMM rounding, scales, and recurrent/last-hidden buffers; it can be larger than the raw size. Small caches can have substantial granularity overhead. `KVPOOL` / `KVSESSION` remain authoritative for managed physical bytes.

Reference checked: [Strata PR #3678](https://github.com/Niko1221/Strata/pull/3678), head `8e7742e8ac2412d59ed3b6a52a9aa7259bc1f128`. Its newer engine has an FP16 baseline and other cache formats, changed attention, Windows trimming and no corresponding unified-VMM ownership. It was **not** merged/cherry-picked. This implementation uses its group-64 INT8 format/GQA approach in a parallel path, with explicit four-plane batch selection and v0.1.45's FP32 default preserved. FP8, Q4 and sampled-MTP changes are not included.

## Focused validation actually performed here

* x86_64, CUDA 12.6.85, SM87: `strata`, `strata-q36`, `q36_kv_int8_test`, `unified_kv_test` built/linked.
* CPU ownership/layout test passed, including the four token distributions and INT8 scale/granule accounting. Non-PIE ASan/UBSan also passed. The PIE sanitizer executable crashed without a diagnostic in this host environment; no production flags were changed to address it.
* Invalid `--kv` selection is rejected explicitly.
* All 63 original compute-kernel machine-code sections still match the Jetson baseline; three new INT8 kernels are added. In this toolchain the new attention uses 64 registers and 24 KiB shared memory, with no register spills. This is not a performance result.
* The native INT8 test **compiled but skipped here (no GPU)**. It checks quantized K/V writes and RoPE, FP64 attention over the stored codes, FP16 prompt staging, row-selected caches, main/draft prefix copies, physical growth including scales, LRU/cache misses and graph replay after eviction. Its CUDA runtime correctness still needs the Orin run.
* No Orin INT8 model inference, long-context residency, model-logit/quality comparison, MTP acceptance or performance result is claimed. No full suites or prolonged benchmarks were run for this change.

The Flash-Next [registered-host-pointer finding](KV_HOST_MAPPING.md) is separate. Its capability-based narrow correction does not assert that devices lacking the guarantee can support flat incremental host mappings. No host-VMM or segmented backend was implemented.

## Build and short smoke tests on Orin

Reuse the **already working CUDA 13.2 / SM87 build directory**, preserving its native-expert/K-quant/MMQ options and ggml pin. Example directory: `build-orin`.

```bash
cmake -S . -B build-orin -DSTRATA_BUILD_TESTS=ON
cmake --build build-orin --target strata strata-q36 strata_kq q36_kv_int8_test unified_kv_test -j4
ctest --test-dir build-orin -R '^(unified_kv_test|q36_kv_int8_test)$' --output-on-failure
```

Keep `libstrata_kq.so` beside the engine. The CUDA INT8 gate must **pass, not skip** on Orin. Then a short actual-model serial MTP2 smoke (no benchmark):

```bash
MODEL=/path/to/Qwen3.6-35B-A3B-first-shard.gguf
TOKENS=$(python3 -c 'print(",".join(["198"]*256))')
build-orin/strata-q36 --model "$MODEL" \
  --max-context 250000 --batch 4 --kv-pool-tokens 500000 \
  --kv int8 --mtp 2 --batch-mtp --chunk 512 \
  --tokens "$TOKENS" --n 8
```

A short four-slot protocol smoke with the required limits and batch-MTP:

```bash
python3 tools/unified_kv_validate.py --exe build-orin/strata-q36 \
  --lengths 256 --gen 8 --pool-tokens 500000 --out results/q36-int8-smoke -- \
  --model "$MODEL" --max-context 250000 --batch 4 \
  --kv int8 --mtp 2 --batch-mtp --chunk 512
```

This sends 256-token prompts, not 250K prompts: the configured ceilings are not proof of full residency. Confirm `kv_format=int8`, scale group 64, graphs on, MMQ and full-GPU expert placement in the logs. INT8 is lossy, so FP32/INT8 generated tokens are not required to be identical. Do not relax existing FP32 numerical checks; establish model quality/logit checks on Orin separately before qualification.
