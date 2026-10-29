# Qwen3.6 INT8 KV on the v0.1.45 Orin branch

Opt in with `--kv int8`. Omitting `--kv`, or selecting `--kv fp32`, retains FP32 KV. This is for Qwen3.6-35B-A3B's 256-dimensional heads and GQA 8. It does not change weight quantization, expert placement, MMQ, or the existing FP32 kernels.

## Format and integration

K **and** V are signed INT8, with separate FP16 scales per 64 dimensions, per token and KV head. K is quantized after its existing RMS normalization and RoPE; V after projection. Codes use round-to-nearest-even against the **stored** half scale, clamped to [-127,127]; zero scales produce zero codes. Queries, softmax and accumulation stay FP32. The new decode kernel shares vector-loaded K and V across the eight GQA heads. Prompt attention dequantizes into the existing shared FP16 operands and retains the cuBLAS prompt path; expert MMQ and GPU-resident experts are not disabled.

Each session owns four head-major VMM planes for every attention layer, including the MTP block: K codes, V codes, K scales, V scales. No persistent FP32 KV mirror is allocated. The pre-existing shared prefill scratch remains. Codes/scales grow together through the same admission policy, byte accounting, idle LRU and replay identities. Copies enqueue each head's code **and scale** prefix, with the existing stream synchronization and donor lease. Batch row tables carry all four pointers; graph arguments keep stable VA across growth and eviction.

Raw bytes per token per attention cache: `2 * 2 * (256 + 4*2) = 1056`, versus 4096 in FP32. Ten main attention caches plus MTP give 11,616 bytes/token: 5,808,000,000 raw bytes for 500K tokens (about 5.41 GiB). This is arithmetic, not a residency measurement. Physical commitment also charges VMM rounding, scales, and recurrent/last-hidden buffers; it can be larger than the raw size. Small caches can have substantial granularity overhead. `KVPOOL` / `KVSESSION` remain authoritative for managed physical bytes.

Reference checked: [Strata PR #3678](https://github.com/Niko1221/Strata/pull/3678), head `8e7742e8ac2412d59ed3b6a52a9aa7259bc1f128`. Its newer engine has an FP16 baseline and other cache formats, changed attention, Windows trimming and no corresponding unified-VMM ownership. It was **not** merged/cherry-picked. This implementation uses its group-64 INT8 format/GQA approach in a parallel path, with explicit four-plane batch selection and v0.1.45's FP32 default preserved. FP8, Q4 and sampled-MTP changes are not included.

For optional startup backing of this same layout, see
[Qwen3.6 physical preallocation](Q36_KV_PREALLOCATION.md). It keeps INT8 kernels,
head strides and the shared logical capacity unchanged, with a separate physical
bank and a page-rounding/distribution bound rather than raw bytes/token.

## Validation scope

The CPU layout/ownership tests and CUDA q36_kv_int8_test cover different scopes.
The device gate checks stored quantized values/scales, attention references,
main/MTP prefix copies, growth and graph replay after eviction. Hardware-required
checks must pass rather than skip. INT8 is lossy; exact FP32-versus-INT8 generated
text is not a quality criterion. Model quality, long-context residency and timing
require separately reported measurements. See [Q36_DEVELOPMENT.md](Q36_DEVELOPMENT.md).
