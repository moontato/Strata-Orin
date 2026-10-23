# Experimental Qwen3.6 sampled MTP

The existing Qwen3.6 speculative path is greedy-only: `Sampling::needs_logits()` and the sequential request dispatch use one token per step when temperature or a penalty is active. This branch adds an opt-in text-only path that keeps the same greedy MTP proposals and the same main-model verify window, but verifies each proposed row with a sample from the main model's actual target distribution.

It is experimental. It changes no deployed default, OWUI setting, temperature, penalty, expert placement, MTP configuration or running service.

## Procedure and why the target distribution is preserved

For confirmed prefix `c`, the MTP block may propose deterministic drafts `d0 ... dD-1` beside the next confirmed token `x`. The engine runs the unchanged verify window `[x, d0 ... dD-1]`.

At verification row `i`:

1. The full target logits after `x, d0 ... d(i-1)` are sampled with the request's existing sampler and history.
2. If that target sample equals `d(i)`, the proposal is committed and verification advances.
3. At the first mismatch, the sampled target token is returned and later rows are not consumed.
4. If every proposal was accepted, row `D` supplies the next token.

Every emitted token is drawn directly from the target distribution conditional on the exact prefix that precedes it. The proposals only decide whether future target rows may be evaluated early; they do not replace target samples or require a residual-distribution correction. With the same target logits and RNG seed, the token sequence and RNG consumption match sequential sampling.

`--mtp-sampled` turns it on. A request can turn it off for A/B measurement with `sampled_mtp=0` (`strata_tune={"sampled_mtp":0}` through the HTTP server). The engine option itself defaults off.

## What is reused

- The original greedy MTP chain, graph launches, verify window and GDN commit path are split into `spec_window()` and `spec_commit()`, then shared by `spec_step()` and `spec_step_sampled()`. No compute kernel is changed.
- Temperature, top-k, top-p, min-p, repetition, frequency and presence penalties use the unchanged production sampler. Its history grows with every accepted proposal, including the token that triggers a history-dependent penalty at the next row.
- Main and MTP KV, INT8 K/V codes and FP16 scales, positions, GDN recurrence, convolution history and the hidden seed follow the existing speculative commit sequence.
- Accepted proposals are reflected in `ss.live` before the answer ends, as existing greedy speculation already does. That lets a later request continue from the tokens the answer actually showed.
- Prefix checkpoints remain a prompt-prefill feature. A decode step does not create a checkpoint. Existing checkpoint metadata stays validated against generation, epoch, live prefix, position and physical coverage; exact-prefix restoration remains the first fast path.
- No KV checkpoint bank or model mirror is allocated for sampled MTP. It reuses existing verify-window and logits buffers.

## Scope limits

Enabled only for a solo text `GEN`, with no active batch window, no request or held-session pictures, and enough room for the whole verify window. BGEN/BGENI, batch MTP, image/M-RoPE and concurrent request windows remain exactly as before. `sampled_mtp=1` sent to an engine not started with `--mtp-sampled` is ignored.

Cancellation is checked before consuming each target row. A target sample taken before a stop is not used to sample future rows. A proposal accepted and printed before the stop is part of the live state. This matches the existing state model for speculation ending inside a window.

## Local validation actually run

On x86_64, no CUDA toolkit was available, so this is CPU/header validation, not native Qwen execution:

- `tests/core/q36_sampled_mtp_test.cpp` calls the same production `Sampling`, sampler and `q36_sample_verify_window()` template. It compares supplied target logits against sequential sampling for 200 seeds and 40 penalty seeds, including history-dependent presence/frequency/repetition behavior, top-k/top-p/min-p, rejection at every position, all-accepted windows, EOS at each decision boundary, max-new boundaries, row-call counts and final RNG state. Passed.
- Same test passed under non-PIE ASan/UBSan.
- Existing `q36_prefix_checkpoint_test` and `unified_kv_test` compiled and passed unchanged.
- `serve/server.py`, `serve/test_server.py`, `tools/q36_sampled_mtp_http_bench.py` and `tools/q36_sampled_mtp_fixture.py` pass `python3 -m py_compile`. The focused server test was not run in this container because `jinja2` is not installed.
- CUDA source compile/link and all 66 kernel-machine-code preservation checks are **pending**: this laptop has no `nvcc`, CMake or prior CUDA build tree.

The native gate `--mtp-sampled-test FILE` runs:

1. 16 sampled tokens directly through sequential and speculative paths, stopping both at the exact same processed prefix.
2. Full emitted sequence, history, live state, RNG state and 8 continuation tokens.
3. Full logits at that common prefix and after continuation. It prints byte-identity flags and maximum absolute difference separately, so distributional equivalence is not confused with byte-identical floating-point execution.
4. 32-token native request protocol with `sampled_mtp=0/1`, comparing exact text emitted by `Serve`.

No model, GPU or performance result is claimed until the Orin gates below pass.

## Build and native gates on Orin

Use a separate build tree. Replace `MODEL` and `LLAMA_SRC` with the actual paths; `CUDA_HOME` defaults to `/usr/local/cuda` in the command.

```bash
cd /path/to/feature/q36-sampled-mtp
cmake -S . -B build-sampled-mtp -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DCMAKE_CUDA_ARCHITECTURES=87 -DSTRATA_GGML_DIR="$LLAMA_SRC" \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_HIP=OFF -DSTRATA_ENABLE_SYCL=OFF \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON \
  -DSTRATA_PREFILL_MMQ=OFF -DSTRATA_BUILD_TESTS=ON

cmake --build build-sampled-mtp --target \
  strata strata-q36 strata_kq unified_kv_test q36_prefix_checkpoint_test q36_sampled_mtp_test -j4
ctest --test-dir build-sampled-mtp \
  -R '^(unified_kv_test|q36_prefix_checkpoint_test|q36_sampled_mtp_test)$' --output-on-failure
```

Make the one-line text fixture with the actual tokenizer and exported chat template:

```bash
python3 tools/q36_sampled_mtp_fixture.py --model "$MODEL" \
  --out /tmp/q36-sampled-mtp-prompt.csv
```

Native correctness gate (isolated, not the live service):

```bash
build-sampled-mtp/strata-q36 --model "$MODEL" \
  --max-context 250000 --batch 4 --kv-pool-tokens 500000 \
  --kv int8 --mtp 2 --batch-mtp --chunk 512 \
  --kv-checkpoint-mib 1024 --kv-checkpoint-interval 1024 \
  --mtp-sampled --mtp-sampled-test /tmp/q36-sampled-mtp-prompt.csv
```

Expected: `SAMPLED_MTP_TEST PASS`, a positive accepted/offered count and `max_common_logit_abs` / `max_continuation_logit_abs` values printed. The gate requires exact sampled tokens, continuation tokens and RNG state; floating-point logits are compared with an explicit reported tolerance, not claimed byte-identical unless `*_bytes=1`.

## HTTP A/B benchmark

Start an independent Orin service, not the deployed one:

```bash
python3 serve/server.py --engine strata --model "$MODEL" \
  --max-context 250000 --batch 4 --kv-pool-tokens 500000 \
  --kv int8 --mtp 2 --batch-mtp --chunk 512 \
  --kv-checkpoint-mib 1024 --kv-checkpoint-interval 1024 \
  --mtp-sampled \
  --host 127.0.0.1 --port 8095 --api-monitor
```

Run short and longer same-prompt arms. The script alternates on/off, uses seed 1234, temperature 1.0, top_p 0.95, top_k 20 and presence_penalty 1.5, and reports native `predicted_ms`, native tok/s and `draft_n`/`draft_n_accepted`.

```bash
python3 tools/q36_sampled_mtp_http_bench.py \
  --base-url http://127.0.0.1:8095 \
  --prompt-text 'Explain, step by step, how an Orin keeps a long Qwen conversation in mixed GPU and host memory.' \
  --max-tokens 96 --reps 5 --warmup 1 --json-out /tmp/q36-sampled-mtp-short.json

printf 'Context: %s\n' "$(python3 -c 'print(("The Qwen runtime records stable system facts and measured cache behavior. " * 220).strip())')" > /tmp/q36-long-prompt.txt
python3 tools/q36_sampled_mtp_http_bench.py \
  --base-url http://127.0.0.1:8095 \
  --prompt-file /tmp/q36-long-prompt.txt \
  --max-tokens 192 --reps 3 --warmup 1 --repeat 4 \
  --json-out /tmp/q36-sampled-mtp-long.json
```

If the server requires a key:

```bash
export STRATA_API_KEY=...
python3 tools/q36_sampled_mtp_http_bench.py --base-url http://127.0.0.1:8095 ...
```

Use the same seed and output cap in both arms; median native decode speed is the controlled comparison. Wall time includes queueing, so native `predicted_per_second` and draft counters are the primary figures.

## Measured context and open questions

The short-prompt figures already measured on the deployed Orin configuration were 30.9 tok/s for sampled requests (no drafts offered), 47.1 tok/s for greedy plus MTP2, and 73.4–84.2% greedy draft acceptance. Those are **different sampling configurations, not a controlled MTP speedup measurement.** Sampled equality acceptance can be much lower than greedy acceptance. The benchmark above must establish:

- sampled proposal acceptance under temperature/top-p/top_k/presence-penalty;
- native decode throughput on/off at short and long context;
- whether extra verify-window work is repaid on the user's model/workload;
- numerical logits from batched verify versus sequential decode.

Until those Orin runs are done, this feature remains experimental and should not be deployed merely because the correctness code is present.
