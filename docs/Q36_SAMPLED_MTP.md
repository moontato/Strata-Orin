# Experimental Qwen3.6 sampled MTP

Default off: `--mtp-sampled` opts into solo text sampled verification of greedy MTP
proposals. A request can disable/re-enable it with native `sampled_mtp=0|1` or HTTP
`strata_tune: {"sampled_mtp": 0|1}`. Without the engine option, the request key has
no effect. Greedy MTP, batched requests, images and prefix checkpoints keep their
existing paths. Hybrid expert placement falls back to sequential sampled decode.
No deployed configuration, branch, OWUI setting or service is changed by this code.

## Target sampling and publication

Run the existing MTP proposal chain and main-model window `[x, d0, ... dD-1]`.
For each row, sample the main model's target logits with the existing sampler,
request RNG and token history. Accept a proposal only when that target sample
is equal to it. On the first mismatch stop consuming rows and retain the target
sample; if all match, retain the final bonus-row sample. No rejection/residual
sampling distribution is needed: every published token is itself a target draw.
This gives sequential output/RNG equivalence **for identical target logits**;
the native gate must separately measure window-versus-sequential numerical error.

The shared production request controller `q36_sampled_request()` is used by Serve,
the native gate and the CPU mock-backend regression. It:

- Tracks a pending mismatch/bonus sample explicitly. It is published once, without
  fetching row-zero logits or drawing the RNG again.
- Tracks the logit row after the actual committed prefix. `Engine::host_logits()`
  reads row zero, which is not the final accepted row of a verify window.
- Verifies provisionally, then publishes proposals one at a time. Only `x` and
  the published proposal prefix are passed to `spec_commit()`. A stop after
  verification or during publication cannot leave hidden drafts in the live state.
- Restores history and RNG to the last published decision when cancellation drops
  provisional decisions or a pending target sample. No later rows are sampled
  after mismatch, EOS, the output cap or an observed cancellation.
- Processes even the terminal published token in this opt-in path. Thus recurrent
  state, convolution history, hidden seed, positions and logical KV progress end
  at exactly the published prefix. This costs a final processing step compared
  with the old sequential path, which can leave its last output unprocessed.

The original `spec_window()` kernels write speculative main/MTP KV (including
INT8 scales) beyond the kept prefix, just as greedy speculation does. Only the
published prefix is logically retained; unused tails are overwritten on later
writes. `spec_commit()` retains recurrence/hidden state for that prefix. No
maximum-size eager allocation, model mirror or persistent checkpoint is added.
Prefix checkpoints remain prompt-only and retain their generation/epoch/backing
validation. Exact-prefix reuse is still tried first.

## Local validation (not model qualification)

CPU checks on x86_64:

- `q36_sampled_mtp_test`: production verifier/sampler against sequential supplied
  logits; penalties, top-k/top-p/min-p, row limits, EOS, cancellation and RNG.
- `q36_sampled_request_test`: the production request controller with a deferred
  mock backend, not another isolated verifier test. Zero/one/two accepted drafts,
  pending mismatch and bonus tokens, full output and RNG equivalence over 80 seeds,
  output caps including an all-accepted terminal window, EOS, fallback, cancellation
  after verification and during multi-draft publication. Retained backend state,
  history and live tokens must all equal the published prefix.
- Both sampled tests pass strict warnings and non-PIE ASan/UBSan.
- Unchanged `q36_prefix_checkpoint_test` and `unified_kv_test` pass, including
  strict-warning and non-PIE ASan/UBSan builds.
- CPU-only CMake Release build and focused CTest: 4/4 passed (`STRATA_NATIVE_EXPERTS=OFF`
  for these standalone CPU targets; this does not qualify native expert inference).
- Focused Python sampling-key, Qwen config and HTTP-harness checks: 13/13 passed;
  Python syntax checks also pass. The HTTP test uses a tiny mock server only; it
  tests model discovery and identical on/off request bodies, **not** inference or performance.
- Negative controls that force pending-token resampling or commit unpublished
  proposals both fail the request-controller regression as expected.
- Native checkpoint-gate source and greedy `spec_window()` / `spec_commit()` /
  `spec_step()` are byte-for-byte unchanged from `e46985e0`.

CUDA compile/link, kernel-machine-code comparison, real-model logits, Orin
checkpoint regression, cancellation/state validation and throughput are **pending**.
There is no usable GPU or `nvcc` frontend on this host. CPU checks are not evidence
of GPU inference correctness. No tolerance has been relaxed.

## Build on Orin (no running-service change)

Use the experimental checkout and a separate build directory. `LLAMA_SRC` must
be the existing pinned ggml checkout, `3cf03257f219afbe7334045ff7c6a06ac68c627d`.

```bash
cd /path/to/feature/q36-sampled-mtp
cmake -S . -B build-sampled-mtp -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DSTRATA_GGML_DIR="$LLAMA_SRC" \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_HIP=OFF -DSTRATA_ENABLE_SYCL=OFF \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON \
  -DSTRATA_PREFILL_MMQ=OFF -DSTRATA_BUILD_TESTS=ON
cmake --build build-sampled-mtp --target strata strata-q36 strata_kq \
  unified_kv_test q36_prefix_checkpoint_test q36_sampled_mtp_test q36_sampled_request_test -j4
ctest --test-dir build-sampled-mtp \
  -R '^(unified_kv_test|q36_prefix_checkpoint_test|q36_sampled_mtp_test|q36_sampled_request_test)$' \
  --output-on-failure
```

## Operator-controlled Orin test window — not deployment

**Do not run another model beside the deployed instance.** The following is a
future maintenance-window procedure requiring the operator's approval. Nothing
here has been executed by the assistant. Pause OWUI/other clients first. Identify
the actual deployed unit and whether it is a user or system unit. Leave its files,
settings, executable, enablement and checkpoint branch at `9597a51a` untouched.
Build and generate fixtures before stopping it.

Set absolute paths to the experimental checkout, actual model first shard and
existing exported tokenizer directory (containing `vocab.json`, `merges.txt`,
`token_type.json` and the correct `chat_template.jinja`). Use the same model and
sampling settings as the baseline; do not lower penalties to inflate acceptance.

```bash
export TEST_REPO=/absolute/path/to/feature/q36-sampled-mtp
export MODEL=/absolute/path/to/model-first-shard.gguf
export TOKENIZER=/absolute/path/to/existing/tokenizer
export TEST_DIR="$(mktemp -d)"
export TEST_CONFIG="$TEST_DIR/sampled-mtp-test.json"
cd "$TEST_REPO"
python3 tools/q36_sampled_mtp_fixture.py --model "$MODEL" \
  --out "$TEST_DIR/sampled.csv"
python3 tools/q36_prefix_checkpoint_fixture.py --model "$MODEL" \
  --out "$TEST_DIR/checkpoints.csv"
python3 - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['TEST_REPO']).resolve()
cfg = {
    'arch': 'qwen35moe', 'model_name': 'qwen3.6-sampled-mtp-test',
    'exe': str(root / 'build-sampled-mtp/strata'), 'cwd': str(root),
    'tokenizer': os.environ['TOKENIZER'], 'parallel': 4,
    'log': str(Path(os.environ['TEST_DIR']) / 'engine.log'),
    'args': ['--native', os.environ['MODEL'], '--max-context', '250000',
             '--kv-pool-tokens', '500000', '--kv', 'int8', '--mtp', '2',
             '--batch-mtp', '--chunk', '512', '--kv-checkpoint-mib', '1024',
             '--kv-checkpoint-interval', '1024', '--mtp-sampled']
}
Path(os.environ['TEST_CONFIG']).write_text(json.dumps(cfg, indent=2) + '\n')
PY
```

All engine options go in this **separate test JSON**, not on `serve/server.py`'s
command line. Its `parallel: 4` supplies `--batch 4`; the server supplies `--serve`.
No live config is edited. The benchmark discovers the actual API model ID through
`GET /v1/models` by default (or pass `--model qwen3.6-sampled-mtp-test`).

For a systemd-managed deployment, use a transient test unit so stopping it also
reaps its engine process. In **one Bash shell**, replace the service name and
choose the correct user/system commands. Review the restoration trap first:

```bash
SERVICE='REPLACE-WITH-ACTUAL-DEPLOYED-UNIT.service'
SYSTEMCTL=(systemctl --user)              # or: (sudo systemctl)
SYSTEMD_RUN=(systemd-run --user)          # or: (sudo systemd-run)
TEST_UNIT="strata-sampled-mtp-review-$$"
PYTHON="$(command -v python3)"             # interpreter with existing server deps
was_active="$("${SYSTEMCTL[@]}" is-active "$SERVICE" || true)"
[[ "$was_active" == active ]] || { echo 'Expected active deployed service; inspect manually'; exit 1; }
restore() {
  local state
  state="$("${SYSTEMCTL[@]}" show "$TEST_UNIT.service" -p LoadState --value 2>/dev/null || true)"
  if [[ "$state" != not-found ]]; then
    "${SYSTEMCTL[@]}" stop "$TEST_UNIT.service" || {
      echo 'Test cleanup failed: inspect its cgroup before manually restoring the deployed unit'
      return 1
    }
  fi
  # A successful stop reaps the transient unit's whole control group.
  "${SYSTEMCTL[@]}" start "$SERVICE"
  "${SYSTEMCTL[@]}" is-active "$SERVICE"
}
trap restore EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"${SYSTEMCTL[@]}" stop "$SERVICE" || exit 1
if "${SYSTEMCTL[@]}" is-active --quiet "$SERVICE"; then exit 1; fi
# Check process/cgroup listings and tegrastats: the deployed engine and its
# memory must be gone. If another launcher restarts it, abort and restore.
```

If deployment is not managed by systemd, use its actual stop/start procedure and
an equivalent cleanup trap; do not guess a unit or use broad `pkill` commands.
Keep clients paused throughout. Within this window, run the native gates **one
at a time** (each exits before the next model is loaded):

```bash
build-sampled-mtp/strata-q36 --model "$MODEL" --serve \
  --max-context 250000 --batch 4 --kv-pool-tokens 500000 \
  --kv int8 --mtp 2 --batch-mtp --chunk 512 \
  --kv-checkpoint-mib 1024 --kv-checkpoint-interval 1024 \
  --mtp-sampled --mtp-sampled-test "$TEST_DIR/sampled.csv" \
  > "$TEST_DIR/sampled-gate.log" 2>&1 || exit 1

build-sampled-mtp/strata-q36 --model "$MODEL" --serve \
  --max-context 250000 --batch 4 --kv-pool-tokens 500000 \
  --kv int8 --mtp 2 --batch-mtp --chunk 512 \
  --kv-checkpoint-mib 1024 --kv-checkpoint-interval 1024 \
  --kv-checkpoint-test "$TEST_DIR/checkpoints.csv" \
  > "$TEST_DIR/checkpoint-gate.log" 2>&1 || exit 1
```

Require `SAMPLED_MTP_TEST PASS` and `PREFIX_CHECKPOINT_TEST PASS` before benchmarking.
The sampled gate compares 16 tokens at a common fully processed prefix, history,
live tokens, position, RNG and eight continuation tokens. It reads logits after
the **eighth** continuation token and from the actual final verification row at
the common prefix. Acceptance and offers must both be positive; a zero-acceptance
fixture fails, rather than passing without testing acceptance. It retains the
native Serve `GEN 32 sampled_mtp=0/1` exact-output check. Both full-logit maximum
errors and byte-identity flags are printed, with unchanged `1e-4` tolerance. The
checkpoint gate still requires byte-identical full logits and MTP2 behavior.

Start only the transient experimental service (never both models):

```bash
"${SYSTEMD_RUN[@]}" --unit="$TEST_UNIT" --property=KillMode=control-group \
  --working-directory="$TEST_REPO" \
  "$PYTHON" "$TEST_REPO/serve/server.py" --engine strata --config "$TEST_CONFIG" \
  --host 127.0.0.1 --port 8095 --api-monitor || exit 1
# Wait for READY and /v1/models in the unit's journal; do not benchmark loading.
# journalctl --user -u "$TEST_UNIT.service" -f   # adapt for a system unit
```

Once ready, alternate on/off at the same prompt, seed, output cap and sampling
fields (temperature 1.0, top_p 0.95, top_k 20, presence_penalty 1.5):

```bash
python3 tools/q36_sampled_mtp_http_bench.py --base-url http://127.0.0.1:8095 \
  --max-tokens 96 --reps 5 --warmup 1 --json-out "$TEST_DIR/short.json" || exit 1
python3 -c 'print("The Qwen runtime records stable facts and measured cache behavior. " * 220)' \
  > "$TEST_DIR/long-prompt.txt"
python3 tools/q36_sampled_mtp_http_bench.py --base-url http://127.0.0.1:8095 \
  --prompt-file "$TEST_DIR/long-prompt.txt" --repeat 4 --max-tokens 192 \
  --reps 3 --warmup 1 --json-out "$TEST_DIR/long.json" || exit 1
# Leaving this shell runs the trap: stop/reap the test unit, restore the original unit.
exit
```

Verify the original service is healthy at its existing address and clients can
resume. Confirm its branch/config still match the pre-window record. Keep logs
and JSON results. If cleanup fails or a test engine survives, stop it and confirm
memory is released **before** restoring the original model; do not overlap them.
No sampled acceptance, speedup, numerical parity or long-context claim is made
until these Orin runs are recorded. Greedy acceptance is not sampled acceptance.
