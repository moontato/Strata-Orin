#!/usr/bin/env bash
# Operator-invoked only, inside the service-stop/transient-unit procedure in docs.
set -euo pipefail
[[ ${STRATA_KV_REVIEW_WINDOW:-} == 1 ]] || { echo 'Use docs/Q36_KV_PREALLOCATION.md service-stop window first' >&2; exit 2; }
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
: "${MODEL:?Set MODEL to the actual GGUF}"
: "${RESULTS:?Set RESULTS to an empty review-results directory}"
BUILD=${BUILD:-"$ROOT/build-kv-preallocation"}
PYTHON=${PYTHON:-python3}
mkdir -p "$RESULTS"
for name in strata strata-q36; do
  if pgrep -x "$name" >/dev/null; then echo "Another $name process exists: abort, never run two models" >&2; exit 1; fi
done
: "${STRATA_GGML_DIR:?Set the pinned llama.cpp source directory}"
[[ $(git -C "$STRATA_GGML_DIR" rev-parse HEAD) == 3cf03257f219afbe7334045ff7c6a06ac68c627d ]]
"$PYTHON" tools/q36_sampled_mtp_fixture.py --model "$MODEL" --out "$RESULTS/sampled.csv"
"$PYTHON" tools/q36_prefix_checkpoint_fixture.py --model "$MODEL" --out "$RESULTS/checkpoints.csv"
# Every native test must exit 0. Only explicit, rollback-checked unsupported
# host-positive cases are partial coverage; device/ring checks remain mandatory.
"$PYTHON" tools/q36_kv_preallocation_native_gate.py --build "$BUILD" --results "$RESULTS"
base=(--model "$MODEL" --serve --max-context 262144 --batch 4 --kv-pool-tokens 500000
      --kv int8 --mtp 2 --batch-mtp --mtp-sampled --chunk 512
      --kv-checkpoint-mib 1024 --kv-checkpoint-interval 1024)
for mode in off on; do
  extra=(); [[ $mode == off ]] || extra=(--kv-preallocate)
  for gate in sampled checkpoints; do
    if [[ $gate == sampled ]]; then option=--mtp-sampled-test; marker='SAMPLED_MTP_TEST PASS'; else option=--kv-checkpoint-test; marker='PREFIX_CHECKPOINT_TEST PASS'; fi
    "$BUILD/strata-q36" "${base[@]}" "${extra[@]}" "$option" "$RESULTS/$gate.csv" \
      --test-state-dump "$RESULTS/$mode-$gate.bin" > "$RESULTS/$mode-$gate.log" 2>&1
    grep -q "$marker" "$RESULTS/$mode-$gate.log"
    grep -q 'every routed expert in VRAM' "$RESULTS/$mode-$gate.log"
    if grep -Eq 'hybrid experts:|given back' "$RESULTS/$mode-$gate.log"; then
      echo 'Expert placement/packed copies changed: not a valid A/B' >&2; exit 1
    fi
    if [[ $mode == on ]]; then grep -q 'KV_PREALLOCATION_MODEL PASS' "$RESULTS/$mode-$gate.log"; fi
  done
done
cmp "$RESULTS/off-sampled.bin" "$RESULTS/on-sampled.bin"
cmp "$RESULTS/off-checkpoints.bin" "$RESULTS/on-checkpoints.bin"
echo 'KV_PREALLOCATION_AB PASS (all logits, outputs, histories, RNG and checkpoint probes byte-identical)'
# Failure gates must not get to a ready native service or silently fall back.
if "$BUILD/strata-q36" "${base[@]}" --kv-preallocate --kv-pool-mib 1 > "$RESULTS/startup-failure.log" 2>&1; then
  echo 'Expected physical-byte-limit startup refusal' >&2; exit 1
fi
grep -q -- '--kv-preallocate: physical bank needs' "$RESULTS/startup-failure.log"
if grep -q '^READY ' "$RESULTS/startup-failure.log"; then echo 'Failure gate unexpectedly became ready' >&2; exit 1; fi
if "$BUILD/strata-q36" "${base[@]}" --kv-preallocate --kv fp32 > "$RESULTS/unsupported.log" 2>&1; then
  echo 'Expected unsupported-format refusal' >&2; exit 1
fi
grep -q -- '--kv-preallocate supports text full-GPU Qwen3.6 only' "$RESULTS/unsupported.log"
"$PYTHON" tools/q36_kv_preallocation_bench.py --exe "$BUILD/strata-q36" --model "$MODEL" \
  --out "$RESULTS/bench" --pairs "${PAIRS:-5}" --lengths 4096,32768 --gen 96
# A separate profiled run, never mixed into the throughput comparison above.
if [[ ${PROFILE:-0} == 1 ]]; then
  nsys profile --trace=cuda,osrt --sample=none --output="$RESULTS/allocator-trace" \
    "$PYTHON" tools/q36_kv_preallocation_bench.py --exe "$BUILD/strata-q36" --model "$MODEL" \
    --out "$RESULTS/profiled" --pairs 1 --lengths 4096 --gen 32
  nsys stats --report cuda_api_sum --format csv "$RESULTS/allocator-trace.nsys-rep" \
    > "$RESULTS/cuda-api-summary.csv"
fi
# Optional, but record explicitly if unavailable. Run inside this same stopped-service window.
if command -v compute-sanitizer >/dev/null; then
  compute-sanitizer --tool memcheck --error-exitcode 99 "$BUILD/kv_preallocation_cuda_test" \
    > "$RESULTS/compute-sanitizer.log" 2>&1
else
  echo 'PENDING: compute-sanitizer unavailable' | tee "$RESULTS/compute-sanitizer.log"
fi
echo 'KV_PREALLOCATION_ORIN PASS (review logs; no deployment)'
