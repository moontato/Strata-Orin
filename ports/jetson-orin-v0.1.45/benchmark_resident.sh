#!/usr/bin/env bash
# Run ON ORIN after the CUDA/correctness gates. Logs are evidence, not a residency proof alone.
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo "usage: $0 /absolute/build-orin/strata-q36 /absolute/model.gguf /absolute/result-dir" >&2
  exit 2
fi
[[ $(uname -m) == aarch64 ]] || { echo "Refusing non-native-ARM64 benchmark" >&2; exit 2; }
bin=$(realpath "$1")
model=$(realpath "$2")
mkdir -p "$3"
out=$(realpath "$3")
[[ -x $bin && -f $model && -f $(dirname "$bin")/libstrata_kq.so ]]
# These release knobs use presence, not value: even '=0' is not a default run.
for name in STRATA_Q36_OLD_KERNELS STRATA_Q36_VERIFY STRATA_Q36_NO_MMQ \
            STRATA_Q36_NO_EXT STRATA_Q36_BLAS_F32 STRATA_Q36_STREAM_ALL \
            STRATA_Q36_FIXED_SLOTS STRATA_Q36_CHUNK; do
  if [[ -v $name ]]; then
    echo "Refusing modified/diagnostic path: unset $name first" >&2
    exit 2
  fi
done
[[ ${STRATA_Q8_PACKED:-1} != 0 && ${STRATA_MMQ_KQ:-1} != 0 && ${CUDA_LAUNCH_BLOCKING:-0} == 0 ]]
[[ ! -e $out/pp4096.txt ]] || { echo "Use a new result directory (do not overwrite a run)" >&2; exit 2; }
{
  date -u
  uname -a
  git -C "$(dirname "$bin")/.." rev-parse HEAD
  sha256sum "$bin" "$(dirname "$bin")/libstrata_kq.so" "$model"
  printf 'CUDA_VISIBLE_DEVICES=%s\nSTRATA_UMA_HEADROOM_GIB=%s\n' "${CUDA_VISIBLE_DEVICES:-0}" "${STRATA_UMA_HEADROOM_GIB:-6}"
  free -h
  ldd "$bin"
} > "$out/metadata.txt"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export STRATA_UMA_HEADROOM_GIB=${STRATA_UMA_HEADROOM_GIB:-6}
# Default graphs, optimized kernels, MMQ and automatic chunk selection. MTP=0
# isolates ordinary prefill/decode; MTP benchmarks are a separate experiment.
for pp in 512 2048 4096; do
  "$bin" --model "$model" --max-context 8192 --mtp 0 \
    --bench-pp "$pp" --bench-tg 128 2>&1 | tee "$out/pp${pp}.txt"
  grep -q 'every routed expert in VRAM' "$out/pp${pp}.txt" || {
    echo "REJECTED: full routed-expert residency not reported" >&2; exit 1;
  }
  grep -q 'prompt experts on MMQ (int8)' "$out/pp${pp}.txt" || {
    echo "REJECTED: optimized MMQ prefill not active" >&2; exit 1;
  }
  if grep -Ei 'hybrid experts:|slot replacements|packed Q8_0 decode copies .*given back|MMQ.*(unavailable|failed)' "$out/pp${pp}.txt"; then
    echo "REJECTED: hybrid placement, optimized-copy give-back, or MMQ failure" >&2
    exit 1
  fi
done
python3 - "$out" <<'PY'
import pathlib, re, statistics, sys
root = pathlib.Path(sys.argv[1])
for pp in (512, 2048, 4096):
    text = (root / f'pp{pp}.txt').read_text()
    values = re.findall(rf'^bench [0-2]: pp{pp} ([0-9.]+) tok/s, tg128 ([0-9.]+) tok/s', text, re.M)
    if len(values) != 3:
        raise SystemExit(f'Missing engine repetitions for pp{pp}')
    print(f'pp{pp}: median {statistics.median(float(p) for p, _ in values):.1f} tok/s; '
          f'tg128 median {statistics.median(float(t) for _, t in values):.1f} tok/s')
print('Log gate accepted; CUDA trace, correctness and physical-memory/thermal evidence are still required.')
PY
