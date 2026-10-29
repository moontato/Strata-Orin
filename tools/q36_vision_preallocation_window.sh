#!/usr/bin/env bash
set -euo pipefail
: "${VISION_STOP_APPROVED:?operator must explicitly set to 1 after review}"
test "$VISION_STOP_APPROVED" = 1
: "${SCOPE:?Set the confirmed system or user manager scope}"
SERVICE=strata.service
LLAMA=llama-server.service
: "${HEALTH:?Set the existing service health endpoint}"
case "$SCOPE" in
 system) CTL=(sudo -n systemctl); RUN=(sudo -n systemd-run --uid="$(id -u)" --gid="$(id -g)") ;;
 user) CTL=(systemctl --user); RUN=(systemd-run --user) ;;
 *) exit 1 ;;
esac
for tool in pgrep curl; do command -v "$tool" >/dev/null; done
for name in strata strata-q36 strata-vision llama-server llama-cli llama-bench; do
 rc=0; pgrep -x "$name" >/dev/null || rc=$?; ((rc<=1)) || exit 1
done
for binary in "$TEST/build-feature-vision/strata-q36" "$TEST/build-baseline-vision/strata-q36"; do
 test -x "$binary"; grep -a -q -- '--kv-vision-test' "$binary"
done
test -x "$TEST/build-vision-encoder/bin/strata-vision"; test -f "$MMPROJ"
test "$("${CTL[@]}" is-active "$SERVICE")" = active
test "$("${CTL[@]}" show "$SERVICE" -p KillMode --value)" = control-group
llama_load=$("${CTL[@]}" show "$LLAMA" -p LoadState --value)
llama_state=absent
if [ "$llama_load" = loaded ]; then
 llama_state=$("${CTL[@]}" show "$LLAMA" -p ActiveState --value)
 case "$llama_state" in active|inactive|failed) ;; *) exit 1 ;; esac
elif [ "$llama_load" != not-found ]; then exit 1; fi
if [ "$llama_state" = active ]; then
 test "$("${CTL[@]}" show "$LLAMA" -p KillMode --value)" = control-group
elif pgrep -x llama-server >/dev/null; then echo 'Unmanaged llama; abort before stop'; exit 1; fi
health() {
 local attempt
 for attempt in $(seq 1 90); do
  if curl --fail --silent --show-error --max-time 5 "$HEALTH" > "$STAGE/health-$$.json" &&
     "$PYTHON" -c 'import json,sys;d=json.load(open(sys.argv[1]));sys.exit(not(d.get("service")=="strata" and d.get("status")=="ok" and d.get("loaded") is True))' "$STAGE/health-$$.json"; then return 0; fi
  sleep 2
 done
 return 1
}
health
test -f /sys/fs/cgroup/cgroup.controllers # reviewed cgroup-v2 host only
UNIT="strata-vision-review-$$"
CGRECORD="$STAGE/test-cgroup-$$.txt"; export CGRECORD
launched=0
RESULTS="$STAGE/results-$(date -u +%Y%m%dT%H%M%SZ)-$$"
printf 'strata=active\nllama=%s\n' "$llama_state" > "$STAGE/prior-state-$$.txt"
engines_alive() {
 local name code
 for name in strata strata-q36 strata-vision llama-server llama-cli llama-bench; do
  code=0; pgrep -x "$name" >/dev/null || code=$?
  [ "$code" = 1 ] || return 0 # inspection errors also block unsafe restart
 done
 return 1
}
restore() {
 local state rc=0
 state=$("${CTL[@]}" show "$UNIT.service" -p LoadState --value 2>/dev/null || true)
 if [ -z "$state" ]; then echo 'Cannot inspect cgroup; manual recovery required'; return 1; fi
 if [ "$state" != not-found ]; then "${CTL[@]}" stop "$UNIT.service" || return 1; fi
 if [ "$launched" = 1 ]; then
  "$PYTHON" - "$CGRECORD" "$UNIT.service" <<'PY' || return 1
from pathlib import Path
import sys
lines=Path(sys.argv[1]).read_text().splitlines()
assert len(lines)==1 and lines[0].startswith('0::/')
name=lines[0][3:]
assert name.endswith('/'+sys.argv[2]) and '..' not in Path(name).parts
root=Path('/sys/fs/cgroup'+name)
if root.exists():
    events=dict(line.split() for line in (root/'cgroup.events').read_text().splitlines())
    assert events['populated']=='0', 'test descendants still alive'
PY
 fi
 "${CTL[@]}" stop "$SERVICE" || return 1
 if [ "$llama_state" = active ]; then "${CTL[@]}" stop "$LLAMA" || return 1; fi
 if engines_alive; then echo 'Test/model survives; DO NOT restart production yet'; return 1; fi
 "${CTL[@]}" start "$SERVICE" || rc=1
 health || rc=1
 if [ "$llama_state" = active ]; then
  "${CTL[@]}" start "$LLAMA" || rc=1
  test "$("${CTL[@]}" is-active "$LLAMA")" = active || rc=1
 elif "${CTL[@]}" is-active --quiet "$LLAMA"; then rc=1; fi
 health || rc=1
 return "$rc"
}
trap 'rc=$?; trap - EXIT; trap "" INT TERM; restore || rc=1; exit "$rc"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"${CTL[@]}" stop "$SERVICE"
if [ "$llama_state" = active ]; then "${CTL[@]}" stop "$LLAMA"; fi
if engines_alive; then echo 'Production not reaped'; exit 1; fi
launched=1
"${RUN[@]}" --unit="$UNIT" --wait --pipe --collect --property=KillMode=control-group \
 --working-directory="$TEST" "$PYTHON" -c \
 'import os,sys;from pathlib import Path;Path(sys.argv[1]).write_text(Path("/proc/self/cgroup").read_text());os.execvp(sys.argv[2],sys.argv[2:])' \
 "$CGRECORD" /usr/bin/env STRATA_VISION_REVIEW_WINDOW=1 STRATA_Q36_STAGE="$STAGE" \
 "$PYTHON" tools/q36_vision_preallocation_gate.py \
 --engine "$TEST/build-feature-vision/strata-q36" \
 --baseline "$TEST/build-baseline-vision/strata-q36" \
 --encoder "$TEST/build-vision-encoder/bin/strata-vision" \
 --model "$MODEL" --mmproj "$MMPROJ" --out "$RESULTS"
# Normal shell exit invokes reaping/restoration too; preserve nonzero test status.
exit
