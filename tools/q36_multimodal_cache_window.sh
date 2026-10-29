#!/usr/bin/env bash
# Isolated GPU validation with prior-service-state restoration.
# Dedicated shell; both named units must use the SAME confirmed manager scope.
set -euo pipefail
: "${MM_CACHE_STOP_APPROVED:?operator approval required after review}"
test "$MM_CACHE_STOP_APPROVED" = 1
: "${SCOPE:?confirmed system OR user manager scope for BOTH services}"
: "${HEALTH:?existing Strata health URL; do not guess/change configuration}"
: "${MODEL:?existing actual GGUF first shard}"
: "${FIXTURES:?already CPU-verified fixtures under staging}"
: "${MM_CACHE_ENGINE:?isolated test build executable; never the live runtime}"
: "${STAGE:?Set the test staging directory}"
TEST=${TEST:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON=${PYTHON:-python3}
ENGINE="$MM_CACHE_ENGINE"
SERVICE=strata.service
LLAMA=llama-server.service
case "$SCOPE" in
 system) CTL=(sudo -n systemctl); RUN=(sudo -n systemd-run --uid="$(id -u)" --gid="$(id -g)") ;;
 user) CTL=(systemctl --user); RUN=(systemd-run --user) ;;
 *) echo 'Unconfirmed manager scope'; exit 1 ;;
esac
cd "$TEST"
test "$(git rev-parse --show-toplevel)" = "$TEST"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)-$$
mkdir -p "$STAGE/mm-window-$STAMP" "$STAGE/mm-tmp-$STAMP"
RECORD="$STAGE/mm-window-$STAMP"; export TMPDIR="$STAGE/mm-tmp-$STAMP"
RESULTS="$STAGE/mm-results-$STAMP"
# These CPU checks MUST finish before the first stop command.
STRATA_Q36_STAGE="$STAGE" "$PYTHON" tools/q36_multimodal_cache_gate.py --preflight --engine "$ENGINE" --model "$MODEL" --fixtures "$FIXTURES" --out "$RESULTS" \
 2>&1 | tee "$RECORD/preflight.log"
test -f /sys/fs/cgroup/cgroup.controllers
for name in "$SERVICE" "$LLAMA"; do
 "${CTL[@]}" show "$name" -p LoadState -p ActiveState -p SubState -p KillMode -p ControlGroup -p MainPID -p Result > "$RECORD/$name.before.txt"
done
test "$("${CTL[@]}" is-active "$SERVICE")" = active
test "$("${CTL[@]}" show "$SERVICE" -p KillMode --value)" = control-group
llama_load=$("${CTL[@]}" show "$LLAMA" -p LoadState --value)
llama_state=absent
if [ "$llama_load" = loaded ]; then
 llama_state=$("${CTL[@]}" show "$LLAMA" -p ActiveState --value)
 case "$llama_state" in active|inactive) ;; *) echo 'Ambiguous llama state; abort BEFORE stop'; exit 1 ;; esac
 if [ "$llama_state" = active ]; then test "$("${CTL[@]}" show "$LLAMA" -p KillMode --value)" = control-group; fi
elif [ "$llama_load" != not-found ]; then exit 1; fi
health() {
 local attempt
 for attempt in $(seq 1 90); do
  if curl --fail --silent --show-error --max-time 5 "$HEALTH" > "$RECORD/health.json" &&
     "$PYTHON" -c 'import json,sys;d=json.load(open(sys.argv[1]));sys.exit(not(d.get("service")=="strata" and d.get("status")=="ok" and d.get("loaded") is True and d.get("images") is True))' "$RECORD/health.json"; then return 0; fi
  sleep 2
 done
 return 1
}
health; cp "$RECORD/health.json" "$RECORD/health.before.json"
printf 'strata=active\nllama=%s\n' "$llama_state" > "$RECORD/prior-states.txt"
UNIT="strata-mm-cache-review-$STAMP"; CGRECORD="$RECORD/cgroup.txt"; launched=0
no_models() { "$PYTHON" -c 'from tools.q36_multimodal_cache_gate import no_models;no_models()'; }
restore() {
 local load rc=0
 load=$("${CTL[@]}" show "$UNIT.service" -p LoadState --value 2>/dev/null || true)
 if [ -z "$load" ]; then echo 'Cannot inspect test unit; MANUAL RECOVERY, no restart'; return 1; fi
 if [ "$load" != not-found ]; then "${CTL[@]}" stop "$UNIT.service" || return 1; fi
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
    assert events['populated']=='0', 'TEST DESCENDANTS SURVIVE; NO PRODUCTION RESTART'
PY
 fi
 # Drain named production units too before the final global model inspection.
 "${CTL[@]}" stop "$SERVICE" || return 1
 if [ "$llama_load" = loaded ]; then "${CTL[@]}" stop "$LLAMA" || return 1; fi
 no_models || { echo 'Model/encoder or inspection error survives: MANUAL RECOVERY, no restart'; return 1; }
 "${CTL[@]}" start "$SERVICE" || rc=1
 health || rc=1
 if [ "$llama_state" = active ]; then
  "${CTL[@]}" start "$LLAMA" || rc=1
  test "$("${CTL[@]}" is-active "$LLAMA")" = active || rc=1
 elif [ "$llama_load" = loaded ]; then
  test "$("${CTL[@]}" show "$LLAMA" -p ActiveState --value)" = inactive || rc=1
 fi
 health || rc=1
 cp "$RECORD/health.json" "$RECORD/health.after.json" || rc=1
 "${CTL[@]}" show "$SERVICE" -p LoadState -p ActiveState -p SubState -p KillMode -p ControlGroup -p MainPID -p Result > "$RECORD/$SERVICE.after.txt" || rc=1
 "${CTL[@]}" show "$LLAMA" -p LoadState -p ActiveState -p SubState -p KillMode -p ControlGroup -p MainPID -p Result > "$RECORD/$LLAMA.after.txt" || rc=1
 return "$rc"
}
# Guaranteed cleanup is installed BEFORE stopping either production service.
trap 'rc=$?; trap - EXIT; trap "" INT TERM; restore || rc=1; exit "$rc"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"${CTL[@]}" stop "$SERVICE"
if [ "$llama_state" = active ]; then "${CTL[@]}" stop "$LLAMA"; fi
no_models # any unmanaged encoder/model or inspection failure aborts, never pkill
launched=1
"${RUN[@]}" --unit="$UNIT" --wait --pipe --collect --property=KillMode=control-group \
 --property=RuntimeMaxSec=300s --property=TimeoutStopSec=20s --working-directory="$TEST" \
 "$PYTHON" -c 'import os,sys;from pathlib import Path;Path(sys.argv[1]).write_text(Path("/proc/self/cgroup").read_text());os.execvp(sys.argv[2],sys.argv[2:])' \
 "$CGRECORD" /usr/bin/env STRATA_MM_CACHE_REVIEW_WINDOW=1 STRATA_Q36_STAGE="$STAGE" \
 LD_LIBRARY_PATH="$(dirname "$ENGINE")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
 "$PYTHON" tools/q36_multimodal_cache_gate.py --engine "$ENGINE" --model "$MODEL" --fixtures "$FIXTURES" --out "$RESULTS" \
 2>&1 | tee "$RECORD/worker.log"
# EXIT performs full test-cgroup reaping + exact prior-state restoration + loaded
# vision health checks even after native failure/timeout. SIGKILL/power loss needs
# manual recovery; never restart blindly if cgroup inspection/reaping failed.
