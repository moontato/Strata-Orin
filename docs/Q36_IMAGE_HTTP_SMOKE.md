# Image-aware Qwen3.6 checkpoints — bounded HTTP integration smoke (prepared, not yet run)

`tools/q36_image_http_smoke.py` drives the **HTTP layer** of the opt-in
`--kv-image-checkpoints` path: the resident GPU encoder getting a real image through
`POST /v1/chat/completions`, the assistant replies that the API actually returned being
carried into later requests, and slot/session choice and reuse read from the **engine's
own log lines**. It is a separate, smaller step than the numerical gate in
[Q36_IMAGE_CHECKPOINTS.md](Q36_IMAGE_CHECKPOINTS.md), and it runs only inside the same
kind of separately approved operator isolation window.

**Status: prepared and tested on the coding host against a loopback stand-in. No model,
encoder, GPU or service was started here. The Orin HTTP window has NOT been run.**

## What this harness proves, and what it never claims

| The harness checks | How |
| --- | --- |
| The image encoder is resident | `GET /health` must report `images=true, loaded=true, service=strata` before any request |
| A real image part reaches the encoder | every image step uploads a deterministic PNG as `image_url` base64 through the OpenAI-compatible endpoint |
| Replies are carried back unchanged | `content`, `reasoning_content` and `tool_calls` from each response are re-sent, in the same conversation, in the next request |
| Slot/session choice is recorded, not guessed | `strata serve: session N: prompt …`, `KVCACHE session=… result=…`, `KVREPLAY … reason=prefix_checkpoint` and `strata batch: slot N …` are parsed from the arm's own engine log and paired with the server's `/metrics` records by prompt size |
| A stable image history reuses | `stable_image/2` must report `reused > 0` (required) |
| Image checkpoints are actually used when the flag is on | at least one restored/prefix-checkpoint line for the arm (required, candidate-on arm only) |
| The flag is the only difference between arms | the image flag is read back from the config the arm **ran with**, and an off arm's log must contain no `IMAGE_CHECKPOINT` line (required) |
| Four clients at once | two bursts of four clients; each burst must be attributed wholly, land on four distinct engine sessions, and the second burst must reuse on every client (required) |
| Nothing leaked from the live service | each arm gets a temporary config: loopback host, temporary port, only the encoder section copied; `api_key`, live host/port, replicas and engine args of the source config are dropped |

| Reported, never required | Why |
| --- | --- |
| Rendered-token and finish divergence between arms | the deployed baseline, the candidate with the flag off and the candidate with the flag on take different prefill schedules; equality is not the goal, the gap is |
| Reuse for the earlier-user rewrite, the early system rewrite and the changed-pixels step | their expected direction is named up front (`prefix_only_reported`, `near_zero_reported`, `image_rows_zero_reported`) and the measured value is reported with it |
| Timings | per step: engine `reused`/`read`/`prompt_ms`, server `duration_s`, client elapsed |
| Per-step engine values for the eight concurrent steps | a concurrent burst is started by four threads; the server begins them in its own order, so naming which client reused what would be invented. They are accounted for **as a burst** instead |

It does **not** re-verify restored-state or logit exactness and never inspects GPU
tensors: that is the completed independent native oracle
(`IMAGE_CHECKPOINT_ORIN_GATE PASS independent=1 exact=1`, 17 measurement lines, see
[Q36_IMAGE_CHECKPOINTS.md](Q36_IMAGE_CHECKPOINTS.md)). It also does not measure actual
OWUI savings — the pass marker says `actual_OWUI_savings=UNMEASURED` on purpose.

## Bounded execution, the same shape as the reviewed gate

The supervisor and worker reuse the mechanisms of `q36_image_checkpoint_gate.py`
(`Budget`, `CleanupState`, `teardown`, `group_kill`, `group_probe`, `worker_status`):

* **One absolute window deadline** (`--window-seconds`, default 1800 s, max 3600 s) bounds
  every phase: server start, each request, log reading and reporting. The HTTP read timeout
  is the same deadline, and a worker that crosses it is `SIGKILL`ed in a process group the
  supervisor created.
* **One cleanup allowance** (`--cleanup-seconds`, default 20 s, max 60 s) covers waiting,
  group killing, reaping and the emptiness proof. It is created once and never re-armed;
  `SIGINT`/`SIGTERM`/`SIGHUP` are recorded and routed through the same single teardown.
* **Every server group this run creates is registered the moment it starts**
  (`owned-groups.jsonl`), and the supervisor reaps exactly those groups — with a
  recycled-PID guard (the recorded kernel start time must still match, or the pid is left
  alone). A worker killed mid-arm therefore cannot leave a model or encoder running.
* **Only one arm runs at a time**, each in its own session, on a loopback port in
  11500–11599; the production port 11436 is refused. The harness never names a service
  manager, OWUI or a deployment path: bringing the live services down and up stays with
  the operator and the existing manager, and `operator_cgroup_required=1` is recorded.
* Progress: the supervisor prints `HTTP_SMOKE_PROGRESS stage=window elapsed_s=… remaining_s=…`
  every 20 s, the worker prints the same cadence while requests run
  (`stage=requests done=n/15 …`) and while a server is loading (`stage=server_start`).
* A result that cannot be recorded exits nonzero, prints
  `HTTP_SMOKE_PROGRESS stage=record action=failed …` and `action=no_restart
  cleanup_state_final=1`, and never restarts cleanup or prints a pass marker.
* The pass marker is bound to the mode that ran: `PASS_BOUNDED_IMAGE_HTTP_SMOKE` (real),
  `PASS_IMAGE_HTTP_SMOKE_PREFLIGHT_ONLY` (preflight), `PASS_IMAGE_HTTP_PLAN_ONLY`
  (plan-only), `PASS_IMAGE_HTTP_SMOKE_SELFTEST` (the loopback stand-in). A preflight or
  plan-only run cannot print the real marker, and vice versa.

## Guards

A real or preflight run is refused unless `STRATA_IMAGE_HTTP_REVIEW_WINDOW=1` is set and
`STRATA_IMAGE_HTTP_ROOT` names the read-only multimodal-cache checkout; the engines, the
model, the source config, the results directory and the server checkout must live in the
reviewed checkout or under `/mnt/ssd/tmp/strata-checkpoints`, the results directory must
be new, and the server checkout may never be the staging directory. `--plan-only` needs no
window because it starts nothing.

## Commands for the Orin window (operator executes; nothing here was run on Orin)

`$TEST` is the reviewed worktree, `$BUILD/strata-q36` the qualified candidate binary from
the completed numerical gate, `$LIVE_ENGINE` the engine the deployed service runs, `$MODEL`
the model file, `$LIVE_CONFIG` a **copy** of the live config (the harness only reads the
`vision` section out of it), `$SERVER_ROOT=$TEST` so every arm uses the same reviewed
`serve/` code and only the engine binary differs.

Transfer the one new file next to the gate scripts that are already on the Orin host (it
imports their reviewed watchdog classes):

```bash
STAGE=/mnt/ssd/tmp/strata-checkpoints
SCRIPTS="$STAGE/scripts-<stamp>"          # the directory already holding q36_image_checkpoint_gate.py
cp tools/q36_image_http_smoke.py "$SCRIPTS/"
sha256sum "$SCRIPTS/q36_image_http_smoke.py"      # must equal the entry in SHA256SUMS
```

Plan inspection — writes the exact commands, temporary configs and the request table, and
starts nothing:

```bash
"$PYTHON" "$SCRIPTS/q36_image_http_smoke.py" --engine "$BUILD/strata-q36" \
  --expect-engine-sha256 eb5ef5de8f7f36de9bd25206d59ec2161a4fb5548e3f74246483a344f8544c3f \
  --baseline-engine "$LIVE_ENGINE" --model "$MODEL" --source-config "$LIVE_CONFIG" \
  --server-root "$TEST" --out "$STAGE/http-smoke-plan-<stamp>" --plan-only
```

Preflight — same guard as the window, still starts nothing; check the three configs and the
flag mapping before committing to the window:

```bash
STRATA_IMAGE_HTTP_ROOT="$TEST" STRATA_IMAGE_HTTP_REVIEW_WINDOW=1 \
"$PYTHON" "$SCRIPTS/q36_image_http_smoke.py" --engine "$BUILD/strata-q36" \
  --baseline-engine "$LIVE_ENGINE" --model "$MODEL" --source-config "$LIVE_CONFIG" \
  --server-root "$TEST" --out "$STAGE/http-smoke-preflight-<stamp>" --preflight
```

The window itself — only after the operator has paused clients, recorded both prior service
states, stopped and reaped production with manager-owned cgroup proof, and owns a transient
test cgroup proven empty before restoring:

```bash
STRATA_IMAGE_HTTP_ROOT="$TEST" STRATA_IMAGE_HTTP_REVIEW_WINDOW=1 \
"$PYTHON" "$SCRIPTS/q36_image_http_smoke.py" --engine "$BUILD/strata-q36" \
  --expect-engine-sha256 eb5ef5de8f7f36de9bd25206d59ec2161a4fb5548e3f74246483a344f8544c3f \
  --baseline-engine "$LIVE_ENGINE" --model "$MODEL" --source-config "$LIVE_CONFIG" \
  --server-root "$TEST" --out "$STAGE/http-smoke-<stamp>" \
  --window-seconds 1500 --cleanup-seconds 30 --max-tokens 24 \
  2>&1 | tee "$STAGE/http-smoke-<stamp>.log"
```

Arm order is `baseline` (deployed engine, no flag), `candidate_off` (qualified engine, no
flag), `candidate_on` (qualified engine plus `--kv-image-checkpoints`). `--arms` takes a
comma-separated subset if an arm must be re-run alone. Interrupt with `SIGINT`: the same
single teardown stops the arm, kills and reaps the group and writes `watchdog.json`.

Report afterwards: `result.json` (status, per-arm checks, the reuse table, the divergence
table), `<arm>-report.json`, `<arm>-engine.log`, `<arm>-server-stdout.txt`,
`config-<arm>.json`, `owned-groups.jsonl`, `watchdog.json`, the raw console log, the exit
records, both engine hashes and the protected-file hashes. The service must come back with
`service=strata,status=ok,loaded=true` **and** `images=true` at
`http://100.100.10.3:11436/health` before clients resume.

## Known limits of this step

* A concurrent burst is accounted for as a burst: per-client engine attribution inside a
  burst is not claimed, and the report marks those rows `attributed=false`.
* If the deployed baseline engine cannot serve through the reviewed `serve/` code, the
  baseline arm is refused and reported — using a different server checkout is not
  authorized by this harness, and a second reviewed step would be needed.
* Reuse for the rewrite and changed-pixel steps is expected to be small or zero. The
  completed numerical gate already measured the same limit on the Orin side: the
  `rendered_header_reasoning` case reused 0 and replayed 1060, while the real earlier-user
  rewrite reused 3584 and replayed 124.
* Nothing here changes the default: `--kv-image-checkpoints` stays off.

## Local validation of the harness itself (coding host, no GPU, no service)

`tools/test_q36_image_http_smoke.py` — 39 tests: the plan and its conversation semantics,
the temporary-config isolation and flag mapping, engine-log parsing and correlation, every
required/reported split, the mode-bound markers, the group attribution, the recycled-PID
guard, the ten loopback stand-in scenarios, the signal teardown, the record-failure path,
and the plan-only/preflight guards.

`tools/check_q36_image_http_smoke_negative.py` — six mutation controls, each run against
both the mutated and the committed harness: the flag reaching every arm, per-client
attribution of a concurrent burst, dropping the reuse requirement, not reaping the server
groups, an unbound pass marker, and removing the window guard. All six bite; none of the
mutations survives, and the committed harness keeps refusing.
