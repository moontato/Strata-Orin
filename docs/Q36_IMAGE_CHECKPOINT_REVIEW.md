# Qwen3.6 image-checkpoint review — resolved, Orin qualification still pending

Reviewed published support tip **72457256fa0367660a8e0849bb19e234c6fc1711** and
implementation **37911119e64aa97b837b2f6a920dbdc8cc3930bd** (parent diagnostics
**765068db267f8cd8f846f8ef45c9cc94f90075d5**). Origin publication and the
coding-server CPU suite are recorded in
[the migration handoff](Q36_MULTIMODAL_CACHE_HANDOFF.md).

Four review blockers and one follow-up teardown defect were confirmed, then **fixed and
regression-tested on the CPU host** in the correction commits on
`feature/q36-image-checkpoints` — R1/R2 in
3958d620 (`observe restored state after selection and hard-bound the gate`) and
f19e2294 (`bind the gate verdict to the run mode under the supervisor`), R3/R4
(supervisor teardown on SIGINT/SIGTERM/SIGHUP, and a cleanup allowance spent twice) in
5d793938, and R5 (cleanup restarted after the worker was already reaped) in
025e13df. The flag remains
**off by default**. Nothing here is CUDA, GPU-output, model or Orin qualification.

## R1 — restored-state observation could read another slot's device rotary delta (resolved)

Audited both `schedule_restored` producers:

- `Engine::checkpoint_restore` — already calls `select(dst)` and restores the
  destination's CPU+device rotary state before observing, so in-place observation is
  correct.
- `Engine::copy_session` — copied another session's CPU fields while `cur` stayed on
  an unrelated slot. The device rotary offset (`d_delta`) is ONE scalar bound to the
  current slot, so an observation there mixed a destination CPU field with an
  unrelated device field. The image oracle's `image_schedule_state` reads both.

Fix (instrumentation placement only; no production selection, caching, kernel,
schedule or sampling change): `src/qwen35/q36_restore_observation.inc` defines an
explicit boundary contract —

```text
copy state -> request_restore_snapshot(dst) -> select(dst) -> fire_restore_snapshot()
           -> upload the incoming rows/table -> process the suffix
```

`copy_session` now only *requests* the observation; `Serve::request_body` fires it
immediately after `E.select(s)` and before `E.set_images(...)` and `E.process(...)`.
A pending snapshot is discarded at the start of every request. Firing while `dst` is
not current throws and is counted (`restore_snapshot_refused`). With no hook installed
(a normal engine run) nothing is requested, counted or thrown.

Evidence, host-only (no CUDA compile, no GPU), copied under
`docs/validation/q36-image-checkpoints/` (`restore-observation-correction.txt`,
`gate-scenarios-correction.txt`, `negative-*-correction.txt`,
`cpu-checks-correction.txt`):

- `python3 tools/q36_restore_observation_harness.py` extracts the **real**
  `copy_session` tail from `src/qwen35/q36_engine.cu`, compiles it with the **real**
  boundary primitives and inert CUDA stubs, and runs the reported geometry
  (donor `-992`, destination empty, current slot `0`):

  ```text
  Q36_RESTORE_OBSERVATION PASS cross_slot_deferred=1 observed_cpu=-992 observed_device=-992
   previous_current_device=0 exactly_once=1 noncurrent_refused=1 same_slot_in_place=1
   counters_balanced=1 no_hook_inert=1
  Q36_RESTORE_OBSERVATION CHECK PASS placement=1 host_harness=0 NOT_GPU_EVIDENCE
  ```

  Coverage: cross-slot complete-state copy (deferred, observed once, one slot's
  CPU+device pair), non-current refusal, same-slot checkpoint restoration observed
  in place, producer/consumer counter balance, and inertness with no hook.
- `python3 tools/check_q36_restore_observation_negative.py` — three mutations, each
  detected (baseline clean, every mutant fails):

  ```text
  RESTORE_OBSERVATION BASELINE PASS placement=1 harness=0 exit=0
  RESTORE_OBSERVATION_NEGATIVE PASS observe_at_copy_time placement_exit=1 harness_exit=-6
  RESTORE_OBSERVATION_NEGATIVE PASS fire_before_select placement_exit=1
  RESTORE_OBSERVATION_NEGATIVE PASS guard_removed exit=1
  ```

  `observe_at_copy_time` re-inserts the pre-fix hook call inside `copy_session`;
  `fire_before_select` moves the deferred observation before the destination
  selection; `guard_removed` deletes the current-slot guard. Reverting either fix
  therefore fails a regression.
- Native gate (pending CUDA run): the image oracle now asserts at **every**
  observation that `e.cur==dst`, that engine position/session length equal the
  restored boundary, that `refused==0`, and that observations exactly match the
  producers (`fires == deferred+immediate`, at most one of each per request). Every
  observation — not just the last — is compared against the independently replayed
  matching schedule, and the fixture must exercise at least one cross-slot
  complete-state observation (`IMAGE_CHECKPOINT_OBSERVATIONS total=... deferred=...
  immediate=... refused=0 balanced=1`). Restored-state coverage was not weakened and
  no expected-failure count was changed.

## R2 — the advertised whole-run budget was not enforced (resolved)

`tools/q36_image_checkpoint_gate.py` is now a **supervisor + worker** pair:

- The supervisor starts the worker in its **own process group**
  (`start_new_session=True`) with a shared absolute `CLOCK_MONOTONIC` deadline, and on
  expiry sends `SIGKILL` to that **whole group** — covering synchronous preflight
  hashing, native execution, log reading, comparisons and final reporting.
- Native gate arms are started **inside the worker's group** so a group kill also
  reaches their descendants. Only supervisor-created groups are ever signalled.
- **Execution budget** (default 300s, explicit maximum 420s) and **cleanup
  allowance** (`--cleanup-seconds`, default 20s, maximum 60s) are separate: reaping
  and the group-emptiness proof get the allowance and are recorded as
  `owned_group_empty` in `watchdog.json`.
- The worker keeps cooperative `budget.check(...)` guards between and inside every
  phase (binary hash, provenance, per-file fixture hashing, per-row audit, arm start,
  arm poll, arm reap, log read, measurement parse, post-run re-hash, verdict).
- Only the supervisor prints a verdict. Its pure `verdict()` refuses: killed worker,
  completion at/after the deadline, nonzero worker exit, non-empty owned group,
  non-PASS worker report, or an incomplete arm set. A late success is never accepted,
  and a selftest never relays a production marker.
- The accepted report status is bound to the mode: a `--preflight` run can only report
  `PASS_PREFLIGHT_ONLY` (it prints `IMAGE_CHECKPOINT_PREFLIGHT PASS
  model_not_started=1`, never the numerical-gate marker), and a numerical-gate report
  is never accepted for a preflight run. A real (non-selftest) preflight still refuses
  to run unless `ROOT` is the reviewed multimodal-cache worktree with results under the
  Orin staging path, so host runs cannot masquerade as Orin preflight.

Deterministic short-budget scenarios (fake children only; no engine, model, encoder
or service; `--selftest`, throwaway staging):

```text
python3 tools/q36_image_checkpoint_gate.py --selftest <normal|slow_preflight|hung_child|slow_postprocess|late_success> ...
normal            -> exit 0, killed=0, deadline_crossed=0, owned_group_empty=1, labelled SELFTEST PASS only
slow_preflight    -> exit 1, killed=1, deadline_crossed=1, owned_group_empty=1
hung_child        -> exit 1, killed=1, owned child AND its marked descendant gone (owned_group_empty=1)
slow_postprocess  -> exit 1, killed=1, deadline_crossed=1
late_success      -> exit 1, killed=1; no ORIN_GATE PASS marker reaches stdout
```

These are asserted in `tools/test_q36_image_checkpoints.py`, together with a unit
check that `verdict()` rejects a late/killed/bad-report run, a slow
**log-read** case (worker-side `TimeoutError`, i.e. a late success is not accepted),
and a case proving an **unowned** process started outside the gate is never touched.

Mutation controls (`python3 tools/check_q36_gate_watchdog_negative.py`) re-enable the
defect and are detected:

```text
GATE_WATCHDOG_NEGATIVE PASS no_group_kill leaked_owned_pids=1 exit=1 group_empty=False
GATE_WATCHDOG_NEGATIVE PASS late_success_accepted
GATE_WATCHDOG_NEGATIVE PASS supervisor_bypass leaked_owned_pids=1 exit=1
```

`no_group_kill` removes the supervisor's group kills, `supervisor_bypass` runs the
worker without a supervisor (cooperative checks only), and `late_success_accepted`
removes the lateness refusal. Both group mutations leave the marked owned descendant
alive, which the real supervisor reaps.

## R3 — an interrupted supervisor left its worker running (resolved)

Reproduced on this host: sending SIGTERM to the supervisor during the fake
`slow_preflight` scenario terminated it immediately (default disposition), leaving the
worker group alive and writing **no** `watchdog.json`; SIGINT reached an
`except BaseException: … raise` path that killed the group but escaped before the
waiting, group-emptiness verification and the failure record.

Fix: SIGINT, SIGTERM and SIGHUP are now handled by one recorder inside `supervise()`.
Nothing is raised from a handler, so no path can escape the teardown: whatever ends the
run (deadline, signal or an unexpected supervisor fault) goes through the same
`teardown()` — kill the owned group, wait for the worker, kill stragglers, reap the
leader, prove the group empty — and the failure is always written to `watchdog.json`
(`interrupted_signals`, `passed=false`, `cleanup_complete`) with a nonzero exit. The
worker's exit status is read with `os.waitid(..., WNOWAIT)` so the group is only ever
signalled while its leader is still unreaped (no recycled-PGID hazard), and the leader
is reaped before the final emptiness proof (a zombie leader no longer makes the group
look alive). An unexpected supervisor fault now also tears the group down and records a
failure before re-raising.

This does **not** weaken the enclosing-cgroup requirement: the supervisor signals only
the session it created, never a name or a foreign pid, and `watchdog.json` repeats
`operator_cgroup_required=true`. The operator-owned cgroup procedure below is unchanged
and still mandatory.

Evidence (fake workers only, no engine/model/encoder/service):

```text
GATE_WATCHDOG_NEGATIVE PASS signal_teardown_removed supervisor_exit=-15 surviving_worker_pids=1 watchdog_record=False
```

(copied as `docs/validation/q36-image-checkpoints/gate-signals-teardown.txt`, which
also prints the per-signal `watchdog.json` fields and the surviving-process checks).
That mutation removes the handler installation, and the reported defect reappears: the
worker group survives and no record is written. The regression tests deliver each of
SIGINT, SIGTERM and SIGHUP to a real supervisor after its worker has started and require
nonzero exit, `passed=false`, the signal named in `interrupted_signals`,
`owned_group_empty=true`, `worker_reaped=true`, no PASS marker, no surviving owned
worker/child/descendant, and a foreign process left untouched.

## R4 — the cleanup allowance was spent twice (resolved)

`supervise()` waited up to `cleanup_seconds` for the worker and then started a **fresh**
`cleanup_seconds` deadline for group cleanup. With a simulated stuck child (execution
budget0.1s, cleanup allowance1s) the run took ≈2.11s — about twice the advertised
bound.

Fix: one absolute cleanup deadline, computed once when teardown begins, bounds waiting,
reaping and the emptiness proof. `teardown()` returns the worker status, the group-kill
status, `group_empty` and `reaped`; success is impossible unless the worker was reaped
**and** the owned group is empty, and `verdict()` now refuses an unreaped worker
separately from a live group.

Evidence (deterministic fake worker whose reaping and group cleanup are blocked,
allowance1.0s, `tools/check_q36_gate_watchdog_negative.py`):

```text
GATE_WATCHDOG BASELINE PASS one_cleanup_allowance_s=1.00 (limit 1.60)
GATE_WATCHDOG_NEGATIVE PASS cleanup_rearmed teardown_s=2.00 (baseline 1.00, one allowance 1.00)
GATE_WATCHDOG_NEGATIVE PASS no_group_kill leaked_owned_pids=1 exit=1 group_empty=False
GATE_WATCHDOG_NEGATIVE PASS supervisor_bypass leaked_owned_pids=1 exit=1
```

`cleanup_rearmed` re-arms the allowance for the verification phase and reproduces the
2× shape; the shipped code stays inside one allowance (measured1.00s for a1.0s
allowance with both reaping and cleanup blocked). The regressions also prove that
neither an unreaped worker nor a non-empty group can be accepted whatever the worker
reported, and that a real `hung_child` run leaves no marked owned process.

Checkpoint logic, numerical comparisons and the existing timeout/mode protections
(execution budget, deadline sharing, preflight/gate status binding, corruption arm,
observation boundary) are unchanged by R3/R4.

## R5 — cleanup was restarted after the worker had already been reaped (resolved)

Reproduced against d9de98d5 with the real supervisor and a fake worker: normal teardown
finished and reaped the worker leader, the `watchdog.json` write then failed (ENOSPC
class), and `supervise()` fell into its fault handler, which called `teardown()` **again
with a fresh `cleanup_seconds` deadline**. That second pass group-signalled the worker's
PGID although its leader had already been reaped — at that point the PGID may belong to
an unrelated process — and the run could spend a second cleanup allowance.

Fix: cleanup progress is now a persistent `CleanupState` per run. Its absolute deadline
is created once and reused by the normal, signal and exceptional paths; `teardown()` is
idempotent (`state.complete` returns the recorded result unchanged); and the ownership
guard `if state.reaped: return state.group_status` means a group whose leader is gone is
only ever **probed**, never signalled, even mid-cleanup. A result that cannot be written
is reported to stderr (`stage=record action=failed`, `action=no_restart
cleanup_state_final=1`), printed as `FAIL result_record_failed_<type>` and exits nonzero
without re-running cleanup; the worker's own PASS can never become a supervisor PASS.

Fault injection through the real supervisor (`--fault record|crash`, selftest-only, fake
children; `gate-fault-injection-postteardown.txt`):

```text
FAULT scenario=normal fault=record exit=1 elapsed_s=0.56 watchdog_record=False owned_probes_after=[]
  IMAGE_CHECKPOINT_GATE_SELFTEST FAIL result_record_failed_NotADirectoryError
FAULT scenario=normal fault=crash exit=1 elapsed_s=0.57 watchdog_record=True
  watchdog: passed=false worker_reaped=true owned_group_empty=true cleanup_complete=true
FAULT scenario=hung_child fault=record exit=1 elapsed_s=3.11 watchdog_record=False owned_probes_after=[]
```

Mutation controls (now seven, `tools/check_q36_gate_watchdog_negative.py`):

```text
GATE_WATCHDOG BASELINE PASS one_cleanup_allowance_s=1.00 (limit 1.60) reentry_s=0.00 signals_after_reap=0 probes_after_reap=1 result_unchanged=True
GATE_WATCHDOG_NEGATIVE PASS cleanup_rearmed teardown_s=2.00 (baseline 1.00, one allowance 1.00)
GATE_WATCHDOG_NEGATIVE PASS cleanup_state_rearmed reentry_s=1.00 (baseline 0.00, one allowance 1.00)
GATE_WATCHDOG_NEGATIVE PASS post_reap_signal signals_after_reap=2 probes=1 (real supervisor: 0 signals)
```

`post_reap_signal` removes the ownership guard and signals twice; `cleanup_state_rearmed`
bypasses the persistent state (the shape of the old fault path) and spends a second
allowance on re-entry. The shipped code signals nothing after a reap and re-entry costs
0.00s. Regressions cover: write failure after completed cleanup, an exception after
reaping (safety-net record still written with `passed=false`), no group signal after
reaping, no renewed allowance, `--fault` refused outside selftest, and the R3/R4 signal,
timeout and unrelated-process protections unchanged (31 Python tests in
python-postteardown.txt). No C++, CUDA or numerical file changed in 025e13df, so the full
CPU suite was deliberately not re-run; see the validation addendum.

## Still true from the first review

The reviewed authorization path (actual FP32 bytes + grid + placement + rotary
semantics, mandatory token LCP, legal complete-image boundaries, contiguous ready
prefix, generation/epoch/backing/owner/pin checks, publish-after-sync, metadata
reconstruction on image restore, default-off bypass) is unchanged in behaviour.
Source observations are still not GPU correctness results.

The native fixture uses explicit `BGENI` slots and one generated token. It does not
qualify engine-selected `GENI`, four-active-slot decode/prefill interleaving, real GPU
eviction pressure, multi-image or later-added-image transitions, or long/full-context
behaviour. `cross_slot` still counts `cp && slot==1`, not donor provenance; the new
observation counters make the complete-state path explicit but do not replace a
provenance assertion when coverage is extended.

## Bounded Orin qualification — preparation, NOT execution

No Orin path is mounted on this host; nothing below was run. See
[Q36_IMAGE_CHECKPOINTS.md](Q36_IMAGE_CHECKPOINTS.md) for the exact commands.

1. CPU checks on the delivered commit are done (`PASS_CPU_ONLY` at f19e2294, raw
   outputs in `docs/validation/q36-image-checkpoints/*-correction.txt`); re-run them
   on Orin alongside the CUDA build.
2. Read-only inventory of the isolated multimodal-cache worktree; protect the three
   f934 follow-up files; preserve the actual diagnostic HEAD/ref/bundle; apply only
   the reviewed post-765068db range plus this fix. Never touch the live
   checkpoint-policy worktree or the preallocation runtime dependencies.
3. CUDA 13.2 / SM87 build with the pinned llama.cpp
   `3cf03257f219afbe7334045ff7c6a06ac68c627d`, CMake provenance naming ONLY the
   multimodal-cache source; CPU fixture preparation and `--preflight` under the
   supervisor; no encoder launch.
4. A **separately approved** operator isolation window: pause clients, record both
   services' prior active/inactive states and vision capability, stop/reap production
   with manager-owned cgroup proof (never `/proc/PID/task/PID/children`), own a
   transient test cgroup, and prove it empty (`cgroup.events populated=0`) before
   restoring BOTH prior states and verifying health plus vision attestation.
5. Numerical gate: corruption arm must fail specifically at
   `IMAGE_CHECKPOINT_FAILURE comparison=restored`; each normal arm must pass its own
   independently replayed exact restored/final state and full logits, including the
   new observation-boundary and counter assertions; positive steady rendered
   after-image rewrite reuse required; legitimate zeros stay zero; allocation,
   staging, cancellation, pin and reset invariants retained.
6. Report raw logs, exit records, per-turn measurements and all hashes. Treat timings
   as synthetic fixture data only. Keep `--kv-image-checkpoints` off by default;
   default-off CUDA parity, other backends, broader lifecycle/pressure coverage and
   actual OWUI divergence/savings remain separate pending work.
