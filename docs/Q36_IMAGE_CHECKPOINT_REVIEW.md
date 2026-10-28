# Qwen3.6 image-checkpoint review — resolved, Orin qualification still pending

Reviewed published support tip **72457256fa0367660a8e0849bb19e234c6fc1711** and
implementation **37911119e64aa97b837b2f6a920dbdc8cc3930bd** (parent diagnostics
**765068db267f8cd8f846f8ef45c9cc94f90075d5**). Origin publication and the
coding-server CPU suite are recorded in
[the migration handoff](Q36_MULTIMODAL_CACHE_HANDOFF.md).

Two review blockers were confirmed, then **fixed and regression-tested on the CPU
host** in the two correction commits on `feature/q36-image-checkpoints` —
3958d620 (`observe restored state after selection and hard-bound the gate`) and
f19e2294 (`bind the gate verdict to the run mode under the supervisor`). The flag remains
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
