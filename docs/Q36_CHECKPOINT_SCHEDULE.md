# Independent schedule-matched checkpoint oracle (developer gate)

Built on vision-qualified `3f3bc4e85e1d44b3501f0dcb8cd73471632753cf`.
This changes no numerical kernel, sampler, production setting or default policy.
CUDA/Orin compilation and model execution are **pending**, not established by the
local CPU tests. The isolated projection investigation is closed absent new,
contradictory evidence; this is a restoration gate, not another matrix study.

## Native implementation

`strata-q36 --kv-schedule-test FILE` uses real `Engine::process()` in reserved
slot 3. FILE contains `# case NAME`, optional `# phase warmup|steady`, and
`SLOT CSV` lines (slots 0/1/2). Each row submits BGEN with one emitted token,
leaving the prompt state idle, without processing that emitted token. This is
interleaved idle-chat replay, **not concurrent decoding or sampled-MTP testing**.
Existing separate native sampled and vision gates remain required.

Instrumentation is inactive unless this test is selected. A host-only schedule
book records each original declared process-call input, suffix flag, consumed
length and actual callback boundaries. Capturing a checkpoint stores only that
metadata alongside its record. Restore/copy inherits the producer recipe; reset
and invalidation discard it. No extra device payload or change to the four-bank
capacity calculation. The original producer input includes an unconsumed tail:
truncating its declared length would run the head at a different point. Chunk
callbacks are NOT converted to separate process calls.

After each tested request, the native reference resets slot 3 to empty, zeros its
recurrence/convolution/hlast through the ordinary reset path, and recomputes all
consumed tokens from zero. It receives ONLY original token IDs and the schedule:
**no checkpoint restore, copy_session, tensor dump loading or tested KV/state is
used as reference input**. The tested sessions are pinned during reference
admission; only the model weights and engine workspace are shared. Every expected
callback/position is checked. The real process implementation supplies ordinary
singleton versus prompt-suffix behavior, head-at-declared-call-end, chunking and
MTP-prefill semantics, including cancellation at a producer checkpoint boundary.
This is independent tensor-state recomputation, not an independent numerical
kernel implementation or a fresh-process determinism claim.

The tested restored state is saved to HOST before its suffix, then compared
against recomputation at the corresponding producer boundary. This catches
corrupted checkpoints even if the suffix later attenuates the error. Full prompt
state and logits/pick are compared again at the final boundary. Every main KV
cell, every initialized MTP KV cell (FP32 or INT8 codes and FP16 scales), GDN,
convolution and hlast is compared by exact bytes. Ordinary singleton calls do not
seed MTP; their undefined cells are excluded explicitly by recipe, never silently
treated as initialized state. Positions and original consumed IDs must agree.
FP32 state/logits are also finite checked. There is no tolerance knob.

`SCHEDULE_FIRST` identifies comparison, category, layer, head, cell-range start,
byte offset and actual/reference bytes. All state/logit/pick results are printed
before final-boundary failure. A restored-boundary failure stops immediately.
`SCHEDULE_CALL` exposes declared/take lengths, head/MTP/suffix flags and boundaries.
A second, one-call-from-empty comparison is labelled `different_schedule`; its
byte differences remain visible but are not mistaken for restoration errors.
No tolerance is relaxed and no harmlessness conclusion follows from that result.
Optional `--test-state-dump` stores original schedule IDs/metadata and full
logits/picks, not reference tensor inputs;
**do not demand cross-policy dump equality when their schedules differ**.

`--kv-schedule-corrupt conv|gdn|hlast` changes one finite word of the selected
checkpoint payload immediately before its first restore. It MUST exit nonzero
with both the corruption marker and `SCHEDULE_FAILURE comparison=restored`, plus
`SCHEDULE_FIRST comparison=restored category=...`. No restore reached, an unrelated
failure, crash, or a PASS marker is not successful negative-control coverage.
Never enable this developer flag in serve. Unknown controls/test combinations
are rejected. It adds no new serving API.

`tools/q36_checkpoint_schedule.py --fixture FILE --out audit.json` audits actual
saved rendered token prefixes per case/slot without loading a model. In the old
fixture turn 2 was a rewrite; only turn 3 extended all prior rendered IDs. Neither
case names nor an appended message proves token-prefix continuation.

## Measurements and scope

`SCHEDULE_MEASURE` reports replayed/reused tokens, bank checkpoint-copy bytes/us,
Serve prefill ms and BGEN-to-first-token admission latency ms, split by declared
warmup/steady phase. The reference and host snapshots are outside these timers.
Checkpoint snapshots inside the tested request ARE timed: diagnostic latency is
not production serving performance. Physical allocation/attempt counters and bank
bytes must remain unchanged after readiness; reset must leave zero mappings.
Startup engine warmup stays enabled and its own log must be preserved. No speedup
is claimed. An adaptive-vs-fixed difference needs the later staged Orin gate.

## Local versus pending

CPU metadata tests cover partial producer calls, original declared length/head,
chunk boundaries, original IDs, same-slot/cross-slot recipe inheritance,
ordinary-singleton/suffix MTP coverage and the original 16384/29/22 call recipe.
A second CPU test executes the actual native oracle include against a small
shape-sensitive Engine/Serve and CUDA-IO stand-in, exercising positive matching
and visible different schedules plus three payload-corruption controls. This
compiles and runs the **gate**, NOT the real Engine/Serve CUDA translation unit.
The ordinary allocator/preallocation, vision staging/leases and sampled-MTP CPU
tests are separate. The export logs identify exactly which were run. No CUDA,
HIP, SYCL or Orin model validation is claimed, and this is not ready for upstream
review/deployment until applicable backend builds and hardware gates pass.
