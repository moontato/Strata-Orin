# Independent schedule-matched checkpoint oracle

Developer validation uses real Engine processing under the recorded producer/suffix
schedule. It is separate from serving and adds no default numerical behavior.

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


`tools/q36_checkpoint_schedule.py --fixture FILE --out audit.json` classifies saved
rendered-token prefixes per case and slot without starting a model. Case names
alone never establish append-only behavior. Native gates, CPU scaffolds and serial
HTTP checks have different scopes; report them separately.
