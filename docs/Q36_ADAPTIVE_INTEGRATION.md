# Qwen3.6 adaptive checkpoint retention

`--kv-checkpoint-policy adaptive` opts into an early safety anchor, up to two
learned rendered-token divergence targets and a recent checkpoint. `fixed` remains
the source default. The global budget, four-payload ceiling, capacity calculator,
donor authorization and numerical processing paths do not change.

Learning requires an actual divergence and replay because suitable past state was
missing. An append-only continuation does not train the policy. Learning chooses
which legal completed prompt callbacks to retain; it cannot create unrecorded
past state or authorize restoration beyond a true token/image prefix.

Example policy options:

```text
--kv-checkpoint-mib 1024 --kv-checkpoint-interval 1024
--kv-checkpoint-policy adaptive
```

Use `tools/q36_adaptive_integration_fixture.py --model MODEL --out NEW_DIRECTORY`
for public synthetic rendered-token cases. `--byte-test` checks tooling only and
is not model/tokenizer evidence. Ordinary continuation fixtures end in a user
message so they can preserve rendered prefixes; the generator still audits actual
tokenization and fails if no genuine append exists. Cases cover earlier-user
rewrites, system-context rewrites, ordinary continuations and interleaved idle chats.

`tools/q36_adaptive_integration_gate.py` compares baseline, fixed and adaptive
retention with each arm's own independently recomputed schedule. Reference smoke
and explicit conv/GDN/hidden-state corruption controls establish that the oracle
rejects damaged restores. Warmup and steady measurements remain separate.
Different schedules do not require byte-identical logits across policies.
See [Q36_CHECKPOINT_SCHEDULE.md](Q36_CHECKPOINT_SCHEDULE.md).

Adaptive retention reduces replay only when retained legal states precede recurring
divergences. Rewrites before the earliest usable anchor can still force a full
prefill. Policy selection is not a frontend-specific normalization rule.
