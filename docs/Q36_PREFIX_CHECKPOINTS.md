# Qwen3.6 prefix checkpoints

Frontends may rewrite earlier messages between requests. The engine first tries
exact live-prefix reuse. On a miss, it can restore the longest retained checkpoint
at or before the actual rendered-token divergence and process the remaining suffix.
Character offsets and message-role boundaries do not authorize cache reuse.

```text
--serve --kv-pool-tokens 524288
--kv-checkpoint-mib 1024 --kv-checkpoint-interval 1024
```

Checkpoint allocation is off when no budget is configured. Payloads contain GDN
recurrence, convolution histories and the last hidden seed. Main/MTP attention KV
prefixes remain in their existing planes, including INT8 scales. Positions, token
lineage, generation, epoch and physical backing must agree before restoration.
Scratch, activations, provisional drafts and logits are not persistent checkpoint
state; suffix processing regenerates the required values.

Captures occur only at eligible existing prompt callbacks. They do not split
chunks or alter numerical schedules. Actively decoding, evicted, faulted or stale
owners are excluded. Cross-slot donors are pinned before admission; both copies
are charged until transfer completes. Rewind/reset/eviction invalidate or rebind
metadata through the original pool lifecycle.

The per-slot payload count is bounded by the global budget and a ceiling of four.
The geometry-derived payload is rounded to the VMM granularity. One payload keeps
a safety anchor; larger budgets retain middle anchors and a rolling recent point.
With fixed retention, middle anchors follow interval multiples. Adaptive retention
is separately opt-in: [Q36_ADAPTIVE_INTEGRATION.md](Q36_ADAPTIVE_INTEGRATION.md).

Text checkpoint logic does not authorize image restoration by itself. Use the
image-aware option only with the required strict vision profile:
[Q36_IMAGE_CHECKPOINTS.md](Q36_IMAGE_CHECKPOINTS.md).

Validate restored state and final logits against independently recomputed matching
schedules using [Q36_CHECKPOINT_SCHEDULE.md](Q36_CHECKPOINT_SCHEDULE.md). A one-call
fresh prefill can have a different floating-point schedule and is a separate
comparison. Do not relax exact restoration checks to hide those differences.
