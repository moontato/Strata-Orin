# Qwen3.6 image-aware prefix checkpoints

`--kv-image-checkpoints` is opt-in. It extends prefix restoration to eligible
image histories with strict preallocation, a checkpoint budget, `--vision`,
`--chunk 512` and an explicit combined image cap no larger than 1024 rows.
The flag remains off by default.

```text
--serve --max-context 262144 --batch 4 --kv-pool-tokens 524288
--kv int8 --mtp 2 --batch-mtp --mtp-sampled --chunk 512 --kv-preallocate
--vision --kv-vision-max-tokens 1024
--kv-checkpoint-mib 1024 --kv-checkpoint-interval 1024
--kv-checkpoint-policy adaptive --kv-image-checkpoints
```

## Identity and authorization

Exact complete-state matches remain first. A checkpoint is considered only when
no exact donor can serve the request. Actual rendered-token LCP is mandatory;
image identity can shorten that prefix but never extend it. Complete images
before a restore boundary must match actual embedding bytes, width, row count,
grid, token-cell placement and rotary start. An advisory key or hash alone grants
no permission, and a boundary inside an image is invalid.

A payload retains bounded CPU proof of image bytes/layout, consumed end, rotary
delta and readiness alongside the existing packed recurrence/conv/hidden snapshot.
Main and MTP KV codes/scales remain in the session planes. Donor generation,
epoch, backing and idle eligibility must match; a donor pin is held through
admission and copying. Shared incoming staging views are not captured or cloned.

Restore reconstructs complete-prefix image metadata and CPU/device rotary state
before publication. Incoming rows and tables for the new request are uploaded
before suffix processing. Unseeded or partially consumed image state does not
become an eligible checkpoint. Cancellation inside an image invalidates the owner;
valid earlier states may survive cancellation outside the image.

## Retention boundaries

Only existing completed prompt callbacks can capture. No chunk is split and no
numerical call shape is changed for checkpoint placement. A target inside an image
maps to a legal completed boundary or is deferred. The four-payload ceiling and
configured physical bank charge remain unchanged. Early divergence before the
first legal retained state can legitimately require replay from zero.

## Validation

The native image-checkpoint gate independently recomputes restored and final
persistent state under the same schedule, compares full logits exactly, checks
finite values and bank invariants, and requires a corrupted-state control to fail.
It exercises rewritten histories, identity/placement changes, cancellation and
cross-slot transfer. It is not just enabled/disabled output equality.

The separate [serial HTTP smoke](Q36_IMAGE_HTTP_SMOKE.md) exercises actual image
parts and server history handling. Native mixed-slot coverage does not establish
concurrent HTTP qualification, and a bounded synthetic case is not a general
frontend speedup benchmark. Other models, backend/device combinations and long
aggregate-pool-pressure workloads require their own validation.
