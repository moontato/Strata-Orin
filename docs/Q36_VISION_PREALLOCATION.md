# Qwen3.6 vision with strict KV preallocation

Enable vision through the server's vision configuration and `--vision`. A matching
encoder/projector produces image embeddings; the text-model GGUF alone does not
supply that encoder. Strict preallocation additionally bounds shared image staging,
M-RoPE positions and row tables before readiness.

```text
--kv-preallocate --vision --kv-vision-max-tokens 1024 --chunk 512
```

`--kv-vision-max-tokens` is the combined image-row cap per request, not per image or
per slot. Image token expansion counts against the request's context and shared
pool admission. Physical KV cells and rotary coordinates are distinct: positional
metadata does not change the KV capacity calculation or token counter.

At width 2048 and cap 1024, FP32 image rows use 8 MiB; the three context-sized
M-RoPE planes use 3 MiB at context 262144. Device and pinned chunk maps and eager
batch-MTP scratch are separate allocations outside the KV bank. `KVVISION` reports
those categories. The resident encoder has its own weights/workspace charge.

The parser checks payload size, width/grid, finite values, exact image pad runs,
keys and aggregate bounds before GPU mutation. Over-cap input is rejected rather
than growing strict staging. A staging lease prevents overwriting shared buffers
while a request uses them. Session-local retained rotary metadata survives later
requests; the shared views are not themselves persistent cache state. Cleanup
requires quiescence before recycling addresses or freeing captured-graph resources.

Native `--kv-vision-test FILE` covers image/text transitions, mixed active slots,
cancellation inside an image and in a text tail, and sampled text after images.
Image-history requests use their existing sampled-MTP exclusion; later eligible
text requests may still offer drafts. Ordinary text checkpointing and image-aware
checkpointing have distinct eligibility rules. See
[Q36_IMAGE_CHECKPOINTS.md](Q36_IMAGE_CHECKPOINTS.md).

`tools/q36_vision_preallocation_gate.py` compares one baseline and two candidate
allocation modes with a resident encoder, exact matching-schedule state/logit
checks and bank invariants. `tools/q36_vision_preallocation_window.sh` provides a
manager-controlled test window with explicit paths and prior-state restoration;
its tests use command shims. Hardware qualification must pass the real gates and
verify actual device capability. CPU test scaffolds do not establish GPU behavior.
