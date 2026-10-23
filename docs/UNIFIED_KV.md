# Unified KV on the Jetson v0.1.45 branch

## Allocation contract (implementation notes)

The source inventory at `ec8149d1` has exactly two GGUF families (`model_family.hpp`):
`qwen35moe` (Qwen3.6, separate engine) and `qwen4exp` (Flash-Next, Coder,
Swift and other compatible fine-tunes/weight formats). Weight quantization does
not select a different cache adapter.

* Qwen3.6: FP32 K and V, `[kv_head][context][head_dim]`, including the MTP
  attention block; per-session GDN convolution/recurrence and last hidden state.
* Flash-Next: `[page][kv_head][page_cell][head_dim]`, FP16, INT8 with FP16
  scales, rotated Q4_0, or hybrid K8V4. QSA indexer, GDN and PLE state are
  separate. Streamed caches have an authoritative mapped host copy, GPU slots
  and a residency map; MTP has a separate cache and may use a ring.

The opt-in common manager owns logical admission, byte ceilings, deterministic
idle-only LRU, session generations, residency and eviction events. Backends own
layout, synchronization, invalidation and physical mappings. All callbacks are
synchronous and must not re-enter the manager. A lease protects the entire request,
including prefill, speculative writes, batch windows and copies. Only a completed
backend synchronization permits the transition back to idle. Failure to synchronize
is fatal, not permission to recycle memory.

CUDA virtual ranges are reserved once. Physical pages cover the union of the
layout's required prefixes, rounded to the device-reported allocation granularity.
In particular, Qwen3.6 needs a prefix for **every head**, not one flat prefix.
Stable addresses preserve graph arguments. Growth happens outside capture, after
previous launches have completed. Reclaimed pages are unmapped and released, not
written to disk. The original allocator remains the default.

A token ceiling is independent of a physical byte ceiling. Rounded physical
commitment and speculative padding count in bytes. Active request admission reserves
logical space for its requested output, but physical commitment grows only with
writes. Idle caches release unused output reservations. CUDA/system free memory on
Orin describes one physical pool, never two capacities to add together.

PR #1614 was inspected as a policy/interface reference only. Its current body is
based on 0.2.0-beta; none of its code or newer engine changes is imported. In
particular, unified-pool exhaustion here must not become a successful truncated
answer.

Further configuration, protocol and validation details are added with the adapters.
