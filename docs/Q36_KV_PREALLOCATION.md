# Qwen3.6 shared KV preallocation

`--kv-preallocate` allocates an engine-owned physical VMM page bank before expert
placement and readiness. Without it, unified KV retains lazy physical allocation.
The feature is opt-in and uses the same logical pool, layouts and numerical kernels.

Example full-GPU profile:

```text
--serve --max-context 262144 --batch 4 --kv-pool-tokens 524288
--kv int8 --mtp 2 --batch-mtp --mtp-sampled --chunk 512
--kv-checkpoint-mib 1024 --kv-checkpoint-interval 1024 --kv-preallocate
```

The context limit includes prompt and generated tokens. The shared pool is an
aggregate reservation ceiling, not a fixed partition: four slots do not imply four
independent full-context physical pools. Admission protects active owners and
pinned donors; idle caches may be evicted. A request must fit both its context and
the available aggregate logical reservation.

## Physical ownership and capacity

Slots retain stable virtual addresses and borrow bank-owned pages. Reset, rewind
and eviction unmap slot pages and return them to the bank. Remapped pages are
initialized before publication. Strict borrowed mappings never allocate fallback
pages or release bank-owned handles. Logical reservations, mapped bytes and held
physical bytes are accounted separately.

The capacity bound uses actual per-head code/scale layouts, VMM granularity,
retained growth/padding and admissible distributions across slots. It is not simply
raw bytes per token times the pool size. Main and loaded MTP attention, recurrent
state, convolution histories, last hidden state and configured checkpoint payloads
are charged. No fixed per-slot token partition is introduced.

For the example geometry and 2 MiB granularity, the conservative bank has 3168
attention pages, 244 state pages and 512 checkpoint pages: 3924 pages, or
8,229,224,448 bytes. Other pool sizes/granularities are recomputed at startup.
`KV_PREALLOCATE`, `KVBANK`, `KVPOOL` and `KVSESSION` expose their separate counters.

The bank does not cover the whole process: weights, expert residency, shared
attention/prefill scratch, library workspaces, graphs, encoder memory and headroom
are additional requirements. Strict startup refuses incompatible placement or
insufficient memory instead of silently reducing the requested profile. Additional
batch graph shapes retain their own headroom checks. Preallocation removes physical
bank allocation after readiness; mapping, initialization, copies and other scratch
work can still occur.

The supported strict inference path is full-GPU Qwen3.6 with INT8 KV, four slots,
MTP-2/batch-MTP, explicit chunking and warmup. Expert-placement overrides and other
unsupported combinations are rejected. Vision adds its own bounded allocations;
see [Q36_VISION_PREALLOCATION.md](Q36_VISION_PREALLOCATION.md).

## Validation

CPU/fake-driver tests exercise capacity, ownership, rollback and allocation failures.
`kv_preallocation_cuda_test` checks real mappings, initialized reused pages, slot
isolation and allocation counters. `tools/q36_kv_preallocation_native_gate.py`
distinguishes unsupported host-addressing positives from mandatory device tests.
Model gates additionally check fixed physical allocation/attempt/release counters
after readiness. Neither host-only tests nor a hardware skip qualify GPU behavior.
