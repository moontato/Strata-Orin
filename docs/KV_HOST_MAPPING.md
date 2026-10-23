# Flash-Next incremental host mapping: bounded finding

`cudaHostGetDevicePointer` failure and a successful, non-identity alias are different cases. The adapter now reports the API error name/code/description, CPU base, chunk, host address, returned alias, expected flat address, neighboring alias relationship, and CUDA capabilities. `STRATA_KV_HOST_TRACE=1` also prints successful registrations.

CUDA explicitly guarantees GPU access through the registered CPU pointer when `cudaDevAttrCanUseHostPointerForRegisteredMem` is nonzero. On that device the adapter publishes only the CPU address, even if CUDA returns a different alias. It never mixes the two aliases. Without that guarantee, every registration/re-registration must return the CPU address itself; otherwise the adapter rolls back and fails explicitly. Query failures are not positive capabilities.

This preserves incremental registration and reclamation; it does not reserve physical storage for the maximum context. Qwen3.6 GPU VMM and Flash-Next kernels are unchanged by this correction.

A focused host-only test with mocked CUDA covered capability 0/1, matching/nonmatching aliases, an actual GetDevicePointer error, rollback and incremental preservation. This is not an Orin runtime pass. The old error alone does not reveal the actual Orin pointers or capability value: rerun with tracing to obtain them.

## If the Orin capability is zero

Independent registrations have no requested device-address parameter. Their aliases cannot be concatenated by assuming a constant offset or by retrying registration until addresses happen to line up. `KvHostPools` is a flat device-pointer layout used by append, prefill mirror, refill and ring-restore operations; replacing its base with the first alias would leave later chunks incorrect and invalidate graph arguments.

Keep the explicit failure. Further Flash-Next work is deferred, not silently replaced with eager pinning, GPU-resident history or disabled streaming. Possible future designs, **not implemented here**:

* A separate capability-checked CPU-backed CUDA host/HOST_NUMA VMM allocator, with independently verified CPU/GPU access and per-buffer granularity accounting. Device VMM support alone does not establish host-VMM support.
* Segmented pinned host backing with an explicit stable device pointer table and adapted host-mirror/refill/staging operations. Existing attention over GPU resident pages could remain unchanged, but this is a real streaming-backend change, not an alias workaround.

Whole-capacity registration violates dynamic commitment. Re-registering a growing prefix can change the alias and requires comprehensive graph/pointer invalidation; it is not this narrow fix.

Reference: [CUDA 13.2 cudaHostGetDevicePointer / cudaHostRegister](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-runtime-api/group__CUDART__MEMORY.html). On host-page-table platforms registration can populate without pinning; qualification there also needs explicit residency/pinning analysis. No new such platform claim is made.
