# Qwen3.6 development and validation

Install the pinned development dependencies in a virtual environment; runtime
installation remains separate:

```bash
python3 -m venv .venv-q36-dev
. .venv-q36-dev/bin/activate
python -m pip install -r requirements-q36-dev.txt
python tools/q36_cpu_checks.py --negative-controls --sanitizers
```

The CPU runner builds the focused ownership, capacity, checkpoint and sampled-MTP
CTest targets and runs Python tests and mutation controls. It does not start an
encoder, model or service. Generated logs and build directories stay untracked.
Optional `--source-preservation` compares immutable numerical methods against full
Git history. A shallow checkout can omit those historical source comparisons;
that is not a numerical or hardware pass.

For CUDA, configure the normal project with the intended toolkit, architecture and
pinned ggml source. Keep the matching K-quant shared library beside its engine.
Do not infer hardware correctness from a CPU build or a successful link. Native
INT8, preallocation, checkpoint, sampled-MTP and vision gates have separate scope.
Use [Q36_CHECKPOINT_SCHEDULE.md](Q36_CHECKPOINT_SCHEDULE.md) for the independent
matching-schedule oracle and [Q36_IMAGE_HTTP_SMOKE.md](Q36_IMAGE_HTTP_SMOKE.md) for
serial API coverage.

## Isolated model-validation tools

Test roots derive from the checkout, not a branch or host-specific mountpoint.
`STRATA_Q36_STAGE` selects a separate directory for test builds, fixtures and output;
the default is `logs/q36-tests` below the checkout. Use fresh result directories.
Real-model tools require their explicit `STRATA_*_REVIEW_WINDOW=1` opt-in and refuse
another running model. That opt-in is not proof that a machine is idle: inspect
actual processes and manage any service isolation deliberately. A manager-owned
cgroup remains necessary when the tool requires one. Preserve original service
states and prove test children gone before restoring them.

The tools record explicit timeouts, exact or bounded numerical acceptance, and
what was tested. Process cleanup tests must not kill unrelated processes. Tests
that need unavailable OS facilities report skips; native device gates required
by a hardware qualification must pass rather than silently skip.

## Coverage limits

Native state and logit comparisons use the same processing schedule. Different
prefill chunk/call geometries may round differently; report those differences
without turning them into restoration errors or loosening the exact oracle.
The image HTTP smoke is serial. Native mixed-slot coverage does not qualify
concurrent HTTP scheduling. Full-context/pool-pressure, model quality across KV
formats and other backends/devices require their own measurements. Never infer
these from a short functional test or an unsupported sanitizer run.
