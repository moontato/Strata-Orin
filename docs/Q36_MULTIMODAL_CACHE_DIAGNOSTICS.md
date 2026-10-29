# Qwen3.6 multimodal cache diagnostics

Native `--kv-cache-diagnostics` and Python `STRATA_Q36_CACHE_DIAGNOSTICS=1` are
independent opt-ins, both off by default. They observe matching/routing decisions;
they do not normalize prompts, modify cache authorization or change numerical work.

Compare actual rendered token prefixes and image identities before diagnosing a
miss. Frontends can rewrite system context, tool results, assistant reasoning or
image placement across turns. Same chat ID or visually similar text does not prove
an unchanged token prefix. Character offsets are not token offsets.

Native diagnostics expose candidate session, prefix length, image compatibility,
eligibility and final reuse/checkpoint resolution. Python diagnostics bind request
preparation and route observations without substituting them for engine evidence.
A session ID alone is not a request boundary: repeated requests and concurrent
admissions can reuse a slot. Do not associate a later restore with an earlier request
based only on that shared slot.

Public fixture tools produce exact-prefix, rendered-history rewrite, system-prefix
rewrite and changed-image controls. The bounded native off/on diagnostic gate checks
that diagnostics do not change schedules, state, logits or emitted tokens; it is not
an independent image-reference oracle. Image checkpoints have a separate independent
restoration test: [Q36_IMAGE_CHECKPOINTS.md](Q36_IMAGE_CHECKPOINTS.md).

Use a dedicated build and fresh results under `STRATA_Q36_STAGE`. Real model gates
require an explicit isolation opt-in and refuse another detected model/server.
Arrange and verify any service isolation outside the worker. The generic managed
window script accepts `TEST`, `STAGE`, `PYTHON`, model/fixture paths and confirmed
manager scope explicitly. Host command-shim tests exercise restoration without
operating real services.
