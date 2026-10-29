# Serial image-checkpoint HTTP validation

`tools/q36_image_http_smoke.py` runs three sequential temporary loopback-server
arms: baseline, candidate with image checkpoints off, and candidate with the flag
on. Each arm uses its own config, port and logs. Production config credentials,
bind address and replica settings are not copied into temporary configs.

There are ten serial requests per arm: three audited eligible rewrite requests,
then stable image history, an earlier-user rewrite, a system rewrite, changed
pixels and text after images. Assistant content, reasoning content and tool calls
returned by the API are carried unchanged into later requests. Short natural
histories may reuse zero; positive restore is required only for the eligible,
actual-tokenizer-audited rewrite controls.

Response usage and complete metrics are checked against bounded engine-log byte
spans. Serial attribution requires one unambiguous summary with matching token
counts; restore provenance must belong to that request. A logical request that
yields and resumes during concurrent admission can emit multiple native summaries.
That requires a scheduler-aware test; this serial harness does not deduplicate
those events or claim concurrent HTTP coverage.

The whole-worker supervisor enforces one monotonic execution deadline and one
cleanup allowance. Owned groups are identified conservatively and reaped; missing
ownership is not permission to signal a recycled process. A manager-owned enclosing
cgroup is still required where the tool's cleanup contract calls for one.
Plan/preflight/selftest markers cannot qualify an actual model run.

## Inputs and modes

`--engine`, `--baseline-engine`, `--model`, `--source-config`, `--server-root`,
`--geometry` and a fresh `--out` are explicit inputs. The server root must be the
checkout, and output must stay within `STRATA_Q36_STAGE` (default logs/q36-tests).
`--plan-only` writes only a plan; `--preflight` validates bindings without starting
an encoder/model/listener. Real execution additionally requires
`STRATA_IMAGE_HTTP_REVIEW_WINDOW=1` and an independently arranged isolated window.
That flag does not stop services or prove the host is idle.

`tools/q36_image_http_preflight.py` consumes an explicit fixture-state file and
config, verifies their source/input hashes, prepares actual-GGUF HTTP geometry,
and runs the CPU plan/preflight modes. Fixture state binds source checkout/commit,
model, candidate build/hash, qualified native fixtures and diagnostic encoder
fixtures. It is generated per test, not tied to an installed service's paths or a
specific binary hash in source. `q36_image_http_geometry.py` checks tokenization,
image identity and legal after-image checkpoint opportunities against those inputs.

This HTTP test does not read GPU tensors or repeat the independent native oracle.
Its results describe that bounded serial case, not broad semantic vision quality,
all frontends or throughput under concurrent HTTP load.
