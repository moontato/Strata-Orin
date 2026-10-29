# Current serial HTTP qualification status

The real Orin serial HTTP smoke PASSED: three arms, ten requests each; positive
controlled after-image checkpoint restoration; cleanup and original vision service
restoration verified. Production image checkpoints remain disabled. Concurrency
and actual OWUI savings are NOT qualified by this smoke. The current continuation
instructions and exact evidence hashes are in
[Q36_IMAGE_HTTP_SERIAL_HANDOFF.md](Q36_IMAGE_HTTP_SERIAL_HANDOFF.md).
Earlier status text and execution recipes below are historical and superseded;
do not run an old preparer or repeat a model window from them.

# Historical preparation notes (superseded by the completed serial run)

The original preparation notes below describe an earlier proposed run, not the
completed serial run. Use the current handoff above for the actual ten-request
scope, results and limits. No production default or native code changed.

The old e57c7a37 transfer and its placeholder-filled commands are superseded.

## Corrections

- Fetch `/metrics?requests=all` through a proxy-free loopback client. Check all
  request counts and the multiset of prompt sizes against response usage. Metrics
  arrival order and prompt size do not establish individual request identity.
- Serial requests have bounded engine-log byte spans. A serial span must have
  exactly one solo/batch summary matching response usage. Duplicate, partial, late,
  wrong-slot or mismatched evidence fails closed. The actual run excludes concurrent
  bursts; those require a separate scheduler-aware contract.
- A restored checkpoint belongs to a serial request only when its event precedes
  that request's sole summary, names that session and matches its reused tokens.
  Read the real stderr `Qwen3.6 prefix checkpoint:` line: stdout KVCACHE/KVREPLAY
  is normally consumed internally by the Python server. Missing cache metadata
  stays unknown. No session-wide association is used by the real worker.
- The real server command receives a config **filename**, not a Python dictionary.
  Each arm has its own loopback listener, config/log files and TMPDIR in staging.
- Preserve returned assistant content/reasoning/tool fields, including null and
  empty values. Save the public request/response JSONs for inspection. No tool
  execution is claimed; semantic image correctness is not graded by this smoke.
- Short natural image-history reuse, second-burst reuse and four occupied slots
  are observations. Four simultaneous clients do not guarantee four occupied slots.
- Positive reuse is checked separately using the already GPU-tested public
  canned-history rewrite, transported as actual HTTP bodies with data-URI PNGs.
  This does not label the canned history as generated model history.

## CPU-only actual-GGUF preparation

Run `tools/q36_image_http_preflight.py` with the service Python after placing this
package under `/mnt/ssd/tmp/strata-checkpoints`. The script starts no listener,
encoder or native engine and does not operate services. It uses the existing
`image-checkpoint-fixture-state.json`, qualified PNG/SVE inputs, native fixture
manifest and actual GGUF/template. It verifies the candidate hash, Orin source
commit and all three protected-file hashes.

The public requests are converted to their exact data-URI HTTP form, then the real
OpenAI conversion and `Service.prepare` execute with frozen, previously encoded
SVE inputs. Actual resulting token IDs and image keys must equal the GPU-tested
fixture. No synthetic tokenizer or image-row count fallback is allowed.

The audit mirrors `Q36ImagePlan` for the first fresh four-payload request with
empty learning, chunk512 and interval1024. It considers only actual legal
callbacks, respects complete image boundaries, and requires a saved checkpoint
AFTER the image and before each genuine rewrite divergence. Known qualified
geometry is image_end2112, LCP3671 and best retained point3584. These numbers are
recomputed, not treated as permission to fabricate a pass.

Prepared bodies, IDs, source configuration, actual encoder binary, candidate and
baseline binaries, source files and prior input evidence are hash-bound. GGUF and
projector identities retain path/size/mtime; the 23 GB GGUF is not copied or fully
hashed. Inputs are rechecked by both the plan/preflight worker and real worker.
HTTP response prompt sizes and actual restored reuse must match the audit.

A minimal source-config snapshot contains only the vision fields, tokenizer path
and model name. It carries no API key, live listener, replicas or production args.
The live configuration is read but never written.

Actual Orin CPU preparation and the serial window were subsequently completed.
Do not repeat a model window solely to reproduce this result.

## Bounded real-window design (not an instruction to execute)

The completed run used the deployed native engine, candidate flag off, and
candidate flag on, all with the same reviewed Python server. Each arm required a
resident encoder and `images=true`. It executed ten serial requests per arm:
controlled rewrites and natural-history observations. Concurrent bursts were not
included; generated histories retained responses from their own conversation.

Each arm has an owned process group. A still-unreaped leader reserves its PID/PGID
until residual group cleanup is signalled. If the leader is already lost, only a
presence probe is allowed; a live or unprovable group is a FAILURE, never empty.
Malformed registry entries fail closed. SIGINT/SIGTERM/SIGHUP use the existing
single-budget watchdog and persistent cleanup state. An outer operator-owned
cgroup and proof of emptiness before restoring production remain mandatory.
The harness itself performs no service-manager operations.

The real stdout marker is `IMAGE_HTTP_SMOKE PASS` with explicit tensor/OWUI limits;
only the supervisor may print it after the correct mode and complete cleanup.
`IMAGE_HTTP_SMOKE_PREFLIGHT PASS` starts no model and is not a GPU result.

## Local evidence and limits

See the serial handoff and the focused tests. Loopback stand-ins are not models
or encoders. The reviewer host skipped six /proc-dependent lifecycle tests; the
coding host subsequently ran all 82 focused tests without skips. The completed
native GPU oracle was not repeated or weakened. No C++/CUDA, include, native
tests or CMake file was changed. Concurrency, broader encoder co-residency and
actual OWUI savings remain unqualified. Production was not switched.
