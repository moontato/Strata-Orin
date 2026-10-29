# Serial image-checkpoint HTTP handoff

The bounded native image-checkpoint gate and the subsequent real serial HTTP smoke
have passed on Orin. Production has NOT switched: image checkpoints remain disabled.
The task now is to integrate and publish the tested harness corrections and evidence,
then prepare one reversible service switch for operator review. Do not start another
synthetic model window merely to reproduce these results.

## Provenance and safe integration

Repository: https://github.com/moontato/Strata-Orin.git
Branch: feature/q36-image-checkpoints.
The supplied coding-agent WIP capture was based on
9154c06d9f1e2d9a51c318b7d65ab2519d6fa848, with uncommitted corrections.
This handback is source snapshots and patches, NOT a new Git commit or bundle.
Inspect current HEAD/status and preserve existing changes; never reset/clean the
repository or overwrite later work to make a patch apply.

The handback includes:

1. patches/original-agent-WIP-vs-9154c06d.patch: the original agent's captured edits.
2. patches/reviewer-changes-vs-WIP.patch: the initial reviewer correction.
3. patches/serial-followup-vs-completed.patch: PNG, serial-scope, artifact-saving,
   tests and current documentation changes after that correction.
4. source/: authoritative final allowlisted file snapshots. Unchanged helper
   modules are included for hash comparison. No native/runtime implementation
   sources or build outputs are included.

Check patches against the actual current tree before applying. If the WIP edits
are already present, do not apply them again. Resolve later differences deliberately;
compare final files to SOURCE_SHA256SUMS. The Orin-tested HTTP harness SHA256 is
bc10928e6d165b0d146b38571791859fdfadbfaeb06bbef8f90d957cdbea4fac.
Do not claim a modified harness is byte-identical to that tested file.

## Results already completed

Actual Orin native source HEAD: 549df89938fe0b2c23c70f2c4ce76ea33d6bedd4
(protected f934 follow-ups preserved; cherry-picked equivalent of reviewed native tree).
Candidate binary SHA256:
eb5ef5de8f7f36de9bd25206d59ec2161a4fb5548e3f74246483a344f8544c3f.
Build: /mnt/ssd/tmp/strata-checkpoints/build-image-checkpoints-96d1a00b.

Native evidence archive SHA256:
79d614ac303c71a9bb8c2ac0e63a8a9ab620f787d570dcab3fc1a86a7edad732.
Its bounded gate used an independent schedule-matched oracle, exact restored/final
state and full logits, a corruption negative control, image identity/lifecycle
checks and stable preallocation accounting. It does not prove broad vision or
all-backend qualification.

HTTP evidence archive SHA256:
9d6e0bfdfe17b3f848ecd52ea08536e9b798e006cca50f3c3747b8158c73b7ea.
Results directory on Orin:
/mnt/ssd/tmp/strata-checkpoints/http-results-20261028T233744Z-3513769.
Execution: 2 min 22.652 s under a 900 s execution cap plus 30 s cleanup allowance.
All 3 arms completed 10 SERIAL requests each. Independent reviewer re-audit matched
30 bounded engine-log spans against 60 saved request/response JSON files, response
prompt usage, flag placement in three loopback configs and restore provenance.
Baseline and candidate_off reuse records match. In candidate_on, qualified rewrite
turns 2 and 3 restored 3584 tokens and read 124; baseline/off both read all 3708.
No required failures. All owned groups were empty/reaped. Per-arm cleanup was
recorded graceful=false; do not reword that as graceful shutdown. The operator
wrapper restored both services' previous states and confirmed the vision model
loaded; services-restored is present.

Tiny naturally generated-history cases still reused zero tokens. They do not
establish actual OWUI savings. The controlled positive case uses public canned
history, with actual GGUF/tokenizer/image geometry independently audited beforehand.
HTTP does NOT repeat the native tensor oracle. Concurrency is explicitly NOT TESTED
by this HTTP run. Prior native mixed-slot coverage remains separate.

## Corrections and why the scope changed

The previous PNG writer encoded chunk lengths including the four-byte chunk type.
That produced the exact rejected image hash seen on Orin. The corrected generator
checks signature, lengths, CRCs and all decoded fixture pixels before writing;
Pillow independently decodes the fixtures in tests. ffprobe was a decode fallback
message, not proof of a missing dependency that needed installation.

During concurrent HTTP admissions, one logical request can yield: a partial solo
summary ending cancel is followed by a resumed batch admission. One summary per
HTTP request is therefore invalid for those bursts. The serial smoke excludes
the bursts rather than dropping/deduplicating evidence. Counts and provenance
remain strict; multi-summary spans still fail closed. A future concurrency test
needs a scheduler-aware logical-request contract of its own.

Other retained corrections: /metrics?requests=all; proxy bypass; config filename
rather than dictionary in the server command; isolated environment and actual
managed-unit preflight; request-local byte spans; real stderr restore parsing;
request JSON saved before POST and HTTP error body saved on failure; bounded
supervision and conservative process-group cleanup. No production model math,
retention policy, native oracle tolerance or native implementation changed here.

## Coding-agent work to do now

1. Verify this package's SHA256SUMS, inspect AGENTS.md and existing Git state,
   and integrate the allowlisted source without disturbing unrelated changes.
2. Run the focused Python tests named in validation/VALIDATION.md and the HTTP
   mutation controls. This is host-only testing, not a request for CUDA/model
   tests or service operation. Preserve honest skips and any new failures.
3. Update old HTTP HOLD/pending/concurrent-qualification wording consistently;
   retain historical failed runs and limits rather than claiming broader coverage.
4. Commit the corrections plus concise evidence on feature/q36-image-checkpoints.
   Use a normal, non-force push to origin when authentication permits, and verify
   the remote branch hash equals local HEAD. If blocked, report the blocker and
   preserve a bundle; never fabricate publication or reset to origin.
5. Return commit/remote hashes, source hash, exact host results and remaining limits.
   Prepare a concrete backup/switch/verification/rollback procedure for review,
   without running it. No rebuild is warranted solely for these harness edits.

Large raw evidence archives belong in project tmp/ or external evidence storage,
not Git. Commit the concise review JSON and documentation, referencing their hashes.
Do not commit credentials, venvs, builds, caches or the raw generated image bodies.

## Orin boundaries

Do not operate services or edit production from the coding container. All Orin
staging is /mnt/ssd/tmp/strata-checkpoints. The isolated feature worktree is
/mnt/ssd/Strata-Orin/worktrees/multimodal-cache. Never change/rebuild/remove the live
checkpoint-policy worktree or the preallocation worktree/vision encoder dependency.
Preserve tools/q36_adaptive_integration_fixture.py,
tools/test_q36_integration_ordinary.py and docs/Q36_ADAPTIVE_ORIN_VALIDATION.json.
Production remains 262144 context/slot, 4 slots, 524288 shared pool, INT8 KV,
preallocation, adaptive checkpoints, sampled MTP/batch-MTP, vision cap 1024 and
fit_max_tokens=true. A later operator switch must preserve these settings, backup
actual config/launcher/unit as appropriate, verify actual engine/encoder and real
API requests, and roll back on failure. No prompt normalization or OWUI edits.
