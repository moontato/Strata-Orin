# Migration evidence

INVENTORY.json is the pre-commit inventory of the original container. It includes
migration files being prepared, not an assertion that they existed at37911119.
The three external source worktrees were verified file-by-file against Git blobs
already in /workspace/.git; all three commits are feature ancestors. No original
file/worktree was removed and no credential/cache directory copied.

fresh-venv-* records the first execution of the new portable runner on implementation
37911119 with migration support not yet committed. Its head field identifies the
functional base, NOT a claim that37911119 already contained the new runner. It
used a newly created Python3.11 venv installed from requirements-q36-dev.txt,
CMake4.4.4/Ninja1.13.2, not the previous dependency directory. All17 selected CPU
CTest cases,86 Python tests +22 subtests,22 mutation controls and the17 sanitizer
CTest cases passed. Future runner records include dirty status and checker hash.

historical-baseline-q2-linker.txt is the older a9837161 native-experts-OFF CPU Q2
linker limitation, not a new feature failure. Unrestricted builds/tests are not
claimed passed. The focused setup intentionally builds/selects only its17 cases.

The adjacent ../q36-image-checkpoints/ directory is a deliberately bounded
portable SUBSET of the original37911119 review's raw evidence. References in its
historical VALIDATION.md to other original logs are available in the optional
original review archive (inside the migration supplement), not hidden runtime
inputs. The final result logs and essential source/mutation/sanitizer/provenance
proofs are committed here; original build dirs, binaries and venvs are not.

Origin-clone verification is performed after the normal feature-branch push and
recorded as a separate delivery proof; it must use the actual network origin,
not --shared/local alternates. Docker daemon unavailable: the dev Dockerfile was
inspected, while the equivalent manual fresh-venv setup was actually executed.
