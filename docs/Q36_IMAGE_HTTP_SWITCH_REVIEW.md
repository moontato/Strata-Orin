# Proposed Orin service switch — review only

**Not authorized or executed.** This is a reversible operator procedure to review,
not a deployment command. The completed native gate and serial HTTP smoke justify
considering the candidate, not claiming concurrent HTTP qualification or OWUI
savings. Keep `--kv-image-checkpoints` disabled until a separate operator decision.

1. Before a separately approved maintenance window, identify the actual active
   manager units, process tree, executable paths, configuration, launcher, model,
   encoder and dependencies; record both Strata and llama service states. Record
   manager-owned cgroup membership and health (`service=strata`, `status=ok`,
   `loaded=true`, `images=true`). Do not assume a unit or config pathname.
2. Under `/mnt/ssd/tmp/strata-checkpoints`, make timestamped byte-for-byte backups
   of the actual configuration, launcher and relevant unit/drop-ins, including
   permissions, ownership and hashes. Verify that backups can be read; retain them.
   Record the original engine and encoder hashes. Ensure the candidate executable is
   the qualified build with SHA256
   `eb5ef5de8f7f36de9bd25206d59ec2161a4fb5548e3f74246483a344f8544c3f` and that all
   three protected f934 files retain the hashes in the serial handoff. Do not
   modify or rebuild the live checkpoint-policy or preallocation worktrees.
3. Pause clients. Stop only the identified services through their manager, then
   prove their manager-owned cgroups empty. Never rely on a process-name kill. If
   isolation or ownership cannot be proved, abort and restore the prior states.
4. Only after explicit operator approval, make the smallest reversible change to
   the identified production launcher/config to select the verified candidate and
   add `--kv-image-checkpoints`. Preserve context 262144, four slots, shared pool
   524288, INT8 KV, preallocation, adaptive checkpoints, sampled MTP/batch-MTP,
   vision cap 1024 and `fit_max_tokens=true`; do not change OWUI or normalize prompts.
   Review the exact diff and record hashes before starting the original service states.
5. Verify the actual running executable and encoder identity, recorded arguments,
   previous service states and health including `images=true`. Under the operator's
   approved client policy, exercise a real image request and its follow-up, a
   text-only request, and inspect logs/errors. Compare against the prior baseline;
   do not treat a functional response as native tensor verification or a savings
   measurement. A concurrent stress window, if desired, is separate work.
6. On any failed health, image request, resource or log check—or on operator
   request—stop through the manager, prove the owned cgroups empty, restore the
   exact backups and their metadata, and restore each service's recorded prior state.
   Verify original hashes, executable/encoder, health and image capability before
   resuming clients. If recovery cannot be proved, leave clients paused and escalate.

Keep the backup, verification output, timestamps and outcome for review. No step
above has been run by this handoff.
