# Experimental Qwen3.6 sampled MTP

Default off: `--mtp-sampled` opts into solo text sampled verification of greedy MTP
proposals. A request can disable/re-enable it with native `sampled_mtp=0|1` or HTTP
`strata_tune: {"sampled_mtp": 0|1}`. Without the engine option, the request key has
no effect. Greedy MTP, batched requests, images and prefix checkpoints keep their
existing paths. Hybrid expert placement falls back to sequential sampled decode.


## Target sampling and publication

Run the existing MTP proposal chain and main-model window `[x, d0, ... dD-1]`.
For each row, sample the main model's target logits with the existing sampler,
request RNG and token history. Accept a proposal only when that target sample
is equal to it. On the first mismatch stop consuming rows and retain the target
sample; if all match, retain the final bonus-row sample. No rejection/residual
sampling distribution is needed: every published token is itself a target draw.
This gives sequential output/RNG equivalence **for identical target logits**;
the native gate must separately measure window-versus-sequential numerical error.

The shared production request controller `q36_sampled_request()` is used by Serve,
the native gate and the CPU mock-backend regression. It:

- Tracks a pending mismatch/bonus sample explicitly. It is published once, without
  fetching row-zero logits or drawing the RNG again.
- Tracks the logit row after the actual committed prefix. `Engine::host_logits()`
  reads row zero, which is not the final accepted row of a verify window.
- Verifies provisionally, then publishes proposals one at a time. Only `x` and
  the published proposal prefix are passed to `spec_commit()`. A stop after
  verification or during publication cannot leave hidden drafts in the live state.
- Restores history and RNG to the last published decision when cancellation drops
  provisional decisions or a pending target sample. No later rows are sampled
  after mismatch, EOS, the output cap or an observed cancellation.
- Processes even the terminal published token in this opt-in path. Thus recurrent
  state, convolution history, hidden seed, positions and logical KV progress end
  at exactly the published prefix. This costs a final processing step compared
  with the old sequential path, which can leave its last output unprocessed.

The original `spec_window()` kernels write speculative main/MTP KV (including
INT8 scales) beyond the kept prefix, just as greedy speculation does. Only the
published prefix is logically retained; unused tails are overwritten on later
writes. `spec_commit()` retains recurrence/hidden state for that prefix. No
maximum-size eager allocation, model mirror or persistent checkpoint is added.
Prefix checkpoints remain prompt-only and retain their generation/epoch/backing
validation. Exact-prefix reuse is still tried first.

## Validation

q36_sampled_mtp_test checks the verifier/sampler and RNG behavior against supplied
sequential logits. q36_sampled_request_test exercises the production request
controller with a mock backend, including mismatch/bonus publication, output caps,
EOS, fallback and cancellation. Native sampled-MTP gates additionally require real
model state/logit and accepted/offered-token checks. Host tests alone do not
establish window-versus-sequential numerical equivalence or performance on a GPU.
