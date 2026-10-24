// Request-level regression: the SAME controller used by Serve, with a mock
// deferred-state backend. This deliberately does not call the isolated verifier.
#include "strata/core/q36_sampled_mtp.hpp"
#include <cassert>
#include <iostream>
#include <string>

using namespace strata::core;

static std::vector<float> target(const std::vector<int32_t>& prefix, int eos_after = -1) {
    const bool eos = eos_after >= 0 && (int) prefix.size() >= eos_after;
    std::vector<float> lg(eos_after >= 0 ? 248047 : 12, -1000.0f);
    uint64_t h = 19;
    for (int32_t t : prefix) h = h * 6364136223846793005ull + (uint64_t) t + 1;
    for (int t = 0; t < 12; ++t) lg[(size_t) t] = (float) ((h >> (t % 8)) % 500) / 100.0f;
    if (eos) lg[248044] = 1000.0f;
    return lg;
}

struct MockBackend {
    const Sampling& smp;
    std::vector<int32_t>& hist;
    std::mt19937_64& rng;
    std::atomic<bool>& cancel;
    std::vector<int32_t> committed, emitted, window_tokens, expected_fix;
    std::vector<std::vector<float>> rows;
    std::vector<int> kept;
    int accept = 2, eos_after = -1, reads = 0, windows = 0, sequential = 0;
    int cancel_before_publish = -1, cancel_after_publish = -1;
    bool window_open = false, allow_windows = true;

    MockBackend(const Sampling& s, std::vector<int32_t>& h, std::mt19937_64& r, std::atomic<bool>& c)
        : smp(s), hist(h), rng(r), cancel(c), committed(h), rows{target(h)} {}
    void logits(int row, std::vector<float>& out) {
        assert(row >= 0 && row < (int) rows.size());
        ++reads;
        out = rows[(size_t) row];
    }
    bool can_window(int) const { return allow_windows; }
    void window(int32_t x, const std::vector<int32_t>& fix, int D, std::vector<int32_t>& drafts) {
        assert(!window_open && fix == expected_fix);
        ++windows;
        window_open = true;
        window_tokens = {x};
        rows.clear();
        auto prefix = committed;
        prefix.push_back(x);
        auto candidate_hist = hist;
        auto candidate_rng = rng;
        drafts.clear();
        for (int i = 0; i <= D; ++i) {
            rows.push_back(target(prefix, eos_after));
            if (i < D) {
                auto lg = rows.back();
                const int y = q36_sample(lg, smp, candidate_hist, candidate_rng);
                // Make the first N proposals match the sequential target path,
                // and force the next to mismatch. Proposal construction never
                // touches the real request RNG. Later conditional rows differ.
                const int d = i < accept ? y : (y + 1) % 12;
                drafts.push_back(d);
                candidate_hist.push_back(d);
                prefix.push_back(d);
                window_tokens.push_back(d);
            }
        }
    }
    void commit(int n) {
        assert(window_open && n >= 1 && n <= (int) window_tokens.size());
        committed.insert(committed.end(), window_tokens.begin(), window_tokens.begin() + n);
        expected_fix.assign(window_tokens.begin() + 1, window_tokens.begin() + n);
        kept.push_back(n);
        window_open = false;
        // Leave row zero as after x, NOT the last kept prefix. A controller
        // which drops its pending token and samples row zero will fail.
    }
    void process(int32_t x) {
        assert(!window_open);
        ++sequential;
        committed.push_back(x);
        expected_fix.clear();
        rows = {target(committed, eos_after)};
    }
    bool publish(int32_t x) {
        if ((int) emitted.size() == cancel_before_publish) {
            cancel.store(true); // after verification, immediately before publication
            return false;
        }
        emitted.push_back(x);
        if ((int) emitted.size() == cancel_after_publish) cancel.store(true);
        return true;
    }
};

struct Reference {
    std::vector<int32_t> emitted, hist;
    std::mt19937_64 rng;
};

static Reference sequential(const Sampling& s, const std::vector<int32_t>& prompt, uint64_t seed,
                            int cap, int eos_after = -1) {
    Reference out{{}, prompt, std::mt19937_64(seed)};
    for (int i = 0; i < cap; ++i) {
        auto lg = target(out.hist, eos_after);
        const int y = q36_sample(lg, s, out.hist, out.rng);
        out.emitted.push_back(y);
        out.hist.push_back(y);
        if (q36_is_eos(y)) break;
    }
    return out;
}

static void run_case(const Sampling& s, int accept, int cap, uint64_t seed,
                     int cancel_before = -1, int cancel_after = -1,
                     int eos_after = -1, bool allow_windows = true) {
    const std::vector<int32_t> prompt{2, 7, 2};
    auto hist = prompt, live = prompt;
    std::mt19937_64 rng(seed);
    std::atomic<bool> stop{false};
    MockBackend b(s, hist, rng, stop);
    b.accept = accept;
    b.cancel_before_publish = cancel_before;
    b.cancel_after_publish = cancel_after;
    b.eos_after = eos_after;
    b.allow_windows = allow_windows;
    b.rows = {target(prompt, eos_after)};
    const auto result = q36_sampled_request(b, s, hist, live, rng, cap, 2, stop);
    const auto ref = sequential(s, prompt, seed, (int) b.emitted.size(), eos_after);
    assert(b.emitted == ref.emitted);
    assert(hist == ref.hist && live == hist && b.committed == live);
    assert(rng == ref.rng); // includes rollback of provisional, unpublished decisions
    assert(result.produced == (long long) b.emitted.size());
    assert(!b.window_open);
    if (cancel_before < 0 && cancel_after < 0) {
        const auto full = sequential(s, prompt, seed, cap, eos_after);
        assert(b.emitted == full.emitted && rng == full.rng);
        assert(b.reads == (int) b.emitted.size()); // pending mismatch/bonus NEVER resampled
        assert(std::string(result.finish) == (q36_is_eos(b.emitted.back()) ? "stop" : "length"));
    } else {
        assert(std::string(result.finish) == "cancel");
        const int count = cancel_before >= 0 ? cancel_before : cancel_after;
        assert((int) b.emitted.size() == count);
    }
    if (accept == 2 && cap == 3 && cancel_before < 0 && cancel_after < 0 && eos_after < 0 && allow_windows) {
        assert(result.accepted == 2 && result.offered == 2);
        assert(b.kept == std::vector<int>{3}); // all accepted exactly at output cap
        assert(result.logits_row == 2);
    }
    if (cap > 3 && allow_windows && eos_after < 0 && cancel_before < 0 && cancel_after < 0) {
        assert(b.windows > 0 && !b.kept.empty());
        assert(b.kept.front() == accept + 1); // explicitly exercise 0, 1 and 2 accepted drafts
    }
    std::vector<float> boundary;
    b.logits(result.logits_row, boundary);
    assert(boundary == target(b.committed, eos_after)); // common/continuation gate must not read row zero blindly
}

int main() {
    Sampling s;
    s.temperature = 0.85f; s.top_k = 7; s.top_p = 0.93f; s.min_p = 0.05f;
    s.last_n = 5; s.repeat = 1.1f; s.freq = 0.2f; s.present = 0.7f;
    for (uint64_t seed = 1; seed <= 80; ++seed) {
        for (int accepted = 0; accepted <= 2; ++accepted) {
            for (int cap : {1, 2, 3, 4, 9, 16}) run_case(s, accepted, cap, seed);
            // Before the first accepted proposal is emitted (verification finished),
            // after one of two accepted proposals is emitted, and after the bonus
            // or mismatch is published. All retained state must equal output.
            run_case(s, accepted, 16, seed, 0); // cancel first sampled token before publication
            run_case(s, accepted, 16, seed, 1);
            run_case(s, accepted, 16, seed, -1, 2);
            run_case(s, accepted, 16, seed, -1, 3);
        }
        run_case(s, 2, 9, seed, -1, -1, -1, false); // placement/admission fallback
    }
    // EOS at the initial, accepted, mismatch and bonus decision boundaries.
    for (int accepted = 0; accepted <= 2; ++accepted)
        for (int eos_after = 3; eos_after <= 7; ++eos_after)
            run_case(s, accepted, 10, 1234, -1, -1, eos_after);
    std::cout << "Q36 request controller: pending mismatch/bonus, 0/1/2 accepts, sequential RNG/output, "
                 "publication cancellation, committed prefix, EOS and caps passed\n";
}
