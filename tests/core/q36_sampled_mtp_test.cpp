#include "strata/core/q36_sampled_mtp.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

using namespace strata::core;

constexpr int kVocab = 16;
constexpr int kEosVocab = 248047;

static uint64_t mix(uint64_t a, uint64_t b) {
    a ^= b + 0x9e3779b97f4a7c15ull + (a << 6) + (a >> 2);
    a *= 0xbf58476d1ce4e5b9ull;
    return a ^ (a >> 31);
}

static std::vector<float> target_logits(const std::vector<int32_t>& hist, uint64_t salt, int forced = -1) {
    uint64_t h = mix(salt, hist.size());
    for (int t : hist) h = mix(h, (uint64_t) t + 1);
    std::vector<float> lg((size_t) kVocab);
    for (int t = 0; t < kVocab; ++t) lg[(size_t) t] = (float) ((h + (uint64_t) t * 1103515245ull) % 900) / 100.0f - 4.5f;
    if (forced >= 0) lg[(size_t) forced] += 1000.0f;
    return lg;
}

static int forced_argmax(const std::vector<float>& lg) {
    return (int) (std::max_element(lg.begin(), lg.end()) - lg.begin());
}

struct RowCounter {
    int calls = 0;
    std::atomic<bool>* stop = nullptr;
    int stop_after = -1;
    const std::vector<std::vector<float>>* rows;
    void operator()(int i, std::vector<float>& out) {
        ++calls;
        assert(i >= 0 && i < (int) rows->size());
        out = (*rows)[(size_t) i];
        if (stop != nullptr && stop_after >= 0 && calls >= stop_after) stop->store(true);
    }
};

static void verify_one_window(const Sampling& s, const std::vector<int32_t>& base, const std::vector<int32_t>& drafts,
                              long long produced_start, long long max_new, uint64_t seed) {
    const int D = (int) drafts.size();
    std::vector<std::vector<float>> rows;
    std::vector<int32_t> speculative_hist = base;
    for (int i = 0; i <= D; ++i) {
        rows.push_back(target_logits(speculative_hist, seed + (uint64_t) i * 77));
        if (i < D) speculative_hist.push_back(drafts[(size_t) i]);
    }

    RowCounter counter;
    counter.rows = &rows;
    std::atomic<bool> stop{false};
    auto spec_hist = base;
    std::mt19937_64 spec_rng(seed);
    const auto r = q36_sample_verify_window(drafts, s, spec_hist, spec_rng, produced_start, max_new, counter, &stop);
    assert(counter.calls == r.accepted + (r.consumed ? 1 : 0) || r.accepted_finish != nullptr || r.next >= 0);

    // Sequential target-sampling reference, using exactly the same rows and history rules.
    int calls = 0, accepted = 0;
    bool consumed = false;
    int32_t next = -1;
    const char* finish = nullptr;
    auto seq_hist = base;
    std::mt19937_64 seq_rng(seed);
    for (int i = 0; i <= D; ++i) {
        std::vector<float> lg = rows[(size_t) i];
        const int y = q36_sample(lg, s, seq_hist, seq_rng);
        ++calls;
        consumed = true;
        if (i == D) {
            next = y;
            break;
        }
        if (y != drafts[(size_t) i]) {
            accepted = i;
            next = y;
            break;
        }
        ++accepted;
        seq_hist.push_back(y);
        if (q36_is_eos(y)) { finish = "stop"; break; }
        if (produced_start + accepted >= max_new) { finish = "length"; break; }
    }
    assert(r.accepted == accepted);
    assert(r.next == next);
    assert(r.consumed == consumed);
    assert(spec_hist == seq_hist);
    assert((r.accepted_finish == nullptr) == (finish == nullptr));
    if (finish != nullptr) assert(std::string(r.accepted_finish) == finish);
    assert(calls == r.accepted + 1);

    auto a = spec_rng, b = seq_rng;
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (int i = 0; i < 12; ++i) assert(u(a) == u(b));
}

static void every_position_and_all_accepted() {
    Sampling s;
    s.temperature = 0.7f; s.top_p = 0.95f; s.top_k = 8; s.min_p = 0.02f;
    s.seed = 99;
    const std::vector<int32_t> base{7, 11, 5};
    for (int D = 1; D <= 4; ++D) {
        std::vector<int32_t> drafts((size_t) D);
        std::vector<int32_t> hist = base;
        for (int i = 0; i < D; ++i) {
            const auto lg = target_logits(hist, 99 + (uint64_t) i * 77);
            drafts[(size_t) i] = forced_argmax(lg);
            hist.push_back(drafts[(size_t) i]);
        }
        // Force the exact rejection position by making the target's row choose the draft up to that row,
        // then one different token at that row.
        for (int reject = 0; reject < D; ++reject) {
            std::vector<std::vector<float>> rows;
            std::vector<int32_t> prefix = base;
            for (int i = 0; i <= D; ++i) {
                int forced = -1;
                if (i < reject) forced = drafts[(size_t) i];
                if (i == reject) {
                    const auto natural = target_logits(prefix, 99 + (uint64_t) i * 77);
                    forced = (drafts[(size_t) i] + 1) % kVocab;
                    rows.push_back([&] { auto lg = natural; lg[(size_t) forced] += 1000.0f; return lg; }());
                } else if (i < D) {
                    const auto natural = target_logits(prefix, 99 + (uint64_t) i * 77);
                    rows.push_back([&] { auto lg = natural; lg[(size_t) drafts[(size_t) i]] += 1000.0f; return lg; }());
                } else {
                    rows.push_back(target_logits(prefix, 99 + (uint64_t) i * 77));
                }
                if (i < D) prefix.push_back(drafts[(size_t) i]);
            }
            RowCounter counter;
            counter.rows = &rows;
            auto spec_hist = base;
            std::mt19937_64 rng(123);
            const auto result = q36_sample_verify_window(drafts, s, spec_hist, rng, 1, 100, counter);
            assert(result.accepted == reject);
            assert(counter.calls == reject + 1);
            assert(result.accepted_finish == nullptr);
            assert(spec_hist.size() == base.size() + (size_t) reject);
        }
        {
            std::vector<std::vector<float>> rows;
            std::vector<int32_t> prefix = base;
            for (int i = 0; i < D; ++i) {
                const auto natural = target_logits(prefix, 99 + (uint64_t) i * 77);
                rows.push_back([&] { auto lg = natural; lg[(size_t) drafts[(size_t) i]] += 1000.0f; return lg; }());
                prefix.push_back(drafts[(size_t) i]);
            }
            rows.push_back(target_logits(prefix, 99 + (uint64_t) D * 77));
            RowCounter counter;
            counter.rows = &rows;
            auto spec_hist = base;
            std::mt19937_64 rng(456);
            const auto result = q36_sample_verify_window(drafts, s, spec_hist, rng, 1, 100, counter);
            assert(result.accepted == D && result.next >= 0 && !result.accepted_finish);
            assert(counter.calls == D + 1);
            assert(spec_hist.size() == base.size() + (size_t) D);
        }
    }
}

static void stopping_and_cancellation() {
    Sampling s;
    s.temperature = 0.8f; s.top_p = 1.0f; s.top_k = kVocab;
    const std::vector<int32_t> base{1, 2};

    // EOS is an accepted proposal: the token is committed, and no later row is sampled.
    {
        std::vector<int32_t> drafts{(int32_t) 248044, 3};
        std::vector<std::vector<float>> rows(3, std::vector<float>(kEosVocab, 0.0f));
        rows[0][(size_t) drafts[0]] += 1000.0f;
        RowCounter counter;
        counter.rows = &rows;
        auto hist = base;
        std::mt19937_64 rng(7);
        const auto r = q36_sample_verify_window(drafts, s, hist, rng, 1, 100, counter);
        assert(r.accepted == 1 && std::string(r.accepted_finish) == "stop" && r.next == drafts[0]);
        assert(counter.calls == 1 && hist.size() == base.size() + 1);
    }
    // A sampled EOS that mismatches is returned, but is not part of the committed proposal prefix.
    {
        std::vector<int32_t> drafts{4, 5};
        std::vector<std::vector<float>> rows(3, std::vector<float>(kEosVocab, 0.0f));
        rows[0][(size_t) 248044] += 1000.0f;
        RowCounter counter;
        counter.rows = &rows;
        auto hist = base;
        std::mt19937_64 rng(11);
        const auto r = q36_sample_verify_window(drafts, s, hist, rng, 1, 100, counter);
        assert(r.accepted == 0 && r.next == 248044 && !r.accepted_finish);
        assert(counter.calls == 1 && hist == base);
    }
    // max_new stops after the accepted token; the row that would produce the bonus token is not sampled.
    {
        std::vector<int32_t> drafts{2, 3};
        std::vector<std::vector<float>> rows(3, std::vector<float>(kEosVocab, 0.0f));
        rows[0][(size_t) 2] += 1000.0f;
        RowCounter counter;
        counter.rows = &rows;
        auto hist = base;
        std::mt19937_64 rng(13);
        const auto r = q36_sample_verify_window(drafts, s, hist, rng, 4, 5, counter);
        assert(r.accepted == 1 && std::string(r.accepted_finish) == "length" && counter.calls == 1);
        assert(hist.size() == base.size() + 1);
    }
    // A pre-request stop consumes no row at all.
    {
        std::atomic<bool> stop{true};
        std::vector<std::vector<float>> rows(3, std::vector<float>(kEosVocab, 0.0f));
        RowCounter counter;
        counter.rows = &rows;
        counter.stop = &stop;
        auto hist = base;
        std::mt19937_64 rng(17);
        const auto r = q36_sample_verify_window({4, 5}, s, hist, rng, 1, 100, counter, &stop);
        assert(counter.calls == 0 && r.accepted == 0 && !r.consumed && std::string(r.accepted_finish) == "cancel");
        assert(hist == base);
    }
    // A stop noticed after one consumed row must not sample row 1, even if that row matched.
    {
        std::atomic<bool> stop{false};
        std::vector<int32_t> drafts{2, 3};
        std::vector<std::vector<float>> rows(3, std::vector<float>(kEosVocab, 0.0f));
        rows[0][(size_t) 2] += 1000.0f;
        RowCounter counter;
        counter.rows = &rows;
        counter.stop = &stop;
        counter.stop_after = 1;
        auto hist = base;
        std::mt19937_64 rng(19);
        const auto r = q36_sample_verify_window(drafts, s, hist, rng, 1, 100, counter, &stop);
        assert(counter.calls == 1 && r.accepted == 1 && r.next == -1 && !r.accepted_finish);
        assert(hist.size() == base.size() + 1);
    }
}

static void history_dependent_penalties() {
    // Presence penalty makes the same logits and seed choose differently after the accepted proposal.
    Sampling s;
    s.temperature = 0.3f; s.top_p = 1.0f; s.top_k = 3; s.last_n = 5; s.present = 0.75f;
    const std::vector<int32_t> base{4, 4, 4, 9};
    for (uint64_t seed = 1; seed <= 40; ++seed)
        verify_one_window(s, base, {2, 3, 5, 6}, 1, 100, seed);
}

int main() {
    // Temperature 0 stays exactly the existing greedy max_element path.
    {
        Sampling greedy;
        std::vector<float> lg = {1, 9, 3};
        std::mt19937_64 rng(5);
        std::vector<int32_t> hist{0};
        assert(q36_sample(lg, greedy, hist, rng) == 1);
        assert(!greedy.needs_logits());
    }
    // Top-k / top-p / min-p and all penalty fields feed sequential/spec comparisons identically.
    Sampling s;
    s.temperature = 1.0f; s.top_p = 0.95f; s.top_k = 20; s.min_p = 0.05f;
    s.repeat = 1.1f; s.freq = 0.2f; s.present = 1.5f; s.last_n = 64;
    for (uint64_t seed = 1; seed <= 200; ++seed)
        verify_one_window(s, {11, 7, 2, 7, 3}, {5, 6, 8}, 2, 100, seed);
    every_position_and_all_accepted();
    stopping_and_cancellation();
    history_dependent_penalties();
    std::cout << "Q36 sampled MTP target-sampling equivalence, penalties, RNG consumption, boundaries and cancellation passed\n";
    return 0;
}
