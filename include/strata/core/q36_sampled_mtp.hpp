#pragma once
// Host-side Qwen3.6 request sampling and the sampled speculative verifier.
// These are deliberately CUDA-free so the production helpers can be tested
// against sequential sampling without a GPU or model.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <map>
#include <random>
#include <vector>

namespace strata::core {

struct Sampling {
    float temperature = 0.f, top_p = 1.f, min_p = 0.f, repeat = 1.f, freq = 0.f, present = 0.f;
    int top_k = 20, last_n = 0;
    unsigned long long seed = 0;
    bool sampled_mtp = true;
    bool needs_logits() const { return temperature > 0.f || repeat != 1.f || freq != 0.f || present != 0.f; }
};

inline bool q36_is_eos(int t) { return t == 248044 || t == 248046; }

inline int q36_sample(std::vector<float>& lg, const Sampling& s, const std::vector<int32_t>& hist, std::mt19937_64& rng) {
    if (s.last_n != 0 && (s.repeat != 1.f || s.freq != 0.f || s.present != 0.f)) {
        std::map<int, int> cnt;
        const size_t from = s.last_n < 0 || (size_t) s.last_n >= hist.size() ? 0 : hist.size() - (size_t) s.last_n;
        for (size_t i = from; i < hist.size(); ++i) ++cnt[hist[i]];
        for (const auto& [id, c] : cnt) {
            float& v = lg[(size_t) id];
            if (s.repeat != 1.f) v = v > 0 ? v / s.repeat : v * s.repeat;
            v -= (float) c * s.freq + s.present;
        }
    }
    if (s.temperature <= 0.f) return (int) (std::max_element(lg.begin(), lg.end()) - lg.begin());
    const int kk = std::max(1, std::min(s.top_k > 0 ? s.top_k : 64, (int) lg.size()));
    std::vector<int> idx(lg.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int) i;
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
    idx.resize((size_t) kk);
    std::vector<double> p(idx.size());
    const double mx = lg[(size_t) idx[0]];
    double sum = 0;
    for (size_t i = 0; i < idx.size(); ++i) sum += (p[i] = std::exp((lg[(size_t) idx[i]] - mx) / s.temperature));
    size_t keep = idx.size();
    double cum = 0;
    for (size_t i = 0; i < idx.size(); ++i) {
        if (p[i] / sum < s.min_p * (p[0] / sum)) { keep = i; break; }
        cum += p[i] / sum;
        if (cum >= s.top_p) { keep = i + 1; break; }
    }
    keep = std::max<size_t>(1, keep);
    double tot = 0;
    for (size_t i = 0; i < keep; ++i) tot += p[i];
    std::uniform_real_distribution<double> u(0.0, tot);
    double r = u(rng);
    for (size_t i = 0; i < keep; ++i) {
        if ((r -= p[i]) <= 0) return idx[i];
    }
    return idx[keep - 1];
}

struct SampledSpecResult {
    int accepted = 0;
    int32_t next = -1;
    bool consumed = false;
    const char* accepted_finish = nullptr;
};

// Verify greedy MTP proposals with target samples. `hist` already contains the
// confirmed token x (and any previously accepted prefix). The caller has emitted
// x, so `produced` includes it. A sample is taken only for the row that follows
// the accepted prefix; future rows are never consumed. On a match the proposal is
// committed and its target sample is already present in `hist`; on a mismatch the
// returned target token has not been emitted/committed.
template <class RowLogits>
inline SampledSpecResult q36_sample_verify_window(const std::vector<int32_t>& drafts, const Sampling& s,
                                                  std::vector<int32_t>& hist, std::mt19937_64& rng,
                                                  long long produced, long long max_new, RowLogits&& row,
                                                  const std::atomic<bool>* cancel = nullptr) {
    SampledSpecResult r;
    const int D = (int) drafts.size();
    std::vector<float> lg;
    int accepted = 0;
    for (int i = 0; i <= D; ++i) {
        if (cancel != nullptr && cancel->load()) {
            if (accepted == 0) r.accepted_finish = "cancel";
            break;
        }
        row(i, lg);
        const int y = q36_sample(lg, s, hist, rng);
        r.consumed = true;
        if (i == D) {
            r.next = y;
            r.accepted = accepted;
            break;
        }
        if (y != drafts[(size_t) i]) {
            r.accepted = accepted;
            r.next = y;
            break;
        }
        ++accepted;
        hist.push_back(y);
        if (q36_is_eos(y)) { r.accepted_finish = "stop"; r.next = y; break; }
        if (produced + accepted >= max_new) { r.accepted_finish = "length"; r.next = y; break; }
    }
    r.accepted = std::max(r.accepted, accepted);
    if (r.accepted > D) r.accepted = D;
    return r;
}

}  // namespace strata::core
