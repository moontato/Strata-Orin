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
                                                  const std::atomic<bool>* cancel = nullptr,
                                                  std::vector<std::mt19937_64>* sampled_rng = nullptr) {
    SampledSpecResult r;
    if (sampled_rng) sampled_rng->clear();
    if (produced >= max_new) { r.accepted_finish = "length"; return r; }
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
        if (sampled_rng) sampled_rng->push_back(rng);
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

// Shared solo sampled-request controller, used by Serve and the native/model
// gate as well as CPU mock-backend tests. Backend supplies logits(row, out),
// process(token), can_window(D), window(x, fix, D, drafts), commit(rows), and
// publish(token) returning whether the token was actually published. window()
// MUST defer recurrent/hidden-state commitment;
// commit() keeps only the first rows. No backend work may overwrite the window
// buffers between window() and commit(). The controller owns history/live edits.
struct SampledRequestResult {
    long long produced = 0, offered = 0, accepted = 0;
    int logits_row = 0; // logits after the actual committed prefix, NOT always row zero
    const char* finish = "length";
};

template <class Backend>
inline SampledRequestResult q36_sampled_request(Backend& backend, const Sampling& s,
                                               std::vector<int32_t>& hist, std::vector<int32_t>& live,
                                               std::mt19937_64& rng, long long max_new, int draft_limit,
                                               const std::atomic<bool>& cancel) {
    SampledRequestResult out;
    std::vector<int32_t> fix, drafts;
    std::vector<float> lg;
    std::vector<std::mt19937_64> sampled_rng;
    int32_t pending = -1;
    auto before_pending = rng;
    auto stopped = [&] { return cancel.load(); };
    auto ending = [&](int32_t t) -> const char* {
        if (q36_is_eos(t)) return "stop";
        if (out.produced >= max_new) return "length";
        return stopped() ? "cancel" : nullptr;
    };
    while (out.produced < max_new) {
        if (stopped()) {
            if (pending >= 0) rng = before_pending; // discard an unpublished target sample
            out.finish = "cancel";
            break;
        }
        if (pending < 0) {
            backend.logits(out.logits_row, lg);
            before_pending = rng;
            pending = q36_sample(lg, s, hist, rng);
        }
        if (stopped()) {
            rng = before_pending;
            out.finish = "cancel";
            break;
        }
        const int32_t x = pending;
        pending = -1; // mismatch/bonus is published exactly once, without another draw
        if (!backend.publish(x)) {
            rng = before_pending;
            out.finish = "cancel";
            break;
        }
        hist.push_back(x);
        ++out.produced;
        const char* finish = ending(x);
        const int D = (int) std::min<long long>(draft_limit, max_new - out.produced);
        if (finish || D <= 0 || !backend.can_window(D)) {
            // Even a terminal published token is processed in this opt-in path.
            // Thus cancellation, EOS and the output cap leave an exact published
            // prefix, not a committed suffix the client has never seen.
            backend.process(x);
            live.push_back(x);
            fix.clear();
            out.logits_row = 0;
            if (finish) { out.finish = finish; break; }
            continue;
        }
        backend.window(x, fix, D, drafts);
        out.offered += (long long) drafts.size();
        const size_t base = hist.size();
        const auto before_window = rng;
        const auto verified = q36_sample_verify_window(
            drafts, s, hist, rng, out.produced, max_new,
            [&](int row, std::vector<float>& logits) { backend.logits(row, logits); }, &cancel, &sampled_rng);
        // Verification is provisional. Publication promotes only a prefix of
        // its decisions. Cancellation can arrive after verification or inside
        // publish(); restore history/RNG for decisions that were never shown.
        hist.resize(base);
        rng = before_window;
        int published = 0;
        finish = nullptr;
        for (; published < verified.accepted;) {
            if (stopped()) { finish = "cancel"; break; }
            const int32_t t = drafts[(size_t) published];
            if (!backend.publish(t)) { finish = "cancel"; break; }
            hist.push_back(t);
            rng = sampled_rng[(size_t) published];
            ++published;
            ++out.produced;
            finish = ending(t);
            if (finish) break;
        }
        // Only x and the proposals actually published are allowed to change
        // recurrence, convolution history, hidden seed, positions and KV progress.
        backend.commit(published + 1);
        live.push_back(x);
        live.insert(live.end(), drafts.begin(), drafts.begin() + published);
        fix.assign(drafts.begin(), drafts.begin() + published);
        out.accepted += published;
        out.logits_row = published;
        if (!finish && stopped()) finish = "cancel";
        if (finish) { out.finish = finish; break; }
        if (verified.next < 0) { out.finish = "cancel"; break; }
        before_pending = rng;
        pending = verified.next;
        rng = sampled_rng.back(); // the already sampled mismatch/bonus, not row zero
    }
    return out;
}

}  // namespace strata::core
