#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace strata::core {
inline uint64_t kv_add(uint64_t a, uint64_t b) {
    if (b > UINT64_MAX - a) throw std::overflow_error("KV size overflow");
    return a + b;
}
inline uint64_t kv_mul(uint64_t a, uint64_t b) {
    if (a && b > UINT64_MAX / a) throw std::overflow_error("KV size overflow");
    return a * b;
}
inline uint64_t kv_ceil(uint64_t n, uint64_t d) {
    if (!d) throw std::invalid_argument("KV zero alignment");
    return n / d + (n % d != 0);
}
// One prefix, repeated at a fixed stride (head-major Q36); one repetition for
// page-major QSA. Allocation rounding is the union, not a sum of rounded spans.
struct KvPrefix {
    uint64_t bytes_per_unit = 0, tokens_per_unit = 1, units = 0;
    uint64_t repeats = 1, stride = 0;
    uint64_t extent() const {
        if (!bytes_per_unit || !tokens_per_unit || !units || !repeats)
            throw std::invalid_argument("invalid KV prefix geometry");
        const uint64_t one = kv_mul(bytes_per_unit, units);
        if (repeats > 1 && stride < one) throw std::invalid_argument("overlapping KV heads");
        return kv_add(kv_mul(repeats - 1, stride), one);
    }
    std::vector<uint64_t> chunks(uint64_t cells, uint64_t granularity) const {
        (void) extent();
        if (!granularity) throw std::invalid_argument("KV VMM is unavailable");
        const uint64_t count = std::min(units, kv_ceil(cells, tokens_per_unit));
        const uint64_t n = kv_mul(count, bytes_per_unit);
        std::vector<uint64_t> out;
        if (!n) return out;
        for (uint64_t h = 0; h < repeats; ++h) {
            const uint64_t begin = kv_mul(h, stride);
            const uint64_t end = kv_ceil(kv_add(begin, n), granularity);
            for (uint64_t c = begin / granularity; c < end; ++c)
                if (out.empty() || out.back() != c) out.push_back(c);
        }
        return out;
    }
};
} // namespace strata::core
