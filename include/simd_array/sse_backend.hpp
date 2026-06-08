#pragma once

/// \file sse_backend.hpp
/// SSE2 implementation of `ArrayBackend<T>`.
///
/// SSE2 is part of the x86-64 baseline ABI, so this backend can be compiled
/// unconditionally for any x86-64 target — no per-function target attribute
/// or `/arch:SSE2` flag is required, and the dispatcher can always fall back
/// to it when AVX/AVX-512 are unavailable.
///
/// One template handles both `float` and `double`: the per-type intrinsic
/// differences (suffix `_ps` vs `_pd`, vector type `__m128` vs `__m128d`,
/// lane count 4 vs 2) are encapsulated in small `if constexpr` helpers. The
/// hot loops themselves are then identical for the two element types.
///
/// Loads/stores are unaligned (`loadu`/`storeu`). On modern x86 cores the
/// performance penalty vs aligned moves on naturally-aligned data is zero,
/// and unaligned ops let the dispatcher accept any user-allocated pointer.
/// The aligned variants will be revisited once the public `simd::Array<T>`
/// guarantees alignment.
///
/// All hot loops follow the same shape: a vectorised body that processes
/// `W` lanes per iteration, then a scalar tail for the remaining
/// `n % W` elements. The tail uses the same arithmetic as the SIMD body, so
/// the output is bit-identical to `ScalarBackend<T>` whenever the inputs
/// avoid floating-point reassociation effects (i.e. always for elementwise
/// ops, and for reductions when intermediate sums stay exactly
/// representable).

#include "backend.hpp"

#include <cstddef>
#include <type_traits>

#include <immintrin.h>

namespace simd {

template<typename T>
class SSEBackend final : public ArrayBackend<T> {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "SSEBackend only supports float and double");

    using vec_t = std::conditional_t<std::is_same_v<T, float>, __m128, __m128d>;
    static constexpr std::size_t W = std::is_same_v<T, float> ? 4 : 2;

    // ------------------------------------------------------------------
    // Type-dispatched intrinsic helpers. Each forwards to the matching
    // `_ps` (single) or `_pd` (double) intrinsic at compile time.
    // ------------------------------------------------------------------
    static vec_t loadu(const T* p) noexcept {
        if constexpr (std::is_same_v<T, float>) return _mm_loadu_ps(p);
        else                                    return _mm_loadu_pd(p);
    }
    static void storeu(T* p, vec_t v) noexcept {
        if constexpr (std::is_same_v<T, float>) _mm_storeu_ps(p, v);
        else                                    _mm_storeu_pd(p, v);
    }
    static vec_t set1(T x) noexcept {
        if constexpr (std::is_same_v<T, float>) return _mm_set1_ps(x);
        else                                    return _mm_set1_pd(x);
    }
    static vec_t setzero() noexcept {
        if constexpr (std::is_same_v<T, float>) return _mm_setzero_ps();
        else                                    return _mm_setzero_pd();
    }
    static vec_t add_v(vec_t a, vec_t b) noexcept {
        if constexpr (std::is_same_v<T, float>) return _mm_add_ps(a, b);
        else                                    return _mm_add_pd(a, b);
    }
    static vec_t sub_v(vec_t a, vec_t b) noexcept {
        if constexpr (std::is_same_v<T, float>) return _mm_sub_ps(a, b);
        else                                    return _mm_sub_pd(a, b);
    }
    static vec_t mul_v(vec_t a, vec_t b) noexcept {
        if constexpr (std::is_same_v<T, float>) return _mm_mul_ps(a, b);
        else                                    return _mm_mul_pd(a, b);
    }
    static vec_t div_v(vec_t a, vec_t b) noexcept {
        if constexpr (std::is_same_v<T, float>) return _mm_div_ps(a, b);
        else                                    return _mm_div_pd(a, b);
    }
    static vec_t min_v(vec_t a, vec_t b) noexcept {
        if constexpr (std::is_same_v<T, float>) return _mm_min_ps(a, b);
        else                                    return _mm_min_pd(a, b);
    }
    static vec_t max_v(vec_t a, vec_t b) noexcept {
        if constexpr (std::is_same_v<T, float>) return _mm_max_ps(a, b);
        else                                    return _mm_max_pd(a, b);
    }

public:
    // ------------------------------------------------------------------
    // Element-wise binary operations.
    // ------------------------------------------------------------------
    void add(T* out, const T* a, const T* b, std::size_t n) const noexcept override {
        std::size_t i = 0;
        for (; i + W <= n; i += W) {
            storeu(out + i, add_v(loadu(a + i), loadu(b + i)));
        }
        for (; i < n; ++i) out[i] = a[i] + b[i];
    }
    void subtract(T* out, const T* a, const T* b, std::size_t n) const noexcept override {
        std::size_t i = 0;
        for (; i + W <= n; i += W) {
            storeu(out + i, sub_v(loadu(a + i), loadu(b + i)));
        }
        for (; i < n; ++i) out[i] = a[i] - b[i];
    }
    void multiply(T* out, const T* a, const T* b, std::size_t n) const noexcept override {
        std::size_t i = 0;
        for (; i + W <= n; i += W) {
            storeu(out + i, mul_v(loadu(a + i), loadu(b + i)));
        }
        for (; i < n; ++i) out[i] = a[i] * b[i];
    }
    void divide(T* out, const T* a, const T* b, std::size_t n) const noexcept override {
        std::size_t i = 0;
        for (; i + W <= n; i += W) {
            storeu(out + i, div_v(loadu(a + i), loadu(b + i)));
        }
        for (; i < n; ++i) out[i] = a[i] / b[i];
    }

    // ------------------------------------------------------------------
    // Scalar broadcast operations.
    // ------------------------------------------------------------------
    void add_scalar(T* out, const T* a, T scalar, std::size_t n) const noexcept override {
        const vec_t vs = set1(scalar);
        std::size_t i = 0;
        for (; i + W <= n; i += W) {
            storeu(out + i, add_v(loadu(a + i), vs));
        }
        for (; i < n; ++i) out[i] = a[i] + scalar;
    }
    void multiply_scalar(T* out, const T* a, T scalar, std::size_t n) const noexcept override {
        const vec_t vs = set1(scalar);
        std::size_t i = 0;
        for (; i + W <= n; i += W) {
            storeu(out + i, mul_v(loadu(a + i), vs));
        }
        for (; i < n; ++i) out[i] = a[i] * scalar;
    }

    // ------------------------------------------------------------------
    // Reductions. The `W` partial sums in the SIMD accumulator are
    // collapsed sequentially after the main loop. This changes the
    // floating-point reduction order vs the scalar backend, so results
    // can differ by one ULP on inputs that trigger rounding — but they
    // remain bit-identical when no intermediate value loses precision.
    // ------------------------------------------------------------------
    T sum(const T* a, std::size_t n) const noexcept override {
        vec_t vs = setzero();
        std::size_t i = 0;
        for (; i + W <= n; i += W) {
            vs = add_v(vs, loadu(a + i));
        }
        alignas(16) T buf[W];
        storeu(buf, vs);
        T s = T{};
        for (std::size_t k = 0; k < W; ++k) s += buf[k];
        for (; i < n; ++i) s += a[i];
        return s;
    }

    T min(const T* a, std::size_t n) const noexcept override {
        if (n == 0) return T{};
        std::size_t i;
        T m;
        if (n >= W) {
            vec_t vm = loadu(a);
            i = W;
            for (; i + W <= n; i += W) {
                vm = min_v(vm, loadu(a + i));
            }
            alignas(16) T buf[W];
            storeu(buf, vm);
            m = buf[0];
            for (std::size_t k = 1; k < W; ++k) if (buf[k] < m) m = buf[k];
        } else {
            m = a[0];
            i = 1;
        }
        for (; i < n; ++i) if (a[i] < m) m = a[i];
        return m;
    }

    T max(const T* a, std::size_t n) const noexcept override {
        if (n == 0) return T{};
        std::size_t i;
        T m;
        if (n >= W) {
            vec_t vm = loadu(a);
            i = W;
            for (; i + W <= n; i += W) {
                vm = max_v(vm, loadu(a + i));
            }
            alignas(16) T buf[W];
            storeu(buf, vm);
            m = buf[0];
            for (std::size_t k = 1; k < W; ++k) if (buf[k] > m) m = buf[k];
        } else {
            m = a[0];
            i = 1;
        }
        for (; i < n; ++i) if (a[i] > m) m = a[i];
        return m;
    }

    // ------------------------------------------------------------------
    // Utilities.
    // ------------------------------------------------------------------
    void fill(T* a, T value, std::size_t n) const noexcept override {
        const vec_t v = set1(value);
        std::size_t i = 0;
        for (; i + W <= n; i += W) {
            storeu(a + i, v);
        }
        for (; i < n; ++i) a[i] = value;
    }

    const char* name() const noexcept override { return "sse2"; }
};

} // namespace simd
