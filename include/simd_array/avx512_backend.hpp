#pragma once

/// \file avx512_backend.hpp
/// AVX-512 Foundation implementation of `ArrayBackend<T>`.
///
/// Same dispatch story as `avx_backend.hpp`: every member function (helpers
/// + virtuals) carries `__attribute__((target("avx512f")))` on GCC / Clang
/// so the `_mm512_*` intrinsics compile into a single TU even when the
/// surrounding project is built without `-mavx512f`. On MSVC the macro is
/// empty and the consuming TU must be compiled with `/arch:AVX512`. Either
/// way the runtime CPUID dispatcher only reaches this backend when both the
/// CPU advertises AVX-512F **and** the OS has enabled the ZMM XSAVE state
/// (the existing `cpu_features().has_avx512f` flag already gates on both).
///
/// `AVX-512F` is sufficient for the float/double arithmetic kernels we need
/// here — DQ/BW/VL extensions are not required, which keeps the binary
/// loadable on a wider set of AVX-512-capable CPUs.
///
/// Loop shape mirrors the SSE / AVX backends: vectorised body of width `W`
/// (16 floats or 8 doubles per `__m512` / `__m512d`), then a scalar tail.

#include "backend.hpp"

#include <cstddef>
#include <type_traits>

#include <immintrin.h>

#if defined(__GNUC__) || defined(__clang__)
    #define SIMD_ARRAY_AVX512_TARGET __attribute__((target("avx512f")))
#else
    #define SIMD_ARRAY_AVX512_TARGET
#endif

namespace simd {

template<typename T>
class AVX512Backend final : public ArrayBackend<T> {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "AVX512Backend only supports float and double");

    using vec_t = std::conditional_t<std::is_same_v<T, float>, __m512, __m512d>;
    static constexpr std::size_t W = std::is_same_v<T, float> ? 16 : 8;

    // ------------------------------------------------------------------
    // Type-dispatched intrinsic helpers.
    // ------------------------------------------------------------------
    SIMD_ARRAY_AVX512_TARGET static vec_t loadu(const T* p) noexcept {
        if constexpr (std::is_same_v<T, float>) return _mm512_loadu_ps(p);
        else                                    return _mm512_loadu_pd(p);
    }
    SIMD_ARRAY_AVX512_TARGET static void storeu(T* p, vec_t v) noexcept {
        if constexpr (std::is_same_v<T, float>) _mm512_storeu_ps(p, v);
        else                                    _mm512_storeu_pd(p, v);
    }
    SIMD_ARRAY_AVX512_TARGET static vec_t set1(T x) noexcept {
        if constexpr (std::is_same_v<T, float>) return _mm512_set1_ps(x);
        else                                    return _mm512_set1_pd(x);
    }
    SIMD_ARRAY_AVX512_TARGET static vec_t setzero() noexcept {
        if constexpr (std::is_same_v<T, float>) return _mm512_setzero_ps();
        else                                    return _mm512_setzero_pd();
    }
    SIMD_ARRAY_AVX512_TARGET static vec_t add_v(vec_t a, vec_t b) noexcept {
        if constexpr (std::is_same_v<T, float>) return _mm512_add_ps(a, b);
        else                                    return _mm512_add_pd(a, b);
    }
    SIMD_ARRAY_AVX512_TARGET static vec_t sub_v(vec_t a, vec_t b) noexcept {
        if constexpr (std::is_same_v<T, float>) return _mm512_sub_ps(a, b);
        else                                    return _mm512_sub_pd(a, b);
    }
    SIMD_ARRAY_AVX512_TARGET static vec_t mul_v(vec_t a, vec_t b) noexcept {
        if constexpr (std::is_same_v<T, float>) return _mm512_mul_ps(a, b);
        else                                    return _mm512_mul_pd(a, b);
    }
    SIMD_ARRAY_AVX512_TARGET static vec_t div_v(vec_t a, vec_t b) noexcept {
        if constexpr (std::is_same_v<T, float>) return _mm512_div_ps(a, b);
        else                                    return _mm512_div_pd(a, b);
    }
    SIMD_ARRAY_AVX512_TARGET static vec_t min_v(vec_t a, vec_t b) noexcept {
        if constexpr (std::is_same_v<T, float>) return _mm512_min_ps(a, b);
        else                                    return _mm512_min_pd(a, b);
    }
    SIMD_ARRAY_AVX512_TARGET static vec_t max_v(vec_t a, vec_t b) noexcept {
        if constexpr (std::is_same_v<T, float>) return _mm512_max_ps(a, b);
        else                                    return _mm512_max_pd(a, b);
    }

public:
    // ------------------------------------------------------------------
    // Element-wise binary operations.
    // ------------------------------------------------------------------
    SIMD_ARRAY_AVX512_TARGET
    void add(T* out, const T* a, const T* b, std::size_t n) const noexcept override {
        std::size_t i = 0;
        for (; i + W <= n; i += W) {
            storeu(out + i, add_v(loadu(a + i), loadu(b + i)));
        }
        for (; i < n; ++i) out[i] = a[i] + b[i];
    }
    SIMD_ARRAY_AVX512_TARGET
    void subtract(T* out, const T* a, const T* b, std::size_t n) const noexcept override {
        std::size_t i = 0;
        for (; i + W <= n; i += W) {
            storeu(out + i, sub_v(loadu(a + i), loadu(b + i)));
        }
        for (; i < n; ++i) out[i] = a[i] - b[i];
    }
    SIMD_ARRAY_AVX512_TARGET
    void multiply(T* out, const T* a, const T* b, std::size_t n) const noexcept override {
        std::size_t i = 0;
        for (; i + W <= n; i += W) {
            storeu(out + i, mul_v(loadu(a + i), loadu(b + i)));
        }
        for (; i < n; ++i) out[i] = a[i] * b[i];
    }
    SIMD_ARRAY_AVX512_TARGET
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
    SIMD_ARRAY_AVX512_TARGET
    void add_scalar(T* out, const T* a, T scalar, std::size_t n) const noexcept override {
        const vec_t vs = set1(scalar);
        std::size_t i = 0;
        for (; i + W <= n; i += W) {
            storeu(out + i, add_v(loadu(a + i), vs));
        }
        for (; i < n; ++i) out[i] = a[i] + scalar;
    }
    SIMD_ARRAY_AVX512_TARGET
    void multiply_scalar(T* out, const T* a, T scalar, std::size_t n) const noexcept override {
        const vec_t vs = set1(scalar);
        std::size_t i = 0;
        for (; i + W <= n; i += W) {
            storeu(out + i, mul_v(loadu(a + i), vs));
        }
        for (; i < n; ++i) out[i] = a[i] * scalar;
    }

    // ------------------------------------------------------------------
    // Reductions. Same caveat as the other SIMD backends: W partial
    // accumulators are collapsed sequentially after the main loop, so
    // the FP reduction order differs from scalar; bit-identical only
    // when intermediate values stay exactly representable.
    // ------------------------------------------------------------------
    SIMD_ARRAY_AVX512_TARGET
    T sum(const T* a, std::size_t n) const noexcept override {
        vec_t vs = setzero();
        std::size_t i = 0;
        for (; i + W <= n; i += W) {
            vs = add_v(vs, loadu(a + i));
        }
        alignas(64) T buf[W];
        storeu(buf, vs);
        T s = T{};
        for (std::size_t k = 0; k < W; ++k) s += buf[k];
        for (; i < n; ++i) s += a[i];
        return s;
    }

    SIMD_ARRAY_AVX512_TARGET
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
            alignas(64) T buf[W];
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

    SIMD_ARRAY_AVX512_TARGET
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
            alignas(64) T buf[W];
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
    SIMD_ARRAY_AVX512_TARGET
    void fill(T* a, T value, std::size_t n) const noexcept override {
        const vec_t v = set1(value);
        std::size_t i = 0;
        for (; i + W <= n; i += W) {
            storeu(a + i, v);
        }
        for (; i < n; ++i) a[i] = value;
    }

    const char* name() const noexcept override { return "avx512f"; }
};

} // namespace simd
