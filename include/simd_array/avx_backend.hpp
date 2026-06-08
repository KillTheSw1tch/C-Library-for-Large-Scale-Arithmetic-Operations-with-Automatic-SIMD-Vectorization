#pragma once

/// \file avx_backend.hpp
/// AVX2 implementation of `ArrayBackend<T>`.
///
/// AVX2 is NOT part of the x86-64 baseline ABI, so simply including this
/// header in a binary built without `-mavx2` would fail to compile (the
/// `_mm256_*` intrinsics expand to instructions the assembler would reject).
/// Two strategies sidestep that:
///
///   * **GCC / Clang** — every function in this backend (helpers + virtuals)
///     carries `__attribute__((target("avx2")))`. The compiler emits AVX2
///     instructions inside those bodies regardless of the project-wide
///     `-march`, while the vtable call sites stay generic. The runtime
///     dispatcher (CPUID-gated) ensures we only ever *call* into this
///     backend on CPUs that actually support AVX2, so the binary remains
///     loadable on older hardware.
///
///   * **MSVC** — has no per-function target. The macro below expands to
///     nothing and the user must build the consuming TU with `/arch:AVX2`.
///     The CPUID dispatcher still picks the right backend at runtime, but
///     the binary as a whole then requires AVX2 to load.
///
/// Loop shape mirrors `sse_backend.hpp` exactly: vectorised body of width
/// `W` (8 floats or 4 doubles per `__m256` / `__m256d`) followed by a
/// scalar tail. Loads/stores are unaligned for the same reason as in the
/// SSE backend.

#include "backend.hpp"

#include <cstddef>
#include <type_traits>

#include <immintrin.h>

#if defined(__GNUC__) || defined(__clang__)
    #define SIMD_ARRAY_AVX2_TARGET __attribute__((target("avx2")))
#else
    #define SIMD_ARRAY_AVX2_TARGET
#endif

namespace simd {

template<typename T>
class AVXBackend final : public ArrayBackend<T> {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "AVXBackend only supports float and double");

    using vec_t = std::conditional_t<std::is_same_v<T, float>, __m256, __m256d>;
    static constexpr std::size_t W = std::is_same_v<T, float> ? 8 : 4;

    // ------------------------------------------------------------------
    // Type-dispatched intrinsic helpers. All tagged with the AVX2 target
    // attribute on GCC/Clang so they may freely use `_mm256_*` intrinsics
    // even when the surrounding TU is compiled without `-mavx2`.
    // ------------------------------------------------------------------
    SIMD_ARRAY_AVX2_TARGET static vec_t loadu(const T* p) noexcept {
        if constexpr (std::is_same_v<T, float>) return _mm256_loadu_ps(p);
        else                                    return _mm256_loadu_pd(p);
    }
    SIMD_ARRAY_AVX2_TARGET static void storeu(T* p, vec_t v) noexcept {
        if constexpr (std::is_same_v<T, float>) _mm256_storeu_ps(p, v);
        else                                    _mm256_storeu_pd(p, v);
    }
    SIMD_ARRAY_AVX2_TARGET static vec_t set1(T x) noexcept {
        if constexpr (std::is_same_v<T, float>) return _mm256_set1_ps(x);
        else                                    return _mm256_set1_pd(x);
    }
    SIMD_ARRAY_AVX2_TARGET static vec_t setzero() noexcept {
        if constexpr (std::is_same_v<T, float>) return _mm256_setzero_ps();
        else                                    return _mm256_setzero_pd();
    }
    SIMD_ARRAY_AVX2_TARGET static vec_t add_v(vec_t a, vec_t b) noexcept {
        if constexpr (std::is_same_v<T, float>) return _mm256_add_ps(a, b);
        else                                    return _mm256_add_pd(a, b);
    }
    SIMD_ARRAY_AVX2_TARGET static vec_t sub_v(vec_t a, vec_t b) noexcept {
        if constexpr (std::is_same_v<T, float>) return _mm256_sub_ps(a, b);
        else                                    return _mm256_sub_pd(a, b);
    }
    SIMD_ARRAY_AVX2_TARGET static vec_t mul_v(vec_t a, vec_t b) noexcept {
        if constexpr (std::is_same_v<T, float>) return _mm256_mul_ps(a, b);
        else                                    return _mm256_mul_pd(a, b);
    }
    SIMD_ARRAY_AVX2_TARGET static vec_t div_v(vec_t a, vec_t b) noexcept {
        if constexpr (std::is_same_v<T, float>) return _mm256_div_ps(a, b);
        else                                    return _mm256_div_pd(a, b);
    }
    SIMD_ARRAY_AVX2_TARGET static vec_t min_v(vec_t a, vec_t b) noexcept {
        if constexpr (std::is_same_v<T, float>) return _mm256_min_ps(a, b);
        else                                    return _mm256_min_pd(a, b);
    }
    SIMD_ARRAY_AVX2_TARGET static vec_t max_v(vec_t a, vec_t b) noexcept {
        if constexpr (std::is_same_v<T, float>) return _mm256_max_ps(a, b);
        else                                    return _mm256_max_pd(a, b);
    }

public:
    // ------------------------------------------------------------------
    // Element-wise binary operations.
    // ------------------------------------------------------------------
    SIMD_ARRAY_AVX2_TARGET
    void add(T* out, const T* a, const T* b, std::size_t n) const noexcept override {
        std::size_t i = 0;
        for (; i + W <= n; i += W) {
            storeu(out + i, add_v(loadu(a + i), loadu(b + i)));
        }
        for (; i < n; ++i) out[i] = a[i] + b[i];
    }
    SIMD_ARRAY_AVX2_TARGET
    void subtract(T* out, const T* a, const T* b, std::size_t n) const noexcept override {
        std::size_t i = 0;
        for (; i + W <= n; i += W) {
            storeu(out + i, sub_v(loadu(a + i), loadu(b + i)));
        }
        for (; i < n; ++i) out[i] = a[i] - b[i];
    }
    SIMD_ARRAY_AVX2_TARGET
    void multiply(T* out, const T* a, const T* b, std::size_t n) const noexcept override {
        std::size_t i = 0;
        for (; i + W <= n; i += W) {
            storeu(out + i, mul_v(loadu(a + i), loadu(b + i)));
        }
        for (; i < n; ++i) out[i] = a[i] * b[i];
    }
    SIMD_ARRAY_AVX2_TARGET
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
    SIMD_ARRAY_AVX2_TARGET
    void add_scalar(T* out, const T* a, T scalar, std::size_t n) const noexcept override {
        const vec_t vs = set1(scalar);
        std::size_t i = 0;
        for (; i + W <= n; i += W) {
            storeu(out + i, add_v(loadu(a + i), vs));
        }
        for (; i < n; ++i) out[i] = a[i] + scalar;
    }
    SIMD_ARRAY_AVX2_TARGET
    void multiply_scalar(T* out, const T* a, T scalar, std::size_t n) const noexcept override {
        const vec_t vs = set1(scalar);
        std::size_t i = 0;
        for (; i + W <= n; i += W) {
            storeu(out + i, mul_v(loadu(a + i), vs));
        }
        for (; i < n; ++i) out[i] = a[i] * scalar;
    }

    // ------------------------------------------------------------------
    // Reductions. As in the SSE backend, the W partial accumulators are
    // collapsed sequentially after the main loop, so the floating-point
    // reduction order differs from the scalar reference; results match
    // bit-for-bit only when intermediate values stay exactly representable.
    // ------------------------------------------------------------------
    SIMD_ARRAY_AVX2_TARGET
    T sum(const T* a, std::size_t n) const noexcept override {
        vec_t vs = setzero();
        std::size_t i = 0;
        for (; i + W <= n; i += W) {
            vs = add_v(vs, loadu(a + i));
        }
        alignas(32) T buf[W];
        storeu(buf, vs);
        T s = T{};
        for (std::size_t k = 0; k < W; ++k) s += buf[k];
        for (; i < n; ++i) s += a[i];
        return s;
    }

    SIMD_ARRAY_AVX2_TARGET
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
            alignas(32) T buf[W];
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

    SIMD_ARRAY_AVX2_TARGET
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
            alignas(32) T buf[W];
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
    SIMD_ARRAY_AVX2_TARGET
    void fill(T* a, T value, std::size_t n) const noexcept override {
        const vec_t v = set1(value);
        std::size_t i = 0;
        for (; i + W <= n; i += W) {
            storeu(a + i, v);
        }
        for (; i < n; ++i) a[i] = value;
    }

    const char* name() const noexcept override { return "avx2"; }
};

} // namespace simd
