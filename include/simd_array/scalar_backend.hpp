#pragma once

/// \file scalar_backend.hpp
/// Plain scalar reference implementation of `ArrayBackend<T>`.
///
/// Two roles:
///   1. Universal fallback when no SIMD ISA is available (the dispatcher
///      always lands here on unsupported hardware).
///   2. **Golden reference** for the correctness tests of every SIMD
///      backend. SSE / AVX / AVX-512 outputs must match the scalar result
///      bit-for-bit on the same inputs (we drive each kernel with the same
///      operation order, so the IEEE-754 result is deterministic).
///
/// The loops are written deliberately straightforwardly — no unrolling, no
/// `restrict`, no compiler hints. The point of this backend is to be the
/// boring, obviously-correct reference, not to compete with the SIMD ones.

#include "backend.hpp"

#include <cstddef>

namespace simd {

template<typename T>
class ScalarBackend final : public ArrayBackend<T> {
public:
    void add(T* out, const T* a, const T* b, std::size_t n) const noexcept override {
        for (std::size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
    }
    void subtract(T* out, const T* a, const T* b, std::size_t n) const noexcept override {
        for (std::size_t i = 0; i < n; ++i) out[i] = a[i] - b[i];
    }
    void multiply(T* out, const T* a, const T* b, std::size_t n) const noexcept override {
        for (std::size_t i = 0; i < n; ++i) out[i] = a[i] * b[i];
    }
    void divide(T* out, const T* a, const T* b, std::size_t n) const noexcept override {
        for (std::size_t i = 0; i < n; ++i) out[i] = a[i] / b[i];
    }

    void add_scalar(T* out, const T* a, T scalar, std::size_t n) const noexcept override {
        for (std::size_t i = 0; i < n; ++i) out[i] = a[i] + scalar;
    }
    void multiply_scalar(T* out, const T* a, T scalar, std::size_t n) const noexcept override {
        for (std::size_t i = 0; i < n; ++i) out[i] = a[i] * scalar;
    }

    T sum(const T* a, std::size_t n) const noexcept override {
        T s = T{};
        for (std::size_t i = 0; i < n; ++i) s += a[i];
        return s;
    }
    T min(const T* a, std::size_t n) const noexcept override {
        if (n == 0) return T{};
        T m = a[0];
        for (std::size_t i = 1; i < n; ++i) if (a[i] < m) m = a[i];
        return m;
    }
    T max(const T* a, std::size_t n) const noexcept override {
        if (n == 0) return T{};
        T m = a[0];
        for (std::size_t i = 1; i < n; ++i) if (a[i] > m) m = a[i];
        return m;
    }

    void fill(T* a, T value, std::size_t n) const noexcept override {
        for (std::size_t i = 0; i < n; ++i) a[i] = value;
    }

    const char* name() const noexcept override { return "scalar"; }
};

} // namespace simd
