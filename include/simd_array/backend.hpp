#pragma once

/// \file backend.hpp
/// Abstract backend interface for element-wise array operations.
///
/// Each backend implements a fixed set of arithmetic kernels on contiguous
/// buffers of `T`. The runtime dispatch layer (see `array.hpp`) selects one
/// backend per program based on the detected CPU features and forwards every
/// call through this interface.
///
/// Design notes:
///   * Backends are stateless — they own no buffers and never allocate.
///     The caller is responsible for allocation, alignment and bounds.
///   * The interface is deliberately narrow and pointer-based to keep the
///     ABI between dispatcher and backend trivial; higher-level conveniences
///     (operator overloads, expression syntax) live in `array.hpp`.
///   * Aliasing rules: `out == a` and `out == b` are supported (in-place
///     update). Partial overlap of `out` with `a` or `b` is undefined.
///   * Reductions on empty input (`n == 0`) return `T{}`. This is a
///     pragmatic convention — `min`/`max` of nothing has no mathematical
///     meaning, but returning a defined value keeps callers crash-free.

#include <cstddef>
#include <type_traits>

namespace simd {

template<typename T>
class ArrayBackend {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "ArrayBackend only supports float and double");

public:
    virtual ~ArrayBackend() = default;

    // ----- Element-wise binary operations: out[i] = a[i] OP b[i] -----
    virtual void add     (T* out, const T* a, const T* b, std::size_t n) const noexcept = 0;
    virtual void subtract(T* out, const T* a, const T* b, std::size_t n) const noexcept = 0;
    virtual void multiply(T* out, const T* a, const T* b, std::size_t n) const noexcept = 0;
    virtual void divide  (T* out, const T* a, const T* b, std::size_t n) const noexcept = 0;

    // ----- Scalar broadcast: out[i] = a[i] OP scalar -----
    virtual void add_scalar     (T* out, const T* a, T scalar, std::size_t n) const noexcept = 0;
    virtual void multiply_scalar(T* out, const T* a, T scalar, std::size_t n) const noexcept = 0;

    // ----- Reductions -----
    virtual T sum(const T* a, std::size_t n) const noexcept = 0;
    virtual T min(const T* a, std::size_t n) const noexcept = 0;
    virtual T max(const T* a, std::size_t n) const noexcept = 0;

    // ----- Utilities -----
    virtual void fill(T* a, T value, std::size_t n) const noexcept = 0;

    /// Short identifier of the backend, e.g. "scalar", "sse2", "avx2".
    virtual const char* name() const noexcept = 0;
};

} // namespace simd
