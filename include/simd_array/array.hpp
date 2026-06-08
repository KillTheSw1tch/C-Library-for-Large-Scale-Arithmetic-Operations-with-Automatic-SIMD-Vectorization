#pragma once

/// \file array.hpp
/// Public entry point of the simd_array library.
///
/// `simd::Array<T>` is a contiguous, 64-byte-aligned container of `T`
/// (`float` or `double`). All arithmetic operators dispatch through the
/// runtime backend selector (`detail::dispatch_backend<T>()`) which picks
/// the highest SIMD ISA the CPU + OS pair supports — AVX-512F, then AVX2,
/// then SSE2, then a scalar fallback. The dispatch table is initialised
/// once per type via `std::call_once` and cached for the program lifetime,
/// so every subsequent operation is a `static const&` load and a virtual
/// call (target: < 5 ns of dispatch overhead, per the thesis spec).
///
/// Above a size threshold (`detail::parallel_min`) operations are split
/// across the global thread pool (`simd::default_pool()`) into chunks
/// rounded up to the SIMD lane count — so each chunk's inner kernel still
/// has the same shape (vector body + scalar tail) as in the single-thread
/// case. Below the threshold the operator runs inline on the caller's
/// thread to avoid pool overhead dwarfing the work.
///
/// Typical usage:
/// \code
///   simd::Array<float> a(1'000'000, 1.0f);
///   simd::Array<float> b(1'000'000, 2.0f);
///   simd::Array<float> c = a + b;          // SIMD + threads automatically
///   const float total   = simd::sum(c);     // 3'000'000.0f
///   const char* backend = simd::Array<float>::backend_name();
/// \endcode

#include "avx512_backend.hpp"
#include "avx_backend.hpp"
#include "backend.hpp"
#include "cpu_features.hpp"
#include "scalar_backend.hpp"
#include "sse_backend.hpp"
#include "thread_pool.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef _WIN32
    #include <malloc.h>
#endif

namespace simd {

namespace detail {

/// Default allocation alignment. 64 bytes covers AVX-512 ZMM loads / a full
/// cache line on every current x86-64 microarchitecture; over-aligning the
/// SSE/AVX cases costs at most one cache line per allocation.
inline constexpr std::size_t default_alignment = 64;

/// Below this element count, parallel dispatch overhead exceeds the SIMD
/// work, so we bypass the thread pool entirely.
inline constexpr std::size_t parallel_min = 32 * 1024;

/// Multiple-of-this lane count for chunk sizing. 16 = AVX-512F float lanes
/// — the widest SIMD width we support. Rounding chunk sizes up to this
/// boundary guarantees each parallel chunk's inner SIMD body still hits
/// full vector width and never has its own per-chunk tail.
inline constexpr std::size_t simd_lane_unit = 16;

template<typename T>
inline T* alloc_aligned(std::size_t count, std::size_t alignment = default_alignment) {
    if (count == 0) return nullptr;
#ifdef _WIN32
    void* p = _aligned_malloc(count * sizeof(T), alignment);
    if (!p) throw std::bad_alloc();
#else
    void* p = nullptr;
    if (posix_memalign(&p, alignment, count * sizeof(T)) != 0 || !p) {
        throw std::bad_alloc();
    }
#endif
    return static_cast<T*>(p);
}

inline void free_aligned(void* p) noexcept {
    if (!p) return;
#ifdef _WIN32
    _aligned_free(p);
#else
    std::free(p);
#endif
}

/// Returns a stable reference to the backend chosen for `T`. The selection
/// runs exactly once per `T` per program (guarded by `std::call_once`), so
/// subsequent calls are a function-local-static load.
template<typename T>
inline const ArrayBackend<T>& dispatch_backend() {
    static std::unique_ptr<ArrayBackend<T>> chosen;
    static std::once_flag flag;
    std::call_once(flag, [] {
        const auto& f = cpu_features();
        if      (f.has_avx512f) chosen = std::make_unique<AVX512Backend<T>>();
        else if (f.has_avx2)    chosen = std::make_unique<AVXBackend<T>>();
        else if (f.has_sse2)    chosen = std::make_unique<SSEBackend<T>>();
        else                    chosen = std::make_unique<ScalarBackend<T>>();
    });
    return *chosen;
}

/// Compute a chunk size that splits `n` elements roughly evenly across
/// (4 × thread_count) chunks, then rounds up to a multiple of
/// `simd_lane_unit` so each chunk is a clean SIMD-width problem.
inline std::size_t parallel_chunk_size(std::size_t n) {
    std::size_t threads = default_pool().thread_count();
    if (threads == 0) threads = 1;
    const std::size_t target = threads * 4;
    std::size_t chunk = (n + target - 1) / target;
    chunk = ((chunk + simd_lane_unit - 1) / simd_lane_unit) * simd_lane_unit;
    if (chunk == 0) chunk = simd_lane_unit;
    return chunk;
}

/// Common control-flow wrapper for elementwise operations: short arrays
/// run inline; long arrays go through the pool with SIMD-aligned chunks.
template<typename Op>
inline void run_partitioned(std::size_t n, Op&& op) {
    if (n == 0) return;
    if (n < parallel_min) {
        op(std::size_t{0}, n);
        return;
    }
    default_pool().parallel_for(n, parallel_chunk_size(n), std::forward<Op>(op));
}

inline void check_same_size(std::size_t a, std::size_t b) {
    if (a != b) throw std::invalid_argument("simd::Array size mismatch");
}

} // namespace detail

// ============================================================================
// Array<T>
// ============================================================================

template<typename T>
class Array {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "simd::Array only supports float and double");

public:
    using value_type = T;

    // ---- construction ------------------------------------------------------

    Array() noexcept = default;

    /// Allocate `n` elements, zero-initialised. Bit-zero is `+0` in
    /// IEEE-754, so a `memset` is a valid value-initialisation for both
    /// `float` and `double`.
    explicit Array(std::size_t n)
        : data_(detail::alloc_aligned<T>(n))
        , size_(n) {
        if (data_) std::memset(data_, 0, n * sizeof(T));
    }

    Array(std::size_t n, T value) : Array(n) { fill(value); }

    Array(std::initializer_list<T> init) : Array(init.size()) {
        std::size_t i = 0;
        for (T v : init) data_[i++] = v;
    }

    Array(const Array& other) : Array(other.size_) {
        if (size_) std::memcpy(data_, other.data_, size_ * sizeof(T));
    }

    Array(Array&& other) noexcept
        : data_(other.data_)
        , size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    Array& operator=(const Array& other) {
        if (this != &other) {
            Array tmp(other);
            swap(tmp);
        }
        return *this;
    }

    Array& operator=(Array&& other) noexcept {
        if (this != &other) {
            detail::free_aligned(data_);
            data_       = other.data_;
            size_       = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    ~Array() { detail::free_aligned(data_); }

    void swap(Array& other) noexcept {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
    }

    // ---- element access ----------------------------------------------------

    T*          data()        noexcept { return data_; }
    const T*    data()  const noexcept { return data_; }
    std::size_t size()  const noexcept { return size_; }
    bool        empty() const noexcept { return size_ == 0; }
    T&          operator[](std::size_t i)       noexcept { return data_[i]; }
    const T&    operator[](std::size_t i) const noexcept { return data_[i]; }

    // ---- mutation ----------------------------------------------------------

    void fill(T value) {
        const auto& be = detail::dispatch_backend<T>();
        detail::run_partitioned(size_, [&](std::size_t b, std::size_t e) {
            be.fill(data_ + b, value, e - b);
        });
    }

    // ---- compound assignment with another Array ---------------------------

    Array& operator+=(const Array& rhs) {
        detail::check_same_size(size_, rhs.size_);
        const auto& be = detail::dispatch_backend<T>();
        detail::run_partitioned(size_, [&](std::size_t b, std::size_t e) {
            be.add(data_ + b, data_ + b, rhs.data_ + b, e - b);
        });
        return *this;
    }
    Array& operator-=(const Array& rhs) {
        detail::check_same_size(size_, rhs.size_);
        const auto& be = detail::dispatch_backend<T>();
        detail::run_partitioned(size_, [&](std::size_t b, std::size_t e) {
            be.subtract(data_ + b, data_ + b, rhs.data_ + b, e - b);
        });
        return *this;
    }
    Array& operator*=(const Array& rhs) {
        detail::check_same_size(size_, rhs.size_);
        const auto& be = detail::dispatch_backend<T>();
        detail::run_partitioned(size_, [&](std::size_t b, std::size_t e) {
            be.multiply(data_ + b, data_ + b, rhs.data_ + b, e - b);
        });
        return *this;
    }
    Array& operator/=(const Array& rhs) {
        detail::check_same_size(size_, rhs.size_);
        const auto& be = detail::dispatch_backend<T>();
        detail::run_partitioned(size_, [&](std::size_t b, std::size_t e) {
            be.divide(data_ + b, data_ + b, rhs.data_ + b, e - b);
        });
        return *this;
    }

    // ---- compound assignment with a scalar --------------------------------

    Array& operator+=(T scalar) {
        const auto& be = detail::dispatch_backend<T>();
        detail::run_partitioned(size_, [&](std::size_t b, std::size_t e) {
            be.add_scalar(data_ + b, data_ + b, scalar, e - b);
        });
        return *this;
    }
    Array& operator*=(T scalar) {
        const auto& be = detail::dispatch_backend<T>();
        detail::run_partitioned(size_, [&](std::size_t b, std::size_t e) {
            be.multiply_scalar(data_ + b, data_ + b, scalar, e - b);
        });
        return *this;
    }

    // ---- reductions --------------------------------------------------------

    T sum() const {
        if (size_ == 0) return T{};
        const auto& be = detail::dispatch_backend<T>();
        if (size_ < detail::parallel_min) {
            return be.sum(data_, size_);
        }
        const std::size_t chunk      = detail::parallel_chunk_size(size_);
        const std::size_t num_chunks = (size_ + chunk - 1) / chunk;
        std::vector<T> partials(num_chunks, T{});
        default_pool().parallel_for(size_, chunk, [&](std::size_t b, std::size_t e) {
            partials[b / chunk] = be.sum(data_ + b, e - b);
        });
        T total = T{};
        for (T p : partials) total += p;
        return total;
    }

    T min() const {
        const auto& be = detail::dispatch_backend<T>();
        if (size_ == 0) return T{};
        if (size_ < detail::parallel_min) {
            return be.min(data_, size_);
        }
        const std::size_t chunk      = detail::parallel_chunk_size(size_);
        const std::size_t num_chunks = (size_ + chunk - 1) / chunk;
        std::vector<T> partials(num_chunks);
        default_pool().parallel_for(size_, chunk, [&](std::size_t b, std::size_t e) {
            partials[b / chunk] = be.min(data_ + b, e - b);
        });
        T m = partials[0];
        for (std::size_t k = 1; k < partials.size(); ++k) if (partials[k] < m) m = partials[k];
        return m;
    }

    T max() const {
        const auto& be = detail::dispatch_backend<T>();
        if (size_ == 0) return T{};
        if (size_ < detail::parallel_min) {
            return be.max(data_, size_);
        }
        const std::size_t chunk      = detail::parallel_chunk_size(size_);
        const std::size_t num_chunks = (size_ + chunk - 1) / chunk;
        std::vector<T> partials(num_chunks);
        default_pool().parallel_for(size_, chunk, [&](std::size_t b, std::size_t e) {
            partials[b / chunk] = be.max(data_ + b, e - b);
        });
        T m = partials[0];
        for (std::size_t k = 1; k < partials.size(); ++k) if (partials[k] > m) m = partials[k];
        return m;
    }

    /// Identifier of the backend chosen for `T` at runtime. Stable for the
    /// lifetime of the program. Useful in tests and diagnostic prints.
    static const char* backend_name() {
        return detail::dispatch_backend<T>().name();
    }

private:
    T*          data_ = nullptr;
    std::size_t size_ = 0;
};

template<typename T>
inline void swap(Array<T>& a, Array<T>& b) noexcept {
    a.swap(b);
}

// ============================================================================
// Free-function operators returning a new Array.
// ============================================================================

template<typename T>
inline Array<T> operator+(const Array<T>& a, const Array<T>& b) {
    detail::check_same_size(a.size(), b.size());
    Array<T> out(a.size());
    const auto& be = detail::dispatch_backend<T>();
    detail::run_partitioned(a.size(), [&](std::size_t bgn, std::size_t end) {
        be.add(out.data() + bgn, a.data() + bgn, b.data() + bgn, end - bgn);
    });
    return out;
}

template<typename T>
inline Array<T> operator-(const Array<T>& a, const Array<T>& b) {
    detail::check_same_size(a.size(), b.size());
    Array<T> out(a.size());
    const auto& be = detail::dispatch_backend<T>();
    detail::run_partitioned(a.size(), [&](std::size_t bgn, std::size_t end) {
        be.subtract(out.data() + bgn, a.data() + bgn, b.data() + bgn, end - bgn);
    });
    return out;
}

template<typename T>
inline Array<T> operator*(const Array<T>& a, const Array<T>& b) {
    detail::check_same_size(a.size(), b.size());
    Array<T> out(a.size());
    const auto& be = detail::dispatch_backend<T>();
    detail::run_partitioned(a.size(), [&](std::size_t bgn, std::size_t end) {
        be.multiply(out.data() + bgn, a.data() + bgn, b.data() + bgn, end - bgn);
    });
    return out;
}

template<typename T>
inline Array<T> operator/(const Array<T>& a, const Array<T>& b) {
    detail::check_same_size(a.size(), b.size());
    Array<T> out(a.size());
    const auto& be = detail::dispatch_backend<T>();
    detail::run_partitioned(a.size(), [&](std::size_t bgn, std::size_t end) {
        be.divide(out.data() + bgn, a.data() + bgn, b.data() + bgn, end - bgn);
    });
    return out;
}

template<typename T>
inline Array<T> operator+(const Array<T>& a, T scalar) {
    Array<T> out(a.size());
    const auto& be = detail::dispatch_backend<T>();
    detail::run_partitioned(a.size(), [&](std::size_t bgn, std::size_t end) {
        be.add_scalar(out.data() + bgn, a.data() + bgn, scalar, end - bgn);
    });
    return out;
}

template<typename T>
inline Array<T> operator+(T scalar, const Array<T>& a) { return a + scalar; }

template<typename T>
inline Array<T> operator*(const Array<T>& a, T scalar) {
    Array<T> out(a.size());
    const auto& be = detail::dispatch_backend<T>();
    detail::run_partitioned(a.size(), [&](std::size_t bgn, std::size_t end) {
        be.multiply_scalar(out.data() + bgn, a.data() + bgn, scalar, end - bgn);
    });
    return out;
}

template<typename T>
inline Array<T> operator*(T scalar, const Array<T>& a) { return a * scalar; }

// ============================================================================
// Free-function reductions and fused kernels.
// ============================================================================

template<typename T> inline T sum(const Array<T>& a) { return a.sum(); }
template<typename T> inline T min(const Array<T>& a) { return a.min(); }
template<typename T> inline T max(const Array<T>& a) { return a.max(); }

/// Dot product. Currently materialises a temporary `a * b` and reduces;
/// a fused multiply-add kernel can replace this once the backend gains an
/// FMA-based dot kernel without changing this signature.
template<typename T>
inline T dot_product(const Array<T>& a, const Array<T>& b) {
    detail::check_same_size(a.size(), b.size());
    if (a.empty()) return T{};
    Array<T> tmp(a.size());
    const auto& be = detail::dispatch_backend<T>();
    detail::run_partitioned(a.size(), [&](std::size_t bgn, std::size_t end) {
        be.multiply(tmp.data() + bgn, a.data() + bgn, b.data() + bgn, end - bgn);
    });
    return tmp.sum();
}

} // namespace simd
