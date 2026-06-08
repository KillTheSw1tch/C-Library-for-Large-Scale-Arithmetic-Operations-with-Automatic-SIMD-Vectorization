/// \file benchmark_multiply.cpp
/// Element-wise floating-point multiplication across all backends and array sizes.
///
/// Run:
///   benchmark_multiply --benchmark_out=mul_results.json --benchmark_out_format=json

#include <simd_array/array.hpp>

#include <benchmark/benchmark.h>
#include <cstddef>
#include <cstdint>

// ============================================================================
// Scalar backend
// ============================================================================

template<typename T>
static void BM_Scalar_Mul(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    simd::Array<T> a(n, T{1.5}), b(n, T{2}), out(n);
    simd::ScalarBackend<T> be;
    for (auto _ : state) {
        be.multiply(out.data(), a.data(), b.data(), n);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(n) * 3 * sizeof(T));
}
BENCHMARK_TEMPLATE(BM_Scalar_Mul, float)
    ->RangeMultiplier(8)->Range(1024, 4 * 1024 * 1024)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_Scalar_Mul, double)
    ->RangeMultiplier(8)->Range(1024, 4 * 1024 * 1024)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// SSE2 backend
// ============================================================================

template<typename T>
static void BM_SSE_Mul(benchmark::State& state) {
    if (!simd::cpu_features().has_sse2) {
        state.SkipWithError("SSE2 not available");
        return;
    }
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    simd::Array<T> a(n, T{1.5}), b(n, T{2}), out(n);
    simd::SSEBackend<T> be;
    for (auto _ : state) {
        be.multiply(out.data(), a.data(), b.data(), n);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(n) * 3 * sizeof(T));
}
BENCHMARK_TEMPLATE(BM_SSE_Mul, float)
    ->RangeMultiplier(8)->Range(1024, 4 * 1024 * 1024)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_SSE_Mul, double)
    ->RangeMultiplier(8)->Range(1024, 4 * 1024 * 1024)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// AVX2 backend
// ============================================================================

template<typename T>
static void BM_AVX_Mul(benchmark::State& state) {
    if (!simd::cpu_features().has_avx2) {
        state.SkipWithError("AVX2 not available");
        return;
    }
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    simd::Array<T> a(n, T{1.5}), b(n, T{2}), out(n);
    simd::AVXBackend<T> be;
    for (auto _ : state) {
        be.multiply(out.data(), a.data(), b.data(), n);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(n) * 3 * sizeof(T));
}
BENCHMARK_TEMPLATE(BM_AVX_Mul, float)
    ->RangeMultiplier(8)->Range(1024, 4 * 1024 * 1024)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_AVX_Mul, double)
    ->RangeMultiplier(8)->Range(1024, 4 * 1024 * 1024)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// AVX-512F backend
// ============================================================================

template<typename T>
static void BM_AVX512_Mul(benchmark::State& state) {
    if (!simd::cpu_features().has_avx512f) {
        state.SkipWithError("AVX-512F not available");
        return;
    }
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    simd::Array<T> a(n, T{1.5}), b(n, T{2}), out(n);
    simd::AVX512Backend<T> be;
    for (auto _ : state) {
        be.multiply(out.data(), a.data(), b.data(), n);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(n) * 3 * sizeof(T));
}
BENCHMARK_TEMPLATE(BM_AVX512_Mul, float)
    ->RangeMultiplier(8)->Range(1024, 4 * 1024 * 1024)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_AVX512_Mul, double)
    ->RangeMultiplier(8)->Range(1024, 4 * 1024 * 1024)
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// simd::Array operator*= (auto-dispatch + thread pool for large N).
// Pre-allocated storage — no heap allocation in the hot loop.
// ============================================================================

template<typename T>
static void BM_Array_Mul(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    simd::Array<T> a(n, T{1.5}), b(n, T{2});
    for (auto _ : state) {
        a *= b;
        benchmark::DoNotOptimize(a.data());
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(n) * 3 * sizeof(T));
}
BENCHMARK_TEMPLATE(BM_Array_Mul, float)
    ->RangeMultiplier(8)->Range(1024, 4 * 1024 * 1024)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_Array_Mul, double)
    ->RangeMultiplier(8)->Range(1024, 4 * 1024 * 1024)
    ->Unit(benchmark::kMicrosecond);
