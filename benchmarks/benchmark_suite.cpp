/// \file benchmark_suite.cpp
/// Comprehensive benchmark suite: add, multiply, sum (reduction), dot product.
/// Float only. Three canonical sizes chosen to stress different cache levels:
///   N0 =   10 000  →  ~40 KB  — fits in L1 data cache
///   N1 =  100 000  → ~400 KB  — fits in L2 cache
///   N2 = 1 000 000 →   ~4 MB  — main-memory bound (thesis headline case)
///
/// Output is consumed by visualize.py to generate the thesis performance charts.
///
/// Run:
///   benchmark_suite --benchmark_out=results.json --benchmark_out_format=json
///   python benchmarks/visualize.py results.json

#include <simd_array/array.hpp>

#include <benchmark/benchmark.h>
#include <cstddef>
#include <cstdint>

static constexpr int64_t N0 =     10'000;
static constexpr int64_t N1 =    100'000;
static constexpr int64_t N2 =  1'000'000;

// ---------------------------------------------------------------------------
// Shared computation helpers — accept a backend reference to avoid duplication.
// ---------------------------------------------------------------------------

static void run_add(benchmark::State& state, simd::ArrayBackend<float>& be) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    simd::Array<float> a(n, 1.0f), b(n, 2.0f), out(n);
    for (auto _ : state) {
        be.add(out.data(), a.data(), b.data(), n);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(n) * 3 * sizeof(float));
}

static void run_mul(benchmark::State& state, simd::ArrayBackend<float>& be) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    simd::Array<float> a(n, 1.5f), b(n, 2.0f), out(n);
    for (auto _ : state) {
        be.multiply(out.data(), a.data(), b.data(), n);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(n) * 3 * sizeof(float));
}

static void run_sum(benchmark::State& state, simd::ArrayBackend<float>& be) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    simd::Array<float> a(n, 1.0f);
    for (auto _ : state) {
        float s = be.sum(a.data(), n);
        benchmark::DoNotOptimize(s);
    }
    // One read pass over the array.
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(n) * sizeof(float));
}

// Dot product: multiply into a pre-allocated tmp, then reduce.
// The tmp allocation is outside the hot loop for a fair comparison
// with the simd::Array version below.
static void run_dot(benchmark::State& state, simd::ArrayBackend<float>& be) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    simd::Array<float> a(n, 1.0f), b(n, 2.0f), tmp(n);
    for (auto _ : state) {
        be.multiply(tmp.data(), a.data(), b.data(), n);
        float s = be.sum(tmp.data(), n);
        benchmark::DoNotOptimize(s);
        benchmark::ClobberMemory();
    }
    // Two reads (a, b) + one read (tmp for sum): ~3 passes.
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(n) * 3 * sizeof(float));
}

// ============================================================================
// ADD
// ============================================================================

static void BM_Suite_Scalar_Add(benchmark::State& state) {
    static simd::ScalarBackend<float> be;
    run_add(state, be);
}
BENCHMARK(BM_Suite_Scalar_Add)->Arg(N0)->Arg(N1)->Arg(N2)->Unit(benchmark::kMicrosecond);

static void BM_Suite_SSE_Add(benchmark::State& state) {
    if (!simd::cpu_features().has_sse2) { state.SkipWithError("SSE2 not available"); return; }
    static simd::SSEBackend<float> be;
    run_add(state, be);
}
BENCHMARK(BM_Suite_SSE_Add)->Arg(N0)->Arg(N1)->Arg(N2)->Unit(benchmark::kMicrosecond);

static void BM_Suite_AVX_Add(benchmark::State& state) {
    if (!simd::cpu_features().has_avx2) { state.SkipWithError("AVX2 not available"); return; }
    static simd::AVXBackend<float> be;
    run_add(state, be);
}
BENCHMARK(BM_Suite_AVX_Add)->Arg(N0)->Arg(N1)->Arg(N2)->Unit(benchmark::kMicrosecond);

static void BM_Suite_AVX512_Add(benchmark::State& state) {
    if (!simd::cpu_features().has_avx512f) { state.SkipWithError("AVX-512F not available"); return; }
    static simd::AVX512Backend<float> be;
    run_add(state, be);
}
BENCHMARK(BM_Suite_AVX512_Add)->Arg(N0)->Arg(N1)->Arg(N2)->Unit(benchmark::kMicrosecond);

// simd::Array operator+= uses the auto-dispatched backend and the thread pool.
static void BM_Suite_Array_Add(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    simd::Array<float> a(n, 1.0f), b(n, 2.0f);
    for (auto _ : state) {
        a += b;
        benchmark::DoNotOptimize(a.data());
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(n) * 3 * sizeof(float));
}
BENCHMARK(BM_Suite_Array_Add)->Arg(N0)->Arg(N1)->Arg(N2)->Unit(benchmark::kMicrosecond);

// ============================================================================
// MULTIPLY
// ============================================================================

static void BM_Suite_Scalar_Mul(benchmark::State& state) {
    static simd::ScalarBackend<float> be;
    run_mul(state, be);
}
BENCHMARK(BM_Suite_Scalar_Mul)->Arg(N0)->Arg(N1)->Arg(N2)->Unit(benchmark::kMicrosecond);

static void BM_Suite_SSE_Mul(benchmark::State& state) {
    if (!simd::cpu_features().has_sse2) { state.SkipWithError("SSE2 not available"); return; }
    static simd::SSEBackend<float> be;
    run_mul(state, be);
}
BENCHMARK(BM_Suite_SSE_Mul)->Arg(N0)->Arg(N1)->Arg(N2)->Unit(benchmark::kMicrosecond);

static void BM_Suite_AVX_Mul(benchmark::State& state) {
    if (!simd::cpu_features().has_avx2) { state.SkipWithError("AVX2 not available"); return; }
    static simd::AVXBackend<float> be;
    run_mul(state, be);
}
BENCHMARK(BM_Suite_AVX_Mul)->Arg(N0)->Arg(N1)->Arg(N2)->Unit(benchmark::kMicrosecond);

static void BM_Suite_AVX512_Mul(benchmark::State& state) {
    if (!simd::cpu_features().has_avx512f) { state.SkipWithError("AVX-512F not available"); return; }
    static simd::AVX512Backend<float> be;
    run_mul(state, be);
}
BENCHMARK(BM_Suite_AVX512_Mul)->Arg(N0)->Arg(N1)->Arg(N2)->Unit(benchmark::kMicrosecond);

static void BM_Suite_Array_Mul(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    simd::Array<float> a(n, 1.5f), b(n, 2.0f);
    for (auto _ : state) {
        // Values accumulate toward +inf after ~127 iters, but float mul stays
        // full-throughput on inf operands — no perf anomaly, no denormals.
        a *= b;
        benchmark::DoNotOptimize(a.data());
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(n) * 3 * sizeof(float));
}
BENCHMARK(BM_Suite_Array_Mul)->Arg(N0)->Arg(N1)->Arg(N2)->Unit(benchmark::kMicrosecond);

// ============================================================================
// SUM (reduction)
// ============================================================================

static void BM_Suite_Scalar_Sum(benchmark::State& state) {
    static simd::ScalarBackend<float> be;
    run_sum(state, be);
}
BENCHMARK(BM_Suite_Scalar_Sum)->Arg(N0)->Arg(N1)->Arg(N2)->Unit(benchmark::kMicrosecond);

static void BM_Suite_SSE_Sum(benchmark::State& state) {
    if (!simd::cpu_features().has_sse2) { state.SkipWithError("SSE2 not available"); return; }
    static simd::SSEBackend<float> be;
    run_sum(state, be);
}
BENCHMARK(BM_Suite_SSE_Sum)->Arg(N0)->Arg(N1)->Arg(N2)->Unit(benchmark::kMicrosecond);

static void BM_Suite_AVX_Sum(benchmark::State& state) {
    if (!simd::cpu_features().has_avx2) { state.SkipWithError("AVX2 not available"); return; }
    static simd::AVXBackend<float> be;
    run_sum(state, be);
}
BENCHMARK(BM_Suite_AVX_Sum)->Arg(N0)->Arg(N1)->Arg(N2)->Unit(benchmark::kMicrosecond);

static void BM_Suite_AVX512_Sum(benchmark::State& state) {
    if (!simd::cpu_features().has_avx512f) { state.SkipWithError("AVX-512F not available"); return; }
    static simd::AVX512Backend<float> be;
    run_sum(state, be);
}
BENCHMARK(BM_Suite_AVX512_Sum)->Arg(N0)->Arg(N1)->Arg(N2)->Unit(benchmark::kMicrosecond);

// simd::Array::sum() uses the dispatched backend + thread pool for large N.
static void BM_Suite_Array_Sum(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    simd::Array<float> a(n, 1.0f);
    for (auto _ : state) {
        float s = simd::sum(a);
        benchmark::DoNotOptimize(s);
    }
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(n) * sizeof(float));
}
BENCHMARK(BM_Suite_Array_Sum)->Arg(N0)->Arg(N1)->Arg(N2)->Unit(benchmark::kMicrosecond);

// ============================================================================
// DOT PRODUCT (multiply + reduce)
// ============================================================================

static void BM_Suite_Scalar_Dot(benchmark::State& state) {
    static simd::ScalarBackend<float> be;
    run_dot(state, be);
}
BENCHMARK(BM_Suite_Scalar_Dot)->Arg(N0)->Arg(N1)->Arg(N2)->Unit(benchmark::kMicrosecond);

static void BM_Suite_SSE_Dot(benchmark::State& state) {
    if (!simd::cpu_features().has_sse2) { state.SkipWithError("SSE2 not available"); return; }
    static simd::SSEBackend<float> be;
    run_dot(state, be);
}
BENCHMARK(BM_Suite_SSE_Dot)->Arg(N0)->Arg(N1)->Arg(N2)->Unit(benchmark::kMicrosecond);

static void BM_Suite_AVX_Dot(benchmark::State& state) {
    if (!simd::cpu_features().has_avx2) { state.SkipWithError("AVX2 not available"); return; }
    static simd::AVXBackend<float> be;
    run_dot(state, be);
}
BENCHMARK(BM_Suite_AVX_Dot)->Arg(N0)->Arg(N1)->Arg(N2)->Unit(benchmark::kMicrosecond);

static void BM_Suite_AVX512_Dot(benchmark::State& state) {
    if (!simd::cpu_features().has_avx512f) { state.SkipWithError("AVX-512F not available"); return; }
    static simd::AVX512Backend<float> be;
    run_dot(state, be);
}
BENCHMARK(BM_Suite_AVX512_Dot)->Arg(N0)->Arg(N1)->Arg(N2)->Unit(benchmark::kMicrosecond);

// simd::Array::dot uses auto-dispatch + thread pool (includes one allocation
// inside simd::dot_product for the temporary multiply buffer).
static void BM_Suite_Array_Dot(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    simd::Array<float> a(n, 1.0f), b(n, 2.0f);
    for (auto _ : state) {
        float s = simd::dot_product(a, b);
        benchmark::DoNotOptimize(s);
    }
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(n) * 3 * sizeof(float));
}
BENCHMARK(BM_Suite_Array_Dot)->Arg(N0)->Arg(N1)->Arg(N2)->Unit(benchmark::kMicrosecond);
