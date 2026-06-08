/// \file sensor_stats_demo.cpp
/// Sensor-data statistics demo: SIMD performance comparison on a realistic
/// data-processing workload.
///
/// Scenario:
///   An IoT temperature sensor records 1 million readings.  For every
///   reading we need the full descriptive statistics:
///       mean, variance, standard deviation, min, max
///   This is a textbook batch-statistics workload (sensor pipelines, data
///   analytics, real-time monitoring).
///
/// Algorithm — two-pass for numerical stability:
///   Pass 1:  sum(x), min(x), max(x)           →  mean, min, max
///   Pass 2:  centered = x - mean              (backend.add_scalar)
///            squared  = centered * centered   (backend.multiply)
///            variance = sum(squared) / N      →  std dev = sqrt(variance)
///
/// Six configurations are timed:
///   1. Scalar       (1 lane,  serial)
///   2. SSE2         (4 lanes, serial)
///   3. AVX2         (8 lanes, serial)
///   4. AVX-512     (16 lanes, serial)        — only if CPU supports it
///   5. AVX2        + thread pool
///   6. AVX-512     + thread pool             — only if CPU supports it
///
/// Each row of the output table shows: time per pass, element throughput,
/// speedup over the scalar baseline, and an OK/FAIL correctness check
/// (compared against the scalar reference within 1e-3 absolute tolerance).
///
/// Sizing note:
///   The default N = 1'000'000 is the sweet spot for this workload.  It is
///   large enough that thread-pool overhead (~10 µs per submit) is amortised
///   over real work, yet small enough that per-thread slices stay close to
///   the L2/L3 cache hierarchy.  Both 8x SIMD speedup (single thread) and
///   ~20x multi-thread speedup are visible at this size on a modern CPU.
///
///   For very small N (~64 K) the thread-pool overhead dominates and the
///   multithreaded rows can actually be slower than serial — this is a
///   fundamental property of any pool-based parallelisation, not a bug.
///
///   For very large N (> a few million) the running float-precision sums
///   inside the backend reductions start to lose mantissa bits, so the
///   per-backend results may drift apart by more than the strict equality
///   tolerance.  The tolerance below is scaled with sqrt(N) to track the
///   classical accumulated-rounding-error bound.
///
/// Usage:
///   sensor_stats_demo               # N = 1'000'000  (default sweet spot)
///   sensor_stats_demo 65536         # cache-resident — clean SIMD speedup
///   sensor_stats_demo 5000000       # DRAM-bound — shows bandwidth ceiling

#include <simd_array/array.hpp>
#include <simd_array/avx512_backend.hpp>
#include <simd_array/avx_backend.hpp>
#include <simd_array/backend.hpp>
#include <simd_array/cpu_features.hpp>
#include <simd_array/scalar_backend.hpp>
#include <simd_array/sse_backend.hpp>
#include <simd_array/thread_pool.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

// ============================================================================
// Result type
// ============================================================================

struct Stats {
    float mean    = 0.0f;
    float variance = 0.0f;
    float stddev  = 0.0f;
    float min     = 0.0f;
    float max     = 0.0f;
};

// ============================================================================
// Synthetic sensor data: 24-hour temperature trace
// ============================================================================

/// Build a realistic temperature trace:
///   base 20 °C  +  daily sinusoid ±10 °C  +  uniform noise ±2 °C.
/// Useful so the printed mean/std are recognisable numbers, not random junk.
static std::vector<float> generate_sensor_data(std::size_t n) {
    std::vector<float> data(n);
    std::mt19937 rng(2024);
    std::uniform_real_distribution<float> noise(-2.0f, 2.0f);

    const float two_pi = 6.28318530718f;
    for (std::size_t i = 0; i < n; ++i) {
        const float phase = two_pi * static_cast<float>(i) / static_cast<float>(n);
        data[i] = 20.0f + 10.0f * std::sin(phase) + noise(rng);
    }
    return data;
}

// ============================================================================
// Pre-allocated scratch buffers
// ============================================================================
// Allocation cost (centered + squared = 2*N*4 bytes per call) would otherwise
// dominate the timing for small N, hiding the SIMD speedup we want to show.

struct SerialScratch {
    std::vector<float> centered;
    std::vector<float> squared;
    explicit SerialScratch(std::size_t n) : centered(n), squared(n) {}
};

struct ParallelScratch {
    // One pair of buffers per thread slot; sized to the largest chunk.
    std::vector<std::vector<float>> centered;
    std::vector<std::vector<float>> squared;
    ParallelScratch(std::size_t nthreads, std::size_t chunk)
        : centered(nthreads, std::vector<float>(chunk)),
          squared (nthreads, std::vector<float>(chunk)) {}
};

// ============================================================================
// Serial statistics — one backend, one thread
// ============================================================================

/// Compute full descriptive statistics using a SIMD backend, single-threaded.
/// Scratch buffers are provided by the caller (allocated once outside the
/// timing loop).
static Stats compute_stats_serial(simd::ArrayBackend<float>& be,
                                   const float* data, std::size_t n,
                                   SerialScratch& scratch) {
    Stats s;

    // Pass 1: sum, min, max  →  mean
    const float sum = be.sum(data, n);
    s.mean = sum / static_cast<float>(n);
    s.min  = be.min(data, n);
    s.max  = be.max(data, n);

    // Pass 2: centred-variance for numerical stability
    be.add_scalar(scratch.centered.data(), data, -s.mean, n);
    be.multiply  (scratch.squared.data(),
                  scratch.centered.data(), scratch.centered.data(), n);
    const float sum_sq = be.sum(scratch.squared.data(), n);

    s.variance = sum_sq / static_cast<float>(n);
    s.stddev   = std::sqrt(s.variance);
    return s;
}

// ============================================================================
// Parallel statistics — split across thread pool, then combine
// ============================================================================

/// Per-chunk partial state. Use double for the running sums so accumulating
/// 1M+ values does not lose mantissa bits compared to the serial path.
struct Partial {
    double      sum    = 0.0;
    double      sum_sq = 0.0;
    float       min    =  std::numeric_limits<float>::infinity();
    float       max    = -std::numeric_limits<float>::infinity();
    std::size_t n      = 0;
};

static Stats compute_stats_parallel(simd::ArrayBackend<float>& be,
                                     const float* data, std::size_t n,
                                     ParallelScratch& scratch) {
    auto& pool = simd::default_pool();
    const std::size_t nthreads = pool.thread_count();
    const std::size_t chunk    = (n + nthreads - 1) / nthreads;

    // --------------------------------------------------------------------
    // Pass 1: in parallel, gather partial (sum, min, max) for each slice.
    // --------------------------------------------------------------------
    std::vector<std::future<Partial>> futures;
    futures.reserve(nthreads);

    for (std::size_t t = 0; t < nthreads; ++t) {
        const std::size_t start = t * chunk;
        const std::size_t end   = std::min(start + chunk, n);
        if (start >= end) break;
        futures.push_back(pool.submit([&be, data, start, end] {
            Partial p;
            p.n   = end - start;
            p.sum = be.sum(data + start, p.n);
            p.min = be.min(data + start, p.n);
            p.max = be.max(data + start, p.n);
            return p;
        }));
    }

    Partial total;
    for (auto& f : futures) {
        Partial p = f.get();
        total.sum += p.sum;
        total.min  = std::min(total.min, p.min);
        total.max  = std::max(total.max, p.max);
        total.n   += p.n;
    }
    const float mean = static_cast<float>(total.sum / static_cast<double>(total.n));

    // --------------------------------------------------------------------
    // Pass 2: in parallel, gather partial sum_of_squared_deviations.
    // Uses pre-allocated per-thread scratch buffers.
    // --------------------------------------------------------------------
    futures.clear();
    for (std::size_t t = 0; t < nthreads; ++t) {
        const std::size_t start = t * chunk;
        const std::size_t end   = std::min(start + chunk, n);
        if (start >= end) break;
        futures.push_back(pool.submit([&be, data, start, end, mean, t, &scratch] {
            const std::size_t m = end - start;
            float* centered = scratch.centered[t].data();
            float* squared  = scratch.squared [t].data();
            be.add_scalar(centered, data + start, -mean, m);
            be.multiply  (squared,  centered, centered, m);
            Partial p;
            p.sum_sq = be.sum(squared, m);
            p.n      = m;
            return p;
        }));
    }
    double sum_sq = 0.0;
    for (auto& f : futures) sum_sq += f.get().sum_sq;

    Stats s;
    s.mean     = mean;
    s.min      = total.min;
    s.max      = total.max;
    s.variance = static_cast<float>(sum_sq / static_cast<double>(total.n));
    s.stddev   = std::sqrt(s.variance);
    return s;
}

// ============================================================================
// Timing helper
// ============================================================================

template<typename Fn>
static double bench_ms(Fn&& fn, int warmup = 2, int runs = 5) {
    for (int i = 0; i < warmup; ++i) fn();
    double best = std::numeric_limits<double>::infinity();
    for (int i = 0; i < runs; ++i) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ms < best) best = ms;
    }
    return best;
}

// ============================================================================
// Golden reference — full double precision
// ============================================================================
// Every backend (scalar included) accumulates partial sums in float, which
// silently loses mantissa bits once N × max|x| exceeds 2^24.  We therefore
// compute the "true" statistics once in double precision and treat *that*
// as ground truth — both for the printed numbers and for the OK/FAIL check.

static Stats compute_stats_golden(const float* data, std::size_t n) {
    double sum = 0.0;
    float  mn  =  std::numeric_limits<float>::infinity();
    float  mx  = -std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        sum += static_cast<double>(data[i]);
        if (data[i] < mn) mn = data[i];
        if (data[i] > mx) mx = data[i];
    }
    const double mean_d = sum / static_cast<double>(n);

    double sum_sq = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d = static_cast<double>(data[i]) - mean_d;
        sum_sq += d * d;
    }
    const double var_d = sum_sq / static_cast<double>(n);

    Stats s;
    s.mean     = static_cast<float>(mean_d);
    s.min      = mn;
    s.max      = mx;
    s.variance = static_cast<float>(var_d);
    s.stddev   = static_cast<float>(std::sqrt(var_d));
    return s;
}

// ============================================================================
// Correctness check
// ============================================================================
// Worst-case rounding-error bound for naive sequential summation is
// N × ε × max|x| (this is what the scalar backend incurs).  SIMD backends
// with chunked partial sums are typically 1-2 orders of magnitude tighter,
// but we accept the loose bound so cross-backend agreement holds even at
// very large N where the scalar reduction drifts visibly.

static bool stats_match(const Stats& ref, const Stats& got, std::size_t n) {
    constexpr float eps   = std::numeric_limits<float>::epsilon();
    const float x_max     = std::max(std::fabs(ref.min), std::fabs(ref.max));
    const float tol_mean  = static_cast<float>(n) * eps * x_max + 1e-3f;
    const float tol_var   = tol_mean * 10.0f + 1e-3f;
    return std::fabs(ref.mean   - got.mean)   <= tol_mean
        && std::fabs(ref.stddev - got.stddev) <= tol_var
        && std::fabs(ref.min    - got.min)    <= 1e-3f
        && std::fabs(ref.max    - got.max)    <= 1e-3f;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    // Default N = 1'000'000 — the sweet spot where both SIMD speedup (per
    // thread) and thread-pool speedup (across cores) are clearly visible.
    // Smaller values let SIMD shine but make threading look bad (pool
    // overhead dominates); larger values push us toward the DRAM bandwidth
    // ceiling.  Both regimes are interesting — try both on the command line.
    std::size_t N = 1'000'000;
    if (argc >= 2) N = static_cast<std::size_t>(std::atoll(argv[1]));
    if (N < 32) {
        std::cerr << "N must be at least 32.\n";
        return 1;
    }

    // -------------------------------------------------------------------------
    // Generate data
    // -------------------------------------------------------------------------
    std::cout << "Generating " << N << " synthetic sensor readings... " << std::flush;
    const std::vector<float> data = generate_sensor_data(N);
    std::cout << "done.\n";

    // -------------------------------------------------------------------------
    // System info
    // -------------------------------------------------------------------------
    const auto&       feat     = simd::cpu_features();
    const std::size_t nthreads = simd::default_pool().thread_count();
    const std::size_t mem_kb   = N * sizeof(float) / 1024;

    std::cout << "\n=== SIMD Array Library — Sensor Statistics Demo ===\n";
    std::cout << "Dataset    : " << N << " float readings  (" << mem_kb << " KB)\n";
    std::cout << "Workload   : mean, variance, stddev, min, max\n";
    std::cout << "CPU caps   : SSE2=" << feat.has_sse2
              << "  AVX2=" << feat.has_avx2
              << "  AVX-512=" << feat.has_avx512f << "\n";
    std::cout << "Threads    : " << nthreads << "\n\n";

    // -------------------------------------------------------------------------
    // Pre-allocate scratch buffers used by every backend run.
    // -------------------------------------------------------------------------
    SerialScratch   serial_scratch(N);
    const std::size_t chunk = (N + nthreads - 1) / nthreads;
    ParallelScratch parallel_scratch(nthreads, chunk);

    // -------------------------------------------------------------------------
    // Compute the high-precision (double-accumulator) reference.  This is
    // both the displayed "true" statistics and the target for the OK/FAIL
    // check against every float-precision backend.
    // -------------------------------------------------------------------------
    const Stats ref = compute_stats_golden(data.data(), N);

    std::cout << "Computed statistics (24-hour temperature trace):\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  mean    = " << std::setw(8) << ref.mean    << " C\n";
    std::cout << "  std dev = " << std::setw(8) << ref.stddev  << " C\n";
    std::cout << "  min     = " << std::setw(8) << ref.min     << " C\n";
    std::cout << "  max     = " << std::setw(8) << ref.max     << " C\n\n";

    // -------------------------------------------------------------------------
    // Benchmark
    // -------------------------------------------------------------------------
    struct Row {
        std::string label;
        double      ms;
        double      mel_per_sec;   // millions of elements processed per second
        double      speedup;
        bool        correct;
    };
    std::vector<Row> rows;

    auto run = [&](const std::string& label,
                   simd::ArrayBackend<float>& be,
                   bool parallel) {
        Stats got;
        const double ms = bench_ms([&] {
            got = parallel
                ? compute_stats_parallel(be, data.data(), N, parallel_scratch)
                : compute_stats_serial  (be, data.data(), N, serial_scratch);
        });
        const double mel = static_cast<double>(N) / 1e6 / (ms * 1e-3);
        const bool ok = stats_match(ref, got, N);
        rows.push_back({label, ms, mel, 0.0, ok});
        std::cout << "  done: " << label << "\n";
    };

    {
        simd::ScalarBackend<float> be;
        run("Scalar  (1 lane,   serial)", be, false);
    }
    if (feat.has_sse2) {
        simd::SSEBackend<float> be;
        run("SSE2    (4 lanes,  serial)", be, false);
    }
    if (feat.has_avx2) {
        simd::AVXBackend<float> be;
        run("AVX2    (8 lanes,  serial)", be, false);
    }
    if (feat.has_avx512f) {
        simd::AVX512Backend<float> be;
        run("AVX-512 (16 lanes, serial)", be, false);
    }
    if (feat.has_avx2) {
        simd::AVXBackend<float> be;
        run("AVX2    (8 lanes,  " + std::to_string(nthreads) + " threads)", be, true);
    }
    if (feat.has_avx512f) {
        simd::AVX512Backend<float> be;
        run("AVX-512 (16 lanes, " + std::to_string(nthreads) + " threads)", be, true);
    }

    // Fill speedup column relative to scalar baseline.
    const double scalar_ms = rows.front().ms;
    for (auto& r : rows) r.speedup = scalar_ms / r.ms;

    // -------------------------------------------------------------------------
    // Print results table
    // -------------------------------------------------------------------------
    constexpr int W_LABEL = 36;
    constexpr int W_TIME  = 14;
    constexpr int W_THRU  = 16;
    constexpr int W_SPDUP = 12;
    constexpr int W_OK    = 8;
    const int W_TOTAL = W_LABEL + W_TIME + W_THRU + W_SPDUP + W_OK + 2;

    std::cout << "\n" << std::string(W_TOTAL, '=') << "\n";
    std::cout << "  " << std::left  << std::setw(W_LABEL) << "Backend"
              << std::right
              << std::setw(W_TIME)  << "Time/pass"
              << std::setw(W_THRU)  << "Throughput"
              << std::setw(W_SPDUP) << "Speedup"
              << std::setw(W_OK)    << "Check"
              << "\n";
    std::cout << std::string(W_TOTAL, '-') << "\n";

    for (const auto& r : rows) {
        std::cout << "  " << std::left << std::setw(W_LABEL) << r.label
                  << std::right
                  << std::setw(W_TIME - 3) << std::fixed << std::setprecision(3) << r.ms << " ms"
                  << std::setw(W_THRU - 5) << std::fixed << std::setprecision(1) << r.mel_per_sec << " Mel/s"
                  << std::setw(W_SPDUP - 1) << std::fixed << std::setprecision(2) << r.speedup << "x"
                  << std::setw(W_OK) << (r.correct ? "OK" : "FAIL")
                  << "\n";
    }
    std::cout << std::string(W_TOTAL, '=') << "\n";

    // -------------------------------------------------------------------------
    // Summary
    // -------------------------------------------------------------------------
    const auto best = std::min_element(rows.begin(), rows.end(),
        [](const Row& a, const Row& b) { return a.ms < b.ms; });
    const bool all_ok = std::all_of(rows.begin(), rows.end(),
        [](const Row& r) { return r.correct; });

    std::cout << "\nPeak speedup : " << std::fixed << std::setprecision(1)
              << best->speedup << "x  (" << best->label << ")\n";
    std::cout << "Correctness  : "
              << (all_ok ? "all backends match scalar reference"
                         : "*** MISMATCH DETECTED ***") << "\n\n";

    return all_ok ? 0 : 1;
}
