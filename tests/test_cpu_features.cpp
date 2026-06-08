#include <simd_array/cpu_features.hpp>

#include <gtest/gtest.h>

#include <iostream>

using simd::cpu_features;

TEST(CpuFeatures, CachedReferenceIsStable) {
    const auto& a = cpu_features();
    const auto& b = cpu_features();
    EXPECT_EQ(&a, &b) << "cpu_features() must return the same cached instance";
}

TEST(CpuFeatures, FlagsAreInternallyConsistent) {
    const auto& f = cpu_features();
    if (f.has_avx2)     EXPECT_TRUE(f.has_avx);
    if (f.has_avx)      EXPECT_TRUE(f.has_sse2);
    if (f.has_sse42)    EXPECT_TRUE(f.has_sse41);
    if (f.has_sse41)    EXPECT_TRUE(f.has_sse2);
    if (f.has_avx512dq) EXPECT_TRUE(f.has_avx512f);
    if (f.has_avx512bw) EXPECT_TRUE(f.has_avx512f);
    if (f.has_avx512vl) EXPECT_TRUE(f.has_avx512f);
}

TEST(CpuFeatures, BestIsaIsNonNull) {
    const auto& f = cpu_features();
    ASSERT_NE(f.best_isa(), nullptr);
    EXPECT_GT(std::string_view(f.best_isa()).size(), 0u);
}

// Diagnostic — always passes, prints the detected feature set so the test
// log doubles as a quick CPU report.
TEST(CpuFeatures, DiagnosticPrint) {
    const auto& f = cpu_features();
    std::cout << "Detected SIMD ISA: " << f.best_isa() << '\n'
              << "  SSE2     : " << f.has_sse2     << '\n'
              << "  SSE4.1   : " << f.has_sse41    << '\n'
              << "  SSE4.2   : " << f.has_sse42    << '\n'
              << "  AVX      : " << f.has_avx      << '\n'
              << "  AVX2     : " << f.has_avx2     << '\n'
              << "  FMA      : " << f.has_fma      << '\n'
              << "  AVX-512F : " << f.has_avx512f  << '\n'
              << "  AVX-512DQ: " << f.has_avx512dq << '\n'
              << "  AVX-512BW: " << f.has_avx512bw << '\n'
              << "  AVX-512VL: " << f.has_avx512vl << '\n';
    SUCCEED();
}
