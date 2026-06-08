#include <simd_array/array.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <vector>

template <typename T>
class ArrayTest : public ::testing::Test {};

using FpTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(ArrayTest, FpTypes);

// ============================================================================
// Construction.
// ============================================================================

TYPED_TEST(ArrayTest, DefaultIsEmpty) {
    using T = TypeParam;
    simd::Array<T> a;
    EXPECT_EQ(a.size(), 0u);
    EXPECT_TRUE(a.empty());
    EXPECT_EQ(a.data(), nullptr);
}

TYPED_TEST(ArrayTest, SizedIsZeroInitialised) {
    using T = TypeParam;
    simd::Array<T> a(8);
    EXPECT_EQ(a.size(), 8u);
    EXPECT_FALSE(a.empty());
    for (std::size_t i = 0; i < a.size(); ++i) EXPECT_EQ(a[i], T{0});
}

TYPED_TEST(ArrayTest, SizedAndFilled) {
    using T = TypeParam;
    simd::Array<T> a(5, T{3.5});
    for (std::size_t i = 0; i < a.size(); ++i) EXPECT_EQ(a[i], T{3.5});
}

TYPED_TEST(ArrayTest, FromInitializerList) {
    using T = TypeParam;
    simd::Array<T> a{1, 2, 3, 4};
    EXPECT_EQ(a.size(), 4u);
    EXPECT_EQ(a[0], T{1});
    EXPECT_EQ(a[3], T{4});
}

TYPED_TEST(ArrayTest, CopyIsIndependent) {
    using T = TypeParam;
    simd::Array<T> a{1, 2, 3};
    simd::Array<T> b(a);
    b[0] = T{99};
    EXPECT_EQ(a[0], T{1});
    EXPECT_EQ(b[0], T{99});
    EXPECT_NE(a.data(), b.data());
}

TYPED_TEST(ArrayTest, MoveTransfersOwnership) {
    using T = TypeParam;
    simd::Array<T> a{1, 2, 3};
    T* original = a.data();
    simd::Array<T> b(std::move(a));
    EXPECT_EQ(b.size(), 3u);
    EXPECT_EQ(b.data(), original);
    EXPECT_EQ(a.size(), 0u);
    EXPECT_EQ(a.data(), nullptr);
}

TYPED_TEST(ArrayTest, CopyAssignment) {
    using T = TypeParam;
    simd::Array<T> a{1, 2, 3};
    simd::Array<T> b(10);
    b = a;
    ASSERT_EQ(b.size(), 3u);
    for (std::size_t i = 0; i < 3; ++i) EXPECT_EQ(b[i], a[i]);
}

TYPED_TEST(ArrayTest, MoveAssignment) {
    using T = TypeParam;
    simd::Array<T> a{1, 2, 3};
    simd::Array<T> b(10);
    b = std::move(a);
    EXPECT_EQ(b.size(), 3u);
    EXPECT_EQ(a.size(), 0u);
}

TYPED_TEST(ArrayTest, SwapExchangesContents) {
    using T = TypeParam;
    simd::Array<T> a{1, 2};
    simd::Array<T> b{10, 20, 30};
    simd::swap(a, b);
    EXPECT_EQ(a.size(), 3u);
    EXPECT_EQ(b.size(), 2u);
    EXPECT_EQ(a[2], T{30});
    EXPECT_EQ(b[0], T{1});
}

// ============================================================================
// Memory alignment.
// ============================================================================

TYPED_TEST(ArrayTest, DataIs64ByteAligned) {
    using T = TypeParam;
    for (std::size_t n : {1u, 7u, 1000u, 100'000u}) {
        simd::Array<T> a(n);
        const auto addr = reinterpret_cast<std::uintptr_t>(a.data());
        EXPECT_EQ(addr % 64u, 0u) << "n=" << n;
    }
}

// ============================================================================
// Backend dispatch.
// ============================================================================

TYPED_TEST(ArrayTest, BackendNameIsKnownAndStable) {
    using T = TypeParam;
    const char* a = simd::Array<T>::backend_name();
    const char* b = simd::Array<T>::backend_name();
    ASSERT_NE(a, nullptr);
    EXPECT_EQ(a, b) << "backend identifier should be a stable string literal";
    const std::string_view name(a);
    EXPECT_TRUE(name == "scalar" || name == "sse2" ||
                name == "avx2"   || name == "avx512f")
        << "unexpected backend: " << name;
}

TYPED_TEST(ArrayTest, BackendMatchesHighestAvailableIsa) {
    using T = TypeParam;
    const std::string_view name(simd::Array<T>::backend_name());
    const auto& f = simd::cpu_features();
    if      (f.has_avx512f) EXPECT_EQ(name, "avx512f");
    else if (f.has_avx2)    EXPECT_EQ(name, "avx2");
    else if (f.has_sse2)    EXPECT_EQ(name, "sse2");
    else                    EXPECT_EQ(name, "scalar");
}

// ============================================================================
// Element-wise binary operators (small N — single-threaded fast path).
// ============================================================================

TYPED_TEST(ArrayTest, AddSmall) {
    using T = TypeParam;
    simd::Array<T> a{1, 2, 3, 4};
    simd::Array<T> b{10, 20, 30, 40};
    simd::Array<T> c = a + b;
    EXPECT_EQ(c[0], T{11});
    EXPECT_EQ(c[1], T{22});
    EXPECT_EQ(c[2], T{33});
    EXPECT_EQ(c[3], T{44});
}

TYPED_TEST(ArrayTest, SubMulDivSmall) {
    using T = TypeParam;
    simd::Array<T> a{10, 20, 30, 40};
    simd::Array<T> b{2, 4, 5, 8};
    simd::Array<T> diff  = a - b;
    simd::Array<T> prod  = a * b;
    simd::Array<T> quot  = a / b;
    EXPECT_EQ(diff[2], T{25});
    EXPECT_EQ(prod[2], T{150});
    EXPECT_EQ(quot[3], T{5});
}

TYPED_TEST(ArrayTest, ScalarBroadcastBothOrders) {
    using T = TypeParam;
    simd::Array<T> a{1, 2, 3};
    simd::Array<T> p1 = a + T{10};
    simd::Array<T> p2 = T{10} + a;
    simd::Array<T> q1 = a * T{3};
    simd::Array<T> q2 = T{3} * a;
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(p1[i], p2[i]);
        EXPECT_EQ(q1[i], q2[i]);
    }
    EXPECT_EQ(p1[2], T{13});
    EXPECT_EQ(q1[2], T{9});
}

TYPED_TEST(ArrayTest, CompoundAssignmentArray) {
    using T = TypeParam;
    simd::Array<T> a{1, 2, 3};
    simd::Array<T> b{10, 20, 30};
    a += b;
    EXPECT_EQ(a[2], T{33});
    a -= b;
    EXPECT_EQ(a[0], T{1});
    a *= b;
    EXPECT_EQ(a[1], T{40});
    a /= b;
    EXPECT_EQ(a[0], T{1});
}

TYPED_TEST(ArrayTest, CompoundAssignmentScalar) {
    using T = TypeParam;
    simd::Array<T> a{1, 2, 3};
    a += T{1};
    EXPECT_EQ(a[0], T{2});
    a *= T{2};
    EXPECT_EQ(a[2], T{8});
}

TYPED_TEST(ArrayTest, FillReplacesAllElements) {
    using T = TypeParam;
    simd::Array<T> a(7, T{1});
    a.fill(T{42});
    for (std::size_t i = 0; i < a.size(); ++i) EXPECT_EQ(a[i], T{42});
}

// ============================================================================
// Reductions (small N).
// ============================================================================

TYPED_TEST(ArrayTest, SumMinMaxSmall) {
    using T = TypeParam;
    simd::Array<T> a{3, 1, 4, 1, 5, 9, 2, 6};
    EXPECT_EQ(a.sum(), T{31});
    EXPECT_EQ(a.min(), T{1});
    EXPECT_EQ(a.max(), T{9});
    EXPECT_EQ(simd::sum(a), T{31});
    EXPECT_EQ(simd::min(a), T{1});
    EXPECT_EQ(simd::max(a), T{9});
}

TYPED_TEST(ArrayTest, ReductionsOnEmpty) {
    using T = TypeParam;
    simd::Array<T> a;
    EXPECT_EQ(a.sum(), T{0});
    EXPECT_EQ(a.min(), T{});
    EXPECT_EQ(a.max(), T{});
}

TYPED_TEST(ArrayTest, DotProductSmall) {
    using T = TypeParam;
    simd::Array<T> a{1, 2, 3, 4};
    simd::Array<T> b{5, 6, 7, 8};
    // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
    EXPECT_EQ(simd::dot_product(a, b), T{70});
}

TYPED_TEST(ArrayTest, DotProductOnEmpty) {
    using T = TypeParam;
    simd::Array<T> a, b;
    EXPECT_EQ(simd::dot_product(a, b), T{0});
}

// ============================================================================
// Size-mismatch handling.
// ============================================================================

TYPED_TEST(ArrayTest, MismatchedSizeBinaryOpThrows) {
    using T = TypeParam;
    simd::Array<T> a(4);
    simd::Array<T> b(5);
    EXPECT_THROW((void)(a + b), std::invalid_argument);
}

TYPED_TEST(ArrayTest, MismatchedSizeCompoundThrows) {
    using T = TypeParam;
    simd::Array<T> a(4), b(5);
    EXPECT_THROW(a += b, std::invalid_argument);
}

TYPED_TEST(ArrayTest, MismatchedSizeDotProductThrows) {
    using T = TypeParam;
    simd::Array<T> a(4), b(5);
    EXPECT_THROW((void)simd::dot_product(a, b), std::invalid_argument);
}

// ============================================================================
// Large-N path (above parallel_min — exercises the thread pool).
// ============================================================================

namespace {
// Use a size that is comfortably above `parallel_min` (32 * 1024) and has a
// non-trivial relationship with the SIMD lane count so the chunk math has
// to work for both float (W=16 on AVX-512) and double (W=8).
constexpr std::size_t LARGE_N = 100'003;
}  // namespace

TYPED_TEST(ArrayTest, LargeAddElementwiseCorrect) {
    using T = TypeParam;
    simd::Array<T> a(LARGE_N);
    simd::Array<T> b(LARGE_N);
    for (std::size_t i = 0; i < LARGE_N; ++i) {
        a[i] = static_cast<T>(i % 64);
        b[i] = static_cast<T>(2 * (i % 64));
    }
    simd::Array<T> c = a + b;
    for (std::size_t i = 0; i < LARGE_N; ++i) {
        ASSERT_EQ(c[i], static_cast<T>(3 * (i % 64))) << "i=" << i;
    }
}

TYPED_TEST(ArrayTest, LargeFillCorrect) {
    using T = TypeParam;
    simd::Array<T> a(LARGE_N);
    a.fill(T{2.5});
    for (std::size_t i = 0; i < LARGE_N; ++i) ASSERT_EQ(a[i], T{2.5});
}

TYPED_TEST(ArrayTest, LargeReductionsCorrect) {
    using T = TypeParam;
    // All values fit in 7 bits (< 128) and there are LARGE_N of them, so
    // the running sum stays well below 2^24 — exact in float as well.
    simd::Array<T> a(LARGE_N);
    T expected_sum = T{};
    for (std::size_t i = 0; i < LARGE_N; ++i) {
        const T v = static_cast<T>(i % 17);
        a[i] = v;
        expected_sum += v;
    }
    EXPECT_EQ(a.sum(), expected_sum);
    EXPECT_EQ(a.min(), T{0});
    EXPECT_EQ(a.max(), T{16});
}

TYPED_TEST(ArrayTest, LargeCompoundAssignmentCorrect) {
    using T = TypeParam;
    simd::Array<T> a(LARGE_N, T{1});
    simd::Array<T> b(LARGE_N, T{2});
    a += b;
    for (std::size_t i = 0; i < LARGE_N; ++i) ASSERT_EQ(a[i], T{3});
    a *= T{4};
    for (std::size_t i = 0; i < LARGE_N; ++i) ASSERT_EQ(a[i], T{12});
}

TYPED_TEST(ArrayTest, LargeDotProductCorrect) {
    using T = TypeParam;
    // 1·1 repeated LARGE_N times → expected = LARGE_N (exact in both float
    // and double for any LARGE_N < 2^24).
    static_assert(LARGE_N < (1u << 24), "LARGE_N must be exact in float");
    simd::Array<T> a(LARGE_N, T{1});
    simd::Array<T> b(LARGE_N, T{1});
    EXPECT_EQ(simd::dot_product(a, b), static_cast<T>(LARGE_N));
}

// ============================================================================
// Diagnostic — always passes, prints which backend the dispatcher picked.
// ============================================================================

TEST(Array, DiagnosticPrintBackends) {
    std::cout << "Backend for simd::Array<float>:  "
              << simd::Array<float>::backend_name()  << '\n'
              << "Backend for simd::Array<double>: "
              << simd::Array<double>::backend_name() << '\n';
    SUCCEED();
}
