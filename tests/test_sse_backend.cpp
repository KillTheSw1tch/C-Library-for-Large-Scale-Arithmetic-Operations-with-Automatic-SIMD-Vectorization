#include <simd_array/scalar_backend.hpp>
#include <simd_array/sse_backend.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

using simd::ScalarBackend;
using simd::SSEBackend;

template <typename T>
class SSEBackendTest : public ::testing::Test {
protected:
    SSEBackend<T>    backend;
    ScalarBackend<T> reference;
};

using FpTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(SSEBackendTest, FpTypes);

TYPED_TEST(SSEBackendTest, NameIsSse2) {
    EXPECT_STREQ(this->backend.name(), "sse2");
}

// ----------------------------------------------------------------------------
// Hand-written cases on small integer inputs that are exact in IEEE-754.
// EXPECT_EQ on the floating-point result is therefore safe.
// ----------------------------------------------------------------------------

TYPED_TEST(SSEBackendTest, Add) {
    using T = TypeParam;
    std::vector<T> a = {1, 2, 3, 4, 5};
    std::vector<T> b = {10, 20, 30, 40, 50};
    std::vector<T> out(a.size());
    this->backend.add(out.data(), a.data(), b.data(), a.size());
    EXPECT_EQ(out, (std::vector<T>{11, 22, 33, 44, 55}));
}

TYPED_TEST(SSEBackendTest, Subtract) {
    using T = TypeParam;
    std::vector<T> a = {10, 20, 30};
    std::vector<T> b = {1, 2, 3};
    std::vector<T> out(a.size());
    this->backend.subtract(out.data(), a.data(), b.data(), a.size());
    EXPECT_EQ(out, (std::vector<T>{9, 18, 27}));
}

TYPED_TEST(SSEBackendTest, Multiply) {
    using T = TypeParam;
    std::vector<T> a = {1, 2, 3, 4};
    std::vector<T> b = {5, 6, 7, 8};
    std::vector<T> out(a.size());
    this->backend.multiply(out.data(), a.data(), b.data(), a.size());
    EXPECT_EQ(out, (std::vector<T>{5, 12, 21, 32}));
}

TYPED_TEST(SSEBackendTest, Divide) {
    using T = TypeParam;
    std::vector<T> a = {10, 20, 30};
    std::vector<T> b = {2, 4, 5};
    std::vector<T> out(a.size());
    this->backend.divide(out.data(), a.data(), b.data(), a.size());
    EXPECT_EQ(out, (std::vector<T>{5, 5, 6}));
}

TYPED_TEST(SSEBackendTest, AddScalar) {
    using T = TypeParam;
    std::vector<T> a = {1, 2, 3};
    std::vector<T> out(a.size());
    this->backend.add_scalar(out.data(), a.data(), T{10}, a.size());
    EXPECT_EQ(out, (std::vector<T>{11, 12, 13}));
}

TYPED_TEST(SSEBackendTest, MultiplyScalar) {
    using T = TypeParam;
    std::vector<T> a = {1, 2, 3};
    std::vector<T> out(a.size());
    this->backend.multiply_scalar(out.data(), a.data(), T{3}, a.size());
    EXPECT_EQ(out, (std::vector<T>{3, 6, 9}));
}

TYPED_TEST(SSEBackendTest, Sum) {
    using T = TypeParam;
    std::vector<T> a = {1, 2, 3, 4, 5};
    EXPECT_EQ(this->backend.sum(a.data(), a.size()), T{15});
}

TYPED_TEST(SSEBackendTest, MinAndMax) {
    using T = TypeParam;
    std::vector<T> a = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3};
    EXPECT_EQ(this->backend.min(a.data(), a.size()), T{1});
    EXPECT_EQ(this->backend.max(a.data(), a.size()), T{9});
}

TYPED_TEST(SSEBackendTest, MinMaxSingleElement) {
    using T = TypeParam;
    std::vector<T> a = {42};
    EXPECT_EQ(this->backend.min(a.data(), 1), T{42});
    EXPECT_EQ(this->backend.max(a.data(), 1), T{42});
}

TYPED_TEST(SSEBackendTest, Fill) {
    using T = TypeParam;
    std::vector<T> a(7, T{0});
    this->backend.fill(a.data(), T{3.5}, a.size());  // 3.5 is exact in IEEE-754
    for (T v : a) EXPECT_EQ(v, T{3.5});
}

// ----------------------------------------------------------------------------
// Edge cases.
// ----------------------------------------------------------------------------

// In-place update on a length that is not a multiple of the SIMD width — the
// vector body and the scalar tail must both land in the right place.
TYPED_TEST(SSEBackendTest, AliasingOutEqualsA) {
    using T = TypeParam;
    constexpr std::size_t N = 11;
    std::vector<T> a(N), b(N), expected(N);
    for (std::size_t i = 0; i < N; ++i) {
        a[i] = static_cast<T>(i);
        b[i] = static_cast<T>(2 * i + 1);
        expected[i] = a[i] + b[i];
    }
    this->backend.add(a.data(), a.data(), b.data(), N);
    EXPECT_EQ(a, expected);
}

TYPED_TEST(SSEBackendTest, EmptyInputIsSafe) {
    using T = TypeParam;
    T* nul = nullptr;
    this->backend.add(nul, nul, nul, 0);
    this->backend.subtract(nul, nul, nul, 0);
    this->backend.multiply(nul, nul, nul, 0);
    this->backend.divide(nul, nul, nul, 0);
    this->backend.add_scalar(nul, nul, T{1}, 0);
    this->backend.multiply_scalar(nul, nul, T{1}, 0);
    this->backend.fill(nul, T{1}, 0);
    EXPECT_EQ(this->backend.sum(nul, 0), T{0});
    EXPECT_EQ(this->backend.min(nul, 0), T{});
    EXPECT_EQ(this->backend.max(nul, 0), T{});
}

// ----------------------------------------------------------------------------
// Cross-checks against ScalarBackend on every length 0..MAX. This is the most
// important set of tests: it exercises the SIMD body, every possible tail
// remainder, and the n < W branch in the reduction kernels in one sweep.
// ----------------------------------------------------------------------------

namespace {
template <typename T>
std::vector<T> make_pattern(std::size_t n) {
    // Fractional but bit-exact-representable values (denominators are powers
    // of two). Stays away from extremes so partial sums don't overflow float.
    std::vector<T> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        v[i] = T{0.25} * static_cast<T>(i) - T{8};
    }
    return v;
}
}  // namespace

TYPED_TEST(SSEBackendTest, ElementwiseMatchesScalarOnAllTailLengths) {
    using T = TypeParam;
    constexpr std::size_t MAX = 20;  // covers all tail remainders for W=4 and W=2
    for (std::size_t n = 0; n <= MAX; ++n) {
        const auto a = make_pattern<T>(n);
        auto b = make_pattern<T>(n);
        for (auto& x : b) x += T{1.5};  // distinct from a, still exact

        std::vector<T> out_sse(n), out_ref(n);

        this->backend.add(out_sse.data(), a.data(), b.data(), n);
        this->reference.add(out_ref.data(), a.data(), b.data(), n);
        ASSERT_EQ(out_sse, out_ref) << "add failed at n=" << n;

        this->backend.subtract(out_sse.data(), a.data(), b.data(), n);
        this->reference.subtract(out_ref.data(), a.data(), b.data(), n);
        ASSERT_EQ(out_sse, out_ref) << "subtract failed at n=" << n;

        this->backend.multiply(out_sse.data(), a.data(), b.data(), n);
        this->reference.multiply(out_ref.data(), a.data(), b.data(), n);
        ASSERT_EQ(out_sse, out_ref) << "multiply failed at n=" << n;

        // Avoid b values that round to zero — the pattern is non-zero by
        // construction (smallest |b| = |0 - 8 + 1.5| = 6.5).
        this->backend.divide(out_sse.data(), a.data(), b.data(), n);
        this->reference.divide(out_ref.data(), a.data(), b.data(), n);
        ASSERT_EQ(out_sse, out_ref) << "divide failed at n=" << n;

        this->backend.add_scalar(out_sse.data(), a.data(), T{2.75}, n);
        this->reference.add_scalar(out_ref.data(), a.data(), T{2.75}, n);
        ASSERT_EQ(out_sse, out_ref) << "add_scalar failed at n=" << n;

        this->backend.multiply_scalar(out_sse.data(), a.data(), T{0.5}, n);
        this->reference.multiply_scalar(out_ref.data(), a.data(), T{0.5}, n);
        ASSERT_EQ(out_sse, out_ref) << "multiply_scalar failed at n=" << n;
    }
}

TYPED_TEST(SSEBackendTest, ReductionsMatchScalarOnAllTailLengths) {
    using T = TypeParam;
    constexpr std::size_t MAX = 20;
    for (std::size_t n = 0; n <= MAX; ++n) {
        // Mix of positive and negative, exactly representable.
        std::vector<T> a(n);
        for (std::size_t i = 0; i < n; ++i) {
            a[i] = static_cast<T>((7 * i + 3) % 17) - T{8};
        }
        ASSERT_EQ(this->backend.sum(a.data(), n),
                  this->reference.sum(a.data(), n)) << "sum at n=" << n;
        ASSERT_EQ(this->backend.min(a.data(), n),
                  this->reference.min(a.data(), n)) << "min at n=" << n;
        ASSERT_EQ(this->backend.max(a.data(), n),
                  this->reference.max(a.data(), n)) << "max at n=" << n;
    }
}

TYPED_TEST(SSEBackendTest, FillMatchesScalarOnAllTailLengths) {
    using T = TypeParam;
    constexpr std::size_t MAX = 20;
    for (std::size_t n = 0; n <= MAX; ++n) {
        std::vector<T> sse_buf(n, T{0});
        std::vector<T> ref_buf(n, T{0});
        this->backend.fill(sse_buf.data(), T{1.25}, n);
        this->reference.fill(ref_buf.data(), T{1.25}, n);
        ASSERT_EQ(sse_buf, ref_buf) << "fill at n=" << n;
    }
}

// Larger run for confidence beyond the brute-force tail sweep.
TYPED_TEST(SSEBackendTest, LargeArrayMatchesScalar) {
    using T = TypeParam;
    constexpr std::size_t N = 1000;
    std::vector<T> a(N), b(N), out_sse(N), out_ref(N);
    for (std::size_t i = 0; i < N; ++i) {
        a[i] = static_cast<T>(i);
        b[i] = static_cast<T>(2 * i);
    }
    this->backend.add(out_sse.data(), a.data(), b.data(), N);
    this->reference.add(out_ref.data(), a.data(), b.data(), N);
    EXPECT_EQ(out_sse, out_ref);
    EXPECT_EQ(this->backend.sum(a.data(), N), this->reference.sum(a.data(), N));
}
