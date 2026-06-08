#include <simd_array/scalar_backend.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

using simd::ScalarBackend;

template <typename T>
class ScalarBackendTest : public ::testing::Test {
protected:
    ScalarBackend<T> backend;
};

using FpTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(ScalarBackendTest, FpTypes);

TYPED_TEST(ScalarBackendTest, NameIsScalar) {
    EXPECT_STREQ(this->backend.name(), "scalar");
}

// ----------------------------------------------------------------------------
// Element-wise binary ops. Inputs are small integers exactly representable in
// both float and double, so EXPECT_EQ on the result is safe (no rounding).
// ----------------------------------------------------------------------------

TYPED_TEST(ScalarBackendTest, Add) {
    using T = TypeParam;
    std::vector<T> a = {1, 2, 3, 4, 5};
    std::vector<T> b = {10, 20, 30, 40, 50};
    std::vector<T> out(a.size());
    this->backend.add(out.data(), a.data(), b.data(), a.size());
    EXPECT_EQ(out, (std::vector<T>{11, 22, 33, 44, 55}));
}

TYPED_TEST(ScalarBackendTest, Subtract) {
    using T = TypeParam;
    std::vector<T> a = {10, 20, 30};
    std::vector<T> b = {1, 2, 3};
    std::vector<T> out(a.size());
    this->backend.subtract(out.data(), a.data(), b.data(), a.size());
    EXPECT_EQ(out, (std::vector<T>{9, 18, 27}));
}

TYPED_TEST(ScalarBackendTest, Multiply) {
    using T = TypeParam;
    std::vector<T> a = {1, 2, 3, 4};
    std::vector<T> b = {5, 6, 7, 8};
    std::vector<T> out(a.size());
    this->backend.multiply(out.data(), a.data(), b.data(), a.size());
    EXPECT_EQ(out, (std::vector<T>{5, 12, 21, 32}));
}

TYPED_TEST(ScalarBackendTest, Divide) {
    using T = TypeParam;
    std::vector<T> a = {10, 20, 30};
    std::vector<T> b = {2, 4, 5};
    std::vector<T> out(a.size());
    this->backend.divide(out.data(), a.data(), b.data(), a.size());
    EXPECT_EQ(out, (std::vector<T>{5, 5, 6}));
}

// ----------------------------------------------------------------------------
// Scalar broadcast ops.
// ----------------------------------------------------------------------------

TYPED_TEST(ScalarBackendTest, AddScalar) {
    using T = TypeParam;
    std::vector<T> a = {1, 2, 3};
    std::vector<T> out(a.size());
    this->backend.add_scalar(out.data(), a.data(), T{10}, a.size());
    EXPECT_EQ(out, (std::vector<T>{11, 12, 13}));
}

TYPED_TEST(ScalarBackendTest, MultiplyScalar) {
    using T = TypeParam;
    std::vector<T> a = {1, 2, 3};
    std::vector<T> out(a.size());
    this->backend.multiply_scalar(out.data(), a.data(), T{3}, a.size());
    EXPECT_EQ(out, (std::vector<T>{3, 6, 9}));
}

// ----------------------------------------------------------------------------
// Reductions.
// ----------------------------------------------------------------------------

TYPED_TEST(ScalarBackendTest, Sum) {
    using T = TypeParam;
    std::vector<T> a = {1, 2, 3, 4, 5};
    EXPECT_EQ(this->backend.sum(a.data(), a.size()), T{15});
}

TYPED_TEST(ScalarBackendTest, MinAndMax) {
    using T = TypeParam;
    std::vector<T> a = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3};
    EXPECT_EQ(this->backend.min(a.data(), a.size()), T{1});
    EXPECT_EQ(this->backend.max(a.data(), a.size()), T{9});
}

TYPED_TEST(ScalarBackendTest, MinMaxSingleElement) {
    using T = TypeParam;
    std::vector<T> a = {42};
    EXPECT_EQ(this->backend.min(a.data(), 1), T{42});
    EXPECT_EQ(this->backend.max(a.data(), 1), T{42});
}

TYPED_TEST(ScalarBackendTest, Fill) {
    using T = TypeParam;
    std::vector<T> a(7, T{0});
    this->backend.fill(a.data(), T{3.5}, a.size());  // 3.5 is exact in IEEE-754
    for (T v : a) EXPECT_EQ(v, T{3.5});
}

// ----------------------------------------------------------------------------
// Edge cases.
// ----------------------------------------------------------------------------

// In-place update: out == a is part of the documented contract.
TYPED_TEST(ScalarBackendTest, AliasingOutEqualsA) {
    using T = TypeParam;
    std::vector<T> a = {1, 2, 3, 4};
    std::vector<T> b = {10, 20, 30, 40};
    this->backend.add(a.data(), a.data(), b.data(), a.size());
    EXPECT_EQ(a, (std::vector<T>{11, 22, 33, 44}));
}

// Empty input must be a no-op for ops, return T{} for reductions, and never
// dereference the (potentially null) pointers.
TYPED_TEST(ScalarBackendTest, EmptyInputIsSafe) {
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

TYPED_TEST(ScalarBackendTest, NegativeValues) {
    using T = TypeParam;
    std::vector<T> a = {-1, -2, 3, -4};
    std::vector<T> b = { 1,  2, -3, 4};
    std::vector<T> out(a.size());
    this->backend.add(out.data(), a.data(), b.data(), a.size());
    EXPECT_EQ(out, (std::vector<T>{0, 0, 0, 0}));
    EXPECT_EQ(this->backend.min(a.data(), a.size()), T{-4});
    EXPECT_EQ(this->backend.max(a.data(), a.size()), T{3});
}

// A larger run to exercise the loop in bulk. N is kept small enough that the
// running float sum stays exact (<<2^24).
TYPED_TEST(ScalarBackendTest, LargeArrayIsConsistent) {
    using T = TypeParam;
    constexpr std::size_t N = 1000;
    std::vector<T> a(N), b(N), out(N);
    for (std::size_t i = 0; i < N; ++i) {
        a[i] = static_cast<T>(i);
        b[i] = static_cast<T>(2 * i);
    }
    this->backend.add(out.data(), a.data(), b.data(), N);
    for (std::size_t i = 0; i < N; ++i) {
        ASSERT_EQ(out[i], static_cast<T>(3 * i));
    }
    const T expected_sum = static_cast<T>(N) * static_cast<T>(N - 1) / T{2};
    EXPECT_EQ(this->backend.sum(a.data(), N), expected_sum);
}
