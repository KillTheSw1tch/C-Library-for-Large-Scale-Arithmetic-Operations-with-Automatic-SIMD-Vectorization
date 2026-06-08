# ---------------------------------------------------------------------------
# GoogleTest via FetchContent (no system install required).
# ---------------------------------------------------------------------------
include(FetchContent)
cmake_minimum_required(VERSION 3.15)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)  # MSVC: match CRT of main project

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

# ---------------------------------------------------------------------------
# Test binaries
# ---------------------------------------------------------------------------
function(simd_array_add_test name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE
        simd_array::simd_array
        GTest::gtest_main
    )
    gtest_discover_tests(${name})
endfunction()

simd_array_add_test(test_cpu_features   test_cpu_features.cpp)
simd_array_add_test(test_scalar_backend test_scalar_backend.cpp)
simd_array_add_test(test_sse_backend    test_sse_backend.cpp)
simd_array_add_test(test_avx_backend    test_avx_backend.cpp)
simd_array_add_test(test_avx512_backend test_avx512_backend.cpp)
simd_array_add_test(test_thread_pool    test_thread_pool.cpp)
simd_array_add_test(test_array          test_array.cpp)
