# ---------------------------------------------------------------------------
# Google Benchmark via FetchContent.
# ---------------------------------------------------------------------------
include(FetchContent)
cmake_minimum_required(VERSION 3.15)

FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.8.3
)
set(BENCHMARK_ENABLE_TESTING     OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL     OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
function(simd_array_add_benchmark name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE
        simd_array::simd_array
        benchmark::benchmark_main
    )
endfunction()

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
simd_array_add_benchmark(benchmark_add      benchmark_add.cpp)
simd_array_add_benchmark(benchmark_multiply benchmark_multiply.cpp)
simd_array_add_benchmark(benchmark_suite    benchmark_suite.cpp)
