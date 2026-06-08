cmake_minimum_required(VERSION 3.15)

project(simd_array_lib
    VERSION 0.1.0
    DESCRIPTION "Header-only C++17 SIMD array library with runtime dispatch"
    LANGUAGES CXX
)

# ---------------------------------------------------------------------------
# Standard
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release if not specified (single-config generators only).
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        Debug Release RelWithDebInfo MinSizeRel)
endif()

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(SIMD_ARRAY_BUILD_TESTS      "Build unit tests"     ON)
option(SIMD_ARRAY_BUILD_BENCHMARKS "Build benchmarks"     OFF)
option(SIMD_ARRAY_BUILD_EXAMPLES   "Build examples"       ON)
option(SIMD_ARRAY_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)

# ---------------------------------------------------------------------------
# Header-only INTERFACE target
# ---------------------------------------------------------------------------
add_library(simd_array INTERFACE
        benchmarks/benchmark_suite.cpp)
add_library(simd_array::simd_array ALIAS simd_array)

target_include_directories(simd_array INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(simd_array INTERFACE cxx_std_17)

find_package(Threads REQUIRED)
target_link_libraries(simd_array INTERFACE Threads::Threads)

# Note on SIMD flags:
#   The library uses per-function target attributes (GCC/Clang) so that
#   AVX2 / AVX-512 code can be compiled inside a single TU without forcing
#   the whole binary to require those instructions. MSVC has no equivalent;
#   on MSVC the user must build with /arch:AVX2 (or /arch:AVX512) to enable
#   the corresponding backends. CPUID dispatch keeps the binary safe to
#   *load* on older CPUs only when built without those flags.
if(MSVC)
    target_compile_options(simd_array INTERFACE /W4 /permissive-)
    if(SIMD_ARRAY_WARNINGS_AS_ERRORS)
        target_compile_options(simd_array INTERFACE /WX)
    endif()
else()
    target_compile_options(simd_array INTERFACE
        -Wall -Wextra -Wpedantic -Wshadow -Wconversion)
    if(SIMD_ARRAY_WARNINGS_AS_ERRORS)
        target_compile_options(simd_array INTERFACE -Werror)
    endif()
endif()

# ---------------------------------------------------------------------------
# Subdirectories
# ---------------------------------------------------------------------------
if(SIMD_ARRAY_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(SIMD_ARRAY_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(SIMD_ARRAY_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "simd_array_lib ${PROJECT_VERSION}")
message(STATUS "  Build type   : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Compiler     : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Tests        : ${SIMD_ARRAY_BUILD_TESTS}")
message(STATUS "  Benchmarks   : ${SIMD_ARRAY_BUILD_BENCHMARKS}")
message(STATUS "  Examples     : ${SIMD_ARRAY_BUILD_EXAMPLES}")
message(STATUS "")
