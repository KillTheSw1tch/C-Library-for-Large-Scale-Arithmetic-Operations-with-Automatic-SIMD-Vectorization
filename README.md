# simd-array-lib

Header-only C++17 library that delivers **guaranteed** SIMD vectorization for
arithmetic operations over floating-point arrays, with runtime CPU capability
detection (CPUID) and integrated multithreading.

Bachelor's diploma thesis, KPI / University of Fribourg.

## Highlights

- **Zero dependencies.** Standard library + compiler intrinsics only.
- **Runtime dispatch.** Single binary picks Scalar / SSE2 / AVX2 / AVX-512 backend at first call (`std::call_once`).
- **Header-only.** Single `#include <simd_array/array.hpp>`.
- **Multithreaded.** Internal thread pool partitions large arrays across cores.
- **Templated.** `simd::Array<float>` and `simd::Array<double>`.

## Quick start

```cpp
#include <simd_array/array.hpp>

simd::Array<float> a(1'000'000, 1.0f);
simd::Array<float> b(1'000'000, 2.0f);
simd::Array<float> c = a + b;     // vectorized + parallelized automatically
```

## Build

Requires CMake ≥ 3.15 and a C++17 compiler (MSVC 2019+, GCC ≥ 9, Clang ≥ 10).

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build --output-on-failure
```

CMake options:

| Option | Default | Effect |
|---|---|---|
| `SIMD_ARRAY_BUILD_TESTS` | `ON` | Compile GoogleTest suite |
| `SIMD_ARRAY_BUILD_BENCHMARKS` | `OFF` | Compile benchmark binaries |
| `SIMD_ARRAY_BUILD_EXAMPLES` | `OFF` | Compile examples |
