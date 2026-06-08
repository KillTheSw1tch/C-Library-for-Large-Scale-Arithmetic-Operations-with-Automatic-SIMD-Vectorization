cmake_minimum_required(VERSION 3.15)

# MSVC does not support per-function target attributes, so AVX2 intrinsics
# require the whole TU to be compiled with /arch:AVX2.
# GCC/Clang use __attribute__((target("avx2"))) inside each backend, so no
# global flag is needed there.
function(simd_array_add_example name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE simd_array::simd_array)
    if(MSVC)
        target_compile_options(${name} PRIVATE /arch:AVX2)
    endif()
endfunction()

simd_array_add_example(image_blur        image_blur.cpp)
simd_array_add_example(sensor_stats_demo sensor_stats_demo.cpp)
