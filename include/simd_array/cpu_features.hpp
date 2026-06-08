#pragma once

/// \file cpu_features.hpp
/// Runtime CPU capability detection via CPUID.
///
/// The probe is executed exactly once (guarded by `std::call_once`) and the
/// result is cached for the lifetime of the program. Subsequent calls to
/// `simd::cpu_features()` are effectively a load of a static reference.
///
/// All flags are reported only when both the CPU advertises support **and**
/// the OS has enabled the corresponding extended state via XSAVE/XCR0 — this
/// is what avoids the classic "AVX is supported but #UD because the OS never
/// saved the YMM/ZMM registers" trap.

#include <cstdint>
#include <mutex>

#if defined(_MSC_VER)
    #include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
    #include <cpuid.h>
#endif

namespace simd {

struct CPUFeatures {
    bool has_sse2     = false;
    bool has_sse41    = false;
    bool has_sse42    = false;
    bool has_avx      = false;
    bool has_avx2     = false;
    bool has_fma      = false;
    bool has_avx512f  = false;  ///< AVX-512 Foundation
    bool has_avx512dq = false;  ///< Doubleword & Quadword instructions
    bool has_avx512bw = false;  ///< Byte & Word instructions
    bool has_avx512vl = false;  ///< Vector Length extensions

    /// Best human-readable name of the highest available SIMD ISA.
    const char* best_isa() const noexcept {
        if (has_avx512f) return "AVX-512";
        if (has_avx2)    return "AVX2";
        if (has_avx)     return "AVX";
        if (has_sse42)   return "SSE4.2";
        if (has_sse41)   return "SSE4.1";
        if (has_sse2)    return "SSE2";
        return "scalar";
    }
};

namespace detail {

inline void cpuid_call(int leaf, int subleaf, int regs[4]) noexcept {
#if defined(_MSC_VER)
    __cpuidex(regs, leaf, subleaf);
#elif defined(__GNUC__) || defined(__clang__)
    unsigned int a = 0, b = 0, c = 0, d = 0;
    __cpuid_count(static_cast<unsigned int>(leaf),
                  static_cast<unsigned int>(subleaf),
                  a, b, c, d);
    regs[0] = static_cast<int>(a);
    regs[1] = static_cast<int>(b);
    regs[2] = static_cast<int>(c);
    regs[3] = static_cast<int>(d);
#else
    regs[0] = regs[1] = regs[2] = regs[3] = 0;
#endif
}

inline std::uint64_t read_xcr0() noexcept {
#if defined(_MSC_VER)
    return _xgetbv(0);
#elif defined(__GNUC__) || defined(__clang__)
    std::uint32_t eax = 0, edx = 0;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return (static_cast<std::uint64_t>(edx) << 32) | eax;
#else
    return 0;
#endif
}

inline CPUFeatures probe() noexcept {
    CPUFeatures f;
    int regs[4] = {0, 0, 0, 0};

    // Leaf 0: highest supported standard leaf.
    cpuid_call(0, 0, regs);
    const int max_leaf = regs[0];
    if (max_leaf < 1) return f;

    // Leaf 1: legacy SSE/AVX/FMA bits + OSXSAVE.
    cpuid_call(1, 0, regs);
    const int ecx1 = regs[2];
    const int edx1 = regs[3];

    f.has_sse2  = (edx1 & (1 << 26)) != 0;
    f.has_sse41 = (ecx1 & (1 << 19)) != 0;
    f.has_sse42 = (ecx1 & (1 << 20)) != 0;
    f.has_fma   = (ecx1 & (1 << 12)) != 0;

    const bool osxsave   = (ecx1 & (1 << 27)) != 0;
    const bool avx_cpuid = (ecx1 & (1 << 28)) != 0;

    bool ymm_enabled = false;
    bool zmm_enabled = false;
    if (osxsave) {
        const std::uint64_t xcr0 = read_xcr0();
        // XMM (bit 1) + YMM (bit 2) must be enabled by the OS for AVX.
        ymm_enabled = (xcr0 & 0x6) == 0x6;
        // For AVX-512 we additionally need opmask (bit 5),
        // ZMM_Hi256 (bit 6) and Hi16_ZMM (bit 7).
        zmm_enabled = (xcr0 & 0xE6) == 0xE6;
    }

    f.has_avx = avx_cpuid && ymm_enabled;

    // Leaf 7, sub-leaf 0: AVX2 + AVX-512 family bits.
    if (max_leaf >= 7) {
        cpuid_call(7, 0, regs);
        const int ebx7 = regs[1];

        f.has_avx2 = f.has_avx && ((ebx7 & (1 << 5)) != 0);

        if (zmm_enabled) {
            f.has_avx512f  = (ebx7 & (1 << 16)) != 0;
            f.has_avx512dq = f.has_avx512f && ((ebx7 & (1 << 17)) != 0);
            f.has_avx512bw = f.has_avx512f && ((ebx7 & (1 << 30)) != 0);
            f.has_avx512vl = f.has_avx512f && ((ebx7 & (1 << 31)) != 0);
        }
    }

    return f;
}

} // namespace detail

/// Returns CPU features detected once on first call and cached thereafter.
inline const CPUFeatures& cpu_features() noexcept {
    static CPUFeatures cached;
    static std::once_flag flag;
    std::call_once(flag, []() noexcept { cached = detail::probe(); });
    return cached;
}

} // namespace simd
