/// \file image_blur.cpp
/// Image box-blur demo: compares Scalar, SSE2, AVX2, and AVX2+Threads backends.
///
/// Algorithm: separable two-pass box blur (horizontal then vertical).
/// For each output row the kernel is:
///   1. For every shift dx in [-R, +R] copy the shifted source row into a
///      scratch buffer, then call backend.add() — this is the SIMD hot-path.
///   2. Scale the accumulator with backend.multiply_scalar().
/// The vertical pass is identical, operating on rows of the horizontal result.
/// All four backends run the same algorithm; only the width of the SIMD
/// register changes (1 / 4 / 8 floats per clock), making the speedup table
/// a clean demonstration of vectorisation gain.
///
/// Usage:
///   image_blur                         # generates a 1920x1080 test image
///   image_blur input.bmp               # blur an existing 24-bit BMP
///   image_blur input.bmp 15            # custom radius (default 10)

#include <simd_array/array.hpp>
#include <simd_array/avx512_backend.hpp>
#include <simd_array/avx_backend.hpp>
#include <simd_array/backend.hpp>
#include <simd_array/cpu_features.hpp>
#include <simd_array/scalar_backend.hpp>
#include <simd_array/sse_backend.hpp>
#include <simd_array/thread_pool.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// ============================================================================
// Minimal 24-bit BMP I/O — no external dependencies
// ============================================================================

struct Image {
    int width  = 0;
    int height = 0;
    /// RGB, top-down order, 3 bytes per pixel (no row padding).
    std::vector<uint8_t> pixels;

    uint8_t& r(int y, int x) noexcept { return pixels[static_cast<size_t>((y * width + x) * 3 + 0)]; }
    uint8_t& g(int y, int x) noexcept { return pixels[static_cast<size_t>((y * width + x) * 3 + 1)]; }
    uint8_t& b(int y, int x) noexcept { return pixels[static_cast<size_t>((y * width + x) * 3 + 2)]; }
    const uint8_t& r(int y, int x) const noexcept { return pixels[static_cast<size_t>((y * width + x) * 3 + 0)]; }
    const uint8_t& g(int y, int x) const noexcept { return pixels[static_cast<size_t>((y * width + x) * 3 + 1)]; }
    const uint8_t& b(int y, int x) const noexcept { return pixels[static_cast<size_t>((y * width + x) * 3 + 2)]; }
};

namespace bmp {

static void put_u16(std::vector<uint8_t>& v, uint16_t x) {
    v.push_back(static_cast<uint8_t>(x));
    v.push_back(static_cast<uint8_t>(x >> 8));
}
static void put_u32(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back(static_cast<uint8_t>(x));
    v.push_back(static_cast<uint8_t>(x >>  8));
    v.push_back(static_cast<uint8_t>(x >> 16));
    v.push_back(static_cast<uint8_t>(x >> 24));
}
static void put_i32(std::vector<uint8_t>& v, int32_t x) { put_u32(v, static_cast<uint32_t>(x)); }

static uint16_t get_u16(const uint8_t* p) { return static_cast<uint16_t>(p[0] | (p[1] << 8)); }
static uint32_t get_u32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8)
         | (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}
static int32_t get_i32(const uint8_t* p) { return static_cast<int32_t>(get_u32(p)); }

bool save(const char* path, const Image& img) {
    const int row_stride  = (img.width * 3 + 3) & ~3; // pad rows to 4 bytes
    const uint32_t px_sz  = static_cast<uint32_t>(row_stride * img.height);
    const uint32_t fsize  = 54u + px_sz;

    std::vector<uint8_t> hdr;
    hdr.reserve(54);
    // File header (14 bytes)
    put_u16(hdr, 0x4D42u);    // 'BM'
    put_u32(hdr, fsize);
    put_u16(hdr, 0); put_u16(hdr, 0); // reserved
    put_u32(hdr, 54u);        // offset to pixel data
    // BITMAPINFOHEADER (40 bytes)
    put_u32(hdr, 40u);
    put_i32(hdr, img.width);
    put_i32(hdr, img.height); // positive → bottom-up storage
    put_u16(hdr, 1);          // color planes
    put_u16(hdr, 24);         // bits per pixel
    put_u32(hdr, 0);          // compression: none
    put_u32(hdr, px_sz);
    put_i32(hdr, 2835);       // ~72 dpi
    put_i32(hdr, 2835);
    put_u32(hdr, 0); put_u32(hdr, 0); // color table

    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(hdr.data()), static_cast<std::streamsize>(hdr.size()));

    // Pixel data: bottom-up rows, BGR order, padded to 4-byte boundary
    std::vector<uint8_t> row(static_cast<size_t>(row_stride), 0);
    for (int y = img.height - 1; y >= 0; --y) {
        for (int x = 0; x < img.width; ++x) {
            row[static_cast<size_t>(x * 3 + 0)] = img.b(y, x);
            row[static_cast<size_t>(x * 3 + 1)] = img.g(y, x);
            row[static_cast<size_t>(x * 3 + 2)] = img.r(y, x);
        }
        f.write(reinterpret_cast<const char*>(row.data()), row_stride);
    }
    return true;
}

bool load(const char* path, Image& img) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    uint8_t hdr[54];
    if (!f.read(reinterpret_cast<char*>(hdr), 54)) return false;
    if (hdr[0] != 'B' || hdr[1] != 'M') { std::cerr << "Not a BMP file.\n"; return false; }

    const uint16_t bpp = get_u16(hdr + 28);
    if (bpp != 24) { std::cerr << "Only 24-bit BMP supported (got " << bpp << " bpp).\n"; return false; }

    const uint32_t px_offset = get_u32(hdr + 10);
    const int32_t  raw_w     = get_i32(hdr + 18);
    const int32_t  raw_h     = get_i32(hdr + 22);
    if (raw_w <= 0) { std::cerr << "Unsupported BMP width.\n"; return false; }

    img.width  = raw_w;
    img.height = std::abs(raw_h);
    img.pixels.resize(static_cast<size_t>(img.width * img.height * 3));

    const int row_stride = (img.width * 3 + 3) & ~3;
    std::vector<uint8_t> row(static_cast<size_t>(row_stride));

    f.seekg(static_cast<std::streamoff>(px_offset), std::ios::beg);

    const bool bottom_up = (raw_h > 0);
    if (bottom_up) {
        for (int y = img.height - 1; y >= 0; --y) {
            if (!f.read(reinterpret_cast<char*>(row.data()), row_stride)) return false;
            for (int x = 0; x < img.width; ++x) {
                img.r(y, x) = row[static_cast<size_t>(x * 3 + 2)];
                img.g(y, x) = row[static_cast<size_t>(x * 3 + 1)];
                img.b(y, x) = row[static_cast<size_t>(x * 3 + 0)];
            }
        }
    } else {
        for (int y = 0; y < img.height; ++y) {
            if (!f.read(reinterpret_cast<char*>(row.data()), row_stride)) return false;
            for (int x = 0; x < img.width; ++x) {
                img.r(y, x) = row[static_cast<size_t>(x * 3 + 2)];
                img.g(y, x) = row[static_cast<size_t>(x * 3 + 1)];
                img.b(y, x) = row[static_cast<size_t>(x * 3 + 0)];
            }
        }
    }
    return true;
}

} // namespace bmp

// ============================================================================
// Test image generator
// ============================================================================

static Image generate_test_image(int width, int height) {
    Image img;
    img.width  = width;
    img.height = height;
    img.pixels.resize(static_cast<size_t>(width * height * 3));

    const float cx  = static_cast<float>(width)  * 0.5f;
    const float cy  = static_cast<float>(height) * 0.5f;
    const float max_r = std::sqrt(cx * cx + cy * cy);
    const int checker = 32;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float dx = static_cast<float>(x) - cx;
            const float dy = static_cast<float>(y) - cy;
            const float r  = std::sqrt(dx * dx + dy * dy) / max_r;
            const float a  = std::atan2(dy, dx);

            float red   = 0.5f + 0.5f * std::sin(a * 3.0f + r * 8.0f);
            float green = 0.5f + 0.5f * std::cos(r * 12.0f);
            float blue  = 1.0f - r;

            if (((x / checker) + (y / checker)) & 1) {
                red   = 1.0f - red;
                green = 1.0f - green;
                blue  = 1.0f - blue;
            }

            img.r(y, x) = static_cast<uint8_t>(std::clamp(red,   0.0f, 1.0f) * 255.0f);
            img.g(y, x) = static_cast<uint8_t>(std::clamp(green, 0.0f, 1.0f) * 255.0f);
            img.b(y, x) = static_cast<uint8_t>(std::clamp(blue,  0.0f, 1.0f) * 255.0f);
        }
    }
    return img;
}

// ============================================================================
// Separable box blur — single-threaded (serial)
// ============================================================================

/// Horizontal blur of every row in `src` → `dst`.
/// Each output pixel = average of 2*radius+1 horizontally adjacent pixels.
///
/// Key: build a padded row [left_border | src_row | right_border] once per
/// row, then for each shift dx simply call be.add() directly into the padded
/// buffer at offset (radius + dx) — no per-shift scalar copy, so SIMD
/// operations dominate the hot path.
static void hblur_serial(simd::ArrayBackend<float>& be,
                          const float* src, float* dst,
                          int W, int H, int radius) {
    const size_t Wsz     = static_cast<size_t>(W);
    const size_t Rsz     = static_cast<size_t>(radius);
    const size_t pad_W   = Wsz + 2 * Rsz; // padded row length
    std::vector<float> padded(pad_W);
    std::vector<float> acc(Wsz);

    for (int y = 0; y < H; ++y) {
        const float* src_row = src + y * W;
        float*       dst_row = dst + y * W;

        // Build padded row: replicate edge pixels into the border regions.
        std::fill(padded.begin(), padded.begin() + static_cast<std::ptrdiff_t>(Rsz), src_row[0]);
        std::memcpy(padded.data() + Rsz, src_row, Wsz * sizeof(float));
        std::fill(padded.begin() + static_cast<std::ptrdiff_t>(Rsz + Wsz), padded.end(), src_row[Wsz - 1]);

        // Accumulate: for each shift dx, padded[radius+dx .. radius+dx+W] is a
        // contiguous W-element slice — pass it directly to SIMD add, no copy.
        std::fill(acc.begin(), acc.end(), 0.0f);
        for (int dx = -radius; dx <= radius; ++dx) {
            be.add(acc.data(), acc.data(),
                   padded.data() + static_cast<size_t>(radius + dx), Wsz);
        }
        be.multiply_scalar(dst_row, acc.data(), 1.0f / static_cast<float>(2 * radius + 1), Wsz);
    }
}

/// Vertical blur: operates on column-oriented rows. Same structure as hblur
/// but shifts are in the y-direction, copying entire rows via memcpy.
static void vblur_serial(simd::ArrayBackend<float>& be,
                          const float* src, float* dst,
                          int W, int H, int radius) {
    const size_t Wsz = static_cast<size_t>(W);
    std::vector<float> acc(Wsz);

    for (int y = 0; y < H; ++y) {
        float* dst_row = dst + y * W;
        std::fill(acc.begin(), acc.end(), 0.0f);

        for (int dy = -radius; dy <= radius; ++dy) {
            int sy = y + dy;
            if (sy < 0) sy = 0;
            else if (sy >= H) sy = H - 1;
            // SIMD: acc[0..W] += src[sy*W .. sy*W+W]
            be.add(acc.data(), acc.data(), src + sy * W, Wsz);
        }
        be.multiply_scalar(dst_row, acc.data(), 1.0f / static_cast<float>(2 * radius + 1), Wsz);
    }
}

// ============================================================================
// Separable box blur — parallel (thread pool over rows)
// ============================================================================

/// Parallel horizontal blur. Rows are distributed across the thread pool;
/// each worker uses the SIMD backend for the inner accumulation.
/// Same padded-row trick as hblur_serial: one memcpy per row, then
/// (2*radius+1) direct SIMD adds into contiguous slices of the padded buffer.
static void hblur_parallel(simd::ArrayBackend<float>& be,
                             const float* src, float* dst,
                             int W, int H, int radius) {
    const size_t Wsz   = static_cast<size_t>(W);
    const size_t Rsz   = static_cast<size_t>(radius);
    const size_t pad_W = Wsz + 2 * Rsz;
    simd::default_pool().parallel_for(
        static_cast<size_t>(H),
        [&](size_t y_start, size_t y_end) {
            // Per-worker scratch, allocated once per chunk.
            std::vector<float> padded(pad_W);
            std::vector<float> acc(Wsz);
            for (size_t y = y_start; y < y_end; ++y) {
                const float* src_row = src + y * Wsz;
                float*       dst_row = dst + y * Wsz;

                std::fill(padded.begin(), padded.begin() + static_cast<std::ptrdiff_t>(Rsz), src_row[0]);
                std::memcpy(padded.data() + Rsz, src_row, Wsz * sizeof(float));
                std::fill(padded.begin() + static_cast<std::ptrdiff_t>(Rsz + Wsz), padded.end(), src_row[Wsz - 1]);

                std::fill(acc.begin(), acc.end(), 0.0f);
                for (int dx = -radius; dx <= radius; ++dx) {
                    be.add(acc.data(), acc.data(),
                           padded.data() + static_cast<size_t>(radius + dx), Wsz);
                }
                be.multiply_scalar(dst_row, acc.data(),
                                   1.0f / static_cast<float>(2 * radius + 1), Wsz);
            }
        });
}

/// Parallel vertical blur. Each output row depends only on (read-only)
/// source rows, so rows are safely distributed across workers.
static void vblur_parallel(simd::ArrayBackend<float>& be,
                             const float* src, float* dst,
                             int W, int H, int radius) {
    const size_t Wsz = static_cast<size_t>(W);
    simd::default_pool().parallel_for(
        static_cast<size_t>(H),
        [&](size_t y_start, size_t y_end) {
            std::vector<float> acc(Wsz);
            for (size_t y = y_start; y < y_end; ++y) {
                float* dst_row = dst + y * Wsz;
                std::fill(acc.begin(), acc.end(), 0.0f);

                for (int dy = -radius; dy <= radius; ++dy) {
                    int sy = static_cast<int>(y) + dy;
                    if (sy < 0) sy = 0;
                    else if (sy >= H) sy = H - 1;
                    be.add(acc.data(), acc.data(),
                           src + static_cast<size_t>(sy) * Wsz, Wsz);
                }
                be.multiply_scalar(dst_row, acc.data(),
                                   1.0f / static_cast<float>(2 * radius + 1), Wsz);
            }
        });
}

// ============================================================================
// Full 3-channel image blur
// ============================================================================

static Image blur_image(const Image& input, int radius,
                         simd::ArrayBackend<float>& be,
                         bool parallel) {
    const int    W  = input.width;
    const int    H  = input.height;
    const size_t N  = static_cast<size_t>(W * H);

    std::vector<float> src_ch(N), h_pass(N), dst_ch(N);

    Image output;
    output.width  = W;
    output.height = H;
    output.pixels.resize(N * 3);

    for (int ch = 0; ch < 3; ++ch) {
        // uint8 → float [0, 1]
        for (size_t i = 0; i < N; ++i)
            src_ch[i] = static_cast<float>(input.pixels[i * 3 + static_cast<size_t>(ch)]) / 255.0f;

        if (parallel) {
            hblur_parallel(be, src_ch.data(), h_pass.data(), W, H, radius);
            vblur_parallel(be, h_pass.data(), dst_ch.data(), W, H, radius);
        } else {
            hblur_serial(be, src_ch.data(), h_pass.data(), W, H, radius);
            vblur_serial(be, h_pass.data(), dst_ch.data(), W, H, radius);
        }

        // float [0, 1] → uint8
        for (size_t i = 0; i < N; ++i) {
            const float v = std::clamp(dst_ch[i], 0.0f, 1.0f);
            output.pixels[i * 3 + static_cast<size_t>(ch)] = static_cast<uint8_t>(v * 255.0f);
        }
    }
    return output;
}

// ============================================================================
// Timing helpers
// ============================================================================

template<typename Fn>
static double bench_ms(Fn&& fn, int warmup = 1, int runs = 3) {
    for (int i = 0; i < warmup; ++i) fn();
    double best = 1e18;
    for (int i = 0; i < runs; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ms < best) best = ms;
    }
    return best;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    // ------------------------------------------------------------------
    // Parse arguments
    // ------------------------------------------------------------------
    Image input;
    std::string in_path, out_path = "blurred_output.bmp";
    int radius = 10;

    if (argc >= 2) {
        in_path = argv[1];
        if (!bmp::load(in_path.c_str(), input)) {
            std::cerr << "Error: cannot load '" << in_path << "'.\n";
            return 1;
        }
    } else {
        std::cout << "No input file given — generating 1920x1080 synthetic image.\n";
        input   = generate_test_image(1920, 1080);
        in_path = "test_input.bmp";
        if (bmp::save(in_path.c_str(), input))
            std::cout << "Saved: " << in_path << "\n";
    }
    if (argc >= 3) radius = std::atoi(argv[2]);

    // ------------------------------------------------------------------
    // System info
    // ------------------------------------------------------------------
    const auto& feat    = simd::cpu_features();
    const size_t nthreads = simd::default_pool().thread_count();
    const size_t pixels   = static_cast<size_t>(input.width * input.height);

    std::cout << "\n=== SIMD Array Library — Image Box Blur Demo ===\n";
    std::cout << "Image     : " << input.width << " x " << input.height
              << " (" << pixels / 1'000 << " K pixels)\n";
    std::cout << "Blur      : box filter, radius " << radius
              << " (kernel " << 2 * radius + 1 << "x" << 2 * radius + 1 << ")\n";
    std::cout << "CPU caps  : SSE2=" << feat.has_sse2
              << "  AVX2=" << feat.has_avx2
              << "  AVX-512=" << feat.has_avx512f << "\n";
    std::cout << "Threads   : " << nthreads << "\n\n";

    // ------------------------------------------------------------------
    // Benchmark each configuration
    // ------------------------------------------------------------------
    struct Result { std::string label; double ms; };
    std::vector<Result> results;
    Image last_output;

    // Scalar
    {
        simd::ScalarBackend<float> be;
        double ms = bench_ms([&] { last_output = blur_image(input, radius, be, false); });
        results.push_back({"Scalar (1 lane)", ms});
        std::cout << "  [1/4] Scalar done\n";
    }

    // SSE2
    if (feat.has_sse2) {
        simd::SSEBackend<float> be;
        double ms = bench_ms([&] { last_output = blur_image(input, radius, be, false); });
        results.push_back({"SSE2  (4 lanes)", ms});
        std::cout << "  [2/4] SSE2 done\n";
    }

    // AVX2 single-thread
    if (feat.has_avx2) {
        simd::AVXBackend<float> be;
        double ms = bench_ms([&] { last_output = blur_image(input, radius, be, false); });
        results.push_back({"AVX2  (8 lanes)", ms});
        std::cout << "  [3/4] AVX2 done\n";
    }

    // AVX2 + thread pool
    if (feat.has_avx2) {
        simd::AVXBackend<float> be;
        double ms = bench_ms([&] { last_output = blur_image(input, radius, be, true); });
        results.push_back({"AVX2  + " + std::to_string(nthreads) + " threads", ms});
        std::cout << "  [4/4] AVX2+MT done\n";
    }

    // AVX-512 (bonus — if available)
    if (feat.has_avx512f) {
        simd::AVX512Backend<float> be;
        double ms = bench_ms([&] { last_output = blur_image(input, radius, be, false); });
        results.push_back({"AVX-512 (16 lanes)", ms});
        simd::AVX512Backend<float> be2;
        ms = bench_ms([&] { last_output = blur_image(input, radius, be2, true); });
        results.push_back({"AVX-512 + " + std::to_string(nthreads) + " threads", ms});
    }

    // ------------------------------------------------------------------
    // Results table
    // ------------------------------------------------------------------
    const double scalar_ms = results.front().ms;
    const size_t col = 26; // label column width

    std::cout << "\n";
    std::cout << "+" << std::string(col + 2, '-') << "+" << std::string(12, '-')
              << "+" << std::string(10, '-') << "+\n";
    std::cout << "| " << std::left << std::setw(static_cast<int>(col)) << "Configuration"
              << " |    Time, ms | Speedup   |\n";
    std::cout << "+" << std::string(col + 2, '-') << "+" << std::string(12, '-')
              << "+" << std::string(10, '-') << "+\n";

    for (const auto& r : results) {
        const double speedup = scalar_ms / r.ms;
        std::cout << "| " << std::left  << std::setw(static_cast<int>(col)) << r.label
                  << " | " << std::right << std::setw(9) << std::fixed << std::setprecision(1) << r.ms
                  << " ms | "
                  << std::setw(6) << std::fixed << std::setprecision(2) << speedup << "x |\n";
    }
    std::cout << "+" << std::string(col + 2, '-') << "+" << std::string(12, '-')
              << "+" << std::string(10, '-') << "+\n";

    // ------------------------------------------------------------------
    // Save output
    // ------------------------------------------------------------------
    if (bmp::save(out_path.c_str(), last_output))
        std::cout << "\nSaved blurred image → " << out_path << "\n";
    else
        std::cerr << "Warning: could not save " << out_path << "\n";

    return 0;
}
