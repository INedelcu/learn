// 36 — Per-pixel RNG seeding: why decorrelation matters
// =====================================================
//
// A path tracer gives every pixel its own random stream. How you SEED those
// streams is easy to get wrong, and a wrong choice shows up as visible structure
// in the image instead of clean noise -- patterns that no amount of averaging
// removes cleanly, because neighboring pixels are drawing CORRELATED numbers.
//
// The renderer seeds its generator by mixing the pixel coordinates, the frame
// index and the convergence step through a hash (a Wang hash, in Utils.hlsl). The
// hash is the important part: it scrambles nearby seeds (pixel i and pixel i+1)
// into totally unrelated states, so adjacent pixels' samples are independent.
//
// This file makes the failure visible. We fill an image where each pixel's value
// is its FIRST random sample under two seeding schemes:
//   * BAD  -- a smooth function of the pixel position, frac(0.123*x + 0.317*y).
//             This is what you get from "low-effort" seeding (e.g. using a linear
//             index or an under-mixed hash): adjacent pixels barely differ, so the
//             image is diagonal banding, not noise.
//   * GOOD -- a Wang hash of the linear pixel index: each pixel is decorrelated,
//             so the image is structureless white noise.
//
// As a number, we report the mean absolute difference between horizontally
// adjacent pixels. For independent U[0,1] samples that expectation is 1/3 ~ 0.333;
// a much smaller value means the neighbors are correlated (structure). We also
// write both images so you can SEE the bands vs the noise.

#include <cstdio>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <filesystem>

static const int W = 256, H = 256;

// Wang hash: scrambles an integer seed thoroughly (the renderer uses this family).
static uint32_t WangHash(uint32_t s) {
    s = (s ^ 61u) ^ (s >> 16);
    s *= 9u;
    s = s ^ (s >> 4);
    s *= 0x27d4eb2du;
    s = s ^ (s >> 15);
    return s;
}

static void WritePPM(const std::string& path, const std::vector<float>& img) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { std::perror(path.c_str()); return; }
    std::fprintf(f, "P6\n%d %d\n255\n", W, H);
    std::vector<unsigned char> row(W * 3);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float v = img[y * W + x];
            unsigned char b = (unsigned char)(v < 0 ? 0 : v > 1 ? 255 : v * 255.0f + 0.5f);
            row[x * 3] = row[x * 3 + 1] = row[x * 3 + 2] = b;
        }
        std::fwrite(row.data(), 1, row.size(), f);
    }
    std::fclose(f);
}

int main() {
    std::filesystem::create_directories("images");

    std::vector<float> bad(W * H), good(W * H);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            uint32_t idx = (uint32_t)(y * W + x);
            // BAD: a smooth, under-mixed function of position -> correlated neighbors.
            float v = 0.123f * x + 0.317f * y; v -= std::floor(v);
            bad[y * W + x] = v;
            // GOOD: hash the index -> decorrelated stream per pixel.
            good[y * W + x] = WangHash(idx) * 0x1p-32f;
        }

    // Mean absolute difference between horizontally adjacent pixels.
    auto adjDiff = [](const std::vector<float>& img) {
        double s = 0; long long n = 0;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W - 1; ++x) { s += std::fabs(img[y * W + x + 1] - img[y * W + x]); ++n; }
        return s / n;
    };

    WritePPM("images/rng_bad.ppm", bad);
    WritePPM("images/rng_good.ppm", good);

    printf("Per-pixel first sample under two seeding schemes (%dx%d).\n\n", W, H);
    printf("  mean |adjacent difference|  (independent noise -> ~0.333):\n");
    printf("    BAD  (smooth seed) = %.4f   <- far below 0.333: neighbors correlated\n", adjDiff(bad));
    printf("    GOOD (Wang hash)   = %.4f   <- ~0.333: neighbors independent\n", adjDiff(good));
    printf("\nWrote images/rng_bad.ppm (diagonal banding -- structure) and\n");
    printf("images/rng_good.ppm (clean white noise). View with view_images.ps1.\n");

    printf("\nThe bad scheme isn't 'more random' or 'less random' on average -- each\n"
           "pixel's value is still in [0,1] -- but its samples are CORRELATED across\n"
           "pixels, so the error has low-frequency structure the eye latches onto and\n"
           "averaging cannot easily remove. Hashing the seed (pixel, frame, step),\n"
           "as the renderer does, is what turns per-pixel streams into independent\n"
           "noise. Good RNG seeding is as important as the sampling math.\n");
    return 0;
}
