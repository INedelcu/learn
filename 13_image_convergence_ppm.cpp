// 13 — Seeing the noise: a 2D image-space Monte Carlo convergence demo
// ====================================================================
//
// This capstone ties the series together by producing actual images you can
// open and look at. It mirrors, in miniature, exactly what the path tracer does:
//
//   THE COLOR OF A PIXEL IS A MONTE CARLO INTEGRAL over the pixel's area.
//
// Here the "scene" is just a handful of flat-shaded circles, and each pixel's
// true value is the AVERAGE SHADE over that pixel's little square — an
// antialiasing integral. Pixels fully inside one shape have zero variance (every
// sample agrees); pixels straddling an EDGE have high variance (samples disagree
// about which shape they hit), and that variance is the speckled noise you see.
// This is the 2D analogue of a path tracer's noisy penumbrae and glossy
// highlights: noise lives wherever the per-pixel integrand varies.
//
// We render the same scene many times and write a PPM image for each setting:
//   * sample counts N = 1, 4, 16, 64, 256 per pixel, and
//   * two samplers: plain RANDOM subpixel jitter vs STRATIFIED (one jittered
//     sample per sqrt(N) x sqrt(N) sub-cell) — the technique from example 07.
//
// For each image we also measure the RMS error against a high-quality reference
// (1024 stratified samples/pixel) and print it. Watch two things:
//   * random RMSE falls like 1/sqrt(N) (4x the samples -> ~2x less noise), and
//   * stratified beats random at matched sample counts — visibly smoother edges.
//
// The progressive frame averaging in PathTracingDemo.cs
// (lerp(prev, new, 1/(step+1))) is the identical idea: each frame adds samples
// to the running per-pixel mean, and the image denoises as 1/sqrt(total samples).
//
// Output: PPM files in .\images\ . PPM is the simplest image format there is
// (a tiny text header + raw RGB bytes); most image viewers, IrfanView, GIMP,
// and VS Code with a PPM extension can open them.

#include <cstdio>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <filesystem>
#include "mc_random.h"

// ---- the "scene": flat-shaded circles in UV space [0,1]^2, painter's order. --
struct Circle { double cx, cy, r, shade; };
static const Circle kScene[] = {
    {0.36, 0.42, 0.24, 0.85},
    {0.62, 0.56, 0.28, 0.45},
    {0.50, 0.74, 0.14, 1.00},
    {0.74, 0.30, 0.12, 0.22},
};
static const double kBackground = 0.06;

// Shade seen along a ray through (u,v): the last (top-most) circle that contains
// the point wins; otherwise the background.
static double SceneShade(double u, double v) {
    double s = kBackground;
    for (const Circle& c : kScene) {
        double dx = u - c.cx, dy = v - c.cy;
        if (dx * dx + dy * dy <= c.r * c.r) s = c.shade;
    }
    return s;
}

static const int W = 240, H = 240;

// Estimate one pixel's average shade with N samples. 'stratified' lays the
// samples on a sqrt(N) x sqrt(N) jittered grid; otherwise they are independent.
// The RNG is seeded per pixel (as in a real renderer) for deterministic images.
static double RenderPixel(int px, int py, int N, bool stratified, uint32_t salt) {
    Rng rng;
    rng.SetSequence((uint64_t)(py * W + px) * 9781u + salt);
    double sum = 0.0;
    if (stratified) {
        int s = (int)std::lround(std::sqrt((double)N));   // N is a perfect square
        for (int j = 0; j < s; ++j)
            for (int i = 0; i < s; ++i) {
                double u = (px + (i + rng.NextFloat()) / s) / W;
                double v = (py + (j + rng.NextFloat()) / s) / H;
                sum += SceneShade(u, v);
            }
        return sum / (s * s);
    }
    for (int k = 0; k < N; ++k) {
        double u = (px + rng.NextFloat()) / W;
        double v = (py + rng.NextFloat()) / H;
        sum += SceneShade(u, v);
    }
    return sum / N;
}

// Write a grayscale buffer (values in [0,1]) as a binary PPM (P6) image.
static void WritePPM(const std::string& path, const std::vector<double>& img) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { std::perror(path.c_str()); return; }
    std::fprintf(f, "P6\n%d %d\n255\n", W, H);
    std::vector<unsigned char> row(W * 3);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            double s = img[y * W + x];
            unsigned char b = (unsigned char)(s < 0 ? 0 : s > 1 ? 255 : s * 255.0 + 0.5);
            row[x * 3 + 0] = row[x * 3 + 1] = row[x * 3 + 2] = b;   // gray
        }
        std::fwrite(row.data(), 1, row.size(), f);
    }
    std::fclose(f);
}

int main() {
    std::filesystem::create_directories("images");

    // --- Reference image: many stratified samples per pixel. ----------------
    std::vector<double> ref(W * H);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            ref[y * W + x] = RenderPixel(x, y, 1024, /*stratified=*/true, /*salt=*/0);
    WritePPM("images/reference.ppm", ref);

    // --- Render each (sampler, N) and report RMS error vs the reference. ----
    printf("Rendering %dx%d images into .\\images\\  (reference = 1024 stratified spp)\n\n",
           W, H);
    printf("%6s | %14s %16s | %14s %16s\n",
           "N(spp)", "random RMSE", "rndRMSE*sqrtN", "stratified RMSE", "stratRMSE*sqrtN");

    const int kCounts[] = {1, 4, 16, 64, 256};
    for (int N : kCounts) {
        std::vector<double> rnd(W * H), strat(W * H);
        double seR = 0.0, seS = 0.0;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                double r = RenderPixel(x, y, N, false, /*salt=*/100u + N);
                double s = RenderPixel(x, y, N, true,  /*salt=*/200u + N);
                rnd[y * W + x] = r;
                strat[y * W + x] = s;
                double dr = r - ref[y * W + x]; seR += dr * dr;
                double ds = s - ref[y * W + x]; seS += ds * ds;
            }
        double rmseR = std::sqrt(seR / (W * H));
        double rmseS = std::sqrt(seS / (W * H));

        char nameR[64], nameS[64];
        std::snprintf(nameR, sizeof nameR, "images/random_%04d.ppm", N);
        std::snprintf(nameS, sizeof nameS, "images/stratified_%04d.ppm", N);
        WritePPM(nameR, rnd);
        WritePPM(nameS, strat);

        printf("%6d | %14.4e %16.4f | %14.4e %16.4f\n",
               N, rmseR, rmseR * std::sqrt((double)N),
               rmseS, rmseS * std::sqrt((double)N));
    }

    printf("\nWrote reference.ppm plus random_NNNN.ppm / stratified_NNNN.ppm.\n"
           "Open random_0001.ppm then random_0256.ppm to watch the edge speckle\n"
           "melt away, and compare random_0016.ppm with stratified_0016.ppm at the\n"
           "same cost. Numerically: 'random RMSE * sqrt(N)' stays ~constant (the\n"
           "1/sqrt(N) law again), while stratified RMSE is markedly lower because\n"
           "the only noisy pixels are the thin edge bands and stratification\n"
           "samples those edges evenly.\n");
    return 0;
}
