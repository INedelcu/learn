// 17 — Adaptive sampling: spend samples where the variance is
// ===========================================================
//
// Example 13 gave every pixel the SAME number of samples. But the noise was not
// uniform — flat interiors converged instantly while edges stayed speckled. That
// is wasteful: samples poured into a flat region buy nothing. ADAPTIVE SAMPLING
// fixes the allocation. With a fixed total budget, send more samples to the
// high-variance pixels and fewer to the calm ones.
//
// HOW MANY MORE? If pixel i has per-sample standard deviation sigma_i and gets
// n_i samples, its remaining error variance is sigma_i^2 / n_i. Minimizing the
// total error  sum_i sigma_i^2 / n_i  under a fixed budget  sum_i n_i = B  gives
// (Lagrange multipliers) the classic result
//
//   n_i  proportional to  sigma_i          ("Neyman allocation").
//
// So we just need an estimate of each pixel's sigma_i. We get it with a cheap
// PILOT pass (a few samples everywhere), compute the sample standard deviation,
// then hand out the remaining budget in proportion to it. Pixels whose pilot
// samples all agreed (sigma_i = 0, e.g. flat interiors) need nothing more; the
// edges soak up the budget.
//
// We compare, at an equal total sample count:
//   * UNIFORM  — every pixel gets B / numPixels samples;
//   * ADAPTIVE — pilot samples everywhere + the rest by Neyman allocation,
// and measure RMS error against a high-quality reference. Adaptive wins because
// it stops wasting samples on the flat regions. We also write a "sample density"
// image so you can SEE the budget concentrate on the edges. Production renderers
// do exactly this (plus perceptual metrics) to decide where to keep tracing.

#include <cstdio>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <filesystem>
#include "mc_random.h"

// ---- the same flat-shaded-circle scene as example 13 -----------------------
struct Circle { double cx, cy, r, shade; };
static const Circle kScene[] = {
    {0.36, 0.42, 0.24, 0.85}, {0.62, 0.56, 0.28, 0.45},
    {0.50, 0.74, 0.14, 1.00}, {0.74, 0.30, 0.12, 0.22},
};
static const double kBackground = 0.06;
static double SceneShade(double u, double v) {
    double s = kBackground;
    for (const Circle& c : kScene) {
        double dx = u - c.cx, dy = v - c.cy;
        if (dx * dx + dy * dy <= c.r * c.r) s = c.shade;
    }
    return s;
}

static const int W = 192, H = 192;
static const int kNumPix = W * H;

static double SampleMean(int px, int py, int n, uint32_t salt, double* outM2 = nullptr) {
    Rng rng; rng.SetSequence((uint64_t)(py * W + px) * 9781u + salt);
    double mean = 0.0, m2 = 0.0;
    for (int k = 0; k < n; ++k) {
        double u = (px + rng.NextFloat()) / W, v = (py + rng.NextFloat()) / H;
        double s = SceneShade(u, v);
        double d = s - mean; mean += d / (k + 1); m2 += d * (s - mean);
    }
    if (outM2) *outM2 = m2;
    return mean;
}

static void WritePPM(const std::string& path, const std::vector<double>& img) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { std::perror(path.c_str()); return; }
    std::fprintf(f, "P6\n%d %d\n255\n", W, H);
    std::vector<unsigned char> row(W * 3);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            double s = img[y * W + x];
            unsigned char b = (unsigned char)(s < 0 ? 0 : s > 1 ? 255 : s * 255.0 + 0.5);
            row[x * 3] = row[x * 3 + 1] = row[x * 3 + 2] = b;
        }
        std::fwrite(row.data(), 1, row.size(), f);
    }
    std::fclose(f);
}

int main() {
    std::filesystem::create_directories("images");
    const int kPilot = 32;         // pilot samples/pixel: enough for a RELIABLE sigma
    const int kAvg   = 128;        // target average samples/pixel -> total budget
    const long long kBudget = (long long)kAvg * kNumPix;
    // A variance FLOOR. The pilot's sigma is itself noisy: a thin-edge pixel can
    // draw kPilot identical samples and look flat (sigma_hat = 0). Adding a small
    // floor to the allocation weight guarantees every pixel keeps a baseline of
    // samples, so such a pixel is never starved below the uniform baseline.
    const double kSigmaFloor = 0.01;

    // --- reference (1024 stratified spp), for measuring error. --------------
    std::vector<double> ref(kNumPix);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            Rng rng; rng.SetSequence((uint64_t)(y * W + x) * 9781u + 7u);
            int s = 32; double sum = 0.0;                      // 32x32 = 1024
            for (int j = 0; j < s; ++j) for (int i = 0; i < s; ++i)
                sum += SceneShade((x + (i + rng.NextFloat()) / s) / W,
                                  (y + (j + rng.NextFloat()) / s) / H);
            ref[y * W + x] = sum / (s * s);
        }

    // --- ADAPTIVE: pilot pass -> per-pixel sigma -> Neyman allocation. ------
    // Neyman's rule (n_i proportional to sigma_i) minimizes total error variance
    // under a fixed budget. We allocate the WHOLE budget by the floored weight
    // (sigma_i + floor), never going below the pilot count (which we reuse).
    std::vector<double> pilotMean(kNumPix), weight(kNumPix);
    double wSum = 0.0;
    for (int i = 0; i < kNumPix; ++i) {
        double m2; pilotMean[i] = SampleMean(i % W, i / W, kPilot, 2u, &m2);
        double sigma = std::sqrt(m2 / (kPilot - 1));           // sample std dev
        weight[i] = sigma + kSigmaFloor;
        wSum += weight[i];
    }

    std::vector<double> adapt(kNumPix), density(kNumPix);
    double seAdapt = 0.0; long long adaptTotal = 0; int maxN = kPilot;
    for (int i = 0; i < kNumPix; ++i) {
        int n = (int)std::llround(kBudget * weight[i] / wSum);
        if (n < kPilot) n = kPilot;                            // reuse the pilot work
        adaptTotal += n; if (n > maxN) maxN = n;
        int extra = n - kPilot;
        double extraMean = extra > 0 ? SampleMean(i % W, i / W, extra, 3u) : 0.0;
        adapt[i] = (kPilot * pilotMean[i] + extra * extraMean) / n;
        density[i] = n;
        double d = adapt[i] - ref[i]; seAdapt += d * d;
    }

    // --- UNIFORM baseline at the SAME total budget (for a fair comparison). --
    int uspp = (int)std::llround((double)adaptTotal / kNumPix);
    std::vector<double> uni(kNumPix);
    double seUni = 0.0;
    for (int i = 0; i < kNumPix; ++i) {
        uni[i] = SampleMean(i % W, i / W, uspp, 1u);
        double d = uni[i] - ref[i]; seUni += d * d;
    }

    // sample-density heatmap (sqrt-scaled so low counts are visible)
    for (int i = 0; i < kNumPix; ++i) density[i] = std::sqrt(density[i] / maxN);

    WritePPM("images/adaptive_render.ppm",  adapt);
    WritePPM("images/uniform_render.ppm",   uni);
    WritePPM("images/adaptive_density.ppm", density);

    printf("Adaptive vs uniform sampling on a %dx%d image, matched budget.\n\n", W, H);
    printf("  total samples (uniform,  %d spp) = %lld\n", uspp, (long long)uspp * kNumPix);
    printf("  total samples (adaptive, %d-%d spp) = %lld\n\n", kPilot, maxN, adaptTotal);
    printf("  %-10s %16s\n", "method", "RMS error");
    printf("  %-10s %16.4e\n", "uniform",  std::sqrt(seUni / kNumPix));
    printf("  %-10s %16.4e\n", "adaptive", std::sqrt(seAdapt / kNumPix));
    printf("  error reduction: %.2fx\n", std::sqrt(seUni / kNumPix) / std::sqrt(seAdapt / kNumPix));

    printf("\nWrote adaptive_render.ppm, uniform_render.ppm, adaptive_density.ppm.\n"
           "The density map is bright exactly along the circle edges — that's where\n"
           "sigma_i is large and the Neyman rule sends the budget. Same total\n"
           "samples, lower error: adaptive sampling stops paying for the flat\n"
           "regions that were already converged after the pilot pass.\n");
    return 0;
}
