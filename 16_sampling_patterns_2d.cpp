// 16 — Seeing sample patterns: random vs stratified vs Halton (2D)
// ================================================================
//
// Examples 07 (stratification) and 11 (low-discrepancy) argued with NUMBERS that
// well-spread samples integrate better. This file shows the same thing with
// PICTURES: it scatters 256 points three ways and writes each to a PPM so you can
// look at *why*. Then it puts all three samplers in one integration-error table
// so the visual and the numeric stories line up.
//
//   RANDOM      independent uniform (x, y). Clumps and voids everywhere — those
//               are the wasted samples, i.e. the variance.
//   STRATIFIED  one jittered point per cell of a sqrt(N) x sqrt(N) grid. No two
//               points share a cell, so the clumping is broken up, but different
//               cells still don't coordinate.
//   HALTON      the deterministic low-discrepancy sequence from example 11
//               (radical inverse in bases 2 and 3). Globally even at every
//               prefix, with no random clumping at all.
//
// The test integral is the same smooth one from example 11,
// integral over [0,1]^2 of e^(x+y) = (e-1)^2, so the error table is directly
// comparable. Watch random sit at O(1/sqrt(N)) while BOTH stratified and Halton
// converge much faster. (For this smooth, separable 2-D integrand jittered
// stratification actually edges out Halton; Halton's real advantages are that it
// is incremental -- you can keep adding samples without fixing N up front -- and
// that it keeps working well in the higher dimensions a path tracer needs, where
// a full stratified grid would require an impossible N^d cells.)
//
// Production path tracers care because every pixel spends its samples on a
// multi-dimensional unit hypercube (pixel xy, lens, light, bounce directions...);
// spreading those samples well is free variance reduction. This renderer uses a
// hashed RNG for simplicity, but the lesson — don't let samples clump — is the
// same one stratification and low-discrepancy sequences enforce.

#include <cstdio>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <utility>
#include <filesystem>
#include "mc_random.h"

static double RadicalInverse(unsigned base, uint64_t a) {
    const double invBase = 1.0 / base; double invBaseN = 1.0, r = 0.0;
    while (a) { r += (a % base) * (invBaseN *= invBase); a /= base; }
    return r;
}

// Draw a point set onto a white canvas with a faint NxN grid, save as PPM.
static void DrawScatter(const std::string& path,
                        const std::vector<std::pair<float, float>>& pts, int grid) {
    const int S = 384;
    std::vector<unsigned char> img(S * S * 3, 255);          // white background
    for (int g = 1; g < grid; ++g) {                          // faint grid lines
        int p = g * S / grid;
        for (int k = 0; k < S; ++k) {
            for (int c = 0; c < 3; ++c) {
                img[(k * S + p) * 3 + c] = 220;
                img[(p * S + k) * 3 + c] = 220;
            }
        }
    }
    for (auto& pt : pts) {                                    // 3x3 black dots
        int cx = (int)(pt.first * S), cy = (int)(pt.second * S);
        for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
            int x = cx + dx, y = cy + dy;
            if (x < 0 || y < 0 || x >= S || y >= S) continue;
            int o = (y * S + x) * 3; img[o] = img[o + 1] = img[o + 2] = 0;
        }
    }
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { std::perror(path.c_str()); return; }
    std::fprintf(f, "P6\n%d %d\n255\n", S, S);
    std::fwrite(img.data(), 1, img.size(), f);
    std::fclose(f);
}

int main() {
    Rng rng(/*sequence=*/16);
    std::filesystem::create_directories("images");

    // --- (1) Scatter pictures of 256 points, three ways. --------------------
    const int N = 256, s = 16;                 // 16 x 16 grid for stratified
    std::vector<std::pair<float, float>> rnd, strat, halt;
    for (int i = 0; i < N; ++i) rnd.push_back({rng.NextFloat(), rng.NextFloat()});
    for (int j = 0; j < s; ++j) for (int i = 0; i < s; ++i)
        strat.push_back({(i + rng.NextFloat()) / s, (j + rng.NextFloat()) / s});
    for (int i = 0; i < N; ++i)
        halt.push_back({(float)RadicalInverse(2, i), (float)RadicalInverse(3, i)});
    DrawScatter("images/pattern_random.ppm",     rnd,   s);
    DrawScatter("images/pattern_stratified.ppm", strat, s);
    DrawScatter("images/pattern_halton.ppm",     halt,  s);
    printf("Wrote images/pattern_random.ppm, pattern_stratified.ppm, pattern_halton.ppm.\n");
    printf("(Random clumps and leaves gaps; stratified fills every cell; Halton is\n"
           " globally even.)\n\n");

    // --- (2) Same picture as numbers: 2D integration error. -----------------
    const double kE = 2.71828182845904523536;
    const double exact = (kE - 1.0) * (kE - 1.0);
    auto f = [](double x, double y) { return std::exp(x + y); };
    const int kTrials = 256;                   // for the randomized methods

    printf("integral_[0,1]^2 e^(x+y) = (e-1)^2 = %.7f\n\n", exact);
    printf("%8s %14s %14s %14s\n", "N", "random RMSE", "stratified RMSE", "Halton err");
    for (int Npts = 16; Npts <= 4096; Npts *= 4) {
        int side = (int)std::lround(std::sqrt((double)Npts));   // perfect squares
        double seR = 0.0, seS = 0.0;
        for (int t = 0; t < kTrials; ++t) {
            double sr = 0.0, ss = 0.0;
            for (int i = 0; i < Npts; ++i) sr += f(rng.NextFloat(), rng.NextFloat());
            for (int j = 0; j < side; ++j) for (int i = 0; i < side; ++i)
                ss += f((i + rng.NextFloat()) / side, (j + rng.NextFloat()) / side);
            double er = sr / Npts - exact, es = ss / Npts - exact;
            seR += er * er; seS += es * es;
        }
        double sh = 0.0;
        for (int i = 0; i < Npts; ++i) sh += f(RadicalInverse(2, i), RadicalInverse(3, i));
        printf("%8d %14.3e %14.3e %14.3e\n",
               Npts, std::sqrt(seR / kTrials), std::sqrt(seS / kTrials),
               std::fabs(sh / Npts - exact));
    }

    printf("\nThe pictures and the table tell one story: random sampling wastes\n"
           "effort on clumps (by far the slowest column), while stratification and\n"
           "Halton both spread the samples out and converge much faster. Open the\n"
           "three PPMs (view_images.ps1) and the difference is obvious by eye -- the\n"
           "random panel has visible clusters and holes the other two simply lack.\n");
    return 0;
}
