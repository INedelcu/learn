// 30 — Blue noise vs white noise: it's not just the variance, it's the spectrum
// ==============================================================================
//
// "White noise" (independent uniform samples) has a flat power spectrum: its
// error contains all frequencies, including low ones the eye is very sensitive to
// -- that is the clumpy, blotchy look of naive random sampling. "BLUE NOISE" is a
// point set whose energy is concentrated in HIGH frequencies and suppressed at
// low ones: the points stay well-separated, leaving no clumps or holes. The error
// it produces is high-frequency too, which the visual system averages away, so
// blue-noise sampling and dithering look dramatically cleaner at the same count --
// the reason it is used for sample placement and for distributing error across a
// frame.
//
// We generate blue noise with MITCHELL'S BEST-CANDIDATE algorithm: to place the
// k-th point, propose several random candidates and keep the one FARTHEST from all
// existing points (distances measured toroidally, so the set tiles seamlessly).
// We then show, against white noise of the same size:
//   * a much larger minimum / mean nearest-neighbor distance (no clumping);
//   * lower variance when integrating a smooth function (like stratification);
//   * scatter images (PPM) so you can see the even spacing -- compare with the
//     random/stratified/Halton panels of example 16.

#include <cstdio>
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>
#include <filesystem>
#include "mc_random.h"

struct P2 { float x, y; };

// Toroidal squared distance on the unit square (wrap-around).
static float Dist2(P2 a, P2 b) {
    float dx = std::fabs(a.x - b.x), dy = std::fabs(a.y - b.y);
    dx = std::fmin(dx, 1.0f - dx); dy = std::fmin(dy, 1.0f - dy);
    return dx * dx + dy * dy;
}

// Best-candidate blue noise: each new point is the best of (count+1) candidates.
static std::vector<P2> BlueNoise(int n, Rng& rng, int capCandidates = 30) {
    std::vector<P2> pts;
    for (int i = 0; i < n; ++i) {
        int m = std::min(i + 1, capCandidates);
        P2 best{0, 0}; float bestMin = -1.0f;
        for (int c = 0; c < m; ++c) {
            P2 cand{rng.NextFloat(), rng.NextFloat()};
            float md = 1e9f;
            for (const P2& p : pts) md = std::fmin(md, Dist2(cand, p));
            if (pts.empty()) md = 1e9f;
            if (md > bestMin) { bestMin = md; best = cand; }
        }
        pts.push_back(best);
    }
    return pts;
}

static double MeanNearest(const std::vector<P2>& p) {
    double s = 0;
    for (size_t i = 0; i < p.size(); ++i) {
        float md = 1e9f;
        for (size_t j = 0; j < p.size(); ++j) if (i != j) md = std::fmin(md, Dist2(p[i], p[j]));
        s += std::sqrt(md);
    }
    return s / p.size();
}

static void DrawScatter(const std::string& path, const std::vector<P2>& pts) {
    const int S = 384;
    std::vector<unsigned char> img(S * S * 3, 255);
    for (const P2& pt : pts) {
        int cx = (int)(pt.x * S), cy = (int)(pt.y * S);
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
    Rng rng(/*sequence=*/30);
    std::filesystem::create_directories("images");

    // --- (1) spacing + scatter images for one set of each. ------------------
    const int N = 600;
    std::vector<P2> white(N);
    for (int i = 0; i < N; ++i) white[i] = {rng.NextFloat(), rng.NextFloat()};
    std::vector<P2> blue = BlueNoise(N, rng);

    printf("Point spacing for %d points (toroidal nearest-neighbor distance):\n", N);
    printf("  white noise: mean nearest = %.4f\n", MeanNearest(white));
    printf("  blue  noise: mean nearest = %.4f  (larger = more evenly spaced)\n\n",
           MeanNearest(blue));
    DrawScatter("images/bluenoise_white.ppm", white);
    DrawScatter("images/bluenoise_blue.ppm", blue);

    // --- (2) integration variance on a smooth function. ---------------------
    const double kE = 2.71828182845904523536;
    const double exact = (kE - 1.0) * (kE - 1.0);
    auto f = [](float x, float y) { return std::exp(x + y); };
    const int n = 64, trials = 2000;
    double mW = 0, vW = 0, mB = 0, vB = 0;
    for (int t = 0; t < trials; ++t) {
        double sw = 0;
        for (int i = 0; i < n; ++i) sw += f(rng.NextFloat(), rng.NextFloat());
        double ew = sw / n; double dW = ew - mW; mW += dW / (t + 1); vW += dW * (ew - mW);

        std::vector<P2> bp = BlueNoise(n, rng);
        double sb = 0;
        for (const P2& p : bp) sb += f(p.x, p.y);
        double eb = sb / n; double dB = eb - mB; mB += dB / (t + 1); vB += dB * (eb - mB);
    }
    printf("integral_[0,1]^2 e^(x+y) = (e-1)^2 = %.6f,  N = %d points:\n", exact, n);
    printf("  white noise variance %.4e\n", vW / (trials - 1));
    printf("  blue  noise variance %.4e\n", vB / (trials - 1));
    printf("  variance reduction: %.1fx\n", (vW / (trials - 1)) / (vB / (trials - 1)));

    printf("\nWrote images/bluenoise_white.ppm and bluenoise_blue.ppm.\n"
           "Blue noise keeps points apart (bigger nearest-neighbor distance, visibly\n"
           "no clumps), which also lowers integration variance like stratification.\n"
           "Its real superpower, though, is the SPECTRUM: pushing error into high\n"
           "frequencies the eye discounts -- why renderers use blue-noise sample sets\n"
           "and dither patterns for perceptually cleaner images at low sample counts.\n");
    return 0;
}
