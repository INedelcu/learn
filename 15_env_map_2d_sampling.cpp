// 15 — Importance-sampling a 2D environment map (marginal + conditional CDFs)
// ===========================================================================
//
// Example 14 sampled a 1D table. A real environment map is 2D: a grid of texels
// whose luminance we want to importance-sample, so that bounce/shadow rays aimed
// at the sky land on the bright sun far more often than on the dim sky. The
// renderer in C:\Work does exactly this with its environment cubemap. The
// standard tool is pbrt's "Distribution2D", built from a MARGINAL and a set of
// CONDITIONAL 1D distributions (the pattern shown in the HDRP CDF-theory demo):
//
//   joint p(u,v)  ==  p(v) * p(u | v)
//
//   p(v)     = sum over u of p(u,v)         -- the MARGINAL over rows
//   p(u | v) = p(u,v) / p(v)                -- the CONDITIONAL within row v
//
// To draw a 2D sample: first pick a row v from the marginal CDF, then pick a
// column u from THAT row's conditional CDF. Because the row was chosen with
// probability p(v) and the column with p(u|v), the pair lands with probability
// p(v)*p(u|v) = p(u,v) — the joint we wanted. This factor-and-chain trick is how
// you sample any 2D (or N-D) tabulated density.
//
// This file:
//   (1) builds a Distribution2D over a small procedural "sky" (a bright sun, a
//       softer glow, a dim sky), and checks the empirical 2D histogram matches
//       the joint pdf;
//   (2) compares estimating an integral over the map with UNIFORM vs map-
//       proportional (importance) sampling — variance drops a lot because the
//       map's dynamic range (sun vs sky) no longer lands in the estimator; and
//   (3) writes two PPM images into .\images\ — the env map with importance
//       samples splatted on it (they pile onto the sun) vs uniform samples
//       (spread evenly). Run view_images.ps1 to see them. This is the picture of
//       "why importance sampling helps": samples follow the energy.

#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <filesystem>
#include "mc_random.h"

// ---- 1D tabulated-distribution helpers (same idea as example 14) ------------
static std::vector<float> BuildCdf(const std::vector<float>& pdf) {
    std::vector<float> cdf(pdf.size() + 1, 0.0f);
    for (size_t i = 0; i < pdf.size(); ++i) cdf[i + 1] = cdf[i] + pdf[i];
    cdf.back() = 1.0f;
    return cdf;
}
static int FindBin(const std::vector<float>& cdf, float u) {
    int lo = 0, hi = (int)cdf.size() - 1;
    while (hi - lo > 1) { int m = (lo + hi) >> 1; if (cdf[m] <= u) lo = m; else hi = m; }
    return lo;
}
// Continuous sample in [0,N): bin index + fractional offset inside it.
static float SampleContinuous(const std::vector<float>& cdf, float u) {
    int i = FindBin(cdf, u);
    float f0 = cdf[i], f1 = cdf[i + 1];
    return i + ((f1 > f0) ? (u - f0) / (f1 - f0) : 0.0f);
}

static const int MW = 64, MH = 32;     // env map resolution (lon x lat)

// A procedural "environment": dim sky gradient + a bright sun + a soft glow.
static float EnvLum(int u, int v) {
    float L = 0.35f + 0.25f * (1.0f - (float)v / MH);   // brighter near horizon
    float du, dv;
    du = u - 46.0f; dv = v - 9.0f;
    if (du * du + dv * dv < 2.4f * 2.4f) L += 200.0f;   // the sun
    du = u - 16.0f; dv = v - 20.0f;
    if (du * du + dv * dv < 4.0f * 4.0f) L += 12.0f;    // a softer glow
    return L;
}

// ---- minimal RGB PPM writer -------------------------------------------------
static void WritePPM(const std::string& path, const std::vector<unsigned char>& rgb,
                     int w, int h) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { std::perror(path.c_str()); return; }
    std::fprintf(f, "P6\n%d %d\n255\n", w, h);
    std::fwrite(rgb.data(), 1, rgb.size(), f);
    std::fclose(f);
}

int main() {
    Rng rng(/*sequence=*/15);
    std::filesystem::create_directories("images");

    // --- Build the joint pdf and the Distribution2D. ------------------------
    std::vector<float> f(MW * MH);
    double total = 0.0;
    for (int v = 0; v < MH; ++v)
        for (int u = 0; u < MW; ++u) { float L = EnvLum(u, v); f[v * MW + u] = L; total += L; }

    std::vector<float> pdf2d(MW * MH);
    for (int i = 0; i < MW * MH; ++i) pdf2d[i] = (float)(f[i] / total);   // normalized joint

    std::vector<float> pV(MH);                              // marginal p(v)
    for (int v = 0; v < MH; ++v) {
        float s = 0.0f; for (int u = 0; u < MW; ++u) s += pdf2d[v * MW + u]; pV[v] = s;
    }
    std::vector<float> cdfV = BuildCdf(pV);

    std::vector<std::vector<float>> cdfU(MH);               // conditional p(u|v) per row
    for (int v = 0; v < MH; ++v) {
        std::vector<float> row(MW);
        for (int u = 0; u < MW; ++u) row[u] = pdf2d[v * MW + u] / pV[v];
        cdfU[v] = BuildCdf(row);
    }

    auto sample2D = [&](int& ui, int& vi) {
        float vc = SampleContinuous(cdfV, rng.NextFloat()); vi = (int)vc; if (vi >= MH) vi = MH - 1;
        float uc = SampleContinuous(cdfU[vi], rng.NextFloat()); ui = (int)uc; if (ui >= MW) ui = MW - 1;
    };

    // --- (1) verify the 2D histogram matches the joint pdf. -----------------
    const long long kHist = 4000000;
    std::vector<long long> hist(MW * MH, 0);
    for (long long s = 0; s < kHist; ++s) { int u, v; sample2D(u, v); hist[v * MW + u]++; }
    double maxErr = 0.0; int sunU = 46, sunV = 9;
    for (int i = 0; i < MW * MH; ++i)
        maxErr = std::fmax(maxErr, std::fabs((double)hist[i] / kHist - pdf2d[i]));
    printf("Distribution2D over a %dx%d env map (sun at u=%d,v=%d).\n\n", MW, MH, sunU, sunV);
    printf("2D histogram vs joint pdf:  max abs error over all texels = %.5f\n", maxErr);
    printf("  sun texel   prob: pdf %.5f  empirical %.5f\n",
           pdf2d[sunV * MW + sunU], (double)hist[sunV * MW + sunU] / kHist);
    printf("  a sky texel prob: pdf %.5f  empirical %.5f\n",
           pdf2d[0], (double)hist[0] / kHist);

    // --- (2) integral over the map: uniform vs importance sampling. ---------
    // Estimate I = sum over texels of f * g, with g a smooth factor in [0,1].
    auto g = [](int u, int /*v*/) { return 0.5f + 0.5f * std::sin(6.2831853f * u / MW); };
    double exactI = 0.0;
    for (int v = 0; v < MH; ++v) for (int u = 0; u < MW; ++u) exactI += f[v * MW + u] * g(u, v);

    const long long kI = 4000000;
    const int Nbins = MW * MH;
    double mU = 0, s2U = 0, mI = 0, s2I = 0;
    for (long long s = 0; s < kI; ++s) {
        // uniform: pick any texel with prob 1/Nbins
        int u = (int)(rng.NextFloat() * MW), v = (int)(rng.NextFloat() * MH);
        if (u >= MW) u = MW - 1; if (v >= MH) v = MH - 1;
        double eU = (double)Nbins * f[v * MW + u] * g(u, v);
        double d = eU - mU; mU += d / (s + 1); s2U += d * (eU - mU);
        // importance: pick texel with prob pdf2d; estimator = f*g/pdf = g*total
        int iu, iv; sample2D(iu, iv);
        double eI = (double)total * g(iu, iv);
        double d2 = eI - mI; mI += d2 / (s + 1); s2I += d2 * (eI - mI);
    }
    printf("\nIntegral of (envmap * smooth factor):  exact = %.2f\n", exactI);
    printf("  %-22s estimate = %.2f   variance = %.3e\n", "uniform texel sampling", mU, s2U / (kI - 1));
    printf("  %-22s estimate = %.2f   variance = %.3e\n", "env-importance sampling", mI, s2I / (kI - 1));
    printf("  variance reduction: %.0fx\n", (s2U / (kI - 1)) / (s2I / (kI - 1)));

    // --- (3) render the two visualizations. ---------------------------------
    const int scale = 9, IW = MW * scale, IH = MH * scale;
    float Lmax = 0.0f; for (float L : f) Lmax = std::fmax(Lmax, L);

    auto makeBackground = [&]() {
        std::vector<unsigned char> img(IW * IH * 3);
        for (int y = 0; y < IH; ++y)
            for (int x = 0; x < IW; ++x) {
                float L = f[(y / scale) * MW + (x / scale)];
                float t = std::pow(L / Lmax, 1.0f / 2.4f);          // tone-map for visibility
                unsigned char gv = (unsigned char)(t * 255.0f + 0.5f);
                int o = (y * IW + x) * 3; img[o] = img[o + 1] = img[o + 2] = gv;
            }
        return img;
    };
    auto splat = [&](std::vector<unsigned char>& img, float uc, float vc) {
        int cx = (int)(uc * scale), cy = (int)(vc * scale);          // sample -> pixel
        for (int dy = 0; dy < 2; ++dy) for (int dx = 0; dx < 2; ++dx) {
            int x = cx + dx, y = cy + dy;
            if (x < 0 || y < 0 || x >= IW || y >= IH) continue;
            int o = (y * IW + x) * 3; img[o] = 255; img[o + 1] = 40; img[o + 2] = 40;   // red
        }
    };

    const int kDots = 1500;
    std::vector<unsigned char> impImg = makeBackground(), uniImg = makeBackground();
    for (int s = 0; s < kDots; ++s) {
        float vc = SampleContinuous(cdfV, rng.NextFloat());
        int vi = (int)vc; if (vi >= MH) vi = MH - 1;
        float uc = SampleContinuous(cdfU[vi], rng.NextFloat());
        splat(impImg, uc, vc);
        splat(uniImg, rng.NextFloat() * MW, rng.NextFloat() * MH);
    }
    WritePPM("images/envmap_importance.ppm", impImg, IW, IH);
    WritePPM("images/envmap_uniform.ppm",   uniImg, IW, IH);

    printf("\nWrote images/envmap_importance.ppm and images/envmap_uniform.ppm.\n"
           "In the importance image the red samples pile onto the sun (where the\n"
           "energy is); in the uniform image they ignore it. That clustering IS the\n"
           "variance reduction in column (2), and it is what the renderer's\n"
           "environment-map sampler does so shadow/bounce rays find the sun.\n");
    return 0;
}
