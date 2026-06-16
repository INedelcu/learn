// 42 — High dimensions: where quasi-Monte Carlo (and scrambling) earn their keep
// ==============================================================================
//
// A companion to example 41. Path-tracing integrals are HIGH-DIMENSIONAL (pixel
// xy, lens, time, then two numbers per bounce -> easily 10-30 dimensions). This is
// where the differences between sampling strategies get dramatic, and also where
// the quasi-Monte Carlo (QMC) sequences of examples 11/35/41 hit their limits.
//
// THE HALTON TRAP IN HIGH DIMENSIONS. Halton uses the radical inverse in the d-th
// prime base for dimension d. Low dimensions (bases 2, 3, 5) are great, but high
// dimensions use large primes, and the radical inverse in a large base b marches
// monotonically (0, 1/b, 2/b, ...) for the first b samples. So two high dimensions
// stay almost perfectly correlated until you have drawn ~b samples -- their 2D
// projection is a few diagonal stripes, not a filled square. An integrand that
// depends on those dimensions jointly is then sampled badly.
//
// SCRAMBLING IS THE FIX, and it matters MOST exactly here. We apply random DIGIT
// SCRAMBLING: for each (dimension, digit position) we draw an independent random
// permutation of the base-b digits and apply it to the radical inverse. A genuine
// permutation (not just an additive shift) of the leading digit breaks the
// monotonic lockstep, so the high-dimension projections fill the square -- and the
// estimator becomes unbiased with error bars (randomized QMC). This is the same
// idea as the Owen scramble in example 35; production renderers use Owen-scrambled
// SOBOL, a better base sequence than Halton in high dimensions (it has no
// large-prime-base problem), randomized the same way.
//
// We show (A) a high-dimension Halton projection collapsing onto stripes and
// scrambling fixing it, and (B) integration error vs dimension for two integrands
// -- an additive one (low "effective dimension", QMC always wins) and a product
// one (high effective dimension, where raw Halton degrades and the QMC edge erodes
// toward plain random: the curse of dimensionality).

#include <cstdio>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <vector>
#include <string>
#include <filesystem>
#include "mc_random.h"

static const int kPrimes[32] = {
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
    59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131
};

static double RadicalInverse(int base, uint32_t i) {
    double invBase = 1.0 / base, invBaseN = 1.0, r = 0.0;
    while (i) { invBaseN *= invBase; r += (i % base) * invBaseN; i /= base; }
    return r;
}
static uint32_t Hash(uint32_t a, uint32_t b) {
    uint32_t h = a * 747796405u + b * 2891336453u; h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; return h;
}

// Random digit scrambling: an independent random permutation of the digits at each
// (dimension, position). The permutation of the leading digit is what breaks the
// high-dimensional correlation; deeper positions randomize the rest (unbiasedness).
struct Scrambler {
    std::vector<std::vector<std::vector<int>>> perm;   // [dim][pos][digit]
    void build(int D, uint32_t seed) {
        perm.assign(D, {});
        for (int d = 0; d < D; ++d) {
            int base = kPrimes[d];
            int K = (int)std::ceil(24.0 / (std::log((double)base) / std::log(2.0)));  // ~24-bit precision
            if (K < 1) K = 1;
            perm[d].resize(K);
            for (int pos = 0; pos < K; ++pos) {
                std::vector<int> p(base);
                for (int k = 0; k < base; ++k) p[k] = k;
                Rng r((uint64_t)Hash(seed, (uint32_t)(d * 257 + pos)) + 1u);   // Fisher-Yates
                for (int k = base - 1; k > 0; --k) { int j = r.NextUInt32() % (k + 1); std::swap(p[k], p[j]); }
                perm[d][pos] = std::move(p);
            }
        }
    }
    double inv(int d, uint32_t i) const {
        int base = kPrimes[d]; double invBase = 1.0 / base, invBaseN = 1.0, r = 0.0;
        for (size_t pos = 0; pos < perm[d].size(); ++pos) {
            int dig = perm[d][pos][i % base];
            invBaseN *= invBase; r += dig * invBaseN; i /= base;
        }
        return r;
    }
};

struct P2 { float x, y; };
static double Chi2(const std::vector<P2>& pts, int g) {
    std::vector<int> c(g * g, 0);
    for (const P2& p : pts) { int ix = (int)(p.x * g); if (ix >= g) ix = g - 1; int iy = (int)(p.y * g); if (iy >= g) iy = g - 1; c[iy * g + ix]++; }
    double e = (double)pts.size() / (g * g), s = 0; for (int v : c) s += (v - e) * (v - e) / e; return s;
}
static void DrawScatter(const std::string& path, const std::vector<P2>& pts) {
    const int S = 384; std::vector<unsigned char> img(S * S * 3, 255);
    for (const P2& pt : pts) {
        int cx = (int)(pt.x * S), cy = (int)(pt.y * S);
        for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
            int x = cx + dx, y = cy + dy; if (x < 0 || y < 0 || x >= S || y >= S) continue;
            int o = (y * S + x) * 3; img[o] = img[o + 1] = img[o + 2] = 0;
        }
    }
    FILE* f = std::fopen(path.c_str(), "wb"); if (!f) { std::perror(path.c_str()); return; }
    std::fprintf(f, "P6\n%d %d\n255\n", S, S); std::fwrite(img.data(), 1, img.size(), f); std::fclose(f);
}

int main() {
    std::filesystem::create_directories("images");
    const double kE = 2.71828182845904523536;

    // --- (A) a high-dimension Halton projection: dims 31 & 32 (bases 127,131). --
    const int dA = 30, dB = 31, Ns = 1024;
    Scrambler scr; scr.build(32, /*seed=*/12345u);
    std::vector<P2> hal(Ns), scram(Ns), rnd(Ns);
    Rng rng(7);
    for (int i = 0; i < Ns; ++i) {
        hal[i]   = {(float)RadicalInverse(kPrimes[dA], i), (float)RadicalInverse(kPrimes[dB], i)};
        scram[i] = {(float)scr.inv(dA, i), (float)scr.inv(dB, i)};
        rnd[i]   = {rng.NextFloat(), rng.NextFloat()};
    }
    DrawScatter("images/highdim_halton.ppm", hal);
    DrawScatter("images/highdim_scrambled.ppm", scram);
    DrawScatter("images/highdim_random.ppm", rnd);
    printf("(A) 2D projection of dimensions %d&%d (primes %d,%d), %d points, 8x8 chi-square:\n",
           dA + 1, dB + 1, kPrimes[dA], kPrimes[dB], Ns);
    printf("    raw Halton      : %8.1f   <- huge: collapses onto diagonal stripes\n", Chi2(hal, 8));
    printf("    scrambled Halton: %8.1f   <- ~random: stripes broken\n", Chi2(scram, 8));
    printf("    plain random    : %8.1f\n", Chi2(rnd, 8));
    printf("    wrote images/highdim_halton.ppm, highdim_scrambled.ppm, highdim_random.ppm\n\n");

    // --- (B) integration error vs dimension, two integrands. ----------------
    // f_add  = (1/d) sum exp(x_j)        -> exact e-1   (LOW effective dimension)
    // f_prod = prod |4 x_j - 2|          -> exact 1     (HIGH effective dimension)
    auto fAdd  = [&](const std::vector<double>& x) { double s = 0; for (double v : x) s += std::exp(v); return s / x.size(); };
    auto fProd = [&](const std::vector<double>& x) { double p = 1; for (double v : x) p *= std::fabs(4 * v - 2); return p; };
    const double exA = kE - 1.0, exP = 1.0;
    const int N = 4096, T = 32;

    for (int which = 0; which < 2; ++which) {
        double exact = which ? exP : exA;
        printf("(B%d) %s   exact = %.5f,  N=%d.  RMS error vs dimension:\n",
               which + 1, which ? "f = prod|4x-2| (high effective dim)" : "f = mean exp(x_j) (low effective dim)",
               exact, N);
        printf("     %4s | %12s %12s %14s\n", "d", "random", "raw Halton", "scrambled Halton");
        for (int d : {2, 4, 8, 16, 32}) {
            std::vector<double> x(d);
            // random RMSE
            double seR = 0;
            for (int t = 0; t < T; ++t) {
                Rng r((uint64_t)t + 1); double sum = 0;
                for (int i = 0; i < N; ++i) { for (int j = 0; j < d; ++j) x[j] = r.NextFloat(); sum += which ? fProd(x) : fAdd(x); }
                double e = sum / N - exact; seR += e * e;
            }
            // raw Halton (deterministic)
            double sumH = 0;
            for (int i = 0; i < N; ++i) { for (int j = 0; j < d; ++j) x[j] = RadicalInverse(kPrimes[j], i); sumH += which ? fProd(x) : fAdd(x); }
            double errH = std::fabs(sumH / N - exact);
            // scrambled Halton RMSE
            double seS = 0;
            for (int t = 0; t < T; ++t) {
                Scrambler s; s.build(d, Hash(t, 99u)); double sum = 0;
                for (int i = 0; i < N; ++i) { for (int j = 0; j < d; ++j) x[j] = s.inv(j, i); sum += which ? fProd(x) : fAdd(x); }
                double e = sum / N - exact; seS += e * e;
            }
            printf("     %4d | %12.3e %12.3e %14.3e\n", d, std::sqrt(seR / T), errH, std::sqrt(seS / T));
        }
        printf("\n");
    }

    printf("(A) shows the Halton trap: a high-dimension pair collapses onto diagonal\n"
           "stripes (huge chi-square), and digit scrambling breaks them. In (B1),\n"
           "scrambled Halton is best at every dimension, while RAW Halton starts\n"
           "strong but slips past even random by d~16 -- its large-prime-base\n"
           "dimensions are poorly distributed at this N; scrambling repairs them. In\n"
           "(B2) the product needs the JOINT distribution: raw Halton degrades\n"
           "catastrophically as d climbs (correlated high dims -> a wildly wrong\n"
           "estimate), scrambling wins most of it back, and every method drifts toward\n"
           "random's 1/sqrt(N) by d=32 -- the curse of dimensionality. Two lessons:\n"
           "scrambling matters most in high dimensions, and QMC's edge depends on the\n"
           "integrand's EFFECTIVE dimension. Production path tracers use Owen-scrambled\n"
           "Sobol -- a better high-dim base sequence than Halton, randomized by the\n"
           "very scrambling shown here.\n");
    return 0;
}
