// 41 — Comparing "random" number sources: PRNGs vs low-discrepancy sequences
// ==========================================================================
//
// A natural question after examples 35 (Owen-scrambled Sobol) and 36 (the Wang
// hash) is: "which is the better random number generator?" The honest answer is
// that they are DIFFERENT CATEGORIES OF THING, and seeing why is the whole point
// of this file.
//
//   * A PRNG -- the Wang hash here, or the PCG in mc_random.h -- produces WHITE
//     NOISE: draws that are independent and uniform, with no structure linking
//     one to the next. That independence is what you need for unbiased estimators
//     and for honest stochastic decisions (Russian roulette, reflect-vs-refract).
//
//   * A LOW-DISCREPANCY SEQUENCE -- Owen-scrambled Sobol -- is deliberately NOT
//     independent: consecutive points repel each other to fill space evenly. That
//     even spread is what makes integrals converge FASTER. Owen scrambling adds
//     just enough randomization to keep it unbiased (example 35).
//
// We compare Wang, PCG and Owen-Sobol three ways, plus a deliberately BAD little
// LCG as a cautionary control:
//   (A) 2D point distribution -- scatter images + a grid chi-square. White noise
//       clumps, Sobol is even, the bad LCG collapses onto a lattice.
//   (B) Monte Carlo convergence -- the punchline: Wang and PCG (both good PRNGs)
//       converge at the SAME 1/sqrt(N); Owen-Sobol converges ~1/N. A better PRNG
//       does not help the rate; better PLACEMENT does.
//   (C) Speed -- ns per sample, so you can weigh it against (B) (efficiency, ex 31).
//
// Conclusion: it is not "PRNG vs Sobol, pick one". Use a hashed PRNG as the
// randomness source and a low-discrepancy sequence for sample placement -- which
// is exactly what production renderers (and this one's per-pixel hashed seeding)
// do together.

#include <cstdio>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <vector>
#include <string>
#include <filesystem>
#include "mc_random.h"

// ---- generators -------------------------------------------------------------
static uint32_t WangHash(uint32_t s) {
    s = (s ^ 61u) ^ (s >> 16); s *= 9u; s = s ^ (s >> 4);
    s *= 0x27d4eb2du; s = s ^ (s >> 15); return s;
}
static uint32_t ReverseBits(uint32_t n) {
    n = (n << 16) | (n >> 16);
    n = ((n & 0x00ff00ffu) << 8) | ((n & 0xff00ff00u) >> 8);
    n = ((n & 0x0f0f0f0fu) << 4) | ((n & 0xf0f0f0f0u) >> 4);
    n = ((n & 0x33333333u) << 2) | ((n & 0xccccccccu) >> 2);
    n = ((n & 0x55555555u) << 1) | ((n & 0xaaaaaaaau) >> 1);
    return n;
}
static uint32_t Sobol0(uint32_t i) { return ReverseBits(i); }
static uint32_t Sobol1(uint32_t i) {
    uint32_t r = 0; for (uint32_t v = 1u << 31; i; i >>= 1, v ^= v >> 1) if (i & 1u) r ^= v; return r;
}
static uint32_t OwenScramble(uint32_t x, uint32_t seed) {
    x = ReverseBits(x);
    x ^= x * 0x3d20adeau; x += seed; x *= (seed >> 16) | 1u;
    x ^= x * 0x05526c56u; x ^= x * 0x53a22864u;
    return ReverseBits(x);
}
static uint32_t HashSeed(uint32_t a, uint32_t b) {
    uint32_t h = a * 747796405u + b * 2891336453u; h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; return h;
}
static float F32(uint32_t u) { return u * 0x1p-32f; }

// A deliberately POOR LCG: full period over a tiny modulus, but a small
// multiplier, so consecutive pairs collapse onto a handful of lattice lines.
struct BadLCG {
    uint32_t s = 1;
    float next() { s = (5u * s + 1u) & 4095u; return s * (1.0f / 4096.0f); }
};

struct P2 { float x, y; };

static double Chi2(const std::vector<P2>& pts, int g) {
    std::vector<int> cnt(g * g, 0);
    for (const P2& p : pts) {
        int ix = (int)(p.x * g); if (ix >= g) ix = g - 1;
        int iy = (int)(p.y * g); if (iy >= g) iy = g - 1;
        cnt[iy * g + ix]++;
    }
    double e = (double)pts.size() / (g * g), s = 0;
    for (int c : cnt) s += (c - e) * (c - e) / e;
    return s;
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
    const double exact = (kE - 1.0) * (kE - 1.0);
    auto f = [](double x, double y) { return std::exp(x + y); };

    // --- (A) 2D point distribution: scatter images + grid chi-square. --------
    const int Ns = 1024, g = 8;
    Rng pcg(1);
    std::vector<P2> wang(Ns), sobol(Ns), bad(Ns);
    BadLCG lcg;
    uint32_t s0 = HashSeed(0, 0), s1 = HashSeed(0, 1);
    for (int i = 0; i < Ns; ++i) {
        // Two coordinates from a hash: hash the index, then hash THAT for y.
        // (Hashing adjacent counters 2i, 2i+1 would leave x,y correlated --
        // Wang doesn't fully avalanche a 1-bit input change.)
        uint32_t hw = WangHash((uint32_t)i);
        wang[i]  = {F32(hw), F32(WangHash(hw))};
        sobol[i] = {F32(OwenScramble(Sobol0(i), s0)), F32(OwenScramble(Sobol1(i), s1))};
        bad[i]   = {lcg.next(), lcg.next()};
    }
    DrawScatter("images/rng_wang.ppm",  wang);
    DrawScatter("images/rng_sobol.ppm", sobol);
    DrawScatter("images/rng_lcg.ppm",   bad);

    printf("(A) 2D uniformity of %d points, %dx%d grid chi-square\n", Ns, g, g);
    printf("    (random ~ %d; much lower = more even; much higher = structured):\n", g * g - 1);
    printf("    Wang hash (white noise) : %8.1f\n", Chi2(wang, g));
    printf("    Owen-Sobol (low-discr.) : %8.1f   <- near 0: perfectly stratified\n", Chi2(sobol, g));
    printf("    bad LCG (lattice)       : %8.1f   <- huge: points on a few lines\n", Chi2(bad, g));
    printf("    wrote images/rng_wang.ppm, rng_sobol.ppm, rng_lcg.ppm\n\n");

    // --- (B) Monte Carlo convergence: the rate is what matters. -------------
    printf("(B) integral_[0,1]^2 e^(x+y) = (e-1)^2 = %.6f.  RMS error over 64 runs:\n", exact);
    printf("    %8s | %12s %12s %14s | %12s %12s\n",
           "N", "Wang", "PCG", "Owen-Sobol", "Wang*sqrtN", "Owen*N");
    const int kTrials = 64;
    for (uint32_t N = 64; N <= 16384; N *= 4) {
        double seW = 0, seP = 0, seO = 0;
        for (int t = 0; t < kTrials; ++t) {
            uint32_t salt = HashSeed(t, 9);
            double sw = 0;
            for (uint32_t i = 0; i < N; ++i) {
                uint32_t hw = WangHash(salt + i);
                sw += f(F32(hw), F32(WangHash(hw)));        // y = hash of the hash
            }
            double ew = sw / N - exact; seW += ew * ew;

            Rng rng((uint64_t)t + 1);
            double sp = 0;
            for (uint32_t i = 0; i < N; ++i) sp += f(rng.NextFloat(), rng.NextFloat());
            double ep = sp / N - exact; seP += ep * ep;

            uint32_t a0 = HashSeed(t, 0), a1 = HashSeed(t, 1);
            double so = 0;
            for (uint32_t i = 0; i < N; ++i)
                so += f(F32(OwenScramble(Sobol0(i), a0)), F32(OwenScramble(Sobol1(i), a1)));
            double eo = so / N - exact; seO += eo * eo;
        }
        double rW = std::sqrt(seW / kTrials), rP = std::sqrt(seP / kTrials), rO = std::sqrt(seO / kTrials);
        printf("    %8u | %12.3e %12.3e %14.3e | %12.4f %12.4f\n",
               N, rW, rP, rO, rW * std::sqrt((double)N), rO * N);
    }

    // --- (C) speed: cost per sample. ----------------------------------------
    const long long M = 50000000;
    auto timeIt = [&](auto gen) {
        volatile float sink = 0; auto t0 = std::chrono::high_resolution_clock::now();
        for (long long i = 0; i < M; ++i) sink += gen((uint32_t)i);
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(t1 - t0).count() / M * 1e9;
    };
    Rng prng(123);
    double tW = timeIt([&](uint32_t i) { return F32(WangHash(i)); });
    double tP = timeIt([&](uint32_t)   { return prng.NextFloat(); });
    double tO = timeIt([&](uint32_t i) { return F32(OwenScramble(Sobol1(i), 12345u)); });
    printf("\n(C) cost per sample:  Wang %.2f ns   PCG %.2f ns   Owen-Sobol %.2f ns\n",
           tW, tP, tO);

    printf("\nThe scatter/chi-square (A) shows the categories: Wang is white noise\n"
           "(random clumps), Owen-Sobol is evenly stratified, the bad LCG is a\n"
           "lattice. Convergence (B) is the punchline: Wang and PCG -- two good but\n"
           "DIFFERENT PRNGs -- land on the same 1/sqrt(N) (their 'error*sqrtN' columns\n"
           "are flat), while Owen-Sobol's 'error*N' stays bounded, i.e. ~1/N. A\n"
           "better PRNG does not converge faster; better sample PLACEMENT does. They\n"
           "are not competitors: use a hashed PRNG (like the renderer's Wang-seeded\n"
           "per-pixel RNG) as the randomness source, and a low-discrepancy sequence\n"
           "for placement, together. (C) shows Sobol+Owen costs a bit more per\n"
           "sample, but (B) means it needs far fewer -- the efficiency tradeoff of\n"
           "example 31.\n");
    return 0;
}
