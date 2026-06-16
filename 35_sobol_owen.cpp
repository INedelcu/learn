// 35 — Sobol sequences and Owen scrambling (randomized quasi-Monte Carlo)
// =======================================================================
//
// Example 11 introduced low-discrepancy sampling with the Halton sequence. The
// SOBOL sequence is the other workhorse, and the one most production renderers
// reach for: it is built from binary "generator matrices" so each new sample is a
// few XORs, it stratifies beautifully in 2D (the first two dimensions form a
// so-called (0,2)-sequence), and it scales to high dimensions better than Halton.
//
// But raw Sobol has two drawbacks: it is DETERMINISTIC (so you cannot get an
// error estimate by re-running it), and stacking many dimensions eventually shows
// correlation artifacts. OWEN SCRAMBLING fixes both. It randomly permutes the
// digits of each sample in a nested ("tree") way that PRESERVES the stratification
// (so you keep the ~1/N quasi-Monte Carlo convergence) while making the whole set
// a random object. Averaging several independent Owen scrambles is RANDOMIZED QMC:
// unbiased, with honest error bars AND near-1/N convergence. We use the cheap
// hash-based Owen scramble of Laine-Karras / Burley (2020).
//
// We integrate the smooth 2D function from example 11, integral_[0,1]^2 e^(x+y) =
// (e-1)^2, and compare:
//   * random         -- RMSE over trials, the usual O(1/sqrt(N));
//   * Sobol (raw)    -- a single deterministic error, ~O(1/N);
//   * Owen-Sobol     -- RMSE over many scrambles: ~O(1/N) AND unbiased.

#include <cstdio>
#include <cmath>
#include <cstdint>
#include "mc_random.h"

static uint32_t ReverseBits(uint32_t n) {
    n = (n << 16) | (n >> 16);
    n = ((n & 0x00ff00ffu) << 8) | ((n & 0xff00ff00u) >> 8);
    n = ((n & 0x0f0f0f0fu) << 4) | ((n & 0xf0f0f0f0u) >> 4);
    n = ((n & 0x33333333u) << 2) | ((n & 0xccccccccu) >> 2);
    n = ((n & 0x55555555u) << 1) | ((n & 0xaaaaaaaau) >> 1);
    return n;
}
// Sobol dimension 0 is the radical inverse base 2 (a bit reversal of the index).
static uint32_t Sobol0(uint32_t i) { return ReverseBits(i); }
// Sobol dimension 1 via its direction numbers v_k = v_{k-1} ^ (v_{k-1} >> 1).
static uint32_t Sobol1(uint32_t i) {
    uint32_t r = 0;
    for (uint32_t v = 1u << 31; i; i >>= 1, v ^= v >> 1)
        if (i & 1u) r ^= v;
    return r;
}
// Hash-based nested-uniform (Owen) scramble of a Sobol integer (Laine-Karras).
static uint32_t OwenScramble(uint32_t x, uint32_t seed) {
    x = ReverseBits(x);
    x ^= x * 0x3d20adeau;
    x += seed;
    x *= (seed >> 16) | 1u;
    x ^= x * 0x05526c56u;
    x ^= x * 0x53a22864u;
    return ReverseBits(x);
}
static uint32_t HashSeed(uint32_t a, uint32_t b) {
    uint32_t h = a * 747796405u + b * 2891336453u;
    h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; return h;
}
static float ToFloat(uint32_t u) { return u * 0x1p-32f; }

int main() {
    Rng rng(/*sequence=*/35);
    const double kE = 2.71828182845904523536;
    const double exact = (kE - 1.0) * (kE - 1.0);
    auto f = [](double x, double y) { return std::exp(x + y); };

    printf("First 8 2D Sobol points (note the even spread):\n");
    for (uint32_t i = 0; i < 8; ++i)
        printf("   (%.4f, %.4f)\n", ToFloat(Sobol0(i)), ToFloat(Sobol1(i)));

    printf("\nintegral_[0,1]^2 e^(x+y) = (e-1)^2 = %.7f\n\n", exact);
    printf("%8s %14s %14s %16s\n", "N", "random RMSE", "Sobol err", "Owen-Sobol RMSE");

    const int kTrials = 64;
    for (uint32_t N = 64; N <= 16384; N *= 4) {
        // random: RMS error over independent runs.
        double sr = 0;
        for (int t = 0; t < kTrials; ++t) {
            double s = 0;
            for (uint32_t i = 0; i < N; ++i) s += f(rng.NextFloat(), rng.NextFloat());
            double e = s / N - exact; sr += e * e;
        }
        double randRMSE = std::sqrt(sr / kTrials);

        // raw Sobol: one deterministic estimate.
        double s = 0;
        for (uint32_t i = 0; i < N; ++i) s += f(ToFloat(Sobol0(i)), ToFloat(Sobol1(i)));
        double sobolErr = std::fabs(s / N - exact);

        // Owen-scrambled Sobol: RMS error over independent scrambles (RQMC),
        // and the mean over scrambles (should be unbiased -> exact).
        double so = 0, meanAcc = 0;
        for (int t = 0; t < kTrials; ++t) {
            uint32_t s0 = HashSeed(t, 0), s1 = HashSeed(t, 1);
            double ss = 0;
            for (uint32_t i = 0; i < N; ++i)
                ss += f(ToFloat(OwenScramble(Sobol0(i), s0)), ToFloat(OwenScramble(Sobol1(i), s1)));
            double est = ss / N; double e = est - exact; so += e * e; meanAcc += est;
        }
        double owenRMSE = std::sqrt(so / kTrials);
        (void)meanAcc;

        printf("%8u %14.3e %14.3e %16.3e\n", N, randRMSE, sobolErr, owenRMSE);
    }

    printf("\nRandom error falls as 1/sqrt(N); both Sobol and Owen-scrambled Sobol\n"
           "fall far faster (~1/N) on this smooth integrand. The Owen version keeps\n"
           "that fast convergence while being randomized -- so unlike raw Sobol it is\n"
           "unbiased and you can estimate its error by averaging scrambles. This\n"
           "randomized-QMC combination is the modern default sampler in renderers.\n");
    return 0;
}
