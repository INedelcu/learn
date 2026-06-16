// 11 — Low-discrepancy sequences and quasi-Monte Carlo
// ====================================================
//
// Plain random sampling (everything before this file) has a problem we already
// saw in example 07: independent points CLUMP. Clumps and gaps are wasted
// information, and they are exactly the 1/sqrt(N) noise. Stratification fought
// this by forcing one sample per grid cell. QUASI-MONTE CARLO goes further: it
// replaces randomness with a deterministic sequence engineered to fill space as
// evenly as possible at EVERY prefix length — a "low-discrepancy" sequence.
//
// DISCREPANCY measures how far a point set's local density strays from perfectly
// uniform. Random points have discrepancy ~ 1/sqrt(N); the sequences below have
// discrepancy ~ (log N)^d / N, which is almost 1/N. The Koksma-Hlawka inequality
// says the integration error is bounded by (variation of f) * (discrepancy), so
// for smooth integrands QMC error falls nearly as 1/N instead of 1/sqrt(N) — a
// huge win.
//
// THE VAN DER CORPUT SEQUENCE (1 dimension): write the index i in base b, then
// mirror its digits across the "decimal" point. This "radical inverse" spreads
// successive points maximally: each new point lands in the largest remaining gap.
//   base 2:  0, 1/2, 1/4, 3/4, 1/8, 5/8, 3/8, 7/8, ...
//
// THE HALTON SEQUENCE (d dimensions): use the van der Corput sequence with a
// DIFFERENT prime base per dimension (2, 3, 5, 7, ...). Coprime bases keep the
// dimensions from lining up into visible lattices. This is the sampler pbrt's
// HaltonSampler is built on, and the same well-distributed-sample principle is
// why renderers prefer stratified / low-discrepancy patterns over naive random
// pixel samples.
//
// CAVEAT (why renderers don't *only* use QMC): in high dimensions the low bases
// run out and pairs of dimensions develop correlation artifacts, so practical
// samplers scramble/randomize the sequence ("randomized QMC") to recover an
// unbiased error estimate while keeping the fast convergence. We keep it
// unscrambled here to show the raw idea.
//
// EXPERIMENT: integrate f(x,y) = e^(x+y) over the unit square. The exact value
// factors as (integral_0^1 e^x dx)^2 = (e - 1)^2. We compare, at each N:
//   * random sampling — reported as RMS error over many independent trials, and
//   * the Halton sequence — a single deterministic error.
// The trailing columns (error * sqrt(N) for random, error * N for Halton) should
// hover around a constant, exposing the two different convergence ORDERS.

#include <cstdio>
#include <cmath>
#include <cstdint>
#include "mc_random.h"

// Radical inverse of 'a' in the given base: mirror a's base-b digits about the
// radix point. Accumulating result += digit * b^-(k+1) avoids any overflow.
static double RadicalInverse(unsigned base, uint64_t a) {
    const double invBase = 1.0 / base;
    double invBaseN = 1.0, result = 0.0;
    while (a) {
        uint64_t digit = a % base;
        invBaseN *= invBase;
        result += digit * invBaseN;
        a /= base;
    }
    return result;
}

int main() {
    const double kE = 2.71828182845904523536;
    const double exact = (kE - 1.0) * (kE - 1.0);
    auto f = [](double x, double y) { return std::exp(x + y); };

    // --- Show the raw sequences, so the "fill the gaps" behavior is visible. -
    printf("van der Corput, base 2 (first 8):  ");
    for (int i = 0; i < 8; ++i) printf("%.4f ", RadicalInverse(2, i));
    printf("\n   note how each point bisects the largest empty gap.\n\n");
    printf("Halton (bases 2 and 3), first 8 2D points:\n");
    for (int i = 0; i < 8; ++i)
        printf("   (%.4f, %.4f)\n", RadicalInverse(2, i), RadicalInverse(3, i));

    // --- Convergence comparison. --------------------------------------------
    printf("\nintegral_[0,1]^2 e^(x+y) dx dy = (e-1)^2 = %.7f\n\n", exact);
    printf("%9s %14s %14s | %14s %14s\n",
           "N", "random RMSE", "Halton err", "rndRMSE*sqrtN", "HaltonErr*N");

    Rng rng(/*sequence=*/11);
    const int kTrials = 64;        // independent random runs, to measure RMS error
    for (long long N = 16; N <= 65536; N *= 4) {
        // Random: root-mean-square error across kTrials independent estimates.
        double sumSq = 0.0;
        for (int t = 0; t < kTrials; ++t) {
            double s = 0.0;
            for (long long i = 0; i < N; ++i)
                s += f(rng.NextFloat(), rng.NextFloat());
            double err = s / N - exact;
            sumSq += err * err;
        }
        double randRMSE = std::sqrt(sumSq / kTrials);

        // Halton: one deterministic estimate from N low-discrepancy points.
        double s = 0.0;
        for (long long i = 0; i < N; ++i)
            s += f(RadicalInverse(2, i), RadicalInverse(3, i));
        double haltonErr = std::fabs(s / N - exact);

        printf("%9lld %14.3e %14.3e | %14.4f %14.4f\n",
               N, randRMSE, haltonErr,
               randRMSE * std::sqrt((double)N), haltonErr * N);
    }

    printf("\nThe 'random RMSE * sqrt(N)' column stays roughly constant — random\n"
           "sampling is stuck at O(1/sqrt(N)). The 'Halton err * N' column also\n"
           "stays bounded, meaning Halton error falls like O(1/N) (up to log\n"
           "factors): nearly the SQUARE of the accuracy for the same N. That is the\n"
           "payoff of well-distributed samples, and why low-discrepancy/stratified\n"
           "patterns are the default in production path tracers.\n");
    return 0;
}
