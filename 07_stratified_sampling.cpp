// 07 — Stratified sampling: stop the samples from clumping
// ========================================================
//
// Plain Monte Carlo scatters samples independently, so by pure chance they
// clump in some places and leave gaps in others. Those gaps and clumps ARE the
// variance. STRATIFICATION removes a lot of it with almost no extra work: split
// the domain into N equal strata (cells) and place exactly one sample, jittered
// at random, inside each cell. The samples can no longer all pile into one region.
//
//   plain:        x_i = U_i                          (U_i ~ Uniform[0,1])
//   stratified:   x_i = (i + U_i) / N,  i = 0..N-1   (one jittered point per cell)
//
// Why it helps: the variance of a stratified estimate is the SUM of the
// within-stratum variances. Because each stratum spans only 1/N of the domain,
// the function varies little across it, so each within-stratum variance is tiny.
// For a smooth 1-D integrand the error can fall faster than the usual 1/sqrt(N).
// (The catch — the "curse of dimensionality" — is that a full grid needs N^d
// cells in d dimensions, which is why high-dimensional rendering uses smarter
// low-discrepancy or per-dimension stratified patterns instead of a full grid.)
//
// This is exactly the jitter a path tracer applies to pixel sample positions
// (and to lens/time/light samples): one well-spread sample per sub-cell beats N
// independent darts. We measure the payoff by running many independent estimates
// of integral_0^1 e^x dx with each method and comparing the spread (variance) of
// the resulting estimates.

#include <cstdio>
#include <cmath>
#include "mc_random.h"

int main() {
    Rng rng(/*sequence=*/7);
    const double kE = 2.71828182845904523536;
    const double I  = kE - 1.0;
    auto f = [](double x) { return std::exp(x); };

    const int kSamples = 64;          // samples per estimate (= number of strata)
    const int kTrials  = 200000;      // independent estimates, to measure spread

    double meanPlain = 0.0, m2Plain = 0.0;
    double meanStrat = 0.0, m2Strat = 0.0;

    for (int t = 0; t < kTrials; ++t) {
        // --- plain Monte Carlo: kSamples independent uniform draws ----------
        double sp = 0.0;
        for (int i = 0; i < kSamples; ++i) sp += f(rng.NextFloat());
        double estPlain = sp / kSamples;

        // --- stratified: one jittered sample inside each of kSamples cells --
        double ss = 0.0;
        for (int i = 0; i < kSamples; ++i) {
            double x = (i + rng.NextFloat()) / kSamples;   // jittered cell center
            ss += f(x);
        }
        double estStrat = ss / kSamples;

        // accumulate the spread of each method's estimates (Welford)
        double dp = estPlain - meanPlain; meanPlain += dp / (t + 1); m2Plain += dp * (estPlain - meanPlain);
        double ds = estStrat - meanStrat; meanStrat += ds / (t + 1); m2Strat += ds * (estStrat - meanStrat);
    }

    double varPlain = m2Plain / (kTrials - 1);
    double varStrat = m2Strat / (kTrials - 1);

    printf("integral_0^1 e^x dx = %.7f,  using %d samples per estimate.\n\n", I, kSamples);
    printf("%-16s %14s %18s %16s\n", "method", "mean estimate", "variance of est.", "std deviation");
    printf("%-16s %14.7f %18.3e %16.3e\n", "plain MC", meanPlain, varPlain, std::sqrt(varPlain));
    printf("%-16s %14.7f %18.3e %16.3e\n", "stratified", meanStrat, varStrat, std::sqrt(varStrat));

    printf("\nvariance reduction factor: %.1fx\n", varPlain / varStrat);
    printf("\nBoth are unbiased (means agree with %.5f), but stratification slashes\n"
           "the variance by spreading the %d samples evenly instead of letting them\n"
           "clump. In a renderer this is the difference between jittered pixel\n"
           "samples and naive random ones: same cost, visibly less noise.\n",
           I, kSamples);
    return 0;
}
