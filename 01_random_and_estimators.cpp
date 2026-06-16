// 01 — Random samples, expectation, and variance: the vocabulary
// ===============================================================
//
// Before we integrate anything, we need three ideas that the rest of the series
// leans on constantly:
//
//   * A RANDOM VARIABLE X is a number produced by a random process. Here X is a
//     uniform draw in [0, 1).
//
//   * The EXPECTATION (mean) E[X] is the average value X takes "in the long run".
//     For X ~ Uniform[0,1], E[X] = integral of x over [0,1] = 1/2.
//
//   * The VARIANCE Var[X] = E[(X - E[X])^2] measures how spread out X is around
//     its mean. For X ~ Uniform[0,1], Var[X] = 1/12 ≈ 0.0833.
//
// The whole point of Monte Carlo is that we usually CANNOT compute E[X] by hand,
// so we ESTIMATE it by averaging many samples. The Law of Large Numbers promises
// that the sample mean converges to the true mean as the number of samples grows.
// This file watches that convergence happen, and also estimates the variance.
//
// We use Welford's online algorithm to accumulate the mean and variance in a
// single pass — it is numerically stable (no catastrophic cancellation from
// "sum of squares minus square of sum") and it is exactly how you would track the
// running error of a render as samples stream in.

#include <cstdio>
#include "mc_random.h"

int main() {
    Rng rng(/*sequence=*/1);

    // --- A quick histogram, to *see* that the draws really are uniform. -------
    // We drop 100,000 samples into 10 equal-width bins. If the generator is
    // uniform, each bin should hold ~10% of the samples.
    const int kBins = 10;
    const int kHistSamples = 100000;
    int hist[kBins] = {0};
    for (int i = 0; i < kHistSamples; ++i) {
        float x = rng.NextFloat();
        hist[(int)(x * kBins)]++;
    }
    printf("Histogram of 100k Uniform[0,1) draws (each bar ~10%% if uniform):\n");
    for (int b = 0; b < kBins; ++b) {
        float frac = (float)hist[b] / kHistSamples;
        printf("  [%.1f,%.1f) %5.2f%%  ", b / 10.0f, (b + 1) / 10.0f, 100 * frac);
        for (int s = 0; s < (int)(frac * 200); ++s) putchar('#');
        putchar('\n');
    }

    // --- Watch the sample mean and variance converge. ------------------------
    // Welford's algorithm: keep a running mean and a running sum of squared
    // deviations M2. After n samples, the (unbiased, Bessel-corrected) variance
    // estimate is M2 / (n - 1).
    printf("\nConverging estimates (true mean = 0.5, true variance = 0.08333...):\n");
    printf("%12s %14s %14s\n", "samples", "mean", "variance");

    double mean = 0.0, m2 = 0.0;
    long long n = 0;
    long long nextReport = 10;
    const long long kTotal = 10000000;  // 10 million
    for (long long i = 0; i < kTotal; ++i) {
        double x = rng.NextFloat();
        ++n;
        double delta = x - mean;
        mean += delta / n;
        m2 += delta * (x - mean);   // uses the updated mean — this is the trick
        if (n == nextReport) {
            double variance = m2 / (n - 1);
            printf("%12lld %14.8f %14.8f\n", n, mean, variance);
            nextReport *= 10;
        }
    }

    printf("\nNotice the mean homing in on 0.5 and the variance on 0.08333.\n"
           "Each extra factor of 10 in samples buys ~1 more correct digit-ish:\n"
           "that sqrt(N) convergence is the subject of example 02 and 04.\n");
    return 0;
}
