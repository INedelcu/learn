// 04 — Variance of the estimator, the Central Limit Theorem, and error bars
// =========================================================================
//
// Example 03 produced a single number per run. But the estimator is itself a
// random variable: run it again with fresh samples and you get a slightly
// different answer. This file studies that randomness directly, because
// understanding it is what lets a renderer say "this pixel has converged" or
// draw a noise threshold.
//
// Setup: we integrate f(x) = e^x on [0,1] (exact value I = e - 1).
//
// KEY FACTS about the estimator F_N = (1/N) sum_i Y_i, where Y_i = f(X_i)/p(X_i):
//
//   * E[F_N] = I                          (it is unbiased — correct on average)
//   * Var[F_N] = Var[Y] / N               (variance falls as 1/N ...)
//   * StdDev[F_N] = sigma_Y / sqrt(N)     (... so the ERROR falls as 1/sqrt(N))
//
// The quantity sigma_Y / sqrt(N) is the STANDARD ERROR. We don't know sigma_Y
// (it depends on the unknown integral), so we ESTIMATE it from the same samples
// using the sample variance. That gives us an error bar essentially for free.
//
// THE CENTRAL LIMIT THEOREM then tells us the shape of the error: for large N,
// F_N is approximately Normal(I, sigma_Y^2/N). A Normal puts 95% of its mass
// within +/-1.96 standard deviations of the mean. So
//
//   [ F_N - 1.96*SE ,  F_N + 1.96*SE ]
//
// is an approximate 95% CONFIDENCE INTERVAL: over many independent runs, it
// should contain the true I about 95% of the time. We test exactly that claim
// at the end by running 100,000 independent estimates and counting how often the
// interval really covers e - 1. (Spoiler: very close to 95%.)

#include <cstdio>
#include <cmath>
#include "mc_random.h"

int main() {
    Rng rng(/*sequence=*/4);
    const double kE = 2.71828182845904523536;
    const double I  = kE - 1.0;          // the true integral
    auto f = [](double x) { return std::exp(x); };

    // --- Part 1: one estimate per N, with a self-computed error bar. ---------
    // We accumulate mean and variance online (Welford), then report the
    // standard error and the 95% confidence interval, and confirm the true
    // value lands inside it.
    printf("integral_0^1 e^x dx = %.8f.  Estimate +/- standard error:\n\n", I);
    printf("%12s %14s %14s %26s\n", "samples", "estimate", "std error", "95%% CI (1.96*SE)");

    double mean = 0.0, m2 = 0.0;
    long long n = 0, nextReport = 100;
    for (long long i = 0; i < 10000000; ++i) {
        double y = f(rng.NextFloat());       // Y = f(X)/p(X), p = 1 on [0,1]
        ++n;
        double delta = y - mean;
        mean += delta / n;
        m2 += delta * (y - mean);
        if (n == nextReport) {
            double var = m2 / (n - 1);                 // sample variance of Y
            double se = std::sqrt(var / n);            // standard error of the mean
            printf("%12lld %14.7f %14.7f   [%9.6f, %9.6f]\n",
                   n, mean, se, mean - 1.96 * se, mean + 1.96 * se);
            nextReport *= 10;
        }
    }

    // --- Part 2: empirically verify the 1/N variance law and CLT coverage. ---
    // Run M independent estimators, each using K samples. Then:
    //   * the spread (sample std-dev) of the M estimates should match the
    //     predicted standard error sigma_Y / sqrt(K), and
    //   * the +/-1.96*SE interval each one builds should cover the truth ~95%.
    printf("\nRunning many independent estimators to test the theory.\n");
    const int kPerEstimate = 1000;
    const int kNumEstimates = 100000;

    double gMean = 0.0, gM2 = 0.0;      // statistics OF the estimates themselves
    int covered = 0;
    for (int e = 0; e < kNumEstimates; ++e) {
        // One estimator over kPerEstimate samples (online mean + variance).
        double m = 0.0, s2 = 0.0;
        for (int i = 0; i < kPerEstimate; ++i) {
            double y = f(rng.NextFloat());
            double d = y - m;
            m += d / (i + 1);
            s2 += d * (y - m);
        }
        double est = m;
        double se = std::sqrt((s2 / (kPerEstimate - 1)) / kPerEstimate);
        if (std::fabs(est - I) <= 1.96 * se) ++covered;

        // Fold this estimate into the population of estimates.
        double d = est - gMean;
        gMean += d / (e + 1);
        gM2 += d * (est - gMean);
    }
    double spreadOfEstimates = std::sqrt(gM2 / (kNumEstimates - 1));

    // Predicted standard error: we need Var[Y]. Var[Y] = E[Y^2] - I^2, and for
    // f = e^x on [0,1], E[Y^2] = integral e^{2x} = (e^2 - 1)/2. So:
    double EY2 = (kE * kE - 1.0) / 2.0;
    double varY = EY2 - I * I;
    double predictedSE = std::sqrt(varY / kPerEstimate);

    printf("\n  %d estimators, %d samples each:\n", kNumEstimates, kPerEstimate);
    printf("    mean of the estimates ........ %.7f  (true I = %.7f)\n", gMean, I);
    printf("    spread of the estimates ...... %.7f  (predicted SE = %.7f)\n",
           spreadOfEstimates, predictedSE);
    printf("    95%% CIs that covered the truth %.2f%%  (CLT predicts ~95%%)\n",
           100.0 * covered / kNumEstimates);

    printf("\nThe spread matches sigma_Y/sqrt(K) and the coverage is ~95%%: the\n"
           "estimator behaves like a Normal centered on the true answer, which is\n"
           "what lets you attach honest error bars to a Monte Carlo result.\n");
    return 0;
}
