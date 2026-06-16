// 12 — Sampling the normal distribution (the Box-Muller transform)
// ================================================================
//
// The normal (Gaussian) distribution is everywhere: it is the limiting shape of
// the Central Limit Theorem (example 04 showed our Monte Carlo estimator becomes
// Normal as N grows), and Gaussian kernels are the standard reconstruction/blur
// filters in rendering. So we should be able to SAMPLE one. But the Gaussian's
// CDF (the "error function") has no elementary inverse, so the inversion method
// from example 05 doesn't apply directly.
//
// THE BOX-MULLER TRANSFORM sidesteps that. It takes TWO independent uniforms and
// produces TWO independent standard normals at once:
//
//   given U1, U2 ~ Uniform(0,1],
//     R     = sqrt(-2 ln U1)        (radius;  R^2 is Exponential -> see ex. 05)
//     theta = 2*pi*U2               (angle, uniform)
//     Z0 = R cos(theta),   Z1 = R sin(theta)        are i.i.d. N(0,1).
//
// The trick is to think in 2D: a pair of independent standard normals has a
// rotationally symmetric density, so its squared radius is exponentially
// distributed (sample R via inverse-CDF of that exponential) and its angle is
// uniform. Box-Muller just samples those polar coordinates and converts back.
//
// To get a general N(mu, sigma^2), shift and scale:  X = mu + sigma * Z.
//
// We verify the sampler three ways: (a) histogram vs the analytic bell curve,
// (b) sample mean ~ 0 and variance ~ 1, and (c) the "68-95-99.7 rule" — the
// fraction of samples within 1, 2, 3 standard deviations of the mean.
//
// (Note: we draw U1 from (0,1] as 1 - NextFloat() so that ln(U1) is never ln(0).
//  A trig-free alternative is Marsaglia's polar method, which uses rejection
//  sampling, exactly the technique from example 05.)

#include <cstdio>
#include <cmath>
#include "mc_random.h"

int main() {
    Rng rng(/*sequence=*/12);
    const double kTwoPi = 6.28318530717958647692;
    const long long kPairs = 5000000;        // -> 10,000,000 normals

    // Histogram over [-4, 4] in 16 bins, plus running mean/variance and the
    // counts within 1/2/3 sigma.
    const int kBins = 16;
    const double lo = -4.0, hi = 4.0, w = (hi - lo) / kBins;
    long long hist[kBins] = {0};
    double mean = 0.0, m2 = 0.0;
    long long n = 0, in1 = 0, in2 = 0, in3 = 0;

    auto tally = [&](double z) {
        ++n;
        double d = z - mean; mean += d / n; m2 += d * (z - mean);
        double az = std::fabs(z);
        if (az < 1.0) ++in1;
        if (az < 2.0) ++in2;
        if (az < 3.0) ++in3;
        int b = (int)((z - lo) / w);
        if (b >= 0 && b < kBins) hist[b]++;
    };

    for (long long i = 0; i < kPairs; ++i) {
        double u1 = 1.0 - rng.NextFloat();   // in (0,1], so ln(u1) is finite
        double u2 = rng.NextFloat();
        double r = std::sqrt(-2.0 * std::log(u1));
        tally(r * std::cos(kTwoPi * u2));
        tally(r * std::sin(kTwoPi * u2));
    }

    // --- Histogram vs the standard normal pdf, phi(x) = exp(-x^2/2)/sqrt(2pi).
    const double invSqrt2pi = 0.39894228040143267794;
    printf("Box-Muller: %lld standard-normal samples vs the bell curve phi(x).\n\n", n);
    printf("  %-14s %10s %10s\n", "bin", "measured", "phi(mid)");
    for (int b = 0; b < kBins; ++b) {
        double x0 = lo + b * w, mid = x0 + 0.5 * w;
        double measured = (double)hist[b] / n / w;        // empirical density
        double phi = invSqrt2pi * std::exp(-0.5 * mid * mid);
        printf("  [%5.2f,%5.2f) %10.4f %10.4f  ", x0, x0 + w, measured, phi);
        for (int s = 0; s < (int)(measured * 90); ++s) putchar('#');
        putchar('\n');
    }

    // --- Moments and the empirical rule. ------------------------------------
    printf("\n  sample mean ....... % .5f   (theory 0)\n", mean);
    printf("  sample variance ... % .5f   (theory 1)\n", m2 / (n - 1));
    printf("\n  within 1 sigma .... %.3f%%   (theory 68.27%%)\n", 100.0 * in1 / n);
    printf("  within 2 sigma .... %.3f%%   (theory 95.45%%)\n", 100.0 * in2 / n);
    printf("  within 3 sigma .... %.3f%%   (theory 99.73%%)\n", 100.0 * in3 / n);

    // --- Shift-and-scale to a general normal N(mu, sigma^2). ----------------
    const double mu = 2.0, sigma = 1.5;
    double m = 0.0, s2 = 0.0; long long k = 0;
    for (long long i = 0; i < 2000000; ++i) {
        double u1 = 1.0 - rng.NextFloat(), u2 = rng.NextFloat();
        double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(kTwoPi * u2);
        double x = mu + sigma * z;                        // X ~ N(mu, sigma^2)
        ++k; double d = x - m; m += d / k; s2 += d * (x - m);
    }
    printf("\n  X = mu + sigma*Z with mu=%.1f, sigma=%.1f:\n", mu, sigma);
    printf("    sample mean = %.4f (theory %.1f),  sample std = %.4f (theory %.1f)\n",
           m, mu, std::sqrt(s2 / (k - 1)), sigma);

    printf("\nThe histogram traces phi(x), the moments are 0 and 1, and the\n"
           "68-95-99.7 fractions land on the textbook values: Box-Muller really is\n"
           "drawing from the normal distribution. This is the same bell curve the\n"
           "CLT (example 04) predicted for the Monte Carlo estimator itself.\n");
    return 0;
}
