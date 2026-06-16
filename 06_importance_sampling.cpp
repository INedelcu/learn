// 06 — Importance sampling: putting samples where they matter
// ===========================================================
//
// Recall the general Monte Carlo formula from example 03:
//
//   I = integral f(x) dx = E[ f(X)/p(X) ],   X ~ p,
//
// which is unbiased for ANY density p > 0 on the support of f. Example 03 used
// the uniform p; here we exploit the freedom to choose p, and that choice is the
// single biggest lever on Monte Carlo noise.
//
// THE IDEA: the estimator averages the ratio f(X)/p(X). If p is shaped like f,
// that ratio is nearly constant from sample to sample, so the average barely
// fluctuates — low variance. In the ideal (unreachable) case p(x) = f(x)/I, the
// ratio is exactly I every time and the variance is ZERO. We cannot use that p
// (it needs the answer I), but any p that roughly tracks f's shape helps a lot.
// Sampling proportionally to "where the integrand is big" is IMPORTANCE SAMPLING.
//
// This is precisely why the path tracer samples the GGX/cosine lobes instead of
// firing rays uniformly: the BRDF concentrates outgoing energy in particular
// directions, and we want our samples to land there. (See BRDF.hlsl —
// SampleSpecularGGX, SampleDiffuseLambert — in the renderer.)
//
// EXPERIMENT: integrate f(x) = e^x on [0,1] (exact = e - 1 ≈ 1.7182818) two ways:
//   (1) Uniform sampling:    p(x) = 1.
//   (2) Importance sampling: p(x) = (2/3)(1 + x), a straight line that leans the
//       same way e^x does on [0,1]. It is easy to sample by inversion:
//         CDF P(x) = (x + x^2/2)/(3/2);  solving P(x) = u gives
//         X = -1 + sqrt(1 + 3u).
//   We give both methods the same sample budget and compare the variance of the
//   per-sample estimator. Lower variance = less noise for the same cost.

#include <cstdio>
#include <cmath>
#include "mc_random.h"

int main() {
    Rng rng(/*sequence=*/6);
    const double kE = 2.71828182845904523536;
    const double I  = kE - 1.0;
    auto f = [](double x) { return std::exp(x); };

    // The importance density and its sampler.
    auto pImp = [](double x) { return (2.0 / 3.0) * (1.0 + x); };      // density
    auto sampleImp = [&rng]() { return -1.0 + std::sqrt(1.0 + 3.0 * rng.NextFloat()); };

    const long long kN = 20000000;

    // --- Method 1: uniform sampling. Estimator value per sample is f(X)/1. ---
    double m1 = 0.0, s1 = 0.0;
    for (long long i = 0; i < kN; ++i) {
        double x = rng.NextFloat();
        double est = f(x);                 // f(X)/p(X), p = 1
        double d = est - m1; m1 += d / (i + 1); s1 += d * (est - m1);
    }
    double var1 = s1 / (kN - 1);

    // --- Method 2: importance sampling. Estimator value is f(X)/pImp(X). -----
    double m2 = 0.0, s2 = 0.0;
    for (long long i = 0; i < kN; ++i) {
        double x = sampleImp();
        double est = f(x) / pImp(x);       // f(X)/p(X), p shaped like f
        double d = est - m2; m2 += d / (i + 1); s2 += d * (est - m2);
    }
    double var2 = s2 / (kN - 1);

    printf("integral_0^1 e^x dx = %.7f\n\n", I);
    printf("%-22s %14s %16s %16s\n", "method", "estimate", "per-sample var", "rel. std error");
    printf("%-22s %14.7f %16.6f %16.6f\n", "uniform  p(x)=1",
           m1, var1, std::sqrt(var1 / kN) / I);
    printf("%-22s %14.7f %16.6f %16.6f\n", "importance p~(1+x)",
           m2, var2, std::sqrt(var2 / kN) / I);

    printf("\nvariance reduction factor: %.2fx\n", var1 / var2);
    printf("\nBoth estimates hit %.5f-ish — importance sampling is still UNBIASED.\n"
           "But its variance is several times smaller, so for the same number of\n"
           "samples the noise (std error) is correspondingly lower. Equivalently,\n"
           "you would need ~%.1fx as many uniform samples to match its accuracy.\n"
           "A crude linear p already helps this much; a p that hugs e^x more\n"
           "tightly would do even better, approaching zero variance in the limit.\n",
           I, var1 / var2);
    return 0;
}
