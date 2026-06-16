// 38 — Fireflies: heavy-tailed estimators and the clamping tradeoff
// =================================================================
//
// Those lone bright white pixels in a noisy render -- "fireflies" -- are not a
// bug, they are the Monte Carlo estimator's heavy TAIL showing through. When a
// rare path carries a huge contribution (a tiny but bright light hit through a
// low-probability direction, a near-singular BSDF/pdf ratio), the estimator value
// f(X)/p(X) can spike enormously. If the tail is heavy enough the estimator can
// even have INFINITE variance: the mean is still correct, but no practical number
// of samples looks converged, because one giant sample dominates the average.
//
// We build exactly that: estimate integral_0^1 1 dx = 1, but draw X from p(x)=2x
// (so X = sqrt(u)) -- a pdf that is tiny near 0 where the integrand is not. The
// estimator Y = f/p = 1/(2X) blows up as X -> 0, and Var[Y] = integral 1/(2x) dx
// DIVERGES. The mean is 1, but the variance is infinite -> fireflies.
//
// The common fix is CLAMPING: cap each sample at some value c. This tames the
// variance dramatically, but it throws away the rare-but-real energy in the tail,
// so the result is BIASED low (example 37's other failure mode). Clamp tightly and
// you get a clean but too-dark image; clamp loosely (or not at all) and you get the
// correct brightness but fireflies. We show the whole tradeoff curve.

#include <cstdio>
#include <cmath>
#include <initializer_list>
#include "mc_random.h"

int main() {
    Rng rng(/*sequence=*/38);
    const long long N = 4000000;

    // Unclamped: correct mean (=1), but watch the max sample and the (unstable)
    // variance -- the signature of a heavy tail.
    double mean = 0, m2 = 0, maxY = 0;
    for (long long i = 0; i < N; ++i) {
        double X = std::sqrt(rng.NextFloat());
        double Y = 1.0 / (2.0 * X);                        // f/p with f=1, p=2x
        if (Y > maxY) maxY = Y;
        double d = Y - mean; mean += d / (i + 1); m2 += d * (Y - mean);
    }
    printf("Estimating integral_0^1 1 dx = 1 with a deliberately bad pdf p(x)=2x.\n");
    printf("Y = 1/(2X) has INFINITE true variance (a heavy tail = fireflies).\n\n");
    printf("  unclamped: mean = %.4f, sample variance = %.3e, largest sample = %.1f\n",
           mean, m2 / (N - 1), maxY);
    printf("  (the single largest sample alone contributed %.4f / %lld to the mean)\n\n",
           maxY, N);

    // Clamped at increasing c: variance becomes finite and small, but the mean is
    // biased low because the tail's energy is discarded.
    printf("  clamp c |  mean (biased) |   bias  |   variance\n");
    for (double c : {2.0, 5.0, 20.0, 100.0, 1000.0}) {
        double mC = 0, vC = 0;
        for (long long i = 0; i < N; ++i) {
            double X = std::sqrt(rng.NextFloat());
            double Y = std::fmin(1.0 / (2.0 * X), c);      // clamp the estimator
            double d = Y - mC; mC += d / (i + 1); vC += d * (Y - mC);
        }
        printf("  %7.0f | %14.5f | %+7.4f | %10.4f\n", c, mC, mC - 1.0, vC / (N - 1));
    }

    printf("\nClamping trades the firefly variance for darkening bias: tight clamps\n"
           "(small c) give a stable but too-dark estimate; loosening c recovers the\n"
           "energy but lets the variance climb back. Production renderers clamp\n"
           "indirect contributions (or use better sampling/MIS, example 24, so the\n"
           "tail never forms) -- a pragmatic, knowingly-biased firefly fix.\n");
    return 0;
}
