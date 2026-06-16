// 37 — Bias vs variance: unbiased, consistent, and the difference
// ===============================================================
//
// Two different things can be wrong with an estimator:
//   * VARIANCE  -- it scatters around its average (noise). Shrinks as 1/N.
//   * BIAS      -- its average is the wrong number. Does NOT shrink with N.
//
// Path tracing is UNBIASED: its expected value is the true integral, so adding
// samples drives the error to zero. Many fast techniques (photon mapping, irradiance
// caching, denoisers) are BIASED but CONSISTENT: they converge to the truth only
// if a smoothing parameter is also driven to zero as N grows. If you hold that
// parameter fixed, more samples just nail down the WRONG (smoothed) answer.
//
// The cleanest example is kernel density estimation -- exactly photon mapping's
// math. To estimate a density p at a point x0 from samples X_i ~ p, count the
// samples landing within a kernel of half-width h and normalize:
//
//   p_hat(x0) = (#samples within h of x0) / (2 h N).
//
// For a curved density this is biased: it reports the AVERAGE of p over the window,
// p(x0) + (curvature)*h^2 + ..., so a fixed h leaves a permanent h^2 bias no matter
// how large N gets. Shrink h with N (h_N -> 0 slowly) and the bias vanishes too --
// that is consistency, and it always converges slower than the unbiased 1/sqrt(N).
//
// We use p(x) = 3x^2 (sampled by X = u^(1/3)) and estimate p(0.5) = 0.75.

#include <cstdio>
#include <cmath>
#include "mc_random.h"

int main() {
    Rng rng(/*sequence=*/37);
    const double x0 = 0.5, truth = 3.0 * x0 * x0;          // p(0.5) = 0.75
    auto sample = [&] { return std::pow((double)rng.NextFloat(), 1.0 / 3.0); };  // X ~ 3x^2

    printf("Estimating the density p(x)=3x^2 at x0=%.1f; true p(x0) = %.4f\n\n", x0, truth);
    printf("Kernel density estimate (photon-mapping style) as N grows:\n");
    printf("  %10s | %-26s | %-26s\n", "N", "FIXED h=0.05", "SHRINKING h = 0.4*N^-0.2");
    printf("  %10s | %12s %12s | %12s %12s\n", "", "estimate", "bias", "estimate", "bias");

    for (long long N = 1000; N <= 10000000; N *= 10) {
        double hFix = 0.05;
        double hShr = 0.4 * std::pow((double)N, -0.2);      // bandwidth -> 0 as N grows
        long long cntFix = 0, cntShr = 0;
        for (long long i = 0; i < N; ++i) {
            double X = sample();
            if (std::fabs(X - x0) < hFix) ++cntFix;
            if (std::fabs(X - x0) < hShr) ++cntShr;
        }
        double estFix = cntFix / (2.0 * hFix * N);
        double estShr = cntShr / (2.0 * hShr * N);
        printf("  %10lld | %12.5f %12.5f | %12.5f %12.5f\n",
               N, estFix, estFix - truth, estShr, estShr - truth);
    }

    printf("\nThe fixed-h estimate gets PRECISE but stays wrong: its bias parks at\n"
           "about h^2 = %.4f no matter how many samples you add. The shrinking-h\n"
           "estimate is consistent -- its bias melts toward 0 -- but it pays for that\n"
           "with slower convergence. This is the unbiased-vs-biased tradeoff between\n"
           "path tracing (slow but converges to the truth) and methods like photon\n"
           "mapping (faster, but you must shrink the kernel to remove the blur).\n",
           0.05 * 0.05);
    return 0;
}
