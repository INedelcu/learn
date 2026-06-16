// 02 — Estimating pi, and the 1/sqrt(N) convergence law
// =====================================================
//
// The "hello world" of Monte Carlo. Throw darts uniformly at the unit square
// [0,1]x[0,1] and count how many land inside the quarter circle x^2 + y^2 <= 1.
//
//   area of quarter circle      (pi/4)
//   ----------------------  =  -------- = pi/4
//   area of unit square            1
//
// so the *fraction* of darts inside the circle is an estimate of pi/4, and
// 4 * fraction is an estimate of pi. This is really just Monte Carlo integration
// of an indicator function — but it gives us a concrete place to study the single
// most important fact about Monte Carlo:
//
//   THE ERROR SHRINKS LIKE 1/sqrt(N).
//
// To halve the error you need 4x the samples. To get one more decimal digit
// (10x less error) you need 100x the samples. This is why path-traced images are
// noisy and why that noise is so stubborn: the noise IS the Monte Carlo error,
// and it only falls off as 1/sqrt(samples-per-pixel).
//
// The fraction p_hat of N Bernoulli trials has standard error sqrt(p(1-p)/N).
// Here p = pi/4 ≈ 0.785, so the standard error of our pi estimate is
//   4 * sqrt(p(1-p)/N).
// We print the actual error alongside this predicted error so you can see them
// track each other, and we print error*sqrt(N), which should hover around a
// constant instead of trending to zero.

#include <cstdio>
#include <cmath>
#include "mc_random.h"

int main() {
    Rng rng(/*sequence=*/2);

    const double kPi = 3.14159265358979323846;
    const double p = kPi / 4.0;                       // true hit probability
    const double predictedConst = 4.0 * std::sqrt(p * (1.0 - p));  // error*sqrt(N) ~ this

    printf("Estimating pi by throwing darts at a quarter circle.\n\n");
    printf("%12s %14s %12s %14s %14s\n",
           "samples", "pi estimate", "abs error", "pred. error", "error*sqrt(N)");

    long long inside = 0;
    long long n = 0;
    long long nextReport = 100;
    const long long kTotal = 100000000;  // 100 million
    for (long long i = 0; i < kTotal; ++i) {
        float x = rng.NextFloat();
        float y = rng.NextFloat();
        if (x * x + y * y <= 1.0f) ++inside;
        ++n;
        if (n == nextReport) {
            double est = 4.0 * (double)inside / n;
            double err = std::fabs(est - kPi);
            double predErr = predictedConst / std::sqrt((double)n);
            printf("%12lld %14.6f %12.6f %14.6f %14.4f\n",
                   n, est, err, predErr, err * std::sqrt((double)n));
            nextReport *= 10;
        }
    }

    printf("\nTwo things to take away:\n"
           "  * 'abs error' and 'pred. error' stay in the same ballpark — the\n"
           "    1/sqrt(N) formula really does predict the noise.\n"
           "  * 'error*sqrt(N)' does NOT march toward zero; it wobbles around a\n"
           "    constant (~%.2f). That constant is what variance-reduction methods\n"
           "    (importance sampling, stratification, MIS) try to shrink — see the\n"
           "    later examples. They cannot beat 1/sqrt(N); they shrink its constant.\n",
           predictedConst);
    return 0;
}
