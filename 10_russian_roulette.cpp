// 10 — Russian roulette: terminating infinite sums without bias
// =============================================================
//
// A path tracer follows a ray as it bounces around: each bounce multiplies a
// running "throughput" by the surface albedo (< 1) and adds in any emission. In
// principle a path can bounce forever, so the radiance along it is an INFINITE
// sum. We obviously cannot trace infinitely many bounces — but if we just stop
// after a fixed number K, we systematically throw away all the later bounces and
// UNDERESTIMATE the result. That is bias, and it shows up as energy loss (overly
// dark interreflections).
//
// RUSSIAN ROULETTE is the trick that lets us stop early while staying UNBIASED.
// At each step, flip a biased coin: with "continuation probability" q keep going,
// otherwise stop. Crucially, whenever we DO continue we divide the running weight
// by q. The division exactly compensates for the chance of having been killed, so
// the expected value is unchanged — we just trade a little extra variance for a
// guaranteed-finite amount of work.
//
//   Claim: term k of the sum is reached with probability q^k, and when reached it
//   is weighted by 1/q^k. Its expected contribution is therefore
//       (value of term k) * q^k * (1/q^k) = (value of term k),
//   so the expectation of the roulette estimator equals the full infinite sum.
//   Meanwhile the expected number of steps is 1 + q + q^2 + ... = 1/(1-q), finite.
//
// MODEL PROBLEM: radiance down a path with constant per-bounce albedo rho and
// constant emission e at every vertex:
//
//   S = e * (1 + rho + rho^2 + rho^3 + ...) = e / (1 - rho).
//
// With e = 1 and rho = 0.8 the true answer is 1/(1-0.8) = 5. We compare:
//   * fixed truncation after K bounces           -> biased (too small),
//   * Russian roulette with continuation prob q   -> unbiased.
// We use q = rho, which is what real path tracers do (set q from the throughput):
// then weight*term is constant and every surviving bounce contributes exactly e,
// so the path length is Geometric and the estimator is beautifully simple.

#include <cstdio>
#include <cmath>
#include <initializer_list>
#include "mc_random.h"

int main() {
    Rng rng(/*sequence=*/10);
    const double e = 1.0, rho = 0.8;
    const double trueS = e / (1.0 - rho);     // = 5
    const long long kPaths = 5000000;

    printf("True sum S = e/(1-rho) = %.0f/(1-%.1f) = %.6f\n\n", e, rho, trueS);

    // --- Biased baseline: always stop after exactly K bounces. ---------------
    // This is deterministic given K (no randomness), so we just sum the series.
    printf("Fixed truncation after K bounces (biased, no roulette):\n");
    printf("  %4s %14s %14s\n", "K", "estimate", "missing");
    for (int K : {1, 2, 4, 8, 16, 32}) {
        double s = 0.0, tp = 1.0;
        for (int k = 0; k <= K; ++k) { s += e * tp; tp *= rho; }
        printf("  %4d %14.6f %14.6f\n", K, s, trueS - s);
    }
    printf("  -> always short of %.3f: truncation loses all the deeper bounces.\n\n", trueS);

    // --- Russian roulette: unbiased, with a self-limiting path length. -------
    // Per path: accumulate e*weight at the current vertex, then roll to continue.
    // On continue, divide weight by q (= rho here) and move on; else stop.
    const double q = rho;       // continuation probability
    double mean = 0.0, m2 = 0.0;
    double totalSteps = 0.0;
    for (long long p = 0; p < kPaths; ++p) {
        double s = 0.0, weight = 1.0;
        int steps = 0;
        for (;;) {
            s += e * weight;               // contribution of the current vertex
            ++steps;
            if (rng.NextFloat() >= q) break;   // roulette says: terminate
            weight /= q;                       // survived: compensate, then...
            weight *= rho;                     // ...apply this bounce's albedo
        }
        totalSteps += steps;
        double d = s - mean; mean += d / (p + 1); m2 += d * (s - mean);
    }
    double var = m2 / (kPaths - 1);

    printf("Russian roulette with continuation prob q = %.2f over %lld paths:\n", q, kPaths);
    printf("  estimate ............ %.6f   (true S = %.6f, UNBIASED)\n", mean, trueS);
    printf("  std error ........... %.6f\n", std::sqrt(var / kPaths));
    printf("  avg bounces / path .. %.4f   (theory 1/(1-q) = %.4f)\n",
           totalSteps / kPaths, 1.0 / (1.0 - q));

    printf("\nFixed truncation is cheap but always too dark; Russian roulette pays a\n"
           "little variance to make the answer correct ON AVERAGE while keeping the\n"
           "work per path finite. Note q=rho makes weight*e constant each step, so\n"
           "this reduces to 'count e per surviving bounce' — exactly the throughput-\n"
           "driven termination the renderer uses in its ray-gen loop.\n");
    return 0;
}
