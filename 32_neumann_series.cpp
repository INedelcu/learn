// 32 — The rendering equation as a Neumann series (per-bounce decomposition)
// ==========================================================================
//
// The rendering equation says outgoing radiance = emitted + reflected:
//
//   L = Le + T L,
//
// where T is the "transport operator" (gather incoming light over the hemisphere,
// weight by the BRDF and cosine, that's one bounce). Solving for L by repeated
// substitution gives the NEUMANN SERIES:
//
//   L = Le + T Le + T^2 Le + T^3 Le + ...   =   sum over k of T^k Le.
//
// Term k is "light that left a source and bounced exactly k times before reaching
// the eye": direct light (k=0), one-bounce indirect (k=1), and so on. A path
// tracer estimates this series by walking paths; capping the path length at K
// (the renderer's g_BounceCount) simply DROPS every term past k=K, which is why
// too few bounces makes indirect lighting too dark.
//
// We make this concrete with a diffuse enclosure where every surface emits E and
// reflects a fraction rho of what it receives. In that uniform setting one bounce
// just multiplies radiance by rho (a cosine-weighted average of a constant field,
// example 08), so term k contributes exactly E*rho^k and the series sums to the
// equilibrium L = E/(1-rho) -- the same geometric series as example 10. We:
//   * print the per-bounce-order contributions and partial sums (the series),
//   * show the truncation error after K bounces is rho^(K+1)/(1-rho) * E, and
//   * run an actual Monte Carlo path tracer (cosine bounces + Russian roulette)
//     that accumulates contributions BY bounce order and recovers each E*rho^k.
// In a real scene T is an integral operator estimated stochastically, but the
// series structure -- and what truncating it costs -- is exactly this.

#include <cstdio>
#include <cmath>
#include "mc_random.h"

int main() {
    Rng rng(/*sequence=*/32);
    const double E = 1.0, rho = 0.6;
    const double Leq = E / (1.0 - rho);              // analytic equilibrium

    // --- the Neumann series, term by term. ----------------------------------
    printf("Diffuse enclosure: E=%.1f, rho=%.1f.  Equilibrium L = E/(1-rho) = %.5f\n\n", E, rho, Leq);
    printf("  %5s | %14s %16s %16s\n", "k", "term E*rho^k", "partial sum", "truncation err");
    double partial = 0.0;
    for (int k = 0; k <= 10; ++k) {
        double term = E * std::pow(rho, k);
        partial += term;
        printf("  %5d | %14.6f %16.6f %16.6f\n", k, term, partial, Leq - partial);
    }
    printf("  -> capping at K bounces (g_BounceCount) drops the tail; the missing\n"
           "     energy is exactly rho^(K+1)/(1-rho)*E.\n\n");

    // --- Monte Carlo path tracer, contributions binned BY bounce order. ------
    // Each path: accumulate E at the current vertex (scaled by throughput), then
    // bounce (throughput *= rho) and continue via Russian roulette (example 10).
    const int kMaxOrder = 12;
    double orderSum[kMaxOrder + 1] = {0};
    const long long paths = 5000000;
    const double q = rho;                            // RR continuation probability
    for (long long p = 0; p < paths; ++p) {
        double tp = 1.0; int k = 0;
        for (;;) {
            if (k <= kMaxOrder) orderSum[k] += E * tp;   // contribution of order k
            if (rng.NextFloat() >= q) break;             // Russian roulette
            tp *= rho / q;                               // reflect (rho) AND RR-compensate (1/q)
            ++k;
            if (k > kMaxOrder + 2) break;                // safety
        }
    }
    printf("  Monte Carlo path tracer (cosine bounces + RR), per-order estimate:\n");
    printf("  %5s | %16s %16s\n", "k", "MC estimate", "analytic E*rho^k");
    double total = 0.0;
    for (int k = 0; k <= 6; ++k) {
        double est = orderSum[k] / paths;
        total += est;
        printf("  %5d | %16.6f %16.6f\n", k, est, E * std::pow(rho, k));
    }
    double grand = 0; for (int k = 0; k <= kMaxOrder; ++k) grand += orderSum[k] / paths;
    printf("  total over all orders = %.5f   (equilibrium %.5f)\n", grand, Leq);

    printf("\nThe per-order MC estimates reproduce the Neumann terms E*rho^k, and they\n"
           "sum to the equilibrium E/(1-rho). 'Direct lighting', '1-bounce GI', etc.\n"
           "are just prefixes of this series; the bounce-count knob in a path tracer\n"
           "chooses how many terms you keep. (Example 10 is the same series viewed\n"
           "through Russian-roulette termination.)\n");
    return 0;
}
