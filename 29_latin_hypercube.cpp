// 29 — Stratifying high dimensions: Latin hypercube sampling
// ==========================================================
//
// Stratification (example 07) crushed variance in 1D by putting one sample per
// cell. The obvious extension to d dimensions -- a full grid -- needs N^d cells:
// for the ~10-dimensional integrals a path tracer faces (pixel xy, lens, light,
// each bounce's two numbers...), even 2 strata per axis is 2^10 = 1024 samples
// just to fill the grid once. That is the CURSE OF DIMENSIONALITY, and it makes
// full stratification useless past a few dimensions.
//
// LATIN HYPERCUBE SAMPLING (LHS, a.k.a. N-rooks) sidesteps it. With only N
// samples it stratifies EACH dimension independently into N strata: every axis
// gets exactly one sample in each of its N slices, and the per-axis slices are
// paired up by a random permutation. So all the 1D projections are perfectly
// stratified with N (not N^d) samples. It cannot capture interactions between
// dimensions, but most of a typical integrand's variance lives in the 1D "main
// effects", and LHS removes exactly that part.
//
//   sample i, dimension j:   x_ij = (perm_j[i] + u_ij) / N,
//   with perm_j an independent random permutation of {0, ..., N-1}.
//
// We integrate an additive function f(x) = sum_j exp(x_j) over [0,1]^d (exact =
// d*(e-1)) and compare the variance of plain random sampling against LHS at the
// same N. LHS should win comfortably, with only N samples in d dimensions.

#include <cstdio>
#include <cmath>
#include <vector>
#include <numeric>
#include "mc_random.h"

int main() {
    Rng rng(/*sequence=*/29);
    const double kE = 2.71828182845904523536;
    const int d = 6, N = 64;
    const double exact = d * (kE - 1.0);
    auto f = [](const std::vector<double>& x) {
        double s = 0; for (double v : x) s += std::exp(v); return s;
    };

    const int trials = 30000;
    double mR = 0, vR = 0, mL = 0, vL = 0;
    std::vector<int> perm(N);
    std::vector<double> x(d);

    for (int t = 0; t < trials; ++t) {
        // --- plain random: N points of d independent uniforms. -------------
        double sr = 0;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < d; ++j) x[j] = rng.NextFloat();
            sr += f(x);
        }
        double estR = sr / N;
        double dR = estR - mR; mR += dR / (t + 1); vR += dR * (estR - mR);

        // --- LHS: each dimension independently stratified into N slices. ----
        // Build, per dimension, a fresh permutation, then read column i out of
        // each permutation to form sample i.
        std::vector<std::vector<int>> perms(d, std::vector<int>(N));
        for (int j = 0; j < d; ++j) {
            std::iota(perms[j].begin(), perms[j].end(), 0);
            for (int k = N - 1; k > 0; --k) {                // Fisher-Yates shuffle
                int m = (int)(rng.NextFloat() * (k + 1));
                std::swap(perms[j][k], perms[j][m]);
            }
        }
        double sl = 0;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < d; ++j) x[j] = (perms[j][i] + rng.NextFloat()) / N;
            sl += f(x);
        }
        double estL = sl / N;
        double dL = estL - mL; mL += dL / (t + 1); vL += dL * (estL - mL);
    }

    printf("integral over [0,1]^%d of sum_j e^(x_j) = d*(e-1) = %.5f,  N = %d samples\n\n",
           d, exact, N);
    printf("  (a full stratified grid would need 2^%d = %d cells just for 2 strata/axis)\n\n",
           d, 1 << d);
    printf("  %-16s estimate %.5f  variance %.4e\n", "random", mR, vR / (trials - 1));
    printf("  %-16s estimate %.5f  variance %.4e\n", "Latin hypercube", mL, vL / (trials - 1));
    printf("  variance reduction: %.1fx\n", (vR / (trials - 1)) / (vL / (trials - 1)));

    printf("\nLHS matches the exact integral and slashes the variance with only N\n"
           "samples in %d dimensions -- because it stratifies every axis's 1D\n"
           "projection, where most of this integrand's variance lives. It is the\n"
           "practical answer to the curse of dimensionality, and the reason renderers\n"
           "stratify per-dimension (often combined with the low-discrepancy patterns\n"
           "of example 11) rather than building an impossible N^d grid.\n", d);
    return 0;
}
