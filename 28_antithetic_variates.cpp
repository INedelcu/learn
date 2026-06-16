// 28 — Antithetic variates: pair each sample with its mirror
// ==========================================================
//
// A tiny, almost-free variance-reduction trick. For every uniform sample u, also
// use its mirror 1 - u, and average the pair:
//
//   estimate per pair = ( f(u) + f(1 - u) ) / 2.
//
// If f is MONOTONIC, then when f(u) is above the mean, f(1 - u) tends to be below
// it -- the two are NEGATIVELY correlated, so their average fluctuates much less
// than either alone. The variance of the pair-average is
//
//   Var[(f(u)+f(1-u))/2] = (1/2) Var[f] (1 + corr(f(u), f(1-u))),
//
// and a negative correlation drives that below the (1/2)Var[f] you'd get from two
// independent samples -- so antithetic sampling beats plain Monte Carlo at the
// SAME number of evaluations.
//
// The catch, which this file also shows: for a function SYMMETRIC about x = 0.5,
// f(u) = f(1-u), the correlation is +1 and the trick gives NO benefit (it can even
// hurt). Antithetic pairing helps exactly when the integrand has a strong
// monotonic trend -- common for many 1D mappings inside a sampler.
//
// We compare plain MC against antithetic at equal evaluation budgets, for a
// monotonic integrand (e^x) and a symmetric one ((x-0.5)^2), reporting the
// variance per evaluation so the comparison is fair.

#include <cstdio>
#include <cmath>
#include "mc_random.h"

// Variance-per-evaluation of plain vs antithetic for a given integrand.
template <typename F>
void Compare(const char* name, F f, double exact, Rng& rng) {
    const long long pairs = 10000000;       // -> 2 * pairs evaluations each

    // Plain MC: treat each of the 2*pairs samples independently.
    double mP = 0, vP = 0; long long nP = 0;
    for (long long i = 0; i < 2 * pairs; ++i) {
        double y = f(rng.NextFloat());
        ++nP; double d = y - mP; mP += d / nP; vP += d * (y - mP);
    }
    double varPerEvalPlain = vP / (nP - 1);

    // Antithetic: pair u with 1-u. One pair = 2 evaluations, so the variance
    // per evaluation is 2 * Var[pair-average].
    double mA = 0, vA = 0; long long nA = 0; double corrAcc = 0, m1 = 0, m2 = 0, c12 = 0;
    for (long long i = 0; i < pairs; ++i) {
        double u = rng.NextFloat();
        double a = f(u), b = f(1.0 - u);
        double pa = 0.5 * (a + b);
        ++nA; double d = pa - mA; mA += d / nA; vA += d * (pa - mA);
        // track corr(f(u), f(1-u)) for reporting
        double d1 = a - m1, d2 = b - m2; m1 += d1 / nA; m2 += d2 / nA; c12 += d1 * (b - m2);
    }
    double varPerEvalAnti = 2.0 * (vA / (nA - 1));   // 2 evals per pair

    printf("  %-18s exact %.6f\n", name, exact);
    printf("    plain      estimate %.6f  variance/eval %.4e\n", mP, varPerEvalPlain);
    printf("    antithetic estimate %.6f  variance/eval %.4e   (corr f(u),f(1-u) = %+.3f)\n",
           mA, varPerEvalAnti, c12 / (nA - 1) / std::sqrt((vP / (nP - 1)) * (vP / (nP - 1))));
    printf("    variance reduction: %.2fx\n\n", varPerEvalPlain / varPerEvalAnti);
}

int main() {
    Rng rng(/*sequence=*/28);
    const double kE = 2.71828182845904523536;

    printf("Antithetic variates: pair u with 1-u, at equal evaluation budgets.\n\n");
    Compare("monotonic e^x", [](double x) { return std::exp(x); }, kE - 1.0, rng);
    Compare("symmetric (x-0.5)^2", [](double x) { return (x - 0.5) * (x - 0.5); }, 1.0 / 12.0, rng);

    printf("For the monotonic e^x the mirrored samples are strongly anti-correlated,\n"
           "so the pair-average barely moves -- a big reduction for one extra subtract\n"
           "and negate. For the symmetric integrand f(u)=f(1-u), the pair is perfectly\n"
           "correlated and antithetic sampling buys nothing. Know your integrand:\n"
           "antithetic pairing is a cheap win for the monotonic 1D maps inside\n"
           "samplers, and a no-op for symmetric ones.\n");
    return 0;
}
