// 27 — Control variates: subtract a function whose integral you know
// ===================================================================
//
// Importance sampling (06) reshaped WHERE we sample. CONTROL VARIATES is the
// other big variance-reduction lever, and it leaves the sampling alone. The idea:
// to estimate I = integral of f, find a cheap "control" function g whose integral
// G you already know AND which is CORRELATED with f. Then estimate
//
//   I = integral f = E[ f(X) - c*(g(X) - G) ],      X ~ uniform,
//
// for any constant c -- because E[g(X) - G] = 0, subtracting it changes nothing
// in expectation, it only cancels the part of f's fluctuation that g can predict.
// The optimal c is  c* = Cov(f,g) / Var(g),  and it cuts the variance by the
// factor  1 / (1 - rho^2),  where rho = corr(f, g). The better g tracks f, the
// closer rho is to 1 and the bigger the win.
//
// (This is the same trick behind "denoising by subtracting an analytic
// approximation" in renderers: integrate the hard residual, add back the part you
// can compute in closed form.)
//
// We estimate integral_0^1 e^x dx = e - 1 using the control g(x) = 1 + x, whose
// integral is exactly 1.5. On [0,1] the line 1+x hugs e^x closely (rho ~ 0.9995),
// so the variance should collapse. We estimate c* from a pilot batch, then
// compare the plain estimator (c = 0) with the control-variate estimator.

#include <cstdio>
#include <cmath>
#include "mc_random.h"

int main() {
    Rng rng(/*sequence=*/27);
    const double kE = 2.71828182845904523536;
    const double I = kE - 1.0;
    auto f = [](double x) { return std::exp(x); };
    auto g = [](double x) { return 1.0 + x; };
    const double G = 1.5;                                   // known integral of g

    // --- pilot pass: estimate c* = Cov(f,g)/Var(g) and the correlation rho. --
    double mf = 0, mg = 0, cff = 0, cgg = 0, cfg = 0; long long np = 0;
    for (int i = 0; i < 200000; ++i) {
        double x = rng.NextFloat(), a = f(x), b = g(x);
        ++np;
        double df = a - mf, dg = b - mg;
        mf += df / np; mg += dg / np;
        cff += df * (a - mf); cgg += dg * (b - mg); cfg += df * (b - mg);
    }
    double cStar = cfg / cgg;
    double rho = cfg / std::sqrt(cff * cgg);
    printf("Estimating integral_0^1 e^x dx = %.7f, control g(x)=1+x (integral = %.1f)\n", I, G);
    printf("  pilot: corr(f,g) = %.5f,  optimal c* = %.5f\n", rho, cStar);
    printf("  predicted variance reduction 1/(1-rho^2) = %.1fx\n\n", 1.0 / (1.0 - rho * rho));

    // --- compare plain vs control-variate estimator. ------------------------
    double mP = 0, vP = 0, mC = 0, vC = 0; const long long N = 20000000;
    for (long long i = 0; i < N; ++i) {
        double x = rng.NextFloat();
        double plain = f(x);
        double ctrl  = f(x) - cStar * (g(x) - G);          // same expectation, less spread
        double dP = plain - mP; mP += dP / (i + 1); vP += dP * (plain - mP);
        double dC = ctrl  - mC; mC += dC / (i + 1); vC += dC * (ctrl  - mC);
    }
    printf("  %-16s estimate %.7f  variance %.4e\n", "plain (c=0)", mP, vP / (N - 1));
    printf("  %-16s estimate %.7f  variance %.4e\n", "control variate", mC, vC / (N - 1));
    printf("  measured variance reduction: %.1fx\n", (vP / (N - 1)) / (vC / (N - 1)));

    printf("\nBoth estimators are unbiased (they agree on %.5f), but subtracting the\n"
           "known-integral control absorbs almost all of e^x's variation, since the\n"
           "line 1+x predicts it so well. Find a cheap g that correlates with your\n"
           "integrand and you get variance reduction for free, orthogonally to\n"
           "importance sampling -- the two combine.\n", I);
    return 0;
}
