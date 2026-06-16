// 03 — The Monte Carlo integration estimator
// ===========================================
//
// We want a number we cannot easily get by hand: a definite integral
//   I = integral over [a,b] of f(x) dx.
//
// The fundamental Monte Carlo estimator says: draw N samples X_i and average f,
// then scale by the width of the domain:
//
//   I ≈ (b - a) * (1/N) * sum_i f(X_i),     X_i ~ Uniform[a,b].
//
// Why does this work? The samples are uniform on [a,b], so their probability
// density is p(x) = 1/(b-a). Rewrite the estimator as the average of f(X_i)/p(X_i):
//
//   (1/N) sum_i f(X_i)/p(X_i)
//
// and its expectation is
//
//   E[f(X)/p(X)] = integral over [a,b] of (f(x)/p(x)) * p(x) dx
//                = integral over [a,b] of f(x) dx = I.    (the p's cancel!)
//
// That last identity is THE Monte Carlo integration formula, and it holds for
// ANY probability density p(x) that is nonzero wherever f is nonzero — not just
// the uniform one. Example 06 (importance sampling) exploits this freedom by
// choosing a clever p; here we use the simplest p, the uniform density, so the
// estimator collapses to "(domain width) * (average of f)".
//
// In a path tracer the very same formula appears as the rendering equation:
//   L_o = integral over the hemisphere of f_r * L_i * cos(theta) dω
// is estimated by sampling directions ω_i from some density p(ω) and averaging
//   f_r(ω_i) * L_i(ω_i) * cos(theta_i) / p(ω_i).
// Everything in this folder is, ultimately, about choosing and analysing that p.
//
// We integrate two functions with known answers so we can check ourselves:
//   f(x) = e^x on [0,1],      exact = e - 1            ≈ 1.7182818
//   g(x) = sin(x) on [0,pi],  exact = 2

#include <cstdio>
#include <cmath>
#include "mc_random.h"

// A small helper: estimate the integral of f over [a,b] with N uniform samples.
template <typename F>
double EstimateIntegral(F f, double a, double b, long long n, Rng& rng) {
    double sum = 0.0;
    for (long long i = 0; i < n; ++i) {
        double x = rng.NextFloat((float)a, (float)b);
        sum += f(x);                 // f(X_i) / p(X_i)  with p = 1/(b-a), folded
    }                                // into the (b-a) factor below
    return (b - a) * sum / n;        // (b-a) * average of f
}

int main() {
    Rng rng(/*sequence=*/3);
    const double kPi = 3.14159265358979323846;
    const double kE  = 2.71828182845904523536;

    auto fExp = [](double x) { return std::exp(x); };
    auto gSin = [](double x) { return std::sin(x); };

    printf("Monte Carlo integration: I ~ (b-a) * mean(f(X_i)), X_i ~ Uniform[a,b]\n\n");

    printf("(A)  integral_0^1 e^x dx          exact = e - 1 = %.7f\n", kE - 1.0);
    printf("%14s %16s %14s\n", "samples", "estimate", "abs error");
    for (long long n = 10; n <= 10000000; n *= 10) {
        double est = EstimateIntegral(fExp, 0.0, 1.0, n, rng);
        printf("%14lld %16.7f %14.7f\n", n, est, std::fabs(est - (kE - 1.0)));
    }

    printf("\n(B)  integral_0^pi sin(x) dx      exact = 2\n");
    printf("%14s %16s %14s\n", "samples", "estimate", "abs error");
    for (long long n = 10; n <= 10000000; n *= 10) {
        double est = EstimateIntegral(gSin, 0.0, kPi, n, rng);
        printf("%14lld %16.7f %14.7f\n", n, est, std::fabs(est - 2.0));
    }

    printf("\nBoth estimates crawl toward the exact value at the familiar 1/sqrt(N)\n"
           "rate. The estimator never needed to know the antiderivative — only how\n"
           "to *evaluate* f at random points. That is exactly why it scales to the\n"
           "high-dimensional, no-closed-form integrals of light transport.\n");
    return 0;
}
