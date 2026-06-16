// 09 — Multiple Importance Sampling (MIS): combining strategies that each win
//      somewhere
// ==========================================================================
//
// Importance sampling (example 06) is great when ONE density matches the whole
// integrand. But real integrands often have several "features", and no single
// easy-to-sample density is good for all of them. The classic rendering case:
// to shade a glossy surface you can either SAMPLE THE BRDF (great for sharp
// reflections, bad for small bright lights) or SAMPLE THE LIGHT (great for small
// lights, bad for near-mirror reflections). Each strategy has catastrophic
// variance exactly where the other shines.
//
// Multiple Importance Sampling (Veach & Guibas, 1995) lets you use BOTH and
// combine them so the result inherits the low variance of whichever strategy was
// appropriate for each sample — and provably never does much worse than the best
// single strategy. The path tracer's per-hit "pick a lobe by luminance" logic and
// its single-scattering estimators are cousins of this idea.
//
// The combined estimator with strategies p1, p2 (one sample each per iteration):
//
//   F = w1(X1) * f(X1)/p1(X1)  +  w2(X2) * f(X2)/p2(X2),   X1~p1, X2~p2,
//
// where the WEIGHTS w_i sum to 1 at every point. The BALANCE HEURISTIC chooses
//
//   w_i(x) = p_i(x) / ( p1(x) + p2(x) ).
//
// Because w1 + w2 = 1, E[F] = integral of f, so MIS is unbiased. The magic is
// that wherever one density is tiny (its f/p blows up), its weight there is also
// tiny — the weight muzzles the spike instead of letting it explode the variance.
//
// SELF-CONTAINED EXPERIMENT (everything has a closed form so we can check it):
//   Two "peaks" built from truncated exponentials on [0,1]:
//     p1(x) = a*e^{-a x}      / (1 - e^{-a})     (peaks at x = 0)
//     p2(x) = a*e^{-a(1-x)}   / (1 - e^{-a})     (peaks at x = 1)
//   Target integrand: f(x) = A*p1(x) + B*p2(x).  Then integral_0^1 f = A + B exactly.
//   p1 samples the left peak well but is starved at the right peak (and vice
//   versa), so each strategy ALONE has high variance. MIS fixes it.
//   Each method is given the same budget of 2 samples per estimate.

#include <cstdio>
#include <cmath>
#include "mc_random.h"

static const double a = 8.0;       // peak sharpness
static const double A = 1.0, B = 4.0;   // unequal peak strengths (true integral = 5)
static const double Z = 1.0 - std::exp(-a);   // normalization of the truncated exp

static double p1(double x) { return a * std::exp(-a * x) / Z; }          // peak at 0
static double p2(double x) { return a * std::exp(-a * (1.0 - x)) / Z; }  // peak at 1
static double f(double x)  { return A * p1(x) + B * p2(x); }             // target

// Sample p1 by inverting its CDF;  p2's sample is the mirror image 1 - X.
static double sampleP1(Rng& rng) { return -std::log(1.0 - rng.NextFloat() * Z) / a; }
static double sampleP2(Rng& rng) { return 1.0 - sampleP1(rng); }

int main() {
    Rng rng(/*sequence=*/9);
    const double I = A + B;                 // the exact integral
    const long long kTrials = 5000000;

    struct Acc { double mean = 0, m2 = 0; long long n = 0;
        void add(double x) { ++n; double d = x - mean; mean += d / n; m2 += d * (x - mean); }
        double var() const { return m2 / (n - 1); } };

    Acc only1, only2, mis;

    for (long long t = 0; t < kTrials; ++t) {
        // (1) Strategy 1 only: 2 samples from p1, average f/p1.
        {
            double xa = sampleP1(rng), xb = sampleP1(rng);
            only1.add(0.5 * (f(xa) / p1(xa) + f(xb) / p1(xb)));
        }
        // (2) Strategy 2 only: 2 samples from p2, average f/p2.
        {
            double xa = sampleP2(rng), xb = sampleP2(rng);
            only2.add(0.5 * (f(xa) / p2(xa) + f(xb) / p2(xb)));
        }
        // (3) MIS: 1 sample from each, combined with the balance heuristic.
        {
            double x1 = sampleP1(rng);
            double x2 = sampleP2(rng);
            double w1 = p1(x1) / (p1(x1) + p2(x1));
            double w2 = p2(x2) / (p1(x2) + p2(x2));
            mis.add(w1 * f(x1) / p1(x1) + w2 * f(x2) / p2(x2));
        }
    }

    printf("integral_0^1 [A*p1 + B*p2] dx = A + B = %.4f   (two peaks, sharpness a=%.0f)\n\n", I, a);
    printf("%-26s %14s %16s\n", "method (2 samples each)", "estimate", "variance of est.");
    printf("%-26s %14.6f %16.3e\n", "sample p1 only",  only1.mean, only1.var());
    printf("%-26s %14.6f %16.3e\n", "sample p2 only",  only2.mean, only2.var());
    printf("%-26s %14.6f %16.3e   <- best\n", "MIS (balance heuristic)", mis.mean, mis.var());

    printf("\nMIS variance is ~%.0fx lower than either single strategy.\n",
           std::fmin(only1.var(), only2.var()) / mis.var());
    printf("\nEach single strategy is unbiased but noisy: it samples its own peak\n"
           "well and the OTHER peak almost never, so the rare hits on the far peak\n"
           "carry a huge f/p value and inflate the variance. With the balance\n"
           "heuristic each MIS term simplifies to f/(p1+p2), a weighted average that\n"
           "stays bounded between A and B instead of exploding — which is precisely\n"
           "why the weights tame the spikes. (Set A == B and that term becomes a\n"
           "constant: MIS would then have *zero* variance here.)\n");
    return 0;
}
