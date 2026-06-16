// 05 — Drawing samples from a chosen distribution (inversion & rejection)
// =======================================================================
//
// So far every random variable was Uniform[0,1]. But the whole power of Monte
// Carlo (example 06 onward) comes from sampling from a *non-uniform* density p(x)
// that you get to choose. This file shows the two workhorse techniques for
// turning a uniform random number into a sample from an arbitrary p(x).
//
// --------------------------------------------------------------------------
// TECHNIQUE 1: THE INVERSION METHOD (a.k.a. inverse-CDF sampling)
// --------------------------------------------------------------------------
// Given a density p(x), form its cumulative distribution function (CDF)
//   P(x) = integral from -inf to x of p(t) dt        (rises from 0 to 1).
// CLAIM: if U ~ Uniform[0,1], then X = P^{-1}(U) has density p.
// Intuition: P spreads probability mass so that "how much mass is below x" equals
// P(x); inverting it pulls a uniform pick of mass back to the matching x.
// This is the method of choice whenever you can invert the CDF in closed form —
// it uses exactly one uniform number per sample and never "wastes" any.
//
//   Worked example A: p(x) = 2x on [0,1].
//     CDF P(x) = x^2, so P^{-1}(u) = sqrt(u).   ->   X = sqrt(U).
//
//   Worked example B: the exponential, p(t) = lambda*e^{-lambda t} on [0, inf).
//     CDF P(t) = 1 - e^{-lambda t}, so P^{-1}(u) = -ln(1 - u) / lambda.
//   This one is everywhere in rendering: it samples the distance a photon travels
//   before scattering/absorption in a medium (Beer-Lambert), where lambda is the
//   extinction coefficient. The path tracer's glass shader uses the same
//   exp(-extinction * t) law for absorption.
//
// --------------------------------------------------------------------------
// TECHNIQUE 2: REJECTION SAMPLING
// --------------------------------------------------------------------------
// When you cannot invert the CDF, you can still sample p if you can (a) evaluate
// p(x) and (b) find a constant M with p(x) <= M*q(x) for an easy-to-sample q.
// Draw a candidate from q, accept it with probability p(x)/(M*q(x)), else throw
// it away and try again. Accepted samples follow p exactly. It is dead simple and
// completely general, but it can be wasteful: the acceptance rate is 1/M, so a
// loose bound means lots of rejected work. (That waste is one reason renderers
// prefer inversion / importance sampling when a closed form is available.)
//
//   Worked example C: p(x) = (3/2)*x^2 on... wait, that integrates to 1/2. We use
//   p(x) = 3x^2 on [0,1] (a valid density: integral of 3x^2 over [0,1] = 1).
//   Proposal q = Uniform[0,1], and since max of 3x^2 on [0,1] is 3, take M = 3.
//   Accept a uniform candidate x with probability p(x)/(M*q(x)) = 3x^2/3 = x^2.
//
// We verify each sampler by histogramming its output and overlaying the density
// the bin "should" have. The bars should match the predicted heights.

#include <cstdio>
#include <cmath>
#include "mc_random.h"

// Print a histogram of samples in [lo,hi) next to the density's predicted
// average height in each bin.
template <typename PDF>
void CheckHistogram(const char* name, const float* samples, int count,
                    float lo, float hi, PDF pdf) {
    const int kBins = 10;
    int hist[kBins] = {0};
    for (int i = 0; i < count; ++i) {
        int b = (int)((samples[i] - lo) / (hi - lo) * kBins);
        if (b >= 0 && b < kBins) hist[b]++;
    }
    float binW = (hi - lo) / kBins;
    printf("  %s\n", name);
    printf("    %-16s %10s %10s\n", "bin", "measured", "predicted");
    for (int b = 0; b < kBins; ++b) {
        float x0 = lo + b * binW, x1 = x0 + binW;
        float measured = (float)hist[b] / count / binW;     // empirical density
        float predicted = 0.5f * (pdf(x0) + pdf(x1));        // density at bin mid-ish
        printf("    [%.2f,%.2f)      %10.4f %10.4f  ", x0, x1, measured, predicted);
        for (int s = 0; s < (int)(measured * 25); ++s) putchar('#');
        putchar('\n');
    }
}

int main() {
    Rng rng(/*sequence=*/5);
    const int kN = 2000000;
    static float buf[2000000];

    printf("Sampling non-uniform densities from Uniform[0,1) draws.\n\n");

    // --- A: inversion, p(x) = 2x on [0,1], X = sqrt(U) -----------------------
    printf("INVERSION  p(x) = 2x on [0,1]   (X = sqrt(U))\n");
    for (int i = 0; i < kN; ++i) buf[i] = std::sqrt(rng.NextFloat());
    CheckHistogram("p(x) = 2x", buf, kN, 0.0f, 1.0f, [](float x) { return 2.0f * x; });

    // --- B: inversion, exponential, X = -ln(1-U)/lambda ----------------------
    const float lambda = 1.5f;
    printf("\nINVERSION  exponential p(t) = %.1f*e^(-%.1f t)   (X = -ln(1-U)/lambda)\n",
           lambda, lambda);
    double sumT = 0.0;
    for (int i = 0; i < kN; ++i) {
        float t = -std::log(1.0f - rng.NextFloat()) / lambda;
        buf[i] = t;
        sumT += t;
    }
    // The exponential has mean 1/lambda; check it, and histogram the body [0,3).
    printf("    sample mean = %.5f   (theory 1/lambda = %.5f)\n",
           sumT / kN, 1.0 / lambda);
    CheckHistogram("exponential", buf, kN, 0.0f, 3.0f,
                   [lambda](float t) { return lambda * std::exp(-lambda * t); });

    // --- C: rejection, p(x) = 3x^2 on [0,1] ----------------------------------
    printf("\nREJECTION  p(x) = 3x^2 on [0,1]   (proposal Uniform, M = 3)\n");
    long long tries = 0;
    int got = 0;
    while (got < kN) {
        float x = rng.NextFloat();          // candidate from q = Uniform[0,1]
        ++tries;
        // accept with probability p(x)/(M*q(x)) = 3x^2 / 3 = x^2
        if (rng.NextFloat() < x * x) buf[got++] = x;
    }
    printf("    acceptance rate = %.3f   (theory 1/M = %.3f)\n",
           (double)kN / tries, 1.0 / 3.0);
    CheckHistogram("p(x) = 3x^2", buf, kN, 0.0f, 1.0f,
                   [](float x) { return 3.0f * x * x; });

    printf("\nMeasured bars track the predicted density in every case. Inversion is\n"
           "cheap and exact when the CDF inverts; rejection always works but pays\n"
           "for it in thrown-away candidates (here ~2 of every 3 tries are wasted).\n");
    return 0;
}
