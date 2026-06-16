// 31 — Efficiency: variance alone is the wrong scoreboard
// =======================================================
//
// Every earlier example compared techniques by VARIANCE. But a method that halves
// the variance is no good if it takes ten times as long per sample -- in that time
// the simpler method could have drawn ten times as many samples and ended up
// ahead. The honest figure of merit is EFFICIENCY (Veach's definition):
//
//   efficiency  =  1 / ( variance-per-sample  *  time-per-sample ).
//
// Equivalently: for a fixed time budget, the achievable error variance is
// (variance-per-sample * time-per-sample) / total_time, so the product V*T is
// what you want to minimize. Two estimators with identical variance are NOT
// equally good if one is more expensive; and a lower-variance estimator can be
// the WORSE choice if it is slow enough.
//
// We estimate integral_0^1 e^x dx three ways, MEASURING wall-clock time:
//   A) uniform sampling          -- cheap, high variance;
//   B) importance sampling       -- one extra sqrt, ~9x less variance (example 06);
//   C) "expensive" importance    -- same samples as B but with a costly pdf
//      evaluation (here simulated with extra transcendental work), standing in for
//      a heavy tabulated/measured pdf.
// B should win. C has the SAME low variance as B yet can be LESS efficient than
// even the cheap uniform method -- the cautionary tale.

#include <cstdio>
#include <cmath>
#include <chrono>
#include "mc_random.h"

// One measurement: variance per sample, elapsed seconds, and the mean estimate.
// (Declared at file scope so every run() call returns the SAME type.)
struct Result { double var, secs, mean; };

int main() {
    Rng rng(/*sequence=*/31);
    const double kE = 2.71828182845904523536;
    const double I = kE - 1.0;
    const long long N = 30000000;
    double sink = 0.0;                      // keeps the "expensive" work from being optimized away

    // Run a sampler N times; return {variance-per-sample, seconds}.
    auto run = [&](auto sampler) {
        double mean = 0, m2 = 0;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (long long i = 0; i < N; ++i) {
            double e = sampler();
            double d = e - mean; mean += d / (i + 1); m2 += d * (e - mean);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        return Result{m2 / (N - 1), secs, mean};
    };

    // A) uniform: estimate is just f(x) = e^x.
    auto A = run([&] { return std::exp((double)rng.NextFloat()); });

    // B) importance sampling with p(x) = (2/3)(1+x); x = -1 + sqrt(1+3u).
    auto B = run([&] {
        double x = -1.0 + std::sqrt(1.0 + 3.0 * rng.NextFloat());
        double p = (2.0 / 3.0) * (1.0 + x);
        return std::exp(x) / p;
    });

    // C) same estimator as B, but with an expensive pdf evaluation (simulated).
    auto C = run([&] {
        double x = -1.0 + std::sqrt(1.0 + 3.0 * rng.NextFloat());
        double p = (2.0 / 3.0) * (1.0 + x);
        double junk = 0;                    // stand-in for a heavy pdf/table lookup
        for (int k = 1; k <= 40; ++k) junk += std::sin(x * k) * std::cos(x * k);
        sink += junk;
        return std::exp(x) / p;
    });

    auto eff = [](double var, double secs) { return 1.0 / (var * (secs)); };  // 1/(V*T_total) ~ relative

    printf("Estimating integral_0^1 e^x dx = %.6f with %lld samples each.\n\n", I, N);
    printf("  %-22s %10s %12s %14s %12s\n", "method", "estimate", "variance", "ns/sample", "efficiency");
    double effA = eff(A.var, A.secs);
    auto row = [&](const char* nm, const Result& r) {
        printf("  %-22s %10.6f %12.4e %14.2f %12.2f\n", nm, r.mean, r.var,
               r.secs / N * 1e9, eff(r.var, r.secs) / effA);
    };
    row("A uniform", A);
    row("B importance", B);
    row("C expensive importance", C);

    printf("\n(efficiency shown relative to A; higher is better)\n");
    printf("\nB wins: a touch more cost per sample buys ~9x less variance, so for a\n"
           "fixed time budget it is far ahead. C has the SAME low variance as B but\n"
           "its expensive per-sample work can drop its efficiency below even cheap\n"
           "uniform sampling. That is the lesson: compare techniques by variance x\n"
           "time, never variance alone -- a fancy sampler that is too slow loses to\n"
           "brute force. (sink=%.3g, printed only so the cost isn't optimized away.)\n", sink);
    return 0;
}
