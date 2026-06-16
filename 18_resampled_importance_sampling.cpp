// 18 — Resampled Importance Sampling (RIS) and reservoirs (the heart of ReSTIR)
// ============================================================================
//
// Importance sampling (06) needs a pdf you can SAMPLE — usually meaning you can
// invert its CDF. But the ideal target in rendering, the full integrand
// f = BRDF * incoming light * visibility * cosine, has no invertible CDF; you can
// only EVALUATE it. RESAMPLED IMPORTANCE SAMPLING (Talbot 2005) gets you samples
// approximately distributed by such a target using only cheap proposals plus
// evaluations of the target. It is the engine under ReSTIR (Bitterli 2020), today's
// state of the art for many-light direct lighting.
//
// THE RIS RECIPE for estimating I = integral of f(x) dx, given:
//   * a "target" p_hat(x) >= 0 we can EVALUATE (ideally shaped like f), and
//   * a cheap "source" pdf q(x) we can SAMPLE (here: uniform).
//   1. Draw M candidates X_1..X_M ~ q.
//   2. Give each a resampling weight  w_i = p_hat(X_i) / q(X_i).
//   3. Pick ONE survivor Y = X_j with probability  w_j / sum_k w_k.
//   4. Estimate:  I_hat = ( f(Y) / p_hat(Y) ) * ( (1/M) * sum_k w_k ).
//
// This is unbiased for every M. At M = 1 it is just ordinary importance sampling
// with the source q. As M grows, Y becomes distributed ever closer to
// p_hat / integral(p_hat), so the estimator's variance falls toward that of
// *ideal* importance sampling with p_hat — WITHOUT ever inverting p_hat's CDF.
// You trade M cheap evaluations for the ability to sample an un-invertible target.
//
// We demonstrate on a product integrand f(x) = p_hat(x) * h(x) where
//   p_hat(x) = e^(8x)   (sharp, dominates the shape — the part we want to follow)
//   h(x)     = 0.5 + x  (a smooth leftover factor)
// so f(Y)/p_hat(Y) = h(Y). Uniform sampling chokes on p_hat's huge dynamic range;
// RIS with growing M tracks p_hat and squeezes the variance down to h's alone.
//
// Then the same selection done as a streaming WEIGHTED RESERVOIR (O(1) memory,
// one pass) — this is the data structure ReSTIR stores per pixel and REUSES
// across neighboring pixels and across frames (each reuse is itself a tiny RIS
// over reservoirs), which is what makes it so powerful in real time.

#include <cstdio>
#include <cmath>
#include <initializer_list>
#include "mc_random.h"

static const double A = 8.0;
static double pHat(double x) { return std::exp(A * x); }   // target (evaluable, sharp)
static double h(double x)    { return 0.5 + x; }           // smooth leftover factor
static double f(double x)    { return pHat(x) * h(x); }    // the integrand

int main() {
    Rng rng(/*sequence=*/18);

    // Reference value by fine deterministic quadrature (midpoint rule).
    double I = 0.0; const int Q = 4000000;
    for (int i = 0; i < Q; ++i) I += f((i + 0.5) / Q);
    I /= Q;

    const double Z = (std::exp(A) - 1.0) / A;   // integral of p_hat over [0,1]
    const long long kTrials = 1000000;

    printf("Estimating I = integral_0^1 e^(8x)*(0.5+x) dx = %.4f\n", I);
    printf("Source q = uniform; target p_hat = e^(8x) (cannot be the final sampler\n"
           "directly in real rendering -- here we only EVALUATE it).\n\n");
    printf("%-26s %14s %16s\n", "method", "mean (-> I)", "variance");

    // --- RIS for a sweep of candidate counts M. -----------------------------
    for (int M : {1, 2, 4, 8, 16, 32, 64}) {
        double mean = 0.0, m2 = 0.0;
        for (long long t = 0; t < kTrials; ++t) {
            double wsum = 0.0, chosen = 0.0;
            for (int i = 0; i < M; ++i) {
                double x = rng.NextFloat();          // X ~ q (uniform, q = 1)
                double w = pHat(x);                  // w_i = p_hat(x)/q(x)
                wsum += w;
                if (rng.NextFloat() < w / wsum) chosen = x;   // reservoir-style pick
            }
            double est = h(chosen) * (wsum / M);     // (f(Y)/p_hat(Y)) * mean(w)
            double d = est - mean; mean += d / (t + 1); m2 += d * (est - mean);
        }
        const char* tag = (M == 1) ? "RIS M=1 (= plain IS)" : "RIS";
        char label[40]; std::snprintf(label, sizeof label, "%s  M=%d", tag, M);
        printf("%-26s %14.4f %16.4e\n", M == 1 ? "RIS M=1 (=uniform MC)" : label, mean, m2 / (kTrials - 1));
    }

    // --- The ceiling RIS is approaching: ideal IS sampling p_hat exactly. ---
    // p_hat is invertible here (it is just an exponential), so we CAN sample it
    // directly to show the limit. Inverse CDF of p_hat/Z: x = ln(1+u(e^A-1))/A.
    {
        double mean = 0.0, m2 = 0.0;
        for (long long t = 0; t < kTrials; ++t) {
            double u = rng.NextFloat();
            double x = std::log(1.0 + u * (std::exp(A) - 1.0)) / A;   // X ~ p_hat/Z
            double est = f(x) / (pHat(x) / Z);                        // = h(x)*Z
            double d = est - mean; mean += d / (t + 1); m2 += d * (est - mean);
        }
        printf("%-26s %14.4f %16.4e   <- the limit RIS converges to\n",
               "ideal IS (sample p_hat)", mean, m2 / (kTrials - 1));
    }

    printf("\nM=1 is ordinary uniform-source sampling (huge variance from e^(8x)).\n"
           "Each doubling of M moves the survivor closer to being distributed by\n"
           "p_hat, and the variance marches down toward the ideal-IS value -- all\n"
           "without inverting p_hat. That is RIS: importance sampling a target you\n"
           "can only evaluate.\n");

    // --- The streaming form: a weighted reservoir (what ReSTIR stores). -----
    // Identical result to the array RIS above, but O(1) memory and one pass:
    // update with each candidate, keeping a running (survivor, weight-sum).
    struct Reservoir {
        double y = 0.0, wsum = 0.0; int M = 0;
        void update(double x, double w, Rng& rng) {
            wsum += w; ++M;
            if (rng.NextFloat() < w / wsum) y = x;   // replace survivor w.p. w/wsum
        }
        double estimate() const { return h(y) * (wsum / M); }
    };
    double mean = 0.0, m2 = 0.0; const int Mres = 16;
    for (long long t = 0; t < kTrials; ++t) {
        Reservoir r;
        for (int i = 0; i < Mres; ++i) { double x = rng.NextFloat(); r.update(x, pHat(x), rng); }
        double est = r.estimate();
        double d = est - mean; mean += d / (t + 1); m2 += d * (est - mean);
    }
    printf("\nWeighted reservoir (streaming RIS), M=%d:  mean = %.4f, variance = %.4e\n",
           Mres, mean, m2 / (kTrials - 1));
    printf("Matches array-RIS at M=%d (same algorithm, O(1) memory). ReSTIR then\n"
           "REUSES these reservoirs across neighbor pixels and across frames -- each\n"
           "merge is another RIS step -- to gather hundreds of effective samples per\n"
           "pixel from just a few traced rays. This is the modern descendant of the\n"
           "importance-sampling (06) and MIS (09) ideas earlier in the series.\n", Mres);
    return 0;
}
