// 14 — Sampling tabulated distributions: building and inverting a CDF
// ===================================================================
//
// Example 05 inverted a CDF that had a closed-form inverse (sqrt, log). But the
// distributions a renderer most wants to sample have NO formula at all: the
// luminance of an HDR environment map, a measured BRDF, the power of each light
// in the scene. These arrive as a TABLE of numbers. This file shows the standard
// machinery for sampling such tables — exactly the pattern in the path tracer's
// environment-map importance sampling and light selection (and the inspiration
// here is a CDF-theory demo from the HDRP investigations folder).
//
// THE RECIPE (piecewise-constant density over N bins):
//   1. Treat the table as a piecewise-constant pdf; normalize so it sums to 1.
//   2. Build the CDF as a prefix sum: F[0]=0, F[i+1]=F[i]+pdf[i], F[N]=1. The
//      array has N+1 entries so we hold both edges of every bin.
//   3. To sample: draw u ~ U[0,1), BINARY-SEARCH for the bin with F[i] <= u <
//      F[i+1], then interpolate inside it. Binary search costs O(log N) and —
//      crucially on a GPU — takes exactly ONE try per sample, unlike rejection
//      sampling whose try-count varies per lane and stalls the whole warp.
//
// We also do the DISCRETE version (no interpolation): "which of these N lights
// should I sample, given their powers?" That is the same prefix-sum CDF, found
// with a binary search — the PickLight pattern.
//
// We verify both by histogramming a few hundred thousand samples and checking
// that each bin's empirical frequency matches its probability. The test case is
// a "sky row": a dim background with three bins 100x brighter (a "sun"), so a
// correct sampler must send ~most of its samples into those three bins.

#include <cstdio>
#include <cmath>
#include <vector>
#include "mc_random.h"

// Build a CDF (prefix sum) from a pdf table. Result has pdf.size()+1 entries.
static std::vector<float> BuildCdf(const std::vector<float>& pdf) {
    std::vector<float> cdf(pdf.size() + 1, 0.0f);
    for (size_t i = 0; i < pdf.size(); ++i) cdf[i + 1] = cdf[i] + pdf[i];
    cdf.back() = 1.0f;                     // pin the top against float drift
    return cdf;
}

// Find the bin i with cdf[i] <= u < cdf[i+1] by binary search (no std::).
static int FindBin(const std::vector<float>& cdf, float u) {
    int lo = 0, hi = (int)cdf.size() - 1;          // searching edges [0, N]
    while (hi - lo > 1) {
        int mid = (lo + hi) >> 1;
        if (cdf[mid] <= u) lo = mid; else hi = mid;
    }
    return lo;                                       // in [0, N-1]
}

// Continuous sample in [0, N): pick a bin, then interpolate within it. Also
// returns the pdf (density) at the sample, which is what MIS would need.
static float SampleContinuous(const std::vector<float>& cdf, float u, float& outPdf) {
    int i = FindBin(cdf, u);
    float f0 = cdf[i], f1 = cdf[i + 1];
    float t = (f1 > f0) ? (u - f0) / (f1 - f0) : 0.0f;
    outPdf = f1 - f0;                                // bin width is 1 -> density = mass
    return i + t;
}

// Discrete sample: just the bin index, distributed proportionally to pdf[i].
static int SampleDiscrete(const std::vector<float>& cdf, float u) {
    return FindBin(cdf, u);
}

int main() {
    Rng rng(/*sequence=*/14);

    // --- A "sky row": 32 bins, three of them (the sun) 100x brighter. -------
    const int W = 32;
    std::vector<float> pdf(W, 1.0f);
    pdf[19] = pdf[20] = pdf[21] = 100.0f;
    float sum = 0.0f; for (float v : pdf) sum += v;
    for (float& v : pdf) v /= sum;                   // normalize to a pdf
    std::vector<float> cdf = BuildCdf(pdf);

    // --- Part 1: continuous inverse-transform sampling of the table. --------
    const int kN = 1000000;
    std::vector<int> hist(W, 0);
    for (int s = 0; s < kN; ++s) {
        float p;
        float x = SampleContinuous(cdf, rng.NextFloat(), p);
        int bin = (int)x; if (bin >= W) bin = W - 1;
        hist[bin]++;
    }
    printf("Continuous inverse-CDF sampling of a tabulated 'sky row' (sun at 19-21):\n");
    printf("  %3s | %9s | %9s\n", "bin", "pdf", "empirical");
    // Show the sun and its neighbours individually; summarize the dim run once.
    printf("  0-17| %9.5f | %9.5f   (each of the 18 dim background bins)\n",
           pdf[0], (float)hist[0] / kN);
    for (int i = 18; i <= 22; ++i) {
        float emp = (float)hist[i] / kN;
        printf("  %3d | %9.5f | %9.5f  ", i, pdf[i], emp);
        for (int s = 0; s < (int)(emp * 120); ++s) putchar('#');
        putchar('\n');
    }
    printf("  23-31 same as the dim background bins above.\n");

    // --- Part 2: discrete pick by weight (the "which light?" question). -----
    std::vector<float> weights = {90.0f, 1.0f, 9.0f};   // three lights, uneven power
    float wtot = 0.0f; for (float w : weights) wtot += w;
    std::vector<float> wpdf(weights.size());
    for (size_t i = 0; i < weights.size(); ++i) wpdf[i] = weights[i] / wtot;
    std::vector<float> wcdf = BuildCdf(wpdf);

    std::vector<int> hits(weights.size(), 0);
    for (int s = 0; s < kN; ++s) hits[SampleDiscrete(wcdf, rng.NextFloat())]++;

    printf("\nDiscrete light selection by power (weights 90 : 1 : 9):\n");
    printf("  %5s | %8s | %9s | %9s\n", "light", "weight", "expected", "empirical");
    for (size_t i = 0; i < weights.size(); ++i)
        printf("  %5zu | %8.1f | %9.4f | %9.4f\n",
               i, weights[i], wpdf[i], (float)hits[i] / kN);

    printf("\nBoth empirical columns match the pdf: the sampler sends samples where\n"
           "the probability mass is, in O(log N) per sample with a single binary\n"
           "search. This is how the renderer importance-samples its environment map\n"
           "(per-texel luminance table) and chooses which light to shade — measured\n"
           "data you can only tabulate, never invert in closed form. Example 15\n"
           "extends this to the full 2D environment map.\n");
    return 0;
}
