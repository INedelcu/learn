// 39 — Pixel reconstruction filters: turning samples into a pixel
// ===============================================================
//
// A pixel's value is not just the average of the samples that landed in it -- it
// is a weighted average defined by a RECONSTRUCTION FILTER centered on the pixel.
// The filter decides how samples (possibly from neighboring pixels) combine, and
// the choice trades sharpness against aliasing and ringing. This is the last step
// of the imaging pipeline, the sibling of the Gaussian sampling in example 12.
//
// Four classic filters (here as 1D kernels; renderers use the 2D product):
//   * BOX     -- flat, width 1. Simplest, but blocky and prone to aliasing.
//   * TENT    -- linear falloff. Smoother, cheap.
//   * GAUSSIAN-- smooth, soft; blurs a little but no ringing.
//   * MITCHELL-NETRAVALI -- has small NEGATIVE lobes, which sharpen edges at the
//     cost of slight overshoot (ringing). The tuned (B=C=1/3) default is the
//     standard "good looking" choice.
//
// We do two things. (1) Verify each kernel is normalized (integrates to 1) -- a
// filter that doesn't sum to 1 changes image brightness. (2) Reconstruct a step
// edge (0 on the left, 1 on the right) by convolving each filter across it, and
// print the transition profile. You can read the character of each filter straight
// off the numbers: the box gives a short hard ramp, the Gaussian a wide soft one,
// and Mitchell a crisp ramp that slightly overshoots past 0 and 1 -- the ringing
// its negative lobes produce.

#include <cstdio>
#include <cmath>

static float Box(float x)  { return std::fabs(x) < 0.5f ? 1.0f : 0.0f; }
static float Tent(float x) { x = std::fabs(x); return x < 1.0f ? 1.0f - x : 0.0f; }
static float Gauss(float x) {                       // sigma = 0.5, negligible past radius 2
    if (std::fabs(x) > 2.0f) return 0.0f;
    return 0.7978846f * std::exp(-2.0f * x * x);     // 0.7978846 = 1/(sigma*sqrt(2pi)) -> integrates to ~1
}
static float Mitchell(float x) {                    // B = C = 1/3, radius 2 (already normalized)
    const float B = 1.0f / 3.0f, C = 1.0f / 3.0f;
    x = std::fabs(x);
    float x2 = x * x, x3 = x2 * x;
    if (x < 1.0f) return ((12 - 9 * B - 6 * C) * x3 + (-18 + 12 * B + 6 * C) * x2 + (6 - 2 * B)) / 6.0f;
    if (x < 2.0f) return ((-B - 6 * C) * x3 + (6 * B + 30 * C) * x2 + (-12 * B - 48 * C) * x + (8 * B + 24 * C)) / 6.0f;
    return 0.0f;
}

// Numerically integrate a filter over [-radius, radius].
template <typename F> static double Norm(F f, double radius) {
    const int M = 200000; double s = 0, dx = 2 * radius / M;
    for (int i = 0; i < M; ++i) s += f((float)(-radius + (i + 0.5) * dx));
    return s * dx;
}

// Reconstruct a step edge s(t)=[t>=0] at center p: integral s(t) f(t-p) dt / norm.
template <typename F> static double EdgeValue(F f, double p, double radius, double norm) {
    const int M = 4000; double s = 0, dx = 2 * radius / M;
    for (int i = 0; i < M; ++i) {
        double t = p - radius + (i + 0.5) * dx;
        if (t >= 0) s += f((float)(t - p));         // s(t)=1 only for t>=0
    }
    return s * dx / norm;
}

int main() {
    printf("Filter normalization (each kernel must integrate to 1):\n");
    double nB = Norm(Box, 0.5), nT = Norm(Tent, 1.0), nG = Norm(Gauss, 2.0), nM = Norm(Mitchell, 2.0);
    printf("  box %.4f   tent %.4f   gaussian %.4f   mitchell %.4f\n\n", nB, nT, nG, nM);

    printf("Reconstructing a step edge (0 left, 1 right). Center position p sweeps\n");
    printf("across the edge; values <0 or >1 are ringing (overshoot):\n\n");
    printf("  %6s | %9s %9s %9s %9s\n", "p", "box", "tent", "gauss", "mitchell");
    for (double p = -1.5; p <= 1.5001; p += 0.25) {
        printf("  %6.2f | %9.4f %9.4f %9.4f %9.4f\n", p,
               EdgeValue(Box, p, 0.5, nB), EdgeValue(Tent, p, 1.0, nT),
               EdgeValue(Gauss, p, 2.0, nG), EdgeValue(Mitchell, p, 2.0, nM));
    }

    printf("\nRead the columns: the box snaps over a narrow 1-pixel ramp (hard, aliasing-\n"
           "prone), the Gaussian eases over a wide soft ramp (no ringing but blurry),\n"
           "and Mitchell makes a crisp ramp that dips slightly below 0 and pokes above\n"
           "1 near the edge -- the overshoot from its negative lobes, which is what\n"
           "makes it look sharp. Choosing the reconstruction filter is the final\n"
           "sharpness-vs-aliasing knob on a rendered image.\n");
    return 0;
}
