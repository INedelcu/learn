// 19 — Importance-sampling a glossy BRDF: GGX with visible-normal sampling
// ========================================================================
//
// Examples 06 and 08 importance-sampled toy functions and the cosine lobe. Real
// specular reflection is harder: the GGX microfacet BRDF concentrates outgoing
// energy into a tight lobe around the mirror direction, and that lobe moves with
// the view. Firing rays uniformly (example 08's uniform hemisphere) almost never
// hits it, so the variance is enormous. The fix the renderer uses is HEITZ 2018
// VISIBLE-NORMAL (VNDF) SAMPLING (this is `SampleSpecularGGX` in BRDF.hlsl).
//
// The microfacet specular BRDF is
//   f_r(wo, wi) = D(h) * G2(wo,wi) * F / (4 (n·wo)(n·wi)),   h = normalize(wo+wi).
// VNDF sampling draws a microfacet normal h from the normals visible to wo, then
// reflects wo about h to get wi. The beautiful part is what happens to the Monte
// Carlo weight f_r·cosθ_i / pdf(wi): almost everything cancels and it collapses to
//
//   weight = F * G2(wo,wi) / G1(wo)             (Heitz 2018, eq. 19).
//
// This file proves that identity numerically and shows the variance win:
//   * We estimate the single-scatter directional albedo
//       E(wo) = integral over the hemisphere of f_r(wo,wi) cosθ_i dwi   (with F=1)
//     TWO independent ways:
//       (A) brute force: uniform hemisphere sampling of f_r·cosθ_i (reference);
//       (B) VNDF: just the average of G2/G1.
//     They must agree — that agreement validates both the BRDF and the sampler.
//   * We compare their variances. VNDF wins by orders of magnitude, especially
//     for smooth surfaces where the uniform sampler keeps missing the lobe.
//
// (We set F=1 to isolate the geometry; E(wo) < 1 then exposes the single-scatter
// energy loss that example 21's white-furnace test studies.)

#include <cstdio>
#include <cmath>
#include <initializer_list>
#include "ggx.h"
#include "mc_random.h"

struct Acc {
    double mean = 0, m2 = 0; long long n = 0;
    void add(double x) { ++n; double d = x - mean; mean += d / n; m2 += d * (x - mean); }
    double var() const { return n > 1 ? m2 / (n - 1) : 0.0; }
};

// Deterministic reference for E(wo): a fine-grid quadrature of the hemisphere
// integral. Unlike a uniform Monte Carlo estimate (which is hopeless for sharp
// specular lobes -- that is the very problem this example is about), quadrature
// stays accurate at any roughness, so it is a trustworthy yardstick.
static double QuadratureAlbedo(const Vec3& wo, float a) {
    const int Nt = 2000, Np = 2000;
    const float PI = kPiGGX;
    double sum = 0.0;
    for (int it = 0; it < Nt; ++it) {
        float theta = (it + 0.5f) / Nt * (0.5f * PI);
        float ct = std::cos(theta), st = std::sin(theta);
        for (int ip = 0; ip < Np; ++ip) {
            float phi = (ip + 0.5f) / Np * (2.0f * PI);
            Vec3 wi(st * std::cos(phi), st * std::sin(phi), ct);
            Vec3 h = normalize(wo + wi);
            sum += GgxBrdfNoF(wo, wi, h, a) * ct * st;        // f_r * cos(theta_i) * sin(theta_i)
        }
    }
    return sum * (0.5 * PI / Nt) * (2.0 * PI / Np);
}

int main() {
    Rng rng(/*sequence=*/19);
    const float PI = kPiGGX;
    const long long N = 2000000;

    printf("GGX single-scatter directional albedo E(wo) (Fresnel set to 1).\n");
    printf("Validating VNDF: its estimator (average of G2/G1) must match a\n");
    printf("deterministic quadrature of the BRDF, and beat uniform MC's variance.\n\n");
    printf("%9s %7s | %12s %12s %8s | %16s\n",
           "theta(deg)", "alpha", "E quad", "E vndf", "match?", "var uniform/vndf");

    for (float thetaDeg : {20.0f, 60.0f}) {
        float th = thetaDeg * PI / 180.0f;
        Vec3 wo(std::sin(th), 0.0f, std::cos(th));
        for (float a : {0.1f, 0.3f, 0.6f}) {
            double Eref = QuadratureAlbedo(wo, a);            // trusted reference
            Acc uni, vndf;
            for (long long i = 0; i < N; ++i) {
                // VNDF estimator: weight is just G2/G1 (F = 1).
                Vec3 hv = SampleGgxVNDF(wo, a, rng.NextFloat(), rng.NextFloat());
                Vec3 wiv = GgxReflect(wo, hv);
                vndf.add(wiv.z > 0.0f ? GgxG2(wo, wiv, a) / GgxG1(wo, a) : 0.0);

                // Uniform hemisphere estimator (for the variance comparison only).
                float z = rng.NextFloat();                    // cos(theta_i)
                float r = std::sqrt(std::fmax(0.0f, 1.0f - z * z));
                float ph = 2.0f * PI * rng.NextFloat();
                Vec3 wi(r * std::cos(ph), r * std::sin(ph), z);
                Vec3 h = normalize(wo + wi);
                uni.add(2.0 * PI * GgxBrdfNoF(wo, wi, h, a) * wi.z);
            }
            bool match = std::fabs(vndf.mean - Eref) < 0.005 * std::fmax(0.05, Eref);
            printf("%9.0f %7.2f | %12.5f %12.5f %8s | %14.0fx\n",
                   thetaDeg, a, Eref, vndf.mean, match ? "yes" : "NO",
                   vndf.var() > 0 ? uni.var() / vndf.var() : 0.0);
        }
    }

    printf("\nThe VNDF estimate matches the quadrature reference to ~3 digits,\n"
           "confirming the F*G2/G1 weight. And its variance is hundreds to\n"
           "thousands of times below uniform MC's -- the gap widening as the surface\n"
           "gets smoother (small alpha), exactly where uniform sampling almost never\n"
           "lands in the specular lobe. (Uniform MC is so noisy there that it makes\n"
           "a useless reference, which is why we trust quadrature instead.) This is\n"
           "why the path tracer samples the BRDF. Note E(wo) < 1: the missing energy\n"
           "is the single-scatter loss that example 21 measures.\n");
    return 0;
}
