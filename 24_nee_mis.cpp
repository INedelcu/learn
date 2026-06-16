// 24 — Direct lighting: next-event estimation and light/BSDF MIS
// ==============================================================
//
// To shade a point you integrate the light arriving over the hemisphere times
// the BRDF. For a small bright light, blindly sampling the BRDF (example 19)
// almost never sends a ray AT the light, so the estimate is mostly zeros with
// occasional huge spikes — terrible variance. NEXT-EVENT ESTIMATION (NEE) fixes
// this by sampling a point ON the light directly and connecting to it. But NEE
// has the opposite failure: for a large light seen in a near-mirror surface, most
// light-area samples land where the BRDF is ~0, again high variance.
//
// Neither strategy wins everywhere — which is exactly the situation Multiple
// Importance Sampling (example 09) was built for. Here we apply MIS to the two
// strategies a renderer actually combines, LIGHT sampling and BSDF sampling, with
// the balance heuristic. This is Veach's canonical result and the spirit of the
// renderer's single-scattering estimators.
//
// Scene (all analytic, full visibility): a shading point at the origin with
// normal +z, a GGX glossy BRDF, and a disk light parallel to the surface centered
// on the mirror-reflection direction. The direct-lighting integral is
//
//   L = integral over the light of  f_r * Le * cos(theta_i) * cos(theta_light) / r^2  dA.
//
// We compute a deterministic quadrature reference, then compare the variance of:
//   * LIGHT only  — sample the disk area, convert to solid angle (example 23);
//   * BSDF only   — VNDF-sample a direction, see if it hits the disk;
//   * MIS         — one of each per iteration, balance-heuristic weighted.
// Two scenes flip which single strategy is good; MIS is robust to both.

#include <cstdio>
#include <cmath>
#include <initializer_list>
#include "ggx.h"
#include "mc_random.h"

static const float PI = kPiGGX;

struct Acc {
    double mean = 0, m2 = 0; long long n = 0;
    void add(double x) { ++n; double d = x - mean; mean += d / n; m2 += d * (x - mean); }
    double var() const { return n > 1 ? m2 / (n - 1) : 0.0; }
};

// Disk light, plane z = H, normal facing down, center (cx, 0, H), radius R.
struct DiskLight { double cx, H, R, Le; };

// BSDF pdf for VNDF GGX sampling: p(wi) = G1(wo) D(h) / (4 (n·wo)).
static double BsdfPdf(const Vec3& wo, const Vec3& wi, float a) {
    if (wi.z <= 0) return 0.0;
    Vec3 h = normalize(wo + wi);
    if (dot(wo, h) <= 0) return 0.0;
    return GgxG1(wo, a) * GgxD(h, a) / (4.0 * wo.z);
}
// Does direction wi from the origin hit the disk? If so return distance r.
static bool HitDisk(const Vec3& wi, const DiskLight& L, double& r) {
    if (wi.z <= 1e-6f) return false;
    double t = L.H / wi.z;                       // reach plane z = H
    double px = t * wi.x - L.cx, py = t * wi.y;
    if (px * px + py * py > L.R * L.R) return false;
    r = t;                                       // wi is unit, so |t*wi| = t
    return true;
}
static double LightPdfSA(double r, const DiskLight& L) {   // area pdf -> solid angle
    return (r * r * r) / (PI * L.R * L.R * L.H);            // r^2/(A cos), cos = H/r
}
// f_r * Le * cos(theta_i) for a direction wi (Fresnel folded to 1).
static double Contribution(const Vec3& wo, const Vec3& wi, float a, const DiskLight& L) {
    Vec3 h = normalize(wo + wi);
    return GgxBrdfNoF(wo, wi, h, a) * L.Le * wi.z;
}

int main() {
    Rng rng(/*sequence=*/24);
    float th = 30.0f * PI / 180.0f;
    Vec3 wo(std::sin(th), 0.0f, std::cos(th));            // view at 30 deg
    // Mirror-reflection direction of wo: (-sin, 0, cos). Put the light on it.
    double dd = 1.1547;
    DiskLight base{ -std::sin(th) * dd, std::cos(th) * dd, 0.0, 5.0 };  // H=cos*dd=1.0

    struct Scene { const char* name; float a; double R; };
    for (Scene sc : { Scene{"smooth surface (a=0.05), LARGE light (R=0.6)", 0.05f, 0.6},
                      Scene{"rough surface  (a=0.30), SMALL light (R=0.08)", 0.30f, 0.08} }) {
        DiskLight L = base; L.R = sc.R; float a = sc.a;

        // --- deterministic reference: quadrature over the disk area. --------
        double ref = 0.0; const int Nr = 1500, Np = 1500;
        for (int i = 0; i < Nr; ++i) {
            double rr = (i + 0.5) / Nr * L.R;
            for (int j = 0; j < Np; ++j) {
                double ph = (j + 0.5) / Np * 2 * PI;
                Vec3 P(L.cx + rr * std::cos(ph), rr * std::sin(ph), L.H);
                double r = length(P); Vec3 wi = P * (1.0f / (float)r);
                if (wi.z <= 0) continue;
                double cosL = L.H / r;
                ref += Contribution(wo, wi, a, L) * (cosL / (r * r)) * rr;
            }
        }
        ref *= (L.R / Nr) * (2 * PI / Np);

        // --- the estimators, variance measured over many samples. -----------
        Acc light, bsdf, mis, misPow;
        const long long N = 3000000;
        for (long long i = 0; i < N; ++i) {
            // LIGHT sample: uniform point on the disk.
            double r0 = L.R * std::sqrt(rng.NextFloat()), ph = 2 * PI * rng.NextFloat();
            Vec3 P(L.cx + r0 * std::cos(ph), r0 * std::sin(ph), L.H);
            double rL = length(P); Vec3 wiL = P * (1.0f / (float)rL);
            double pL = LightPdfSA(rL, L);
            double cL = (wiL.z > 0) ? Contribution(wo, wiL, a, L) / pL : 0.0;
            light.add(cL);

            // BSDF sample: VNDF direction; contributes only if it hits the light.
            Vec3 h = SampleGgxVNDF(wo, a, rng.NextFloat(), rng.NextFloat());
            Vec3 wiB = GgxReflect(wo, h);
            double rB; double cB = 0.0, pB = 0.0;
            if (wiB.z > 0 && HitDisk(wiB, L, rB)) {
                pB = BsdfPdf(wo, wiB, a);
                if (pB > 0) cB = Contribution(wo, wiB, a, L) / pB;
            }
            bsdf.add(cB);

            // MIS: combine the same two samples, weighting each technique by its
            // pdf at the OTHER technique's sample. Two heuristics:
            double pB_atL = BsdfPdf(wo, wiL, a);             // bsdf pdf at the light dir
            double cLval = (wiL.z > 0) ? Contribution(wo, wiL, a, L) : 0.0;
            double pL_atB = 0.0, cBval = 0.0;
            if (wiB.z > 0 && pB > 0) { pL_atB = LightPdfSA(rB, L); cBval = Contribution(wo, wiB, a, L); }

            // balance heuristic: w_i = p_i / sum_j p_j
            double wLb = (pL + pB_atL > 0) ? pL / (pL + pB_atL) : 0.0;
            double wBb = (pB + pL_atB > 0) ? pB / (pB + pL_atB) : 0.0;
            mis.add((cLval > 0 ? wLb * cLval / pL : 0.0) + (cBval > 0 ? wBb * cBval / pB : 0.0));

            // power heuristic (beta=2): w_i = p_i^2 / sum_j p_j^2 -- pushes the
            // weak technique's weight down harder, trimming MIS's overhead.
            double wLp = (pL * pL + pB_atL * pB_atL > 0) ? pL * pL / (pL * pL + pB_atL * pB_atL) : 0.0;
            double wBp = (pB * pB + pL_atB * pL_atB > 0) ? pB * pB / (pB * pB + pL_atB * pL_atB) : 0.0;
            misPow.add((cLval > 0 ? wLp * cLval / pL : 0.0) + (cBval > 0 ? wBp * cBval / pB : 0.0));
        }

        double best = std::fmin(light.var(), bsdf.var());
        double worst = std::fmax(light.var(), bsdf.var());
        printf("Scene: %s\n", sc.name);
        printf("  reference L = %.5f\n", ref);
        printf("  %-14s mean %.5f  variance %.4e\n", "light only", light.mean, light.var());
        printf("  %-14s mean %.5f  variance %.4e\n", "bsdf only",  bsdf.mean,  bsdf.var());
        printf("  %-14s mean %.5f  variance %.4e\n", "MIS balance", mis.mean, mis.var());
        printf("  %-14s mean %.5f  variance %.4e\n", "MIS power",  misPow.mean, misPow.var());
        printf("  -> picking the WRONG single strategy costs %.0fx variance here;\n", worst / best);
        printf("     MIS(power) stays within %.1fx of the BEST and is %.0fx better than the WORST.\n\n",
               misPow.var() / best, worst / misPow.var());
    }

    printf("All estimators are unbiased (means match the reference). The point of\n"
           "MIS is ROBUSTNESS: each single strategy is catastrophic in the scene that\n"
           "doesn't suit it (BSDF misses the small light; light sampling wastes\n"
           "samples across the large one), and you do NOT know in advance which a\n"
           "given pixel/material/light needs. MIS is automatically within a small\n"
           "factor of whichever is best -- never the disastrous one. The power\n"
           "heuristic suppresses the losing technique's weight harder than the\n"
           "balance heuristic, shrinking MIS's small overhead over the best. This is\n"
           "why production direct lighting samples both lights and BSDFs and\n"
           "MIS-combines them (example 09's heuristic, applied to real lobes).\n");
    return 0;
}
