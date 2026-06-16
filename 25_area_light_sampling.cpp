// 25 — Sampling area lights: triangles and spheres
// =================================================
//
// To light a point with next-event estimation (example 24) you must draw a
// uniform (or better) sample ON the light. Different light shapes need different
// maps. Two that every renderer needs:
//
// TRIANGLE (the universal mesh-light primitive). A uniform point uses barycentric
// coordinates with a square-root warp:
//   su = sqrt(u1);  b0 = 1 - su;  b1 = u2 * su;  b2 = 1 - b0 - b1;
//   P  = b0*A + b1*B + b2*C.
// The sqrt is the Jacobian fix (example 23) for the triangle's shape; without it
// points bunch toward one vertex. Check: the expected barycentric coordinate is
// 1/3 for each vertex, and the mean sample position is the centroid.
//
// SPHERE, sampled over the SOLID ANGLE it subtends (not its surface). From a
// shading point a distance d from a sphere of radius R, the sphere fills a cone
// of half-angle theta_max with sin(theta_max) = R/d. Sample directions uniformly
// inside that cone:
//   cos(theta) = 1 - u1*(1 - cos theta_max);   phi = 2*pi*u2;   pdf = 1/(2*pi*(1-cos theta_max)).
// This is far better than sampling the sphere's surface, because it never
// generates a direction that misses the sphere or points at its hidden back.
// Check: estimating the unshadowed irradiance from an overhead sphere gives the
// closed form  I = pi * (R/d)^2,  and cone sampling has dramatically lower
// variance than naive uniform-hemisphere sampling when the sphere is small.

#include <cstdio>
#include <cmath>
#include "vec3.h"
#include "mc_random.h"

static const float PI = 3.14159265358979323846f;

struct Acc {
    double mean = 0, m2 = 0; long long n = 0;
    void add(double x) { ++n; double d = x - mean; mean += d / n; m2 += d * (x - mean); }
    double var() const { return n > 1 ? m2 / (n - 1) : 0.0; }
};

int main() {
    Rng rng(/*sequence=*/25);

    // --- Triangle: uniform barycentric sampling. ----------------------------
    Vec3 A(0, 0, 0), B(2, 0, 0), C(0, 3, 0);
    const long long Nt = 4000000;
    double sb0 = 0, sb1 = 0, sb2 = 0; Vec3 csum(0, 0, 0);
    bool allInside = true;
    for (long long i = 0; i < Nt; ++i) {
        float su = std::sqrt(rng.NextFloat());
        float b0 = 1.0f - su, b1 = rng.NextFloat() * su, b2 = 1.0f - b0 - b1;
        sb0 += b0; sb1 += b1; sb2 += b2;
        Vec3 P = A * b0 + B * b1 + C * b2;
        csum = csum + P;
        if (b0 < -1e-5f || b1 < -1e-5f || b2 < -1e-5f) allInside = false;
    }
    printf("Triangle uniform sampling (%lld samples):\n", Nt);
    printf("  mean barycentric = (%.4f, %.4f, %.4f)   (each should be 0.3333)\n",
           sb0 / Nt, sb1 / Nt, sb2 / Nt);
    printf("  mean position    = (%.4f, %.4f, %.4f)\n", csum.x / Nt, csum.y / Nt, csum.z / Nt);
    printf("  centroid (A+B+C)/3 = (%.4f, %.4f, %.4f)\n",
           (A.x + B.x + C.x) / 3, (A.y + B.y + C.y) / 3, (A.z + B.z + C.z) / 3);
    printf("  all samples inside triangle: %s\n\n", allInside ? "yes" : "NO");

    // --- Sphere: solid-angle (cone) vs uniform-hemisphere sampling. ---------
    // Sphere of radius R centered overhead at distance d (axis = +z = normal).
    // Unshadowed irradiance I = integral over directions to the sphere of
    // cos(theta) dω = pi * (R/d)^2 exactly.
    double R = 0.2, d = 1.0;
    double cosMax = std::sqrt(1.0 - (R / d) * (R / d));   // cos(theta_max), sin=R/d
    double I = PI * (R / d) * (R / d);

    Acc cone, hemi;
    const long long Ns = 4000000;
    for (long long i = 0; i < Ns; ++i) {
        // cone sampling: direction within the sphere's cone, always useful.
        double ct = 1.0 - rng.NextFloat() * (1.0 - cosMax);
        double st = std::sqrt(std::fmax(0.0, 1.0 - ct * ct));
        double ph = 2 * PI * rng.NextFloat();
        Vec3 wc(st * std::cos(ph), st * std::sin(ph), ct);          // theta from +z
        double pdfCone = 1.0 / (2 * PI * (1.0 - cosMax));
        cone.add(wc.z / pdfCone);                                   // cos(theta)/pdf

        // uniform-hemisphere sampling: only directions inside the cone hit it.
        double z = rng.NextFloat(), r = std::sqrt(std::fmax(0.0, 1.0 - z * z));
        double ph2 = 2 * PI * rng.NextFloat();
        Vec3 wh(r * std::cos(ph2), r * std::sin(ph2), z);
        double hit = (wh.z >= cosMax) ? 1.0 : 0.0;                  // within the cone?
        hemi.add(hit * wh.z / (1.0 / (2 * PI)));                    // cos/pdf if it hits
    }

    printf("Sphere light (R=%.2f, d=%.2f): unshadowed irradiance, exact I = pi(R/d)^2 = %.6f\n", R, d, I);
    printf("  cone (solid-angle)  mean %.6f  variance %.4e\n", cone.mean, cone.var());
    printf("  uniform hemisphere  mean %.6f  variance %.4e\n", hemi.mean, hemi.var());
    printf("  variance reduction: %.1fx\n", hemi.var() / cone.var());
    printf("  subtended solid angle = 2pi(1-cos theta_max) = %.6f sr\n", 2 * PI * (1.0 - cosMax));

    printf("\nThe triangle map is uniform (barycentrics average 1/3); cone sampling a\n"
           "sphere recovers the exact irradiance with far less variance than firing\n"
           "rays at the whole hemisphere and hoping to hit -- the smaller the light,\n"
           "the bigger the win. These are the per-shape samplers NEE (24) calls.\n");
    return 0;
}
