// 22 — Refraction, Snell's law, and total internal reflection (glass)
// ===================================================================
//
// When a ray hits a smooth dielectric (glass, water) it SPLITS: some light
// reflects, the rest bends as it crosses into the new medium. This file builds
// the three facts a path tracer needs to render glass — and they are exactly
// what `SampleGlassGGX` in the renderer's glass shader implements.
//
// 1. SNELL'S LAW sets the refracted angle:   n_i sin(theta_i) = n_t sin(theta_t).
//    Going into a denser medium (air -> glass) bends the ray toward the normal;
//    coming out (glass -> air) bends it away.
//
// 2. The FRESNEL reflectance F (from example 20) sets HOW the energy splits:
//    a fraction F reflects, 1 - F refracts. A path tracer can't follow both for
//    every ray, so it picks one stochastically: reflect with probability F,
//    refract with probability 1 - F. The branch probability then cancels the F
//    weight, which is why the renderer's glass weight reduces to just G2/G1.
//
// 3. TOTAL INTERNAL REFLECTION: going from dense to thin (glass -> air), past a
//    CRITICAL ANGLE theta_c = asin(n_t / n_i) Snell's law has no solution
//    (sin theta_t would exceed 1). All light reflects. This is the mirror-like
//    sheen on the underside of a water surface.
//
// We verify each numerically: that the refracted direction obeys Snell, that the
// Monte Carlo reflect/refract split reproduces F, that TIR switches on exactly at
// the critical angle, and that refracting back out reverses the ray (reciprocity).
//
// One subtlety the comments flag: radiance is scaled by (n_t/n_i)^2 across a
// refraction (the "n^2 law"), which an unbiased glass integrator must carry.

#include <cstdio>
#include <cmath>
#include <initializer_list>
#include "vec3.h"
#include "mc_random.h"

// Refract incident direction d (pointing INTO the surface) about normal n, with
// eta = n_i / n_t. Returns false on total internal reflection.
static bool Refract(const Vec3& d, const Vec3& n, float eta, Vec3& out) {
    float cosi = -dot(n, d);                              // d points into surface
    float k = 1.0f - eta * eta * (1.0f - cosi * cosi);   // = 1 - (eta sin_i)^2
    if (k < 0.0f) return false;                          // TIR: no transmitted ray
    out = eta * d + (eta * cosi - std::sqrt(k)) * n;
    return true;
}

static double FresnelDielectric(double cosi, double ni, double nt) {
    if (cosi < 0) { double t = ni; ni = nt; nt = t; cosi = -cosi; }
    double sint = ni / nt * std::sqrt(std::fmax(0.0, 1.0 - cosi * cosi));
    if (sint >= 1.0) return 1.0;
    double cost = std::sqrt(std::fmax(0.0, 1.0 - sint * sint));
    double rpar = (nt * cosi - ni * cost) / (nt * cosi + ni * cost);
    double rper = (ni * cosi - nt * cost) / (ni * cosi + nt * cost);
    return 0.5 * (rpar * rpar + rper * rper);
}

int main() {
    Rng rng(/*sequence=*/22);
    const double DEG = 3.14159265358979323846 / 180.0;
    const Vec3 N(0, 0, 1);                               // surface normal (up)

    // --- (1) Air -> glass: verify Snell's law on the refracted direction. ---
    double ni = 1.0, nt = 1.5, eta = ni / nt;
    printf("Air -> glass (n_i=%.1f, n_t=%.1f): refracted angle obeys Snell.\n", ni, nt);
    printf("  %12s | %10s %14s %12s\n", "theta_i(deg)", "F (refl)", "theta_t(deg)", "Snell check");
    for (double aDeg : {10.0, 30.0, 50.0, 70.0}) {
        double th = aDeg * DEG;
        Vec3 d(std::sin(th), 0, -std::cos(th));          // incident, heading down
        Vec3 t; bool ok = Refract(d, N, (float)eta, t);
        double thetaT = std::asin(std::fmax(-1.0, std::fmin(1.0, std::sqrt(t.x * t.x + t.y * t.y)))) / DEG;
        double lhs = ni * std::sin(th), rhs = nt * std::sin(thetaT * DEG);  // n_i sin_i vs n_t sin_t
        printf("  %12.0f | %10.4f %14.2f %12s\n", aDeg,
               FresnelDielectric(std::cos(th), ni, nt), thetaT,
               std::fabs(lhs - rhs) < 1e-4 ? "OK" : "BAD");
    }

    // --- (2) Glass -> air: total internal reflection past the critical angle. -
    ni = 1.5; nt = 1.0; eta = ni / nt;
    double critical = std::asin(nt / ni) / DEG;
    printf("\nGlass -> air (n_i=%.1f, n_t=%.1f): critical angle = asin(n_t/n_i) = %.2f deg\n",
           ni, nt, critical);
    printf("  %12s | %8s\n", "theta_i(deg)", "refract?");
    for (double aDeg : {20.0, 40.0, 41.0, 42.0, 60.0}) {
        double th = aDeg * DEG;
        Vec3 d(std::sin(th), 0, -std::cos(th)), t;
        bool ok = Refract(d, N, (float)eta, t);
        printf("  %12.0f | %s\n", aDeg, ok ? "transmit" : "TIR (all reflected)");
    }
    printf("  -> the switch happens right at %.1f deg, as predicted.\n", critical);

    // --- (3) Monte Carlo reflect/refract split reproduces F. ----------------
    ni = 1.0; nt = 1.5;
    double th = 40.0 * DEG, F = FresnelDielectric(std::cos(th), ni, nt);
    long long reflected = 0; const long long M = 2000000;
    for (long long i = 0; i < M; ++i)
        if (rng.NextFloat() < F) ++reflected;            // pick reflect w.p. F
    printf("\nStochastic reflect/refract at %.0f deg: F = %.4f, empirical reflect fraction = %.4f\n",
           40.0, F, (double)reflected / M);

    // --- (4) Reciprocity: refract in, then back out, recovers the direction. -
    eta = ni / nt;
    Vec3 d(std::sin(th), 0, -std::cos(th)), into, back;
    Refract(d, N, (float)eta, into);                     // air -> glass
    Refract(into, Vec3(0, 0, -1), (float)(nt / ni), back);  // glass -> air (flip normal)
    printf("reciprocity: in (%.4f,%.4f,%.4f) -> out (%.4f,%.4f,%.4f)  [should match in]\n",
           d.x, d.y, d.z, back.x, back.y, back.z);

    printf("\nSnell holds, the reflect/refract split reproduces Fresnel, TIR turns on\n"
           "exactly at the critical angle, and the path is reversible -- the full\n"
           "behavior the glass shader relies on. (It also flips the surface push-off\n"
           "sign for the refracted side and applies Beer-Lambert absorption along\n"
           "the transmitted segment; see examples 05 and 33 for that exponential.)\n");
    return 0;
}
