// 20 — Fresnel reflectance: the exact equations vs Schlick's approximation
// ========================================================================
//
// Every surface reflects more light at grazing angles — that bright rim on the
// edge of a sphere, the mirror-like sheen of a road at a low sun. That angular
// ramp is FRESNEL reflectance F(theta): the fraction of light reflected (rather
// than transmitted/absorbed) as a function of the angle between the incoming
// ray and the surface normal. It is a multiplier on essentially every BRDF, and
// the renderer evaluates it constantly (Schlick form, in Utils.hlsl/BRDF.hlsl).
//
// The EXACT equations come from electromagnetics (the boundary conditions on a
// plane interface):
//   * Dielectrics (glass, water, plastic): a function of the two indices of
//     refraction n_i, n_t and the angle, averaging the two polarizations.
//   * Conductors (metals): the same idea but with a COMPLEX index n + i k; the
//     absorption k is why metals are reflective and tinted.
//
// SCHLICK'S APPROXIMATION replaces all of that with one cheap formula:
//   F(theta) = F0 + (1 - F0) * (1 - cos theta)^5,
// where F0 is the reflectance at normal incidence (theta = 0). It is exact at
// both endpoints (0 and 90 degrees) by construction; the question this file
// answers is how good it is in between.
//
// We tabulate exact vs Schlick for (a) glass (n = 1.5) and (b) a metal, and
// report the largest error. Takeaways you can read off the numbers: Schlick is
// superb for dielectrics (sub-percent error), and noticeably looser for metals
// in the mid-angles — which is why physically-based renderers either accept that
// small error or feed Schlick a tuned F0, exactly as this renderer does
// (F0 from IOR for dielectrics, from base color for metals).

#include <cstdio>
#include <cmath>

// Exact unpolarized Fresnel reflectance for a dielectric interface.
static double FresnelDielectric(double cosi, double ni, double nt) {
    if (cosi < 0) { double t = ni; ni = nt; nt = t; cosi = -cosi; }   // crossing the other way
    double sint = ni / nt * std::sqrt(std::fmax(0.0, 1.0 - cosi * cosi));
    if (sint >= 1.0) return 1.0;                                       // total internal reflection
    double cost = std::sqrt(std::fmax(0.0, 1.0 - sint * sint));
    double rpar = (nt * cosi - ni * cost) / (nt * cosi + ni * cost);
    double rper = (ni * cosi - nt * cost) / (ni * cosi + nt * cost);
    return 0.5 * (rpar * rpar + rper * rper);
}

// Exact unpolarized Fresnel reflectance for a conductor (complex index n + i k).
static double FresnelConductor(double cosi, double n, double k) {
    if (cosi < 0) cosi = 0;
    double c2 = cosi * cosi, s2 = 1.0 - c2;
    double n2 = n * n, k2 = k * k;
    double t0 = n2 - k2 - s2;
    double a2b2 = std::sqrt(std::fmax(0.0, t0 * t0 + 4.0 * n2 * k2));
    double t1 = a2b2 + c2;
    double a = std::sqrt(std::fmax(0.0, 0.5 * (a2b2 + t0)));
    double t2 = 2.0 * a * cosi;
    double Rs = (t1 - t2) / (t1 + t2);
    double t3 = c2 * a2b2 + s2 * s2;
    double t4 = t2 * s2;
    double Rp = Rs * (t3 - t4) / (t3 + t4);
    return 0.5 * (Rp + Rs);
}

static double Schlick(double cosi, double F0) {
    double m = 1.0 - cosi;
    return F0 + (1.0 - F0) * (m * m) * (m * m) * m;   // (1-cos)^5
}

int main() {
    const double angles[] = {0, 15, 30, 45, 60, 75, 85, 89};
    const double DEG = 3.14159265358979323846 / 180.0;

    // --- (a) Glass, n = 1.5 (dielectric). F0 = ((nt-ni)/(nt+ni))^2 = 0.04. ---
    double ni = 1.0, nt = 1.5;
    double F0d = ((nt - ni) / (nt + ni)) * ((nt - ni) / (nt + ni));
    printf("Dielectric glass (n_i=%.1f, n_t=%.1f).  Schlick F0 = %.4f\n", ni, nt, F0d);
    printf("  %9s | %10s %10s %10s\n", "angle(deg)", "exact", "schlick", "abs err");
    double maxErrD = 0.0;
    for (double ad : angles) {
        double c = std::cos(ad * DEG);
        double ex = FresnelDielectric(c, ni, nt), sc = Schlick(c, F0d);
        maxErrD = std::fmax(maxErrD, std::fabs(ex - sc));
        printf("  %9.0f | %10.5f %10.5f %10.5f\n", ad, ex, sc, std::fabs(ex - sc));
    }
    printf("  max abs error = %.5f  (Schlick is excellent for dielectrics)\n\n", maxErrD);

    // --- (b) A metal (conductor), n=0.2, k=3.0 per channel. ------------------
    double n = 0.2, k = 3.0;
    double F0m = ((n - 1.0) * (n - 1.0) + k * k) / ((n + 1.0) * (n + 1.0) + k * k);
    printf("Conductor metal (n=%.2f, k=%.2f).  Schlick F0 = %.4f\n", n, k, F0m);
    printf("  %9s | %10s %10s %10s\n", "angle(deg)", "exact", "schlick", "abs err");
    double maxErrM = 0.0;
    for (double ad : angles) {
        double c = std::cos(ad * DEG);
        double ex = FresnelConductor(c, n, k), sc = Schlick(c, F0m);
        maxErrM = std::fmax(maxErrM, std::fabs(ex - sc));
        printf("  %9.0f | %10.5f %10.5f %10.5f\n", ad, ex, sc, std::fabs(ex - sc));
    }
    printf("  max abs error = %.5f  (larger -- conductors bend differently)\n\n", maxErrM);

    // --- endpoint sanity checks --------------------------------------------
    printf("Endpoint checks (Schlick is exact here by construction):\n");
    printf("  glass  @0deg : exact %.5f, schlick %.5f, F0 %.5f\n",
           FresnelDielectric(1.0, ni, nt), Schlick(1.0, F0d), F0d);
    printf("  glass @90deg : exact %.5f, schlick %.5f (both -> 1)\n",
           FresnelDielectric(std::cos(89.999 * DEG), ni, nt), Schlick(std::cos(89.999 * DEG), F0d));

    printf("\nFresnel is the angular ramp that brightens every surface toward grazing\n"
           "angles. Schlick trades a tiny dielectric error (and a modest metallic\n"
           "one) for a single pow() -- a trade the renderer makes everywhere it\n"
           "needs reflectance.\n");
    return 0;
}
