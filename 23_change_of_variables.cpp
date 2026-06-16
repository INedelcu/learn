// 23 — Change of variables: the Jacobian, and area vs solid-angle measure
// ========================================================================
//
// Every sampling routine is a change of variables: you feed in uniform numbers
// and warp them into points/directions. The rule that keeps things unbiased is
//
//   if y = T(x) and X has density p_x, then p_y(y) = p_x(x) / |J_T(x)|,
//
// where |J_T| is the Jacobian determinant of the map (how much it stretches
// area/volume). Get the Jacobian wrong and your samples have the wrong density —
// a silent bias. Two places this bites in a renderer:
//
// PART 1 — SAMPLING A DISK. The "obvious" map r = u1, phi = 2*pi*u2 is WRONG: it
// puts equal numbers of points at every radius, so they pile up at the center
// (the Jacobian of polar coordinates is r, which this ignores). The correct
// area-uniform map is r = sqrt(u1). We also show Shirley's CONCENTRIC mapping,
// which is area-uniform AND low-distortion (it keeps nearby square samples nearby
// on the disk, so stratification survives). Disk sampling is how a renderer
// samples a thin lens (depth of field) or a disk light.
//
// PART 2 — AREA vs SOLID-ANGLE MEASURE. To light a point from an area light you
// can integrate over the light's AREA or over the SOLID ANGLE it subtends. The
// two are related by the geometry ("G") term:
//
//   dω = cos(theta_light) / r^2  dA          so   p_solidangle = p_area * r^2 / cos(theta_light).
//
// That r^2/cos factor is THE conversion behind next-event estimation (example 24).
// We verify it by estimating the solid angle a disk light subtends two ways —
// sampling its area and converting, vs a deterministic quadrature — and checking
// they agree.

#include <cstdio>
#include <cmath>
#include "vec3.h"
#include "mc_random.h"

static const double kPi = 3.14159265358979323846;

int main() {
    Rng rng(/*sequence=*/23);

    // --- PART 1: disk sampling, three maps, judged by the radius histogram. --
    // For an area-uniform disk the fraction of points with radius in an annulus
    // is proportional to that annulus's area, i.e. proportional to r. We bin by
    // radius and print the per-bin density (count / annulus-area); a correct
    // sampler gives a FLAT density across bins.
    const int kBins = 8, kN = 2000000;
    auto radiusTest = [&](const char* name, auto sampler) {
        int hist[8] = {0};
        for (int i = 0; i < kN; ++i) {
            float x, y; sampler(x, y);
            float r = std::sqrt(x * x + y * y);
            int b = (int)(r * kBins); if (b >= kBins) b = kBins - 1;
            hist[b]++;
        }
        printf("  %-26s per-bin density (flat = uniform):\n    ", name);
        for (int b = 0; b < kBins; ++b) {
            float r0 = (float)b / kBins, r1 = (float)(b + 1) / kBins;
            float area = (float)(kPi * (r1 * r1 - r0 * r0));   // annulus area
            printf("%5.2f ", (hist[b] / (float)kN) / area);
        }
        printf("\n");
    };
    printf("Disk sampling (unit disk). Density should be FLAT if area-uniform:\n");
    radiusTest("naive   r=u  (WRONG)", [&](float& x, float& y) {
        float r = rng.NextFloat(), p = 2 * (float)kPi * rng.NextFloat();
        x = r * std::cos(p); y = r * std::sin(p);
    });
    radiusTest("correct r=sqrt(u)", [&](float& x, float& y) {
        float r = std::sqrt(rng.NextFloat()), p = 2 * (float)kPi * rng.NextFloat();
        x = r * std::cos(p); y = r * std::sin(p);
    });
    radiusTest("concentric (Shirley)", [&](float& x, float& y) {
        float a = 2 * rng.NextFloat() - 1, b = 2 * rng.NextFloat() - 1;   // square [-1,1]^2
        if (a == 0 && b == 0) { x = y = 0; return; }
        float r, phi;
        if (std::fabs(a) > std::fabs(b)) { r = a; phi = (float)kPi / 4 * (b / a); }
        else                            { r = b; phi = (float)kPi / 2 - (float)kPi / 4 * (a / b); }
        x = r * std::cos(phi); y = r * std::sin(phi);
    });
    printf("  -> the naive map's density spikes at small radius (points clump at the\n"
           "     center); the two correct maps are flat.\n\n");

    // --- PART 2: area -> solid-angle conversion for a disk light. ------------
    // Disk light parallel to the surface: center C=(0,0,H), normal facing down,
    // radius R. Solid angle subtended from the origin:
    //   Omega = integral over disk of cos(theta_light)/r^2 dA,  cos(theta_light)=H/r.
    const double H = 1.2, R = 0.5;
    const double A = kPi * R * R;

    // (a) deterministic quadrature reference over the disk (polar grid).
    double quad = 0.0; const int Nr = 2000, Np = 2000;
    for (int i = 0; i < Nr; ++i) {
        double rr = (i + 0.5) / Nr * R;
        for (int j = 0; j < Np; ++j) {
            double ph = (j + 0.5) / Np * 2 * kPi;
            double x = rr * std::cos(ph), y = rr * std::sin(ph);
            double r = std::sqrt(x * x + y * y + H * H);
            quad += (H / r) / (r * r) * rr;                  // (cos/r^2) * (r dr dphi)
        }
    }
    quad *= (R / Nr) * (2 * kPi / Np);

    // (b) Monte Carlo by sampling the AREA uniformly and converting each sample's
    // area pdf (1/A) to a solid-angle pdf via r^2/cos(theta_light).
    double mc = 0.0; const int M = 4000000;
    for (int i = 0; i < M; ++i) {
        float r0 = (float)(R * std::sqrt(rng.NextFloat())), ph = 2 * (float)kPi * rng.NextFloat();
        double x = r0 * std::cos(ph), y = r0 * std::sin(ph);
        double r = std::sqrt(x * x + y * y + H * H);
        double cosLight = H / r;
        mc += cosLight / (r * r);                            // integrand; * A/M done below
    }
    mc *= A / M;                                             // E[g]*A, since p_area = 1/A

    printf("Disk light (H=%.1f, R=%.1f): solid angle it subtends\n", H, R);
    printf("  quadrature reference ...... %.6f sr\n", quad);
    printf("  area-sampling + r^2/cos ... %.6f sr  (should match)\n", mc);
    printf("  analytic 2pi(1-cos a_max).. %.6f sr  (a_max = atan(R/H))\n",
           2 * kPi * (1 - H / std::sqrt(H * H + R * R)));

    printf("\nThe Jacobian is not bookkeeping you can skip: the wrong disk map biases\n"
           "your lens/area-light samples, and the r^2/cos factor is exactly what lets\n"
           "next-event estimation (example 24) integrate a light over solid angle.\n");
    return 0;
}
