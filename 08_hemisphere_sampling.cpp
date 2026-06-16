// 08 — Sampling the hemisphere: uniform vs cosine-weighted
// ========================================================
//
// This is where the series meets the path tracer head-on. The rendering equation
// integrates incoming light over the hemisphere above a surface point, weighted by
// the cosine of the angle to the normal (Lambert's cosine law):
//
//   L_out = integral over hemisphere of  f_r(omega) * L_in(omega) * cos(theta) d(omega)
//
// We estimate it by sampling directions omega from some density p(omega) and
// averaging  f_r * L_in * cos(theta) / p(omega). The question is: which p?
//
// Two ways to sample a direction on the hemisphere (theta measured from the
// normal, so cos(theta) = z):
//
//   UNIFORM over solid angle:   every direction equally likely.  p(omega) = 1/(2*pi).
//     Sample:  z = u1;  r = sqrt(1 - z*z);  phi = 2*pi*u2;
//              omega = (r cos phi, r sin phi, z).
//
//   COSINE-WEIGHTED:            directions near the normal more likely.
//     p(omega) = cos(theta)/pi.   Malley's method: sample a disk uniformly and
//     lift it onto the hemisphere.
//     Sample:  r = sqrt(u1);  phi = 2*pi*u2;  (x,y) = (r cos phi, r sin phi);
//              z = sqrt(1 - u1);   omega = (x, y, z).
//
// Why cosine-weighting is the natural choice for diffuse surfaces: the density
// is PROPORTIONAL TO the cos(theta) factor in the integrand. That factor then
// cancels in f_r*L_in*cos(theta)/p, removing it as a source of variance. In the
// renderer's SampleDiffuseLambert this is exactly why the cos(theta)/pi pdf
// cancels the albedo/pi Lambertian BRDF — the sampled weight reduces to the
// albedo with no leftover cosine.
//
// We demonstrate the payoff on two integrals with known answers:
//
//   (1) integral of cos(theta) d(omega)            = pi      (the bare cosine factor)
//   (2) integral of cos^3(theta) d(omega)          = pi/2    (a directional integrand)
//
// For (1), cosine sampling gives the answer with LITERALLY ZERO variance — every
// sample reports exactly pi. For (2) it cannot be zero, but it still cancels one
// cosine and beats uniform sampling handily.

#include <cstdio>
#include <cmath>
#include "mc_random.h"

int main() {
    Rng rng(/*sequence=*/8);
    const double kPi = 3.14159265358979323846;
    const long long kN = 10000000;

    // We accumulate, for each integral and each strategy, the running mean and
    // variance of the per-sample estimator f(omega)/p(omega).
    struct Acc { double mean = 0, m2 = 0; long long n = 0;
        void add(double x) { ++n; double d = x - mean; mean += d / n; m2 += d * (x - mean); }
        double var() const { return m2 / (n - 1); } };

    Acc uni1, cos1, uni2, cos2;

    for (long long i = 0; i < kN; ++i) {
        // ---- one uniform-hemisphere sample ----
        {
            double u1 = rng.NextFloat(), u2 = rng.NextFloat();
            double z = u1;                       // cos(theta)
            double pdf = 1.0 / (2.0 * kPi);
            // integrand (1): f = cos(theta) = z
            uni1.add(z / pdf);
            // integrand (2): f = cos^3(theta) = z^3
            uni2.add((z * z * z) / pdf);
        }
        // ---- one cosine-weighted sample ----
        {
            double u1 = rng.NextFloat(), u2 = rng.NextFloat();
            double z = std::sqrt(1.0 - u1);      // cos(theta) from Malley's method
            double pdf = z / kPi;                // cos(theta)/pi
            // integrand (1): f = z  ->  estimator = z / (z/pi) = pi   (constant!)
            cos1.add(z / pdf);
            // integrand (2): f = z^3 -> estimator = z^3/(z/pi) = pi*z^2
            cos2.add((z * z * z) / pdf);
        }
    }

    printf("Hemisphere integrals via %lld samples each.\n\n", kN);

    printf("(1)  integral of cos(theta) dω   exact = pi = %.7f\n", kPi);
    printf("     %-18s %14s %16s\n", "strategy", "estimate", "per-sample var");
    printf("     %-18s %14.7f %16.3e\n", "uniform",          uni1.mean, uni1.var());
    printf("     %-18s %14.7f %16.3e   <- exactly zero!\n", "cosine-weighted", cos1.mean, cos1.var());

    printf("\n(2)  integral of cos^3(theta) dω exact = pi/2 = %.7f\n", kPi / 2.0);
    printf("     %-18s %14s %16s\n", "strategy", "estimate", "per-sample var");
    printf("     %-18s %14.7f %16.3e\n", "uniform",          uni2.mean, uni2.var());
    printf("     %-18s %14.7f %16.3e\n", "cosine-weighted",  cos2.mean, cos2.var());
    printf("     variance reduction: %.1fx\n", uni2.var() / cos2.var());

    printf("\nFor integral (1) the cosine pdf is *proportional to the integrand*, so\n"
           "every sample returns the exact answer pi — zero variance, the ideal\n"
           "importance-sampling case from example 06. For integral (2) it cancels\n"
           "one of the three cosines, flattening the estimator and cutting variance\n"
           "several-fold. This is the principle behind cosine-weighted diffuse\n"
           "bounce sampling in the path tracer.\n");
    return 0;
}
