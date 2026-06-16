// 34 — Equiangular sampling for volumetric single scattering
// ==========================================================
//
// Picture a camera ray passing through fog near a small bright light (the classic
// "god rays" / light-shaft setup). The in-scattered light gathered along the ray
// is dominated by the inverse-square falloff to the light:
//
//   contribution(t) ~ 1 / d(t)^2 = 1 / (h^2 + t^2),
//
// where t runs along the ray and h is the ray's closest distance to the light.
// This integrand spikes sharply near the closest point. Sampling t UNIFORMLY (or
// even by transmittance) wastes nearly all samples far from the spike -- high
// variance. EQUIANGULAR SAMPLING (Kulla & Conty 2012) samples t with a density
// proportional to 1/(h^2 + t^2), which has an analytic inverse CDF (the integral
// of 1/(h^2+t^2) is an arctangent):
//
//   t = h * tan( atan(a/h) + u * (atan(b/h) - atan(a/h)) ).
//
// Because the density matches the integrand's dominant factor, the estimator
// f(t)/p(t) is nearly constant -- variance collapses, just like importance
// sampling in example 06 (here for the geometric falloff instead of a BRDF).
//
// We integrate the falloff over a ray segment [a,b] (analytic value
// (1/h)(atan(b/h) - atan(a/h))) two ways and compare variance: uniform sampling
// (noisy) vs equiangular (essentially exact, since the density IS the integrand).

#include <cstdio>
#include <cmath>
#include "mc_random.h"

int main() {
    Rng rng(/*sequence=*/34);
    const double h = 0.1;            // closest approach of the ray to the light (small -> sharp)
    const double a = -1.0, b = 1.5;  // ray segment in t

    auto f = [&](double t) { return 1.0 / (h * h + t * t); };   // inverse-square falloff
    double atanA = std::atan(a / h), atanB = std::atan(b / h);
    double exact = (atanB - atanA) / h;                          // analytic integral

    struct Acc { double mean = 0, m2 = 0; long long n = 0;
        void add(double x){ ++n; double d=x-mean; mean+=d/n; m2+=d*(x-mean);}
        double var() const { return n>1? m2/(n-1):0; } };
    Acc uni, equi;
    const long long N = 5000000;
    for (long long i = 0; i < N; ++i) {
        // uniform: t ~ U[a,b], estimator f(t)*(b-a).
        double tu = a + (b - a) * rng.NextFloat();
        uni.add(f(tu) * (b - a));

        // equiangular: t with pdf proportional to 1/(h^2+t^2). The normalized
        // density is exactly f(t)/integral, so f(t)/pdf = integral every time.
        double te = h * std::tan(atanA + rng.NextFloat() * (atanB - atanA));
        double pe = f(te) / exact;          // normalized density
        equi.add(f(te) / pe);               // = exact -> zero variance
    }

    printf("Single-scattering falloff integral_%.1f^%.1f 1/(h^2+t^2) dt, h=%.2f\n", a, b, h);
    printf("  exact = (atan(b/h)-atan(a/h))/h = %.6f\n\n", exact);
    printf("  %-14s estimate %.6f  variance %.4e\n", "uniform",     uni.mean,  uni.var());
    printf("  %-14s estimate %.6f  variance %.4e\n", "equiangular", equi.mean, equi.var());
    printf("  variance reduction: %.3e x\n", uni.var() / std::fmax(equi.var(), 1e-300));

    printf("\nEquiangular sampling matches the integrand's 1/d^2 shape exactly here,\n"
           "so every sample returns the true value -- variance collapses to ~0, while\n"
           "uniform sampling sprays most samples far from the spike and stays noisy.\n"
           "With transmittance and a phase function folded in the cancellation is no\n"
           "longer perfect, but equiangular still wins big; it is the standard way to\n"
           "sample distance for volumetric single scattering (god rays).\n");
    return 0;
}
