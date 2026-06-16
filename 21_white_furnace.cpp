// 21 — The white furnace test: is your BRDF conserving energy?
// ============================================================
//
// A BRDF must never reflect MORE light than arrives — that would create energy
// from nothing. The standard way to check is the WHITE FURNACE TEST: imagine the
// surface sitting inside a uniformly glowing white sphere (incoming radiance = 1
// in every direction). The outgoing radiance then equals the DIRECTIONAL ALBEDO
//
//   E(wo) = integral over the hemisphere of f_r(wo, wi) * cos(theta_i) dwi.
//
// With perfectly reflective microfacets (Fresnel = 1), energy conservation
// demands E(wo) <= 1. If E(wo) == 1 the object is invisible against the furnace
// (it reflects exactly its surroundings); if E(wo) < 1 it looks DARKER than the
// background, and the missing 1 - E(wo) is energy the model lost.
//
// Here is the catch this test exposes: the SINGLE-SCATTER GGX BRDF (the one in
// example 19 and in most real-time renderers) loses energy at high roughness,
// because it only counts light that bounces off the microsurface ONCE. In
// reality, light that hits a microfacet and is masked often scatters again and
// still escapes; ignoring those multiple bounces makes rough metals/dielectrics
// too dark. We measure that deficit as a function of roughness.
//
// We reuse the VNDF machinery from example 19: with Fresnel = 1, the estimator
// for E(wo) is simply the average of G2/G1. As validation we first run the test
// on a Lambertian surface, whose albedo we know exactly (E = rho), then report
// the GGX deficit and assert E <= 1 (a pass/fail energy-conservation check).

#include <cstdio>
#include <cmath>
#include <initializer_list>
#include "ggx.h"
#include "mc_random.h"

// E(wo) for GGX with F = 1, via VNDF: average of G2/G1 over reflected samples.
static double DirectionalAlbedo(const Vec3& wo, float a, Rng& rng, long long N) {
    double sum = 0.0; double g1 = GgxG1(wo, a);
    for (long long i = 0; i < N; ++i) {
        Vec3 h = SampleGgxVNDF(wo, a, rng.NextFloat(), rng.NextFloat());
        Vec3 wi = GgxReflect(wo, h);
        if (wi.z > 0.0f) sum += GgxG2(wo, wi, a) / g1;
    }
    return sum / N;
}

int main() {
    Rng rng(/*sequence=*/21);
    const float PI = kPiGGX;

    // --- Sanity: a Lambertian surface of albedo rho must give E = rho. ------
    // f_r = rho/pi, so E = integral (rho/pi) cos dwi = rho. We confirm the
    // furnace machinery on this known case before trusting it on GGX.
    double rho = 0.8, sum = 0.0; const long long Nl = 2000000;
    for (long long i = 0; i < Nl; ++i) {                 // uniform hemisphere
        float z = rng.NextFloat();
        sum += 2.0 * PI * (rho / PI) * z;                // 2pi * f_r * cos
    }
    printf("Lambertian furnace check: rho = %.3f, measured E = %.4f (should match)\n\n",
           rho, sum / Nl);

    // --- GGX: directional albedo at normal incidence and hemisphere average. -
    // E0   = E(wo) at theta = 0.
    // Eavg = cosine-weighted average of E(wo) over all view directions:
    //        sample wo ~ cosine, take ONE VNDF sample, average G2/G1.
    printf("GGX single-scatter albedo (Fresnel = 1). Energy is conserved iff E <= 1.\n\n");
    printf("  %7s | %10s %10s %12s %8s\n", "alpha", "E(normal)", "E avg", "energy lost", "E<=1?");
    bool allConserved = true;
    for (float a : {0.05f, 0.1f, 0.2f, 0.4f, 0.8f}) {
        double E0 = DirectionalAlbedo(Vec3(0, 0, 1), a, rng, 2000000);

        double avg = 0.0; const long long M = 4000000;
        for (long long i = 0; i < M; ++i) {
            float u1 = rng.NextFloat(), u2 = rng.NextFloat();         // cosine-weighted wo
            float r = std::sqrt(u1), ph = 2.0f * PI * u2;
            Vec3 wo(r * std::cos(ph), r * std::sin(ph), std::sqrt(std::fmax(0.0f, 1.0f - u1)));
            Vec3 h = SampleGgxVNDF(wo, a, rng.NextFloat(), rng.NextFloat());
            Vec3 wi = GgxReflect(wo, h);
            if (wi.z > 0.0f) avg += GgxG2(wo, wi, a) / GgxG1(wo, a);
        }
        double Eavg = avg / M;
        bool ok = E0 <= 1.0001 && Eavg <= 1.0001;
        allConserved = allConserved && ok;
        printf("  %7.2f | %10.4f %10.4f %12.4f %8s\n", a, E0, Eavg, 1.0 - Eavg, ok ? "yes" : "NO");
    }

    printf("\nEnergy conservation (E <= 1): %s\n", allConserved ? "PASS" : "FAIL");
    printf("\nNo row exceeds 1 (the model never invents energy), but the deficit\n"
           "1 - E grows with roughness: at alpha=0.8 a large fraction of energy is\n"
           "simply missing. That is the single-scatter approximation showing its\n"
           "seams -- rough metals rendered this way look too dark. The fix is a\n"
           "multiple-scattering term (Kulla-Conty 2017) that adds the lost energy\n"
           "back so E -> 1; the renderer approximates this cheaply by tinting the\n"
           "diffuse lobe with (1 - F0). The furnace test is how you'd catch a BRDF\n"
           "that got it wrong (E > 1) -- a standard sanity check when writing one.\n");
    return 0;
}
