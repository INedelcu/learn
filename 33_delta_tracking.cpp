// 33 — Participating media: delta (Woodcock) tracking
// ====================================================
//
// Example 05 sampled the distance a ray travels before colliding in a HOMOGENEOUS
// medium: the transmittance is exp(-sigma*t), invertible as t = -ln(1-u)/sigma.
// But fog, smoke and clouds have a HETEROGENEOUS extinction sigma(x) that varies
// along the ray, so the transmittance
//
//   T(t) = exp( - integral_0^t sigma(s) ds )
//
// has no closed-form inverse to sample from. DELTA TRACKING (Woodcock 1965) solves
// this with a beautiful trick: pad the medium with FICTITIOUS "null" collisions so
// the TOTAL collision rate is a constant majorant sigma_max >= sigma(x) everywhere.
// Now distances are exponential with rate sigma_max (easy to sample), and at each
// tentative collision you flip a coin:
//
//   accept as a REAL collision with probability sigma(x)/sigma_max,
//   otherwise it was a null collision -- ignore it and keep marching.
//
// The null collisions exactly compensate for the constant majorant, so the
// accepted distances follow the true heterogeneous transmittance -- unbiased, with
// no integral to invert. (The same idea, "ratio tracking", gives a low-variance
// transmittance estimate.) This is how production volume renderers handle clouds;
// the homogeneous Beer-Lambert of the glass shader is the sigma = const case.
//
// We use a linear profile sigma(x) = s0 + s1*x on [0,L], whose optical depth
// integral is analytic, and verify: (1) the sampled collision-distance histogram
// matches the true pdf sigma(t)*T(t); (2) the transmittance through the slab from
// delta tracking and from ratio tracking both match exp(-optical depth).

#include <cstdio>
#include <cmath>
#include "mc_random.h"

static const double s0 = 0.5, s1 = 1.5, L = 2.0;
static double Sigma(double x)        { return s0 + s1 * x; }                 // extinction
static double OpticalDepth(double t) { return s0 * t + 0.5 * s1 * t * t; }   // integral_0^t sigma
static double Transmittance(double t){ return std::exp(-OpticalDepth(t)); }

int main() {
    Rng rng(/*sequence=*/33);
    const double sMax = Sigma(L);                    // majorant: sigma never exceeds this
    const long long N = 4000000;

    // --- (1) sample collision distance by delta tracking; histogram it. -----
    const int B = 10; long long hist[B] = {0}; long long transmits = 0;
    for (long long i = 0; i < N; ++i) {
        double t = 0.0; bool collided = false;
        for (;;) {
            t += -std::log(1.0 - rng.NextFloat()) / sMax;     // free flight at rate sMax
            if (t >= L) break;                                // exited the slab
            if (rng.NextFloat() < Sigma(t) / sMax) { collided = true; break; }  // real collision
            // else: null collision -- continue marching
        }
        if (collided) { int b = (int)(t / L * B); if (b >= B) b = B - 1; hist[b]++; }
        else ++transmits;
    }

    printf("Heterogeneous slab sigma(x)=%.1f+%.1f x on [0,%.1f], majorant sigma_max=%.1f\n\n",
           s0, s1, L, sMax);
    printf("Collision-distance histogram vs true pdf sigma(t)*T(t):\n");
    printf("  %-14s %12s %12s\n", "bin", "measured", "predicted");
    double binW = L / B;
    for (int b = 0; b < B; ++b) {
        double t0 = b * binW, t1 = t0 + binW, mid = 0.5 * (t0 + t1);
        double measured = (double)hist[b] / N / binW;          // empirical density
        double predicted = Sigma(mid) * Transmittance(mid);    // sigma(t) exp(-tau)
        printf("  [%.2f,%.2f)    %12.5f %12.5f\n", t0, t1, measured, predicted);
    }

    // --- (2) transmittance through the whole slab, two estimators. ----------
    // Delta tracking: T = P(no real collision before L) -> the transmit fraction.
    double Tdelta = (double)transmits / N;
    // Ratio tracking: march with the majorant but instead of accepting, multiply
    // a running weight by (1 - sigma/sigma_max) at each tentative collision. Its
    // expectation is the transmittance, with much lower variance.
    double Tratio = 0.0;
    const long long M = 1000000;
    for (long long i = 0; i < M; ++i) {
        double t = 0.0, w = 1.0;
        for (;;) {
            t += -std::log(1.0 - rng.NextFloat()) / sMax;
            if (t >= L) break;
            w *= (1.0 - Sigma(t) / sMax);
        }
        Tratio += w;
    }
    Tratio /= M;

    printf("\nTransmittance through the slab (analytic exp(-tau), tau=%.4f -> %.6f):\n",
           OpticalDepth(L), Transmittance(L));
    printf("  delta tracking (transmit fraction) = %.6f\n", Tdelta);
    printf("  ratio tracking (weighted)          = %.6f\n", Tratio);

    printf("\nThe collision distances match sigma(t)*T(t) and both transmittance\n"
           "estimates match exp(-tau) -- all without inverting the varying optical\n"
           "depth. Null collisions are the trick that turns an un-invertible\n"
           "heterogeneous medium into simple exponential free flights.\n");
    return 0;
}
