// 26 — Building an orthonormal basis around a normal (Duff et al. 2017)
// =====================================================================
//
// Sampling routines (cosine hemisphere in example 08, GGX in 19) produce
// directions in a LOCAL frame where the normal is (0,0,1). To use them you must
// rotate those directions into world space around the actual surface normal —
// you need two tangent vectors completing an orthonormal basis (a "frame") with
// the normal. The renderer does this in `BuildOrthonormalBasis` (Utils.hlsl).
//
// The naive approach ("cross the normal with some fixed axis") breaks when the
// normal happens to be near that axis. DUFF, BURGESS, LICEA-KANE & THEOBALD
// (2017) give a BRANCHLESS construction that is numerically stable for EVERY
// normal, including the poles, using a sign trick:
//
//   sign = +/-1 from n.z;  a = -1/(sign + n.z);  b = n.x*n.y*a;
//   b1 = ( 1 + sign*n.x*n.x*a,   sign*b,        -sign*n.x );
//   b2 = ( b,                    sign + n.y*n.y*a, -n.y    );
//
// (b1, b2, n) is then orthonormal and right-handed. This file verifies that:
//   (1) for thousands of random normals, b1/b2/n are mutually perpendicular and
//       unit length (residuals at the floating-point floor), including the hard
//       cases n = +/-z;
//   (2) cosine-weighted samples generated in the local frame and rotated through
//       the basis land in the correct hemisphere and have mean cos(theta) = 2/3
//       about the chosen normal (the known cosine-weighted average).

#include <cstdio>
#include <cmath>
#include "vec3.h"
#include "mc_random.h"

// Duff et al. 2017, branchless orthonormal basis. n must be unit length.
static void OrthonormalBasis(const Vec3& n, Vec3& b1, Vec3& b2) {
    float sign = std::copysignf(1.0f, n.z);
    float a = -1.0f / (sign + n.z);
    float b = n.x * n.y * a;
    b1 = Vec3(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
    b2 = Vec3(b, sign + n.y * n.y * a, -n.y);
}

int main() {
    Rng rng(/*sequence=*/26);

    // --- (1) orthonormality over many random normals + the polar cases. -----
    double maxErr = 0.0;
    auto check = [&](const Vec3& n) {
        Vec3 b1, b2; OrthonormalBasis(n, b1, b2);
        double e = 0.0;
        e = std::fmax(e, std::fabs(dot(b1, b2)));         // perpendicular pairs
        e = std::fmax(e, std::fabs(dot(b1, n)));
        e = std::fmax(e, std::fabs(dot(b2, n)));
        e = std::fmax(e, std::fabs(length(b1) - 1.0));    // unit length
        e = std::fmax(e, std::fabs(length(b2) - 1.0));
        maxErr = std::fmax(maxErr, e);
    };
    const int kNormals = 200000;
    for (int i = 0; i < kNormals; ++i) {
        float z = 2.0f * rng.NextFloat() - 1.0f;          // uniform point on sphere
        float r = std::sqrt(std::fmax(0.0f, 1.0f - z * z));
        float ph = 2.0f * 3.14159265f * rng.NextFloat();
        check(Vec3(r * std::cos(ph), r * std::sin(ph), z));
    }
    check(Vec3(0, 0, 1));                                 // the cases naive code fails on
    check(Vec3(0, 0, -1));
    printf("Orthonormality over %d random normals + the poles:\n", kNormals);
    printf("  worst residual (dot/length error) = %.2e   (machine-epsilon small)\n\n", maxErr);

    // --- (2) rotate cosine-weighted samples into a tilted frame. ------------
    // Pick a slanted normal; generate cosine-weighted local directions and map
    // them through (b1,b2,n). They must all lie in n's hemisphere, and the mean
    // of cos(theta) = dot(sample, n) must be 2/3.
    Vec3 n = normalize(Vec3(0.4f, -0.7f, 0.6f));
    Vec3 b1, b2; OrthonormalBasis(n, b1, b2);
    double sumCos = 0.0; int below = 0; const long long N = 4000000;
    for (long long i = 0; i < N; ++i) {
        float u1 = rng.NextFloat(), u2 = rng.NextFloat();
        float r = std::sqrt(u1), ph = 2.0f * 3.14159265f * u2;
        Vec3 local(r * std::cos(ph), r * std::sin(ph), std::sqrt(std::fmax(0.0f, 1.0f - u1)));
        Vec3 world = b1 * local.x + b2 * local.y + n * local.z;     // rotate into world
        double c = dot(world, n);
        sumCos += c;
        if (c < -1e-5) ++below;
    }
    printf("Cosine-weighted samples rotated into the frame of n=(%.2f,%.2f,%.2f):\n", n.x, n.y, n.z);
    printf("  mean cos(theta) about n = %.5f   (cosine-weighted theory = 0.66667)\n", sumCos / N);
    printf("  samples below the surface = %d   (should be 0)\n", below);

    printf("\nThe basis is orthonormal everywhere (no pole blow-up) and correctly\n"
           "carries a local sampling distribution onto an arbitrary normal -- the\n"
           "small but essential plumbing that lets every hemisphere/BRDF sampler in\n"
           "the renderer work in a convenient local frame, then transform out.\n");
    return 0;
}
