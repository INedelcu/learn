// ggx.h — the GGX / Trowbridge-Reitz microfacet model with Smith masking-
// shadowing and Heitz-2018 visible-normal (VNDF) sampling. Shared by the
// microfacet examples (19 sampling, 21 white furnace). This mirrors the
// renderer's BRDF.hlsl, and the implementation IS part of the lesson, so it is
// commented rather than hidden.
//
// Everything is in a LOCAL frame: the surface normal is (0,0,1), so n·v == v.z.
// A microfacet specular surface is modeled as a field of tiny mirror facets;
// D gives the statistical distribution of their orientations (half-vectors h),
// and the Smith G terms account for facets occluding each other.

#pragma once
#include <cmath>
#include "vec3.h"

constexpr float kPiGGX = 3.14159265358979323846f;

// Disney/URP convention used by the renderer: alpha = (1 - smoothness)^2.
inline float SmoothnessToAlpha(float smoothness) { float a = 1.0f - smoothness; return a * a; }

// GGX normal distribution function D(h): how densely microfacet normals cluster
// around the surface normal. Sharp (small alpha) -> mirror; broad -> rough.
inline float GgxD(const Vec3& h, float a) {
    float a2 = a * a;
    float c = h.z;                                  // n·h
    float d = c * c * (a2 - 1.0f) + 1.0f;
    return a2 / (kPiGGX * d * d);
}

// Smith Lambda for GGX: auxiliary term for masking/shadowing of a direction v.
inline float GgxLambda(const Vec3& v, float a) {
    float c = std::fabs(v.z);
    if (c >= 0.99999f) return 0.0f;
    float tan2 = (1.0f - c * c) / (c * c);          // tan^2(theta_v)
    return 0.5f * (-1.0f + std::sqrt(1.0f + a * a * tan2));
}

// Smith G1 (one direction is masked) and height-correlated G2 (both directions).
inline float GgxG1(const Vec3& v, float a) { return 1.0f / (1.0f + GgxLambda(v, a)); }
inline float GgxG2(const Vec3& wo, const Vec3& wi, float a) {
    return 1.0f / (1.0f + GgxLambda(wo, a) + GgxLambda(wi, a));
}

// Heitz 2018, "Sampling the GGX Distribution of Visible Normals." Returns a
// microfacet normal h drawn from the normals VISIBLE from wo (not just D): this
// is what makes specular sampling efficient, because facets pointing away from
// the viewer can never contribute and shouldn't be sampled.
inline Vec3 SampleGgxVNDF(const Vec3& wo, float a, float u1, float u2) {
    Vec3 Vh = normalize(Vec3(a * wo.x, a * wo.y, wo.z));               // 1. stretch view
    Vec3 T1 = (Vh.z < 0.9999f) ? normalize(cross(Vec3(0, 0, 1), Vh)) : Vec3(1, 0, 0);
    Vec3 T2 = cross(Vh, T1);                                          // 2. orthonormal basis
    float r = std::sqrt(u1), phi = 2.0f * kPiGGX * u2;                // 3. sample the
    float t1 = r * std::cos(phi), t2 = r * std::sin(phi);            //    projected disk
    float s = 0.5f * (1.0f + Vh.z);
    t2 = (1.0f - s) * std::sqrt(std::fmax(0.0f, 1.0f - t1 * t1)) + s * t2;
    float t3 = std::sqrt(std::fmax(0.0f, 1.0f - t1 * t1 - t2 * t2));
    Vec3 Nh = T1 * t1 + T2 * t2 + Vh * t3;                            // 4. lift to hemisphere
    return normalize(Vec3(a * Nh.x, a * Nh.y, std::fmax(1e-6f, Nh.z))); // 5. unstretch
}

// Reflect the view direction wo about the half-vector h to get incident dir wi.
inline Vec3 GgxReflect(const Vec3& wo, const Vec3& h) { return h * (2.0f * dot(wo, h)) - wo; }

// Single-scatter specular BRDF value WITHOUT the Fresnel factor (caller adds F):
//   f_r = D * G2 / (4 (n·wo)(n·wi)).
inline float GgxBrdfNoF(const Vec3& wo, const Vec3& wi, const Vec3& h, float a) {
    float denom = 4.0f * wo.z * wi.z;
    if (denom <= 0.0f) return 0.0f;
    return GgxD(h, a) * GgxG2(wo, wi, a) / denom;
}
