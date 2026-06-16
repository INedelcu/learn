// vec3.h — a minimal 3D vector, shared by the geometry-heavy tutorials
// (microfacet sampling, refraction, light sampling, orthonormal bases).
//
// Deliberately tiny: just the operations those examples need. All the rendering
// math in this folder is done in a LOCAL frame where the surface normal is
// (0,0,1), so a "z" component is the same as the cosine of the angle to the
// normal (n·v == v.z). Keep that in mind while reading the examples.

#pragma once
#include <cmath>

struct Vec3 {
    float x = 0, y = 0, z = 0;
    Vec3() {}
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator-(Vec3 a)         { return {-a.x, -a.y, -a.z}; }
inline Vec3 operator*(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline Vec3 operator*(float s, Vec3 a) { return a * s; }

inline float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3  cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline float length(Vec3 a)    { return std::sqrt(dot(a, a)); }
inline Vec3  normalize(Vec3 a) { float l = length(a); return l > 0 ? a * (1.0f / l) : a; }
