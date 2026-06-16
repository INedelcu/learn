// mc_random.h — a tiny, reproducible random number generator shared by all the
// tutorials in this folder.
//
// We implement PCG32 (Melissa O'Neill, 2014), the same family of generator pbrt
// uses. Why not std::mt19937? Three reasons that matter for a rendering course:
//
//   1. It is *small and fast* — one 64-bit multiply-add per number. In a path
//      tracer you draw billions of random numbers, so the cost of the generator
//      is not negligible.
//   2. It is *statistically excellent* — it passes the TestU01 suite, unlike the
//      classic LCGs people often reach for first.
//   3. It supports *streams*: two generators seeded with different "sequence"
//      values produce independent streams. In a renderer you give every pixel
//      its own stream so the image is deterministic regardless of threading.
//
// Everything here is deterministic: the same seed gives the same numbers on every
// machine and every run. That is exactly what you want while *learning*, because
// you can compare your output to the numbers quoted in the comments.

#pragma once
#include <cstdint>

// The largest float strictly less than 1. We clamp to this so NextFloat() never
// returns exactly 1.0f — a value of 1.0 can index one-past-the-end of a table or
// make a "u in [0,1)" assumption fail. pbrt uses the same constant.
constexpr float kOneMinusEpsilon = 0x1.fffffep-1f;

class Rng {
public:
    // 'sequence' selects an independent stream; 'seed' offsets within it.
    explicit Rng(uint64_t sequence = 1, uint64_t seed = 0x853c49e6748fea9bULL) {
        SetSequence(sequence, seed);
    }

    void SetSequence(uint64_t sequence, uint64_t seed = 0x853c49e6748fea9bULL) {
        state_ = 0u;
        inc_ = (sequence << 1u) | 1u;   // inc must be odd
        NextUInt32();
        state_ += seed;
        NextUInt32();
    }

    // The core PCG32 step: advance an LCG, then apply an output permutation
    // (xorshift + random rotation) that hides the LCG's weak low bits.
    uint32_t NextUInt32() {
        uint64_t oldstate = state_;
        state_ = oldstate * 6364136223846793005ULL + inc_;
        uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
        uint32_t rot = (uint32_t)(oldstate >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((0u - rot) & 31u));
    }

    // Uniform float in [0, 1). Multiplying a 32-bit integer by 2^-32 maps
    // {0, ..., 2^32-1} onto evenly spaced points in [0, 1).
    float NextFloat() {
        float r = NextUInt32() * 0x1p-32f;
        return r < kOneMinusEpsilon ? r : kOneMinusEpsilon;
    }

    // Uniform float in [a, b).
    float NextFloat(float a, float b) { return a + (b - a) * NextFloat(); }

private:
    uint64_t state_ = 0;
    uint64_t inc_ = 0;
};
