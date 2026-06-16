// 40 — Average in linear space: the gamma trap
// ============================================
//
// Monitors are non-linear: an 8-bit pixel value is sRGB-ENCODED, not proportional
// to light. Encoding is roughly value = linear^(1/2.2). This bites Monte Carlo
// rendering in one specific, very common way:
//
//   YOU MUST AVERAGE RADIANCE IN LINEAR SPACE, then encode once at the end.
//
// If you accidentally average already-encoded (sRGB) values -- or, equivalently,
// blend/downsample/anti-alias in the display space -- you get the wrong answer,
// always too DARK in the midtones. The classic symptom: a 50%-coverage edge
// between black and white that renders as a dim gray instead of the correct bright
// gray. This is why the renderer accumulates LINEAR radiance
// (lerp(prev, new, 1/(step+1)) on linear values) and only converts to sRGB at the
// final blit.
//
// We show the error numerically across coverage fractions, and write a PPM whose
// top strip averages correctly (in linear) and bottom strip averages wrongly (in
// sRGB) -- the bottom midtones come out visibly darker.

#include <cstdio>
#include <cmath>
#include <initializer_list>
#include <vector>
#include <string>
#include <filesystem>

static double ToSRGB(double c) {                    // linear -> display encoding
    c = c < 0 ? 0 : c > 1 ? 1 : c;
    return c <= 0.0031308 ? 12.92 * c : 1.055 * std::pow(c, 1.0 / 2.4) - 0.055;
}

int main() {
    std::filesystem::create_directories("images");

    // --- numeric: a pixel that is fraction 'cov' white (L=1) over black (L=0). --
    // CORRECT: average light (= cov), THEN encode.
    // WRONG:   average the encoded pixels = cov*sRGB(1) + (1-cov)*sRGB(0) = cov.
    printf("A pixel that is fraction 'cov' covered by white (L=1) over black (L=0).\n");
    printf("Displayed 8-bit gray, computed correctly vs by averaging encoded values:\n\n");
    printf("  %5s | %16s | %16s | %s\n", "cov", "correct (linear)", "wrong (encoded)", "error");
    for (double cov : {0.1, 0.25, 0.5, 0.75, 0.9}) {
        int correct = (int)std::lround(ToSRGB(cov) * 255);   // encode(linear average)
        int wrong   = (int)std::lround(cov * 255);           // average of encoded {0,255}
        printf("  %5.2f | %12d/255 | %12d/255 | %d levels too dark\n",
               cov, correct, wrong, correct - wrong);
    }

    // --- visual: coverage gradient, top correct, bottom wrong. --------------
    const int W = 384, H = 120, half = H / 2;
    std::vector<unsigned char> img(W * H * 3);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            double cov = (x + 0.5) / W;                       // 0..1 coverage across width
            double encoded = (y < half) ? ToSRGB(cov)         // correct: encode the linear average
                                        : cov;                // wrong: the average of encoded values
            unsigned char b = (unsigned char)(encoded * 255.0 + 0.5);
            int o = (y * W + x) * 3; img[o] = img[o + 1] = img[o + 2] = b;
        }
    FILE* f = std::fopen("images/gamma_compare.ppm", "wb");
    if (f) { std::fprintf(f, "P6\n%d %d\n255\n", W, H); std::fwrite(img.data(), 1, img.size(), f); std::fclose(f); }

    printf("\nWrote images/gamma_compare.ppm: identical coverage gradient, top averaged\n");
    printf("in LINEAR space (correct), bottom averaged in ENCODED space (wrong). The\n");
    printf("bottom is visibly darker through the midtones -- the same gradient, lit\n");
    printf("'wrong'. At 50%% coverage the error is ~%d/255 levels.\n",
           (int)std::lround(ToSRGB(0.5) * 255) - 128);
    printf("\nThe rule: keep all accumulation, blending and filtering in linear\n"
           "radiance; apply the sRGB curve only when writing the final pixel. Mixing\n"
           "the two spaces is one of the most common and most visible rendering bugs.\n");
    return 0;
}
