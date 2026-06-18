# Monte Carlo & Probability for Path Tracing — a C++ tutorial series

**📖 Interactive companion site: https://inedelcu.github.io/Learn/** — each example explained with typeset math, annotated code, and live in-browser figures.

Forty-two small, self-contained C++ programs that build up the probability and
statistics behind Monte Carlo integration, in the order you'd want to learn
them, and connect each idea back to how the [PathTracingDemo](https://github.com/INedelcu/PathTracingDemo)
path tracer actually works. Inspired by the analysis in **pbrt** (*Physically Based Rendering*,
Pharr, Jakob & Humphreys) and a CDF-theory demo from the HDRP investigations —
see References below.

Every example is one `.cpp` file that uses only the standard library plus a few
tiny shared headers (`mc_random.h` for the RNG; `vec3.h` and `ggx.h` for the
geometry/BRDF examples). Each one **prints results you can check against a known
answer**, and the runs are fully deterministic (fixed RNG seeds), so your output
will match the numbers quoted in the comments.

## Build & run

The folder ships with a PowerShell build script that finds a compiler for you
(`g++`/`clang++` if on `PATH`, otherwise MSVC `cl` via the Visual Studio
developer environment):

```powershell
.\build.ps1            # compile all examples into .\build\
.\build.ps1 -Run       # compile, then run every example in order
.\build.ps1 -Run 06    # compile all, run only the example(s) matching "06"
```

Prefer CMake? `cmake -S . -B build && cmake --build build --config Release`.

Or compile a single file by hand:

```powershell
# MSVC (from a "x64 Native Tools" developer prompt)
cl /std:c++17 /EHsc /O2 06_importance_sampling.cpp

# or g++ / clang++
g++ -std=c++17 -O2 06_importance_sampling.cpp -o importance
```

The `build\` directory is disposable — delete it any time.

## The examples

Read them in order; each leans on the previous ones.

| # | File | Concept | What it demonstrates |
|---|------|---------|----------------------|
| 01 | `01_random_and_estimators.cpp` | random variables, mean, variance | uniformity histogram; sample mean → 1/2 and variance → 1/12 via Welford's online algorithm |
| 02 | `02_estimating_pi.cpp` | the **1/√N convergence law** | estimate π by dart-throwing; error tracks `√(p(1-p)/N)`, and `error·√N` stays ~constant |
| 03 | `03_monte_carlo_integration.cpp` | the estimator `I = E[f(X)/p(X)]` | integrate `eˣ` and `sin x` with no antiderivative; the core formula behind the rendering equation |
| 04 | `04_variance_and_clt.cpp` | variance of the estimator, **CLT**, confidence intervals | standard error `σ/√N`; a 100k-run experiment confirms 95% CIs cover the truth ~95% of the time |
| 05 | `05_sampling_distributions.cpp` | **inversion** & **rejection** sampling | draw from `2x`, an exponential (Beer–Lambert distances!), and `3x²`; histograms match the target densities |
| 06 | `06_importance_sampling.cpp` | **importance sampling** | same integral, two densities; a `p` shaped like `f` cuts variance ~9× while staying unbiased |
| 07 | `07_stratified_sampling.cpp` | **stratification / jitter** | one jittered sample per cell beats independent darts by ~1000× variance here |
| 08 | `08_hemisphere_sampling.cpp` | **uniform vs cosine-weighted** hemisphere sampling | cosine-weighting gives *zero* variance for the bare cosine integral — the diffuse-bounce trick |
| 09 | `09_multiple_importance_sampling.cpp` | **MIS** (balance heuristic) | two strategies, each bad on one peak; MIS combines them for ~2000× lower variance than either alone |
| 10 | `10_russian_roulette.cpp` | **Russian roulette** | terminate an infinite bounce sum early without bias; fixed truncation is shown to be too dark |
| 11 | `11_low_discrepancy.cpp` | **low-discrepancy / quasi-Monte Carlo** | van der Corput & Halton sequences; QMC error falls ~`1/N` vs random's `1/√N` on a smooth 2D integral |
| 12 | `12_gaussian_box_muller.cpp` | **sampling the normal distribution** | Box–Muller transform; histogram matches the bell curve and the 68-95-99.7 rule lands on target |
| 13 | `13_image_convergence_ppm.cpp` | **per-pixel MC, visualized** | renders PPM images at 1→256 spp, random vs stratified; *watch* the 1/√N edge noise vanish |
| 14 | `14_tabulated_cdf_sampling.cpp` | **tabulated & discrete CDF inversion** | build a CDF from a table, binary-search to sample it; continuous "sky row" + discrete light pick |
| 15 | `15_env_map_2d_sampling.cpp` | **2D env-map importance sampling** | pbrt's `Distribution2D` (marginal + conditional CDFs); PPMs *show* samples clustering on the sun |
| 16 | `16_sampling_patterns_2d.cpp` | **patterns visualized** | random vs stratified vs Halton point sets to PPM, plus a unified 2D error table (ties 07 & 11 together) |
| 17 | `17_adaptive_sampling.cpp` | **adaptive sampling** | Neyman allocation — spend the budget where variance is; ~3.7× lower error at matched cost + a density map |
| 18 | `18_resampled_importance_sampling.cpp` | **RIS & reservoirs (ReSTIR)** | importance-sample a target you can only *evaluate*; variance → ideal IS as M grows; streaming reservoir |

### Group A — BRDFs & energy (closest to `BRDF.hlsl`)
| # | File | Concept | What it demonstrates |
|---|------|---------|----------------------|
| 19 | `19_ggx_vndf_sampling.cpp` | **GGX visible-normal sampling** | Heitz VNDF; weight collapses to `F·G2/G1`, validated vs quadrature, ~1600× less variance than uniform |
| 20 | `20_fresnel.cpp` | **Fresnel: exact vs Schlick** | dielectric & conductor reflectance curves; Schlick's sub-percent (dielectric) / few-percent (metal) error |
| 21 | `21_white_furnace.cpp` | **energy conservation** | white-furnace test; single-scatter GGX loses energy at high roughness (E≤1, deficit grows) |
| 22 | `22_dielectric_refraction.cpp` | **refraction, Snell, TIR** | refract direction, Fresnel reflect/refract split, critical angle, reciprocity |

### Group B — geometry of sampling
| # | File | Concept | What it demonstrates |
|---|------|---------|----------------------|
| 23 | `23_change_of_variables.cpp` | **Jacobian; area↔solid angle** | disk maps (naive clumps vs correct/concentric); the `r²/cos` conversion behind light sampling |
| 24 | `24_nee_mis.cpp` | **NEE + light/BSDF MIS** | the Veach scene; wrong single strategy costs 200–27000×, MIS(power) stays near the best |
| 25 | `25_area_light_sampling.cpp` | **triangle & sphere lights** | barycentric `√` triangle sampling; sphere cone (solid-angle) sampling beats hemisphere by ~10⁶× |
| 26 | `26_orthonormal_basis.cpp` | **frame from a normal** | Duff 2017 branchless ONB; orthonormal to machine epsilon, carries cosine samples (mean cosθ = 2/3) |

### Group C — more variance reduction
| # | File | Concept | What it demonstrates |
|---|------|---------|----------------------|
| 27 | `27_control_variates.cpp` | **control variates** | subtract a known-integral correlated function; 61× reduction (matches `1/(1-ρ²)`) |
| 28 | `28_antithetic_variates.cpp` | **antithetic variates** | pair u with 1−u; 31× on a monotonic integrand, a no-op (0.5×) on a symmetric one |
| 29 | `29_latin_hypercube.cpp` | **Latin hypercube (N-rooks)** | stratify every axis with only N samples; 3700× reduction in 6D, beating the curse of dimensionality |
| 30 | `30_blue_noise.cpp` | **blue vs white noise** | best-candidate blue noise; wider spacing, lower variance, high-frequency error spectrum (+ PPMs) |
| 31 | `31_efficiency.cpp` | **efficiency = 1/(var·time)** | timed comparison; a low-variance but slow sampler loses to brute force — variance alone misleads |

### Group D — light transport & media
| # | File | Concept | What it demonstrates |
|---|------|---------|----------------------|
| 32 | `32_neumann_series.cpp` | **rendering eqn as a series** | `L = ΣTᵏLe`; per-bounce decomposition, truncation error = the bounce-count knob |
| 33 | `33_delta_tracking.cpp` | **heterogeneous media** | Woodcock/delta tracking with null collisions; matches analytic transmittance with no integral to invert |
| 34 | `34_equiangular_sampling.cpp` | **volumetric single scatter** | sample distance ∝ 1/d² (Kulla–Conty); variance collapses ~10³³× vs uniform |

### Group E — RNG & QMC infrastructure
| # | File | Concept | What it demonstrates |
|---|------|---------|----------------------|
| 35 | `35_sobol_owen.cpp` | **Sobol + Owen scrambling** | the (0,2)-sequence; randomized QMC keeps ~1/N convergence while being unbiased |
| 36 | `36_rng_decorrelation.cpp` | **per-pixel RNG seeding** | bad seeds → structured banding, Wang hash → clean noise (+ PPMs); adjacent-pixel correlation |

### Group F — estimator pathologies & the image
| # | File | Concept | What it demonstrates |
|---|------|---------|----------------------|
| 37 | `37_bias_vs_variance.cpp` | **bias vs consistency** | kernel density estimate (photon-mapping style); fixed bandwidth parks at a permanent bias |
| 38 | `38_fireflies_clamping.cpp` | **fireflies & clamping** | a heavy-tailed (infinite-variance) estimator; clamping trades variance for darkening bias |
| 39 | `39_reconstruction_filters.cpp` | **reconstruction filters** | box/tent/Gaussian/Mitchell; normalization + edge profile showing blur vs ringing |
| 40 | `40_linear_vs_gamma.cpp` | **average in linear space** | averaging sRGB-encoded values is ~60/255 too dark; accumulate linear, encode last (+ PPM) |

### Capstone
| # | File | Concept | What it demonstrates |
|---|------|---------|----------------------|
| 41 | `41_rng_comparison.cpp` | **PRNG vs low-discrepancy** | Wang/PCG (white noise) vs Owen-Sobol; scatter + chi-square + convergence + speed. A better PRNG does *not* converge faster — placement does; use both |
| 42 | `42_high_dimensional_qmc.cpp` | **high-dimensional QMC** | Halton collapses to stripes in high dims; digit scrambling fixes it; error vs dimension shows the curse and effective dimension (raw Halton hits 1e6 error at d=32) |

Several examples write images into `.\images\`. PPM viewers are scarce on
Windows, so `.\view_images.ps1` re-encodes every PPM to PNG (`-Open` also opens
`reference.png`). Things to look at:
- `random_0001.png` (hard, aliased edges) vs `random_0256.png` (smooth), and
  `random_0016.png` vs `stratified_0016.png` at equal cost (example 13);
- `envmap_importance.png` vs `envmap_uniform.png` — samples pile onto the sun (15);
- `pattern_random.png` vs `pattern_halton.png` — clumps vs even coverage (16);
- `adaptive_density.png` — the budget glows on the circle edges (17);
- `bluenoise_white.png` vs `bluenoise_blue.png` — clumps vs even spacing (30);
- `rng_bad.png` (a regular weave) vs `rng_good.png` (clean noise) (36);
- `gamma_compare.png` — the bottom (wrongly averaged) strip is darker (40);
- `rng_wang.png` (white-noise clumps) vs `rng_sobol.png` (even) vs `rng_lcg.png`
  (a lattice of lines) — the three categories side by side (41);
- `highdim_halton.png` (a high-dim pair collapsed onto diagonal stripes) vs
  `highdim_scrambled.png` (stripes broken by digit scrambling) (42).

## How each idea shows up in the [path tracer](https://github.com/INedelcu/PathTracingDemo)

These aren't abstract — every concept maps onto a specific piece of the renderer:

- **The estimator `E[f/p]` (03)** *is* the rendering equation in
  `RayGenerator.raytrace`: `radiance += emission * throughput` accumulates the
  Monte Carlo average, and `throughput *= albedo` carries the `f/p` weights down
  the path.
- **1/√N noise (02, 04)** is the per-pixel noise you watch shrink as
  convergence steps accumulate; the renderer averages frames with
  `lerp(prev, new, 1/(step+1))`, which is exactly the running sample mean.
- **Importance sampling (06)** is why `BRDF.hlsl` uses GGX **VNDF** sampling and
  cosine-weighted diffuse sampling instead of firing rays uniformly — sample
  where the BRDF is large.
- **Cosine-weighted hemisphere sampling (08)** is `SampleDiffuseLambert`: the
  `cos θ / π` pdf cancels the `albedo / π` Lambertian BRDF, leaving just the
  albedo as the weight — the zero-variance cancellation in example 08.
- **Inversion sampling of an exponential (05)** is the `exp(-extinction · t)`
  Beer–Lambert absorption in the glass shader.
- **Russian roulette (10)** is the throughput-based path termination in the
  ray-gen loop (`g_BounceCount*` plus the max-channel-throughput roulette).
- **MIS (09)** is the spiritual parent of the per-hit "pick a lobe by luminance"
  logic and the single-scattering estimators in the closest-hit shaders.
- **Low-discrepancy sampling (11)** is the well-distributed-samples principle
  behind production samplers (pbrt's `HaltonSampler`); this renderer's hashed
  per-pixel RNG trades that for simplicity, but the goal — avoid clumped samples
  — is the same one stratification (07) chases.
- **Sampling the normal (12)** underlies Gaussian pixel-reconstruction filters,
  and is the very distribution the CLT (04) says the estimator converges to.
- **Per-pixel convergence (13)** is the whole renderer in miniature: each pixel
  is a Monte Carlo integral, and the progressive `lerp(prev, new, 1/(step+1))`
  accumulation in `PathTracingDemo.cs` is the running mean these images visualize.
- **Tabulated CDF inversion (14) and Distribution2D (15)** are exactly how the
  renderer importance-samples its environment cubemap (a per-texel luminance
  table you can only tabulate) and selects among lights — measured data with no
  closed-form inverse, sampled by building a CDF and binary-searching it.
- **Sample patterns (16)** revisit stratification (07) and low-discrepancy (11)
  visually: the goal is the well-spread per-pixel sample sets a renderer wants
  across pixel/lens/light/bounce dimensions.
- **Adaptive sampling (17)** is the principle behind stopping early on converged
  pixels and pouring samples into noisy ones — a standard production optimization
  on top of the uniform per-pixel budget this renderer uses.
- **RIS / ReSTIR (18)** is the modern evolution of importance sampling (06) and
  MIS (09): when the ideal target (BRDF·light·visibility·cosine) can only be
  evaluated, resample cheap candidates toward it — the basis of real-time
  many-light direct lighting.
- **GGX/VNDF, Fresnel, energy, refraction (19–22)** are `BRDF.hlsl` line by line:
  `SampleSpecularGGX`'s `F·G2/G1` weight, the Schlick Fresnel, the single-scatter
  energy loss the diffuse `(1−F0)` tint compensates for, and `SampleGlassGGX`'s
  Snell/Fresnel/TIR.
- **Light-sampling geometry (23–26)** is direct lighting: the `r²/cos` area↔solid-
  angle term, light/BSDF **MIS**, per-shape light samplers, and the
  `BuildOrthonormalBasis` (Duff 2017) every hemisphere sampler rotates through.
- **Variance reduction & efficiency (27–31)** are the levers (and the honest
  scoreboard) behind every sampling choice: control variates, antithetic pairing,
  per-dimension stratification, blue-noise placement, and variance×time.
- **Transport & media (32–34)** are the bounce-count series (`g_BounceCount*`),
  heterogeneous-volume tracking, and volumetric single-scatter distance sampling —
  the general case of the glass shader's Beer–Lambert.
- **RNG & QMC (35–36)** are the sampler and the per-pixel seeding: randomized
  low-discrepancy sampling, and why the renderer hashes (pixel, frame, step).
- **Pathologies & imaging (37–40)** are the things that go visibly wrong: biased-
  but-consistent estimators, fireflies and clamping, reconstruction filters, and
  the linear-vs-sRGB averaging rule the renderer follows by accumulating linearly.

## A note on the RNG (`mc_random.h`)

All examples share a tiny **PCG32** generator — the same family pbrt uses.
It's small, fast, statistically excellent, and supports independent *streams*
(each example seeds a different sequence), so results are deterministic and
reproducible across machines. That determinism is the point while learning:
you can trust that a mismatch means a real difference, not RNG luck.

## References

- M. Pharr, W. Jakob, G. Humphreys — *Physically Based Rendering: From Theory to
  Implementation* (4th ed.), the Monte Carlo and sampling chapters.
  Free online: <https://pbr-book.org>.
- E. Veach, L. Guibas — *Optimally Combining Sampling Techniques for Monte Carlo
  Rendering* (SIGGRAPH 1995). The MIS / balance-heuristic paper (example 09).
- M. O'Neill — *PCG: A Family of Simple Fast Space-Efficient Statistically Good
  Algorithms for Random Number Generation* (2014). The RNG in `mc_random.h`.
- J. H. Halton — *On the efficiency of certain quasi-random sequences of points
  in evaluating multi-dimensional integrals* (1960). The Halton sequence (11).
- G. Box, M. Muller — *A Note on the Generation of Random Normal Deviates*
  (1958). The Box–Muller transform (example 12).
- J. Talbot, D. Cline, P. Egbert — *Importance Resampling for Global
  Illumination* (EGSR 2005). Resampled Importance Sampling (example 18).
- B. Bitterli et al. — *Spatiotemporal Reservoir Resampling for Real-Time Ray
  Tracing with Dynamic Direct Lighting* (ReSTIR, SIGGRAPH 2020). Example 18.
- E. Heitz — *Sampling the GGX Distribution of Visible Normals* (JCGT 2018).
  The VNDF sampler and `F·G2/G1` weight (example 19).
- T. Duff et al. — *Building an Orthonormal Basis, Revisited* (JCGT 2017). The
  branchless basis in example 26 and the renderer's `BuildOrthonormalBasis`.
- E. Veach — *Robust Monte Carlo Methods for Light Transport Simulation* (PhD
  thesis, 1997). NEE, the power heuristic, and efficiency (examples 24, 31).
- C. Kulla, A. Conty — *Importance Sampling of Many Lights* / *Sampling the
  Distance...* (2012). Equiangular volumetric sampling (example 34).
- E. Woodcock et al. — delta tracking (1965); B. Burley — *Practical Hash-based
  Owen Scrambling* (JCGT 2020). Examples 33 and 35.
- D. Mitchell, A. Netravali — *Reconstruction Filters in Computer Graphics*
  (SIGGRAPH 1988). The Mitchell filter in example 39.
- The `Distribution2D` marginal/conditional pattern (examples 14–15) follows
  pbrt's marginal/conditional CDF sampling.
- See the path tracer's own `README.md` "References" for the BRDF/sampling
  publications (Heitz 2018 VNDF, Duff et al. 2017, Wächter & Binder 2019, etc.).
