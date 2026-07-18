# Realistic Initial Conditions

## Status

Deliverable C1 implemented. C2 is a documented future direction (see below).

## C1 — GRF + Zel'dovich (IMPLEMENTED)

`init_distribution="grf"` in `SimulationParameters` activates a Gaussian random field seeded
with BBKS approximate LCDM P(k) + Zel'dovich displacement.

**SUPPORT GEOMETRY (the WS5 fix).** `sample_grf(..., support="sphere")` is the DEFAULT.
It confines the cloud to the SAME sphere radius `(box/2)/sqrt(3/5)` that `uniform_sphere`
uses (masks the Lagrangian grid by undisplaced radius; radial-clips excursions), so GRF is a
CLUSTERED SPHERE and the ONLY difference vs uniform is clustering. The legacy `support="box"`
(a GRF-perturbed CUBE) is kept for reproducibility/comparison only — DO NOT use it for sweeps.
The box variant has a fat radial tail (max ~1.64 R, ~11% of particles beyond 1.3 R vs uniform's
~0.1%), which is what produced the WS5 swing (below).

**Cache axis (keyed==run).** Because the default changed box->sphere, the two supports get
DISTINCT sweep cache keys so a pre-fix box cache is never served for the new sphere a(t):
`SweepConfig.grf_support` (JSON `grf_support`, default `"sphere"`) threads into the sim via
`init_kwargs={"support": ...}` (grf only; uniform_sphere unaffected) AND into
`build_cache_name`, which appends a `sphsup` token ONLY for grf+sphere. Legacy `"box"` keeps
the bare `grfinit` key (old caches stay valid as box). `PHYSICS_CACHE_VERSION` stays `"v3"`
(no bump). See [../scripts/parameter-sweep.md](../scripts/parameter-sweep.md).

**Is GRF broken? NO — diagnosed (WS5, item 11).** The GRF density field is healthy: mean(delta)
~1e-21, no NaN, P(k) decays large->small scale, Zel'dovich displacement RMS-controlled to exactly
0.5 cell, RMS-norm exact, COM~0. The handover's "swing" (uniform ~0.52 vs grf ~1.56 chi2/dof at
M=1500/S=30) was REAL and split between two causes:
  - **H4 (setup, NOW FIXED, ~40% of the swing):** the legacy GRF filled a CUBE, not a sphere.
    cube-vs-sphere geometry (not clustering) put extra mass in the corners; in a strong tidal
    field those far-out particles change the bulk a(t). `support="sphere"` cuts delta(grf-uni)
    at M=1500/S=30 from ~+0.53 (box) to ~+0.30 (sphere).
  - **H1 (real physics, residual ~+0.30):** even a properly sphere-confined CLUSTERED cloud fits
    slightly worse than a smooth sphere in the STRONG-field cell — particles piled at density
    peaks feel different differential node tidal forcing. It is LOCALIZED to the strong-field
    cell (weak cell M=100/S=60: delta(grf-uni) ~+0.001, i.e. GRF == uniform) and does NOT shrink
    with N (N=1000..4000 stable, ~0.01-0.06 seed spread) — so NOT H3 (noise) and NOT H2 (not at
    runaway: growth ~2.75 vs anchor ~3.30).

**Key invariant (CLARIFIED).** In the WEAK / isotropic regime the background (growth, chi2_dof)
IS clustering-insensitive: GRF == uniform. The clustering signal otherwise lives in STRUCTURE
(shear / Hubble dipole, `anisotropy_report.py`). The strong-field cell is the EXCEPTION: there,
clustering moves the bulk a(t) by a real, N-converged amount — characterized, not buried.
Evidence: `_generate_ws5_grf.py` -> `results/figures/ws5/grf_vs_uniform.{png,csv}` (gitignored).

**Convergence check** (`convergence_check.py`): sweeps N ∈ {1 000, 10 000} (+ optionally 100 000
with `--full`). At N=1 000 and N=10 000, t_start=5.8 Gyr, 80 steps (dt=0.025 Gyr):

| N | growth | anchor | ≤LCDM | dt<0.05 | wall(s) |
|---|--------|--------|-------|---------|---------|
| 1 000 | 1.2183 | PASS | PASS | PASS | ~2 s |
| 10 000 | 1.2183 | PASS | PASS | PASS | ~20 s |

Growth spread across N-ladder: **0.000 %** — isotropic background fully converged.

**never-exceed-LCDM reference is ANALYTIC** (particle-count independent): the `≤LCDM`
gate compares the GRF run's RMS at every step against the analytic LambdaCDM Friedmann
a(t) (`solve_friedmann_at_times`, Omega_Lambda=0.7), scaled to the GRF run's OWN initial
RMS — mirroring `tests/test_early_time_behavior.py` (test_lcdm_nbody_vs_analytic_lcdm,
test_matter_only_decelerates_correctly). It is NOT an N-body LCDM proxy, so the gate
reflects physics, not shot noise. The `max LCDM ratio` is a stable 1.000000 for all N.

Run the full check:
```
PYTHONIOENCODING=utf-8 python convergence_check.py --full --png
```

## C2 — Galaxy Catalog Sampling (NOT IMPLEMENTED — FUTURE DIRECTION)

Sampling a real galaxy catalog (SDSS, 2MRS, etc.) to initialise particle positions.

**Recipe**:
1. Download a public redshift survey subset (e.g. 2MRS ~45 000 galaxies).
2. Convert (RA, Dec, z) → comoving Cartesian coordinates with the LCDM angular diameter
   distance.
3. Rescale comoving positions into the simulation box: x_sim = x_comoving / x_max × box/2.
4. Subsample to N particles (importance sampling weighted by luminosity or random).
5. Feed raw positions to the existing shared centering + RMS-norm block (unchanged).

**CRITICAL CAVEATS**:
- Survey geometry (angular mask, magnitude limit, redshift completeness) and selection
  function imprint a SPURIOUS anisotropy that is mask-geometry, not physics.
- A real-catalog init MUST be kept OFF the shear/dipole anisotropy diagnostic path
  (`anisotropy_report.py`) or clearly flagged as "mask-contaminated".
- Input validation: accept only plain CSV/FITS paths — NO pickle/eval paths (untrusted
  local input, security invariant from Section-6 plan).
- Keep C2 behind a `--catalog` CLI flag so default (uniform_sphere) and grf are
  unaffected.

## Files

- `cosmo/initial_distributions.py` — GRF sampler (`sample_grf`, BBKS P(k), Zel'dovich) with
  `support="sphere"` (default) / `"box"` (legacy); read-only DIAGNOSTIC helpers
  `grf_density_field` + `grf_field_stats` (delta moments, P(k) decay, displacement RMS/cell)
- `cosmo/constants.py` — `SimulationParameters.init_distribution`, `init_kwargs` (forwards `Ng`,
  `support` to `sample_grf`)
- `cosmo/particles.py` — `_init_uniform_sphere`, `_init_grf` branch in `_initialize_particles`
- `cosmo/simulation.py` — passes `init_distribution` / `init_kwargs` to `ParticleSystem`
- `cosmo/cli.py` — `--init-distribution` arg + `args_to_sim_params` forwarding
- `convergence_check.py` — standalone N-ladder convergence harness (barnes_hut auto)
- `_generate_ws5_grf.py` — the WS5 GRF investigation: Part A field statistics (no sim) + Part B
  swing decomposition (authoritative `compute_pantheon_metrics`, uniform vs grf-sphere vs
  grf-box across {config, seed, N}). Output `results/figures/ws5/grf_vs_uniform.{png,csv}`.
- `tests/test_realistic_init.py` — fast unit tests (incl. `TestGRFFieldStats`, `TestGRFSupport`);
  slow N-ladder marked `@pytest.mark.slow`. `tests/test_matter_only_consistency.py` —
  `test_eds_invariant_holds_for_grf_init` (M=0 == EdS holds for GRF).

## Links

- [initial-conditions.md](./initial-conditions.md) — damping, calibration, safe t_start floor
- [barnes-hut-optimization.md](./barnes-hut-optimization.md) — force method auto-selection
