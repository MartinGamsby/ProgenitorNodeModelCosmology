# Realistic Initial Conditions

## Status

Deliverable C1 implemented. C2 is a documented future direction (see below).

## C1 — GRF + Zel'dovich (IMPLEMENTED)

`init_distribution="grf"` in `SimulationParameters` activates a Gaussian random field seeded
with BBKS approximate LCDM P(k) + Zel'dovich displacement.

**Key invariant**: The isotropic background (growth factor, chi2_dof) is INSENSITIVE to
clustering. The value of GRF realism shows up in STRUCTURE — shear / Hubble dipole in
`anisotropy_report.py` — not in chi2_dof or a(t)/mu(z). Do not treat a stable isotropic
metric as a bug.

**Convergence check** (`convergence_check.py`): sweeps N ∈ {1 000, 10 000} (+ optionally 100 000
with `--full`). At N=1 000 and N=10 000, t_start=5.8 Gyr, 80 steps (dt=0.025 Gyr):

| N | growth | anchor | ≤LCDM | dt<0.05 | wall(s) |
|---|--------|--------|-------|---------|---------|
| 1 000 | 1.2183 | PASS | PASS | PASS | ~2 s |
| 10 000 | 1.2183 | PASS | PASS | PASS | ~20 s |

Growth spread across N-ladder: **0.000 %** — isotropic background fully converged.

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

- `cosmo/initial_distributions.py` — GRF sampler (BBKS P(k), Zel'dovich, `sample_grf`)
- `cosmo/constants.py` — `SimulationParameters.init_distribution`, `init_kwargs`
- `cosmo/particles.py` — `_init_uniform_sphere`, `_init_grf` branch in `_initialize_particles`
- `cosmo/simulation.py` — passes `init_distribution` / `init_kwargs` to `ParticleSystem`
- `cosmo/cli.py` — `--init-distribution` arg + `args_to_sim_params` forwarding
- `convergence_check.py` — standalone N-ladder convergence harness (barnes_hut auto)
- `tests/test_realistic_init.py` — fast unit tests; slow N-ladder marked `@pytest.mark.slow`

## Links

- [initial-conditions.md](./initial-conditions.md) — damping, calibration, safe t_start floor
- [barnes-hut-optimization.md](./barnes-hut-optimization.md) — force method auto-selection
