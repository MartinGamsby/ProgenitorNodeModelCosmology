# From-Sim N-body Hubble Diagram vs Real Pantheon+ (the NON-circular test)

The genuine from-DATA test: build the External-Node model's mu(z) directly from
the REAL N-body simulation's scale-factor history a(t), then score it against
real Pantheon+SH0ES SNe. Sibling of the semi-analytic test in
[./hubble-diagram.md](./hubble-diagram.md).

## Why this is the real test (circularity distinction — READ THIS)

The semi-analytic `external_node` curve in [./hubble-diagram.md](./hubble-diagram.md)
uses `Omega_Lambda_eff = G*M/(S^3 H0^2)` plugged into the LCDM E(z) form. At the
data-matching config (M=855, S=37.8 -> Omega_Lambda_eff ~= 0.70) that curve is
**mathematically identical to LCDM by construction** — it is CIRCULAR. It cannot
fail to look like LCDM; it IS LCDM with a relabeled Omega.

The from-sim curve here is **NOT circular**: it integrates whatever the N-body
particles actually did. The dynamics (internal gravity + 26 HMEA tidal nodes +
Hubble drag) produce a(t); if the mechanism does NOT mimic Lambda, the curve
deviates and the chi^2 worsens. This is the test that can actually fail.

| Curve | Source | Circular? |
|-------|--------|-----------|
| `external_node` (analytic shortcut) | Omega_Lambda_eff in LCDM E(z) | YES — == LCDM at 0.70 |
| `external_node_nbody` (from sim)    | integrate real a(t) | NO — real dynamics |

The Stage-1 script plots BOTH so the circular shortcut and the honest from-sim
curve sit side by side.

## sim a(t) -> mu(z) kernel (cosmo/sim_distance.py)

`sim_to_distance_modulus(z_target, a, t_Gyr, t_start_Gyr, today_index=-1, today_tol_Gyr=0.2)`
Pure function, no I/O / plotting / sim imports. Steps:

1. Absolute time: `t_abs = t_start_Gyr + t_Gyr` (sim t_Gyr is RELATIVE, starts 0).
2. Guard: snapshot at `today_index` must be within `today_tol_Gyr` of 13.8 Gyr,
   else ValueError. (Caller MUST run the sim so t_start + t_duration == 13.8.)
3. Renormalize to today: `a_today = a / a[today_index]` (sim ships a[0]=1 at
   t_start; we need a=1 TODAY). `z_snap = 1/a_today - 1` (z=0 today).
4. Comoving distance WITHOUT differentiating a (avoids the smoothing edge
   artifacts in [../numerics/expansion-rate-calculation.md](../numerics/expansion-rate-calculation.md)):

   ```
   D_C(t_i) = c * integral_{t_i}^{t_today} dt / a_today(t)
   ```

   via `scipy.integrate.cumulative_trapezoid` on the snapshot grid.
5. `d_L = (1+z) * D_C` (flat — RMS-isotropic sim has no curvature term),
   `mu = 5*log10(d_L/Mpc) + 25`.
6. Interpolate onto `z_target`, HARD-CLIP to the sim-covered z-range (no
   extrapolation). Returns `in_range` mask aligned to `z_target`, `z_cover`.

H0 is NOT a parameter: integrating physical `c*dt/a` carries the scale; the
absolute mu offset is marginalized downstream (`fit_offset`). Only the SHAPE of
a(t) matters. Validated: feeding analytic-LCDM a(t) reproduces
`model_distance_modulus(z, "lcdm")` to < 0.03 mag after offset removal.

## Stage 1 — gating script (hubble_diagram_nbody.py)

Runs ONE External-Node N-body config, converts a(t)->mu(z), clips Pantheon+ to
the sim z-range, evaluates 4 curves via the precomputed-mu engine entry point
`cosmo.hubble_diagram.evaluate_precomputed(z, mu_obs, sigma, mu_model)`
(same offset/chi^2/R^2 helper as `evaluate_model`), and prints a DEVIATION
diagnostic (max/RMS of mu_sim - mu_LCDM vs typical data sigma).

Timing constraint: `t_duration = 13.8 - t_start` so the last snapshot is z=0.
dt = t_duration/n_steps must stay < 0.05 Gyr (leapfrog limit) — script errors
with the needed n_steps otherwise.

**Full-coverage DEFAULT: t_start=2.9 Gyr, M=855, S=37.8, particles=2000,
n_steps=273** (dt~0.040 Gyr, z up to ~2.3 — full Pantheon+). For a quick run use
`--t-start 5.8 --particles 80 --n-steps 300`.

CLI extras:
- `--from-best-config <sweep_csv>` loads the best row (lowest `chi2_dof`, else
  `diff_pct`) from a pantheon sweep CSV and uses its M/S/centerM as defaults;
  explicit `--M/--S/--center-node-mass` still override.
- A **JSON sidecar** `<png>.summary.json` is written next to the PNG with config
  (M,S,centerM,t_start,particles,n_steps), coverage (n_in_range, z_cover),
  per-model {chi2,dof,chi2_dof,R2,DeltaM}, deviation diagnostics, and the growth
  anchor {growth_factor, growth_target, anchor_ok} — the machine-readable source
  the paper cites. Canonical numbers: [./pantheon-comparison-results.md](./pantheon-comparison-results.md).
- The residual panel uses LCDM as the zero reference (Delta-mu vs LCDM). The
  Einstein-de Sitter null (Omega_m=1) is the plotted "no dark energy" comparison
  (see EdS-null section below). `node_mass_amplitude` / `init_distribution` are
  also exposed as flags (uniform / uniform_sphere defaults).

### Honest Stage-1 result (the gating answer)

At t_start=5.8 (sim reaches only z ~ 0.96), 1464 in-range SNe:

| Curve | chi2/dof | R2 |
|-------|----------|----|
| Ext-Node N-body (from sim) | 0.476 | 0.9968 |
| LCDM (analytic) | 0.442 | 0.9970 |
| Analytic shortcut (==LCDM) | 0.442 | 0.9970 |
| Matter-only | 0.523 | 0.9965 |

Max |mu_sim - mu_LCDM| = 0.16 mag, RMS 0.097 mag; typical sigma 0.21 mag
=> ~0.77 sigma. (The "Matter-only" row above was the OLD open Omega_m=0.3 null,
since replaced by Einstein-de Sitter — see anchor/null sections below.)

**Full-resolution run** (2000 particles, t_start=2.9 Gyr, full z to ~2.1, M=855):
from-sim chi2/dof=0.494 vs LCDM 0.431; max|mu_sim-mu_LCDM|=0.21 mag, RMS 0.156,
sigma~0.21 => ~1.0 sigma — a DETECTABLE (not conclusive) deviation, and the model
fits the SNe slightly WORSE than LCDM over the full range. CONCLUSION (honest):
the real mechanism produces effective dark energy (sits with LCDM, far from the
matter-only null) but does NOT beat LCDM; at observable z it is ~indistinguishable
from / marginally worse than LCDM. Real, not circular.

## Stage 2 — earlier t_start for full Pantheon+ coverage

Validated (no production-code change) that the initial-conditions + auto-damping
machinery `(t_start/13.8)^0.135` and the never-exceed-LCDM constraint hold down
to a **safe floor t_start = 2.9 Gyr** (a~0.310, z_max ~ 2.23 — covers full
Pantheon+ z <= ~2.3). `n_steps = ceil((13.8 - t_start)/0.04)` keeps dt <= 0.04
Gyr. Damping formula needed NO modification; t_start=5.8 behavior unchanged. See
[./initial-conditions.md](./initial-conditions.md).

## Stage 3 — from-data chi^2 sweep objective

`SweepConfig(objective="pantheon")` scores each config by chi^2 of its sim-derived
mu(z) vs real Pantheon+ instead of R^2 vs the LCDM baseline. Additive; the lcdm
objective stays the default and unchanged. See
[../scripts/parameter-sweep.md](../scripts/parameter-sweep.md).

## Physical expansion ANCHOR (critical — fixes an under-constrained comparison)

`sim_to_distance_modulus` renormalizes a=1 at the LAST snapshot ("today") and only
compares the SHAPE over the observed z-range. By itself that lets a RUNAWAY config
(e.g. M=100000 expanding ~25000x in the 10.9 Gyr) renormalize and fit the z<z_start
window while predicting a nonsensical history — so the bare Stage-3 sweep was nearly
INSENSITIVE to (M,S) and wandered to absurd M. The missing constraint: total
expansion over [t_start, today] must equal the real `1+z(t_start)` (~3.2x for
t_start=2.9 Gyr).

`cosmo.parameter_sweep.expected_growth_factor(t_start_Gyr)` returns the physical
`a(today)/a(t_start)` from the LCDM background. `compute_pantheon_metrics` rejects
(worst-case score) any config whose `a_curve[-1]/a_curve[0]` deviates from it by
more than `GROWTH_ANCHOR_TOL` (0.20). Rejected metrics carry `growth_factor` and
`growth_target` for transparency; both are also CSV columns. The from-sim script
prints the same anchor check ("PHYSICAL" / "UNPHYSICAL, would be rejected").
This is NOT velocity calibration masking the nodes — a(t) is in fact hugely
sensitive to M (growth 3.1x at M=20 vs 25000x at M=100000); the flaw was the
floating-"today" normalization, now anchored.

## The matter-only null: use Einstein-de Sitter, NOT open Omega_m=0.3

`model_distance_modulus(z, "matter_only")` is Omega_m=0.3 / Omega_k=0.7 (OPEN). An
open low-density universe is nearly degenerate with LCDM in the SN Hubble diagram
(negative curvature's sinh term mimics dark energy) — a MISLEADING null that hugs
LCDM. The meaningful "dark energy is required" null is
`model_distance_modulus(z, "einstein_de_sitter")` = flat Omega_m=1, which the SN
data decisively rule out (sweeps to ~-0.6 mag vs LCDM by z~2). `hubble_diagram_nbody.py`
plots Einstein-de Sitter as its matter-only null.

## Module map

| File | Role |
|------|------|
| `cosmo/sim_distance.py` | Pure a(t) -> mu(z) kernel (`sim_to_distance_modulus`). No I/O. |
| `cosmo/hubble_diagram.py` | `evaluate_precomputed` (precomputed-mu entry) + shared `_evaluate_from_precomputed_mu` helper used by `evaluate_model` too. |
| `cosmo/factories.py` | `results_to_sim_result` populates `SimResult.a_curve` from the sim's full a array. |
| `hubble_diagram_nbody.py` | Comparison tool: run sim -> mu(z) -> clip -> 4-curve compare (incl. LCDM residual reference + EdS null) -> deviation diagnostic -> 2-panel PNG + JSON sidecar. `--from-best-config` loads M/S/centerM from a sweep CSV; t_start=2.9 full-coverage default. |

## Tests

- `tests/test_sim_distance.py` (14) — LCDM round-trip < 0.03 mag, normalization,
  z-mapping, range clipping, today_tol guard, input validation, mu monotonicity.
- `tests/test_hubble_diagram_nbody.py` (15) — evaluate_precomputed == evaluate_model;
  deviation-metric sanity; 3 slow integration smokes (real Pantheon+, tiny sim).
- `tests/test_early_start_validation.py` (9) — Stage-2 safe-floor / dt / damping.
- `tests/test_parameter_sweep_pantheon.py` (18) — pantheon scorer, end-to-end
  pantheon sweep, cache-key objective isolation (lcdm vs pantheon disjoint), and
  the growth anchor (runaway rejected, physical accepted, expected_growth_factor).

## Related
- [./hubble-diagram.md](./hubble-diagram.md) — the semi-analytic sibling test
- [./initial-conditions.md](./initial-conditions.md) — Stage-2 validated t_start range
- [../scripts/parameter-sweep.md](../scripts/parameter-sweep.md) — Stage-3 objective
- [../numerics/expansion-rate-calculation.md](../numerics/expansion-rate-calculation.md) — why we integrate a(t), not differentiate it
