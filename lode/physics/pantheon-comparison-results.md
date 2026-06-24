# Canonical From-Sim vs Pantheon+ Comparison Results

The publication-style numbers the paper cites for the from-sim N-body mu(z) vs
REAL Pantheon+SH0ES test. Produced by the comparison tool `hubble_diagram_nbody.py`
(see [./hubble-diagram-nbody.md](./hubble-diagram-nbody.md) for the kernel/method).

## The tool

`hubble_diagram_nbody.py` runs ONE External-Node N-body sim, converts a(t)->mu(z)
via `cosmo.sim_distance.sim_to_distance_modulus` (D_C = c*integral(dt/a), no
differentiation), clips Pantheon+ to the sim-covered z-range, offset-marginalizes
each model's additive magnitude (`DeltaM`), and reports chi2/dof + R2. It saves a
2-panel PNG (mu(z) + Delta-mu-vs-LCDM residual panel) and a **machine-readable JSON
sidecar** (`<png>.summary.json`) with config + per-model stats + growth anchor.
`--from-best-config <sweep_csv>` loads the best (M,S,centerM) from a pantheon sweep.

## The default config

Full-coverage default: `t_start=2.9 Gyr` (z up to ~2.3), UNIFORM external nodes
(`node_mass_amplitude=0` => all 26 nodes = M_ext_kg), `init_distribution=uniform_sphere`,
M=855, S=37.8 (Omega_Lambda_eff=0.699), centerM=1. Particles/steps only need to be
enough for a(t) to converge; chi2/dof is shape-driven and stable across particle count.

## Canonical numbers (offset-marginalized chi2/dof)

Config that produced the table below: **t_start=2.9, M=855, S=37.8, centerM=1,
particles=400, n_steps=273, seed=42, uniform nodes, uniform_sphere init**.
(400p/273steps is the smallest config giving stable chi2/dof; the 2000p default
reproduces these to ~0.01. z coverage = [0.024, 2.095], 1333 SNe in range, dof=1332.)

| Model | chi2 | chi2/dof | R2 |
|-------|------|----------|-----|
| Ext-Node N-body (from sim) | 666.9 | **0.5007** | 0.99607 |
| LCDM (analytic) | 582.3 | **0.4372** | 0.99651 |
| Analytic shortcut (==LCDM) | 582.2 | 0.4371 | 0.99652 |
| Einstein-de Sitter NULL (Omega_m=1) | 1124.8 | **0.8444** | 0.99340 |

Growth anchor: model a(today)/a(t_start) = **3.095** vs physical target **3.304**
(6.3% off, within GROWTH_ANCHOR_TOL=0.20 => **anchor_ok = True / PHYSICAL**).
Deviation from LCDM: max 0.205 mag, RMS 0.152 mag vs typical sigma 0.206 mag
=> ~1.0 sigma (marginal, borderline detectable, not conclusive).

## Honest verdict

The from-sim External-Node model produces **effective dark energy**: it sits
right with LCDM (0.50 vs 0.43 chi2/dof) and is FAR from the Einstein-de Sitter
no-dark-energy null (0.84, which the SN data decisively disfavor). But it does
**NOT beat LCDM** — it fits Pantheon+ slightly WORSE (0.50 vs 0.43). And it is
**degenerate in M/S^3**: SNe constrain the effective expansion (~effective
Omega_Lambda), not M and S separately, so chi2/dof ~ 0.48-0.50 is essentially
flat across M from 20 to 200000 once each M is paired with its growth-anchored S
(see [../scripts/parameter-sweep.md](../scripts/parameter-sweep.md) Stage-3 table).

## Related
- [./hubble-diagram-nbody.md](./hubble-diagram-nbody.md) — kernel, anchor, EdS null, module map
- [../scripts/parameter-sweep.md](../scripts/parameter-sweep.md) — Stage-3 M/S^3 degeneracy table
- [./hubble-diagram.md](./hubble-diagram.md) — the semi-analytic (circular) sibling
