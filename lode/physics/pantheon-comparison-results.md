# Canonical From-Sim vs Pantheon+ Comparison Results

> **PARTIALLY UPDATED 2026-06-25**: The WS1 targeted_near_lcdm sweep (400p,
> t_start=2.9, uniform_sphere, eds_consistent=True) has RE-PINNED the isotropic
> numbers on one consistent kernel/anchor. See "Re-pinned WS1 numbers" section below.
> The Section-1 knob sweep numbers (~0.50-0.51 for GRF runs, n_sne=1339) used a
> different Pantheon cut and should not be mixed with the WS1 isotropic band (0.52,
> n_sne=1425). The "canonical numbers" table at M=855/S=37.8 is still STALE; that
> config gives chi2/dof~0.69 with current ICs (below). Do NOT cite the 0.50 table below.

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

## Re-pinned WS1 numbers (2026-06-25)

One consistent kernel: uniform_sphere, 400p, 273 steps, t_start=2.9, eds_consistent=True, n_sne=1425.

**Best isotropic (amp=0)**:
M=200/S=20: chi2/dof=0.519, growth=2.88
M=1500/S=30: chi2/dof=0.522, growth=2.88
Isotropic band: 0.52-0.54 across M=200-3000 at matched low-S (20-35 Gpc).
LCDM=0.436, EdS=0.843. Gap to LCDM: ~0.08 chi2/dof — real, not noise.

Particle-count stability at best isotropic cells: N=400→0.52, N=1000→0.53, N=2000→0.53.
No large convergence shift at low-S.

**Best overall (amp=0.5)**: M=1500/S=55, chi2/dof=0.487, growth=2.99 (PF3: growth nudge).

**Nominal M=855/S=37.8**: chi2/dof ≈ 0.69 at current ICs (not near-LCDM; see note).

**Figures**: results/figures/ws1/ mu_z_panel_M200_S20_targeted_near_lcdm_iso_{400,2000}p.png

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

The from-sim External-Node model produces **effective dark energy**: its best
physical configs land NEAR LCDM (~0.48-0.51 vs 0.43 chi2/dof) and DECISIVELY
reject the Einstein-de Sitter no-dark-energy null (~0.85, which the SN data
strongly disfavor). For a toy model, landing near LCDM and rejecting no-DE is a
genuine success. It does **not EXACTLY match LCDM** (fits Pantheon+ slightly
worse), and it is **degenerate in M/S^3**: SNe constrain the effective expansion
(~effective Omega_Lambda), not M and S separately, so chi2/dof is essentially
flat across M from 20 to 200000 once each M is paired with its growth-anchored S
(see [../scripts/parameter-sweep.md](../scripts/parameter-sweep.md) Stage-3 table).
NOTE: the exact chi2 numbers in this file are config/init-sensitive and predate the
current kernel/anchor; they are being **re-pinned on one consistent kernel/anchor
with graphs** by the deeper-exploration sweep — see
[../plans/pinned-findings.md](../plans/pinned-findings.md) (PF4) and
[../plans/overarching-sweep.md](../plans/overarching-sweep.md). Do not over-claim an
LCDM match OR a failure until the re-pinned numbers + figures land.

## Section-1 Pantheon knob sweep (910 sims, June 2026)

Full-factorial sweep over M ∈ {50,100,250,500,700,750,800,850,900,1000},
S ∈ {20..80 step 5}, amplitude ∈ {0,0.25,0.5,0.75}, seed ∈ {42,7},
init_distribution="grf", particles=400, n_steps=273, t_start=2.9. Growth anchor ON.
Executed by `pantheon_knob_sweep.py` → `results/sweep_results_pantheon.csv` (isotropic
rows, load_best_config-compatible) + `results/knob_sweep_summary.csv` (all 910 rows).
Runtime: ~1.8 s/sim probe → 1638 s estimated; actual ~25 min (cache assisted for amplitude>0).

### HEADLINE (defensible, paper-quotable)
The from-sim Pantheon+ result is the best **isotropic** (amplitude=0) config:
**M=50, S=80, chi2/dof=0.5056**, R2=0.99607, growth=3.085, n_sne=1339. This is the
number to cite. It controls for selection bias (min over 130 isotropic configs, not
the min over the full 910-config pool). The M/S³ degeneracy holds: per-M best chi2/dof
at amp=0 spans only 0.5056..0.5063; the good band (chi2/dof<0.55) is 100/130 rows,
all M, S=25..80.

### Verified status of node_mass_amplitude / node_mass_seed (skeptical re-audit)
The "amplitude>0 systematically beats isotropic" claim was re-investigated with
controlled experiments. Conclusion: the effect is **REAL but indirect — it is a growth
nudge, not an anisotropy fit improvement** — and the earlier framing was misleading.

What is TRUE (proven):
- **No bug.** End-to-end through `CosmologicalSimulation`, `HMEAGrid.get_masses()` sums
  to exactly 26·M_ext_kg and mean==M_ext_kg for amp=0 AND amp=0.5/0.75 (std/mean≈0.38).
  Mean-preservation is intact in the real force path, not just the unit test.
- **No particle confound.** Particle positions AND velocities are byte-identical across
  node_mass_amplitude and across node_mass_seed (node_masses() uses an independent
  default_rng drawn AFTER the particle cloud; it never touches the global RNG). So
  "seed=42 wins" is NOT a lucky cloud. Guarded by
  `tests/test_node_masses.py::TestSimPathNodeMassInvariants`.
- **Amplitude raises the realized growth factor toward the physical target** wherever the
  tidal field is strong (small S / large M): e.g. M=1000,S=50 growth 3.078→3.100→3.139→
  3.191 as amp 0→0.25→0.5→0.75. At weak tidal field (M=50,S=80) growth is FLAT
  (3.0849→3.0852) and chi2/dof is FLAT (0.5054..0.5057). The traceless/shear argument
  holds to LINEAR order; amplitude injects variance that back-reacts on the bulk
  RMS-radius a(t) at SECOND order only when nodes are close.

Why the chi2 "improvement" is mostly selection + a growth artifact:
- The "overall best" M=1000,S=50,amp=0.5 (chi2/dof=0.4619) is computed over only **1295
  SNe**, vs **1339** for the isotropic best — the higher growth raises z_min_cover and
  drops the lowest-z SNe. On a COMMON (identical 1295-SNe) window the gap shrinks:
  amp=0 → 0.5025, amp=0.5 → 0.4619 (Δ=0.040). Of the raw 0.054 delta, ~0.013 is the
  changing-window/selection effect; ~0.040 survives as a genuine better fit ON the same
  SNe — but it is genuine **only because all from-sim runs sit BELOW the physical growth
  3.304, so any growth INCREASE moves toward the target** (Pearson corr |growth−3.304|
  vs chi2/dof = 0.31 over 821 configs).
- **Noise floor** (chi2/dof across particle realizations at amp=0, M=1000/S=50) = 0.0008,
  so 0.040 is far above shot noise — the effect is not noise, it is real growth physics.

The "seed=42 ~0.005 advantage" is **NOT systematic** — it is misread. Paired seed42−seed7
(341 pairs, same M/S/amp): median +0.0031 (seed42 better in the typical pair, 295/341)
but MEAN −0.018 (seed7 better on average; outliers dominate), std 0.12. nm_seed is a
noisy orientation knob with no defensible preferred value; do NOT claim a seed preference.

**Interpretation for the paper**: amplitude is NOT an independent fit knob that "beats"
the isotropic model. It is degenerate with M/S — it modestly raises the realized growth
toward physical and the SN fit follows, but the model still sits above LCDM (0.46–0.51 vs
0.44) and never crosses it. Quote the isotropic 0.5056 as the headline. The grf
init_distribution at 400p is chi2/dof-indistinguishable from uniform_sphere.

**Anisotropy-showcase config** (Section 4 — its job is to MOVE shear/dipole, not the
chi2): **M=1000, S=50, amplitude=0.75, nm_seed=42, init='grf'** — the strongest-tidal,
largest-spread config (std/mean≈0.38), where anisotropy is maximal. (The flat-chi2 point
M=50,S=80,amp=0.75 has the cleanest "same chi2, only shear moves" story but the weakest
shear; pick by what the figure needs to show.)

## Related
- [./hubble-diagram-nbody.md](./hubble-diagram-nbody.md) — kernel, anchor, EdS null, module map
- [../scripts/parameter-sweep.md](../scripts/parameter-sweep.md) — Stage-3 M/S^3 degeneracy table + knob sweep harness
- [./hubble-diagram.md](./hubble-diagram.md) — the semi-analytic (circular) sibling
