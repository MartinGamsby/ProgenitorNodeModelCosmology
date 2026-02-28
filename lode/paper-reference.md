# Paper Reference

**File**: docs/paper.tex

Draft paper defining theoretical framework and goals the code tests.

## Core Thesis

Replace dark energy (Λ) with classical tidal forces from trans-observable Hyper-Massive External Attractors (HMEAs) in virialized meta-structure.

**Key claim**: Dark energy not inevitable—classical gravity can reproduce ΛCDM expansion.

**Differentiator from prior work**: Single discrete lattice geometry simultaneously explains cosmic acceleration AND three independent anomalies (Hubble Tension, Axis of Evil, dark flow) with no additional parameters. Prior proposals (Mersini-Houghton 2009, modified inflation models, local void models) address individual anomalies with purpose-built mechanisms. Unification under one geometric structure is novel. Introduction explicitly states this.

## Physical Model

**Virialized Meta-Structure**: Crystal-lattice-like arrangement of HMEA nodes beyond observable universe. Stable, gravitationally balanced (like relaxed galaxy cluster), irregular in detail. HMEAs described as ancient black holes far older than 13.8 Gyr (possibly quadrillions of years).

**Progenitor Hypothesis** (sec:progenitor): Big Bang = destabilization of a node in this meta-structure. Pre-existing equilibrium explains isotropy without fine-tuning. Introduction forward-references this.

**Tidal Acceleration** (sec:tidal): Expanding universe climbs potential gradient of nearest HMEA. Only the closest node matters (virialized equilibrium cancels at center; nearest dominates via 1/d³ scaling; edge nodes ~35%, corner ~19% contribution vs face). Single-node derivation is conservative—adding nodes would increase tidal force, predicting smaller M or S. Linear scaling: a_tidal ≈ (2 G M_ext / S³) × R. Mathematically equivalent to Λ term: H₀² Ω_Λ R. N-body sim uses all 26 nodes.

## Parameter Estimates (from paper)

Analytical target: M_ext ≈ 5×10⁵⁵ kg, S ≈ 39 Gpc

### Optimization Tradeoff (sec:tradeoffs)
Size R² is an integrated quantity (smooth, forgiving). Expansion rate R² is a derivative quantity (physically demanding). Optimizing only for size R² can produce configs that nail the size curve but follow a different H(t) path. Balanced optimization weights both.

**Balanced-optimization configs** (primary):
- M=9000×M_obs, S=38 Gpc: 98.02% endpoint, R²_size=0.9954, R²_rate=0.9630
- M=875×M_obs, S=24 Gpc: 97.92% endpoint, R²_size=0.9951, R²_rate=0.9603
- M=92×M_obs, S=15 Gpc: 98.67% endpoint, R²_size=0.9966, R²_rate=0.9565

**Size-only optimization** (secondary):
- M=800×M_obs, S=22 Gpc: 99.85% endpoint, R²_size=0.9991, R²_rate=poor

**Matter-only baseline**: 96.06% endpoint, R²_size=0.9890, R²_rate=0.8350 (full 8 Gyr period)

Sweep range: M=92-9000+, S=15-70 Gpc, 2000 particles, 8 Gyr (t=5.8→13.8)
Expansion rate comparison inherently approximate: isotropic RMS vs Friedmann H(t), model predicts anisotropy, Hubble tension means target H(t) itself uncertain.

## What Code Tests

**Success criteria** (from sec:numerical):
- Late-time acceleration (8 Gyr: t=5.8→13.8 Gyr) ✓
- ΛCDM expansion match: R²_size>0.99, R²_rate>0.95 (balanced) ✓
- Realistic H₀ ≈ 70 km/s/Mpc ✓
- Classical gravity only (no exotic physics) ✓
- Multiple parameter solutions (mechanism robustness) ✓
- Matter-only comparison (validates acceleration mechanism) ✓

**Validation checks**:
- Matter-only never exceeds ΛCDM at any timestep (physics constraint)
- R² metric (coefficient of determination) for both size and expansion rate
- RMSE for absolute deviation
- COM drift monitoring, runaway particle detection

**Explicit non-goals** (sec:limitations):
- Early universe (inflation, nucleosynthesis, baryogenesis)
- CMB power spectrum / Planck constraints
- Structure formation / density perturbations
- BAO standard ruler measurements
- Full GR treatment (currently Newtonian)

## Scope Statement

**Toy model**: Proof-of-concept mechanism validation, not complete cosmological theory.

**Philosophy**: "We do not claim this model represents physical reality. Rather, we demonstrate that alternatives to dark energy exist using only classical physics."

**Code purpose**: Generate data validating viability of tidal acceleration mechanism. Explore parameter space (M, S). Test theoretical claims before investing in full relativistic formulation.

## Key Sections for Code Alignment

- sec:tidal / sec:estimation: Tidal acceleration formula
- sec:framework: Grid configuration (3×3×3 lattice justification)
- sec:results: Primary balanced configs M=9000/S=38, M=875/S=24, M=92/S=15
- sec:tradeoffs: Optimization tradeoff (size vs expansion rate)
- sec:scope / sec:limitations: What code doesn't need to address

## Workflow

**parameter_sweep.py**: Systematic exploration using LINEAR_SEARCH with adaptive step-skipping. Computes R² metrics for size and expansion rate. Saves best configurations to results/best_config.pkl.

**run_simulation.py**: Reproduction script using specific (M, S) parameters. Generates comparison plots showing ΛCDM, External-Node, and Matter-only evolution.

**Matter-only comparison**: Critical validation. Matter-only achieves ~96% endpoint but R²_rate=0.835 vs external-node R²_rate=0.963. External-node R²_size=0.995 vs matter-only 0.989. Proves external-nodes provide genuine acceleration mechanism, not coincidental endpoint.

**CSV metrics note**: In sweep CSV, match_curve_rmse_pct is stored as 100−RMSE×100. Actual RMSE = 1−(match_curve_rmse_pct/100).

## Usage

When requirements unclear, check paper for ground truth on:
- What physical regime we're modeling
- What success looks like
- What we explicitly don't care about (yet)

When paper contradicts code, paper defines intent; code is implementation-in-progress.

## Quantified Predictions (sec:predictions)

Predictions section includes quantitative results computed by `compute_predictions.py`:

### 1. Phantom Energy Behavior (w < -1)
**Mechanism**: As R→S, tidal force scales as (S-R)⁻². Effective w = -1 - (2/3)(d ln H / d ln a) should drift below ΛCDM.

**Quantitative results** (M=9000, S=38 Gpc, R/S=0.19):
- Phantom threshold at t≈11.3 Gyr, w_ext ≈ -0.64
- At t=12 Gyr: w_ext ≈ -0.84, w_ΛCDM ≈ -0.63, Δw ≈ -0.21
- Configs with smaller S (e.g. S=15, R/S≈0.48) would show stronger present-day phantom signatures
- Observable via precision w(z) measurements at low redshift

### 2. Dipole Anisotropy in H₀
**Mechanism**: Virialized structure has both position irregularity (5%) AND mass variation (20%). Both contribute to asymmetric tidal field, adding in quadrature.

**Quantitative results** (computed by `compute_predictions.py`, M=9000, S=38 Gpc, Ω_Λ,eff=7.24):
- Position only (grid): ΔH₀/H₀ ≈ 2.4%
- Mass only (grid): ΔH₀/H₀ ≈ 3.9%
- **Combined (grid)**: ΔH₀/H₀ ≈ 4.6% (~3.2 km/s/Mpc)
- **Combined (single node, worst case)**: ΔH₀/H₀ ≈ 11.3% (~7.9 km/s/Mpc)
- Hubble Tension: ~8.6% (6 km/s/Mpc)
- Testable by Euclid/LSST: dipole at 5-11% level with no local structure correlation
- Prediction robust across M/S configs (depends on GM/S³ ratio, ~constant for matching configs)

### 3. CMB Axis of Evil (Large-Angle Multipole Alignment)
**Mechanism**: 3×3×3 lattice has nodes at 3 distinct distances (face S, edge S√2, corner S√3). Discrete geometry imprints quadrupole (l=2) from face-node asymmetries and octopole (l=3) from corner-node tetrahedral sub-symmetry. All multipoles share lattice axes → guaranteed alignment.

**Qualitative predictions**:
- Low quadrupole power (cubic symmetry partially cancels l=2)
- Planar octopole (corner contributions project onto lattice planes)
- Quadrupole-octopole mutual alignment (both derive from same lattice)
- H₀ dipole aligned with Axis of Evil direction (cross-check with sec:dipole)

**References**: Land & Magueijo 2005 (PRL 95:071301), de Oliveira-Costa+ 2004 (PRD 69:063516), Copi+ 2006 (MNRAS 367:79), Copi+ 2010 (Adv.Astron. 2010:847541)

### 4. Dark Flow (Large-Scale Bulk Motions)
**Mechanism**: Asymmetric HMEA lattice → net tidal pull on entire bubble. Scale-independent (externally sourced). ΔH₀/H₀ ~ 4.6% → v_bulk ~ 320 km/s (grid avg), ~790 km/s (single node). Consistent with observed ~400 km/s (Watkins+ 2023, 4.8σ ΛCDM tension).

**Key evidence**: Watkins & Feldman 2025 show flow dominated by external sources beyond survey volume — exactly what HMEAs predict.

**Directional coherence**: Dark flow (l~290-298°), Axis of Evil (l~260°), H₀ dipole should all point toward nearest HMEA. Approximate concordance observed.

**References**: Kashlinsky+ 2008 (ApJ 686:L49), Planck 2014 (A&A 561:A97), Watkins+ 2023 (MNRAS 524:1885), Watkins & Feldman 2025 (arXiv:2512.03168)

### Speculative Sections (marked in paper)
- sec:fossil: Fossil Black Holes—speculative, no simulation support
- sec:metabolism: Great Metabolism—speculative extension, not derived from simulation

## LaTeX Structure Notes

Paper uses `\label`/`\ref` for all cross-references and `hyperref` for clickable links.
Figures include matter-only (green dotted) alongside ΛCDM (blue solid) and External-Node (red dashed).
Hawking radiation defense uses T_H ∝ 1/M → T ~ 10⁻⁸³ K (negligible at source).
Tables wrapped in `\begin{table}[htbp]` floats with `\caption`/`\label`: tab:results (parameter configs), tab:comparison (external-node vs matter-only).
All R² formatted as `$R^2$` (fully in math mode) throughout.
Hubble Tension discussed as potential consequence of directional dipole: CMB is all-sky avg, distance ladder is directional → dipole could manifest as method-dependent discrepancy.
Numerical robustness paragraph covers seed variation, particle count convergence, resolution threshold.
w_eff formula clarified as total effective equation of state (not DE component alone).
