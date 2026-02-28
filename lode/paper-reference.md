# Paper Reference

**File**: docs/paper.tex

Draft paper defining theoretical framework and goals the code tests.

## Core Thesis

Replace dark energy (Λ) with classical tidal forces from trans-observable Hyper-Massive External Attractors (HMEAs) in virialized meta-structure.

**Key claim**: Dark energy not inevitable—classical gravity can reproduce ΛCDM expansion.

**Differentiator from prior work**: Single discrete lattice geometry simultaneously explains cosmic acceleration AND three independent anomalies (Hubble Tension, Axis of Evil, dark flow) with no additional parameters. Prior proposals (Mersini-Houghton 2009, modified inflation models, local void models) address individual anomalies with purpose-built mechanisms. Unification under one geometric structure is novel. Introduction explicitly states this.

## Physical Model

**Virialized Meta-Structure**: Crystal-lattice-like arrangement of HMEA nodes beyond observable universe. Stable, gravitationally balanced (like relaxed galaxy cluster), irregular in detail.

**Progenitor Hypothesis**: Big Bang = destabilization of a node in this meta-structure. Pre-existing equilibrium explains isotropy without fine-tuning.

**Tidal Acceleration**: Expanding universe climbs potential gradient of nearest HMEAs. Linear scaling: a_tidal ≈ (2 G M_ext / S³) × R. Mathematically equivalent to Λ term: H₀² Ω_Λ R.

## Parameter Estimates (from paper)

Analytical target: M_ext ≈ 5×10⁵⁵ kg, S ≈ 39 Gpc

Numerical exploration found MULTIPLE close matches (not single optimal):
- M=800×M_obs, S=22 Gpc: 99.85% endpoint, R²=0.999 (size), RMSE≈0.006
- M=50000×M_obs, S=50 Gpc: ~100% endpoint, R²=0.999 (size)
- M=500×M_obs, S=20 Gpc: ~100% endpoint, R²=0.999 (size)

Result: ~100% endpoint match, R²=0.999 size match over 8 Gyr (t=5.8→13.8 Gyr)
Sweep range: M=20-20000 (up to 200000), S=20-70 Gpc, 2000 particles

## What Code Tests

**Success criteria** (from paper Section 4):
- Late-time acceleration (8 Gyr: t=5.8→13.8 Gyr) ✓
- ΛCDM expansion match to ~100% endpoint, R²=0.999 size ✓
- Realistic H₀ ≈ 70 km/s/Mpc ✓
- Classical gravity only (no exotic physics) ✓
- Multiple parameter solutions (mechanism robustness) ✓
- Matter-only comparison (validates acceleration mechanism) ✓

**Validation checks**:
- Matter-only never exceeds ΛCDM at any timestep (physics constraint)
- R² metric (coefficient of determination) for size evolution
- RMSE for absolute deviation
- COM drift monitoring, runaway particle detection
- Multi-seed testing (R²>0.999 achievable with any seed by adjusting M/S)

**Explicit non-goals** (Section 5.2):
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

- Section 3.2: Tidal acceleration formula
- Section 4.1: Grid configuration (3×3×3 lattice justification)
- Section 4.2: Primary parameters M=800×M_obs, S=22 Gpc
- Section 5: Limitations (what code doesn't need to address)

## Workflow

**parameter_sweep.py**: Systematic exploration using LINEAR_SEARCH with adaptive step-skipping. Computes R² metrics for size. Saves best configurations to results/best_config.pkl.

**run_simulation.py**: Reproduction script using specific (M, S) parameters. Generates comparison plots showing ΛCDM, External-Node, and Matter-only evolution.

**Matter-only comparison**: Critical validation. Matter-only achieves ~96% endpoint but size R²=0.927 (last 4 Gyr) vs external-node R²=0.992. Proves external-nodes provide genuine acceleration mechanism, not coincidental endpoint.

**CSV metrics note**: In sweep CSV, match_curve_rmse_pct is stored as 100−RMSE×100. Actual RMSE = 1−(match_curve_rmse_pct/100).

## Usage

When requirements unclear, check paper for ground truth on:
- What physical regime we're modeling
- What success looks like
- What we explicitly don't care about (yet)

When paper contradicts code, paper defines intent; code is implementation-in-progress.

## Quantified Predictions (Paper Section 7)

Paper Section 7 includes quantitative predictions computed by `compute_predictions.py`:

### 1. Phantom Energy Behavior (w < -1)
**Mechanism**: As R→S, tidal force scales as (S-R)⁻². Effective w = -1 - (2/3)(d ln H / d ln a) should drift below ΛCDM.

**Quantitative results** (M=800, S=22 Gpc):
- Today (t=13.8 Gyr): w_ext ≈ -0.74 vs w_ΛCDM ≈ -0.70, Δw ≈ -0.04 (**STALE** — from old M=855/S=25 run)
- Current R/S ratio ≈ 0.32
- Significant phantom deviation (Δw < -0.05) requires R/S → 1
- Observable via precision w(z) measurements at low redshift
- **NOTE**: Extended sim (20 Gyr) with S=22 breaks down — universe blows past nodes (R/S>>1). Paper Section 7.1 values need recomputation from 8 Gyr run or different approach.

### 2. Dipole Anisotropy in H₀
**Mechanism**: Virialized structure has both position irregularity (5%) AND mass variation (20%). Both contribute to asymmetric tidal field, adding in quadrature.

**Quantitative results** (computed by `compute_predictions.py`, M=800, S=22 Gpc, Ω_Λ,eff=3.32):
- Position only (grid): ΔH₀/H₀ ≈ 2.8%
- Mass only (grid): ΔH₀/H₀ ≈ 3.7%
- **Combined (grid)**: ΔH₀/H₀ ≈ 4.7% (~3.3 km/s/Mpc)
- **Combined (single node, worst case)**: ΔH₀/H₀ ≈ 11.4% (~8.0 km/s/Mpc)
- Hubble Tension: ~8.6% (6 km/s/Mpc)
- Testable by Euclid/LSST: dipole at 5-11% level with no local structure correlation

### 3. CMB Axis of Evil (Large-Angle Multipole Alignment)
**Mechanism**: 3×3×3 lattice has nodes at 3 distinct distances (face S, edge S√2, corner S√3). Discrete geometry imprints quadrupole (l=2) from face-node asymmetries and octopole (l=3) from corner-node tetrahedral sub-symmetry. All multipoles share lattice axes → guaranteed alignment.

**Qualitative predictions**:
- Low quadrupole power (cubic symmetry partially cancels l=2)
- Planar octopole (corner contributions project onto lattice planes)
- Quadrupole-octopole mutual alignment (both derive from same lattice)
- H₀ dipole aligned with Axis of Evil direction (cross-check with Section 7.2)

**References**: Land & Magueijo 2005 (PRL 95:071301), de Oliveira-Costa+ 2004 (PRD 69:063516), Copi+ 2006 (MNRAS 367:79), Copi+ 2010 (Adv.Astron. 2010:847541)

### 4. Dark Flow (Large-Scale Bulk Motions)
**Mechanism**: Asymmetric HMEA lattice → net tidal pull on entire bubble. Scale-independent (externally sourced). ΔH₀/H₀ ~ 4.7% → v_bulk ~ 330 km/s (grid avg), ~800 km/s (single node). Consistent with observed ~400 km/s (Watkins+ 2023, 4.8σ ΛCDM tension).

**Key evidence**: Watkins & Feldman 2025 show flow dominated by external sources beyond survey volume — exactly what HMEAs predict.

**Directional coherence**: Dark flow (l~290-298°), Axis of Evil (l~260°), H₀ dipole should all point toward nearest HMEA. Approximate concordance observed.

**References**: Kashlinsky+ 2008 (ApJ 686:L49), Planck 2014 (A&A 561:A97), Watkins+ 2023 (MNRAS 524:1885), Watkins & Feldman 2025 (arXiv:2512.03168)

### Speculative Sections (marked in paper)
- Section 6.2 Fossil Black Holes: Speculative, no simulation support
- Section 8 Great Metabolism: Speculative extension, not derived from simulation
