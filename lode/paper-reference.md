# Paper Reference

**File**: docs/paper.tex

Draft paper defining theoretical framework and goals the code tests.

## Core Thesis

Replace dark energy (Λ) with classical tidal forces from trans-observable Hyper-Massive External Attractors (HMEAs) in virialized meta-structure.

**Key claim**: Dark energy not inevitable—classical gravity can reproduce ΛCDM expansion.

## Physical Model

**Virialized Meta-Structure**: Crystal-lattice-like arrangement of HMEA nodes beyond observable universe. Stable, gravitationally balanced (like relaxed galaxy cluster), irregular in detail.

**Progenitor Hypothesis**: Big Bang = destabilization of a node in this meta-structure. Pre-existing equilibrium explains isotropy without fine-tuning.

**Tidal Acceleration**: Expanding universe climbs potential gradient of nearest HMEAs. Linear scaling: a_tidal ≈ (G M_ext / S³) × R. Mathematically equivalent to Λ term: H₀² Ω_Λ R.

## Parameter Estimates (from paper)

Analytical target: M_ext ≈ 5×10⁵⁵ kg, S ≈ 31 Gpc

Numerical exploration found MULTIPLE close matches (not single optimal):
- M=855×M_obs, S=25 Gpc: 99.36% endpoint, R²=0.9979 (size), R²=0.8976 (expansion rate)
- M=97000×M_obs, S=65 Gpc: 99.46% endpoint, R²=0.9983 (size), R²=0.9041 (expansion rate)
- M=69×M_obs, S=15 Gpc: 99.91% endpoint, R²=0.9992 (size), R²=0.8120 (expansion rate)

Result: >99% endpoint match, R²>0.89 expansion rate match over 10 Gyr (t=3.8→13.8 Gyr)

## What Code Tests

**Success criteria** (from paper Section 4):
- Late-time acceleration (10 Gyr: t=3.8→13.8 Gyr) ✓
- ΛCDM expansion match to >99% endpoint, R²>0.89 expansion rate ✓
- Realistic H₀ ≈ 70 km/s/Mpc ✓
- Classical gravity only (no exotic physics) ✓
- Multiple parameter solutions (mechanism robustness) ✓
- Matter-only comparison (validates acceleration mechanism) ✓

**Validation checks**:
- Matter-only never exceeds ΛCDM at any timestep (physics constraint)
- R² metric (coefficient of determination) for statistical rigor
- Last-half R² (5 Gyr) isolates late-time acceleration behavior
- COM drift monitoring, runaway particle detection
- Dual-seed testing (parameter_sweep.py validates with multiple random seeds)

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

- Section 3.2: Tidal acceleration formula (paper.tex:62-71)
- Section 4.1: Grid configuration (3×3×3 lattice justification)
- Section 4.2: Optimal parameters M=800×M_obs, S=24 Gpc
- Section 5: Limitations (what code doesn't need to address)

## Workflow

**parameter_sweep.py**: Systematic exploration using LINEAR_SEARCH with adaptive step-skipping. Computes R² metrics for size and expansion rate. Saves best configurations to results/best_config.pkl.

**run_simulation.py**: Reproduction script using specific (M, S) parameters. Generates comparison plots showing ΛCDM, External-Node, and Matter-only evolution.

**Matter-only comparison**: Critical validation showing matter-only achieves ~96% endpoint but R²_expansion=-0.48 (catastrophic). Proves external-nodes provide genuine acceleration mechanism, not coincidental endpoint.

## Usage

When requirements unclear, check paper for ground truth on:
- What physical regime we're modeling
- What success looks like
- What we explicitly don't care about (yet)

When paper contradicts code, paper defines intent; code is implementation-in-progress.

## Quantified Predictions (Paper Section 7)

Paper Section 7 now includes quantitative predictions computed by `compute_predictions.py`:

### 1. Phantom Energy Behavior (w < -1)
**Mechanism**: As R→S, tidal force scales as (S-R)⁻². Effective w = -1 - (2/3)(d ln H / d ln a) should drift below ΛCDM.

**Quantitative results** (M=855, S=25 Gpc):
- Today (t=13.8 Gyr): w_ext ≈ -0.74 vs w_ΛCDM ≈ -0.70, Δw ≈ -0.04
- Current R/S ratio ≈ 0.3
- Significant phantom deviation (Δw < -0.05) requires R/S → 1
- Observable via precision w(z) measurements at low redshift

### 2. Dipole Anisotropy in H₀
**Mechanism**: 5% HMEA grid irregularity creates asymmetric tidal field.

**Quantitative results** (analytical, M=855, S=25 Gpc):
- Full grid (statistical average): ΔH₀/H₀ ≈ 2.6% (~1.8 km/s/Mpc)
- Single nearest node (worst case): ΔH₀/H₀ ≈ 6.3% (~4.4 km/s/Mpc)
- Hubble Tension (~8.6%) would need ~17% grid irregularity
- Testable by Euclid/LSST: dipole at 2-6% level with no local structure correlation

### Speculative Sections (marked in paper)
- Section 6.2 Fossil Black Holes: Speculative, no simulation support
- Section 8 Great Metabolism: Speculative extension, not derived from simulation
