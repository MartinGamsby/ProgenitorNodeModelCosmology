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

Numerical best-fit: M_ext = 800×M_obs ≈ 8×10⁵⁵ kg, S = 24 Gpc

Result: 99.4% match to ΛCDM over 6 Gyr

## What Code Tests

**Success criteria** (from paper Section 4):
- Late-time acceleration (past ~6 Gyr) ✓
- ΛCDM expansion match to ~99% ✓
- Realistic H₀ ≈ 70 km/s/Mpc ✓
- Classical gravity only (no exotic physics) ✓

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

## Discrepancies to Watch

Paper uses Ω_Λ_eff = 2.555 (paper.tex:123). Check if code calculations align.

Paper mentions "6 Gyr period (t=10.8 → 16.8 Gyr)" but actual simulations may use different ranges—verify against run_simulation.py defaults.

## Usage

When requirements unclear, check paper for ground truth on:
- What physical regime we're modeling
- What success looks like
- What we explicitly don't care about (yet)

When paper contradicts code, paper defines intent; code is implementation-in-progress.
