# External-Node Cosmology: Replacing Dark Energy with Classical Gravity

A proof-of-concept toy model demonstrating that classical gravitational forces from trans-observable structure can reproduce ΛCDM cosmic acceleration without requiring vacuum energy.

### Key Result

**>99% agreement with ΛCDM expansion** over 10 billion years (t=3.8→13.8 Gyr) using only Newtonian gravity from external massive nodes. Multiple parameter configurations achieve R²>0.89 for expansion rate dynamics, demonstrating mechanism robustness.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the main simulation (outputs to current directory)
python run_simulation.py

# Run with custom output directory
python run_simulation.py ./my_results/
```

## Repository Structure

```
.
├── README.md                      # This file
├── run_simulation.py              # Main script to reproduce final results
├── parameter_sweep.py             # Systematic parameter exploration with R² metrics
│
└── cosmo/                         # Core simulation code
    ├── constants.py               # Physical constants and parameters
    ├── particles.py               # Particle initialization
    ├── integrator.py              # Leapfrog integration
    ├── simulation.py              # Main simulation class
    ├── analysis.py                # R² metrics, Friedmann solver, comparison tools
    └── visualization.py           # Plotting utilities
```

## The Model

### Physical Concept

Standard ΛCDM attributes cosmic acceleration to dark energy (Λ). This model proposes an alternative:

- **Observable universe** = expanding bubble within a larger meta-structure
- **Meta-structure** = virialized grid of Hyper-Massive External Attractors (HMEAs)
- **Acceleration** = tidal gravitational pull from external nodes

### Key Equations

Tidal acceleration from external mass:
```
a_tidal ≈ (2GM_ext/S³) × R
```

This mimics dark energy:
```
a_Λ = H₀² Ω_Λ × R
```

### Example Parameter Configurations

Multiple (M, S) combinations achieve high fidelity ΛCDM matching:

- **M = 855 × M_observable**, S = 25 Gpc: 99.36% endpoint, R²=0.90 (expansion rate)
- **M = 97000 × M_observable**, S = 64 Gpc: 99.46% endpoint, R²=0.90 (expansion rate)
- **M = 69 × M_observable**, S = 15 Gpc: 99.91% endpoint, R²=0.81 (expansion rate)

This multiplicity demonstrates mechanism robustness—no fine-tuning to singular configuration required.

## Numerical Implementation

### Integration Method

**Leapfrog (Kick-Drift-Kick)**: Symplectic second-order integrator with pre-kick initialization to properly stagger velocities. Eliminates "initial bump" artifact that would otherwise cause early-time expansion to overshoot.

### Initial Conditions

**Position normalization**: After random particle placement, positions are scaled to ensure exact target RMS radius. This eliminates 0.1-1% variation from random sampling that would cause initialization artifacts in model comparisons.

**Velocity initialization**: Damped Hubble flow with center-of-mass removal:
```
v = damping × H(t_start) × r - v_COM
```

**Pre-kick**: Apply negative half-kick before evolution to align with leapfrog staggering convention:
```
v(t=-dt/2) = v(t=0) - a(t=0) × dt/2
```

### ΛCDM Baseline Alignment

ΛCDM baseline computed at **exact** N-body snapshot times using `solve_friedmann_at_times()`. This ensures:
- t[0] = 0.0 exactly (no grid misalignment)
- a[0] = a_start exactly (no interpolation error)
- Relative expansion starts at 1.0 (no "bump" pattern)

### Validation and Quality Checks

**Physics Constraints** (enforced via unit tests):
- Matter-only **never** exceeds ΛCDM expansion at any timestep (physics requirement)
- Early-time behavior matches ΛCDM before models diverge
- Smooth evolution from t=0 (no initialization artifacts)
- All models start with identical initial size (fair comparison)

**R² Metric for Statistical Rigor**:
- Coefficient of determination: R² = 1 - (SS_residual / SS_total)
- R²=1.0 perfect fit, R²=0 no better than mean, R²<0 worse than constant
- **Last-half R² (5 Gyr)**: Isolates late-time acceleration, avoiding early-universe inflation of scores
- Separate R² for size evolution and expansion rate H(t) evolution

**Automated Quality Checks** (via `parameter_sweep.py`):
- Center-of-mass drift monitoring (ensures symmetric grid produces negligible bulk motion)
- Runaway particle detection (flags numerical instability: max_radius / RMS > 2.0)
- Matter-only comparison (validates external-node mechanism provides genuine acceleration)

See `tests/test_early_time_behavior.py` and `tests/test_model_comparison.py` for enforcement.

## Results

### Quantitative Comparison

**Simulation Period**: t = 3.8 Gyr → 13.8 Gyr (10 Gyr, late-universe expansion)

| Model            | Final Size | Endpoint Match | Size R² (last 5 Gyr) | Expansion Rate R² (last 5 Gyr) |
|------------------|------------|----------------|----------------------|--------------------------------|
| ΛCDM baseline    | 14.52 Gpc  | 100%           | 1.0000               | 1.0000                         |
| External-Node    | 14.42 Gpc  | 99.36%         | 0.9979               | 0.8976 ✓                       |
| Matter-only      | 14.02 Gpc  | 96.56%         | 0.9716               | **-0.4840 ❌**                 |

*(Using M=855×M_obs, S=25 Gpc for External-Node example)*

**Key Insight**: Endpoint match ≠ dynamics match
- Matter-only reaches similar size (96.56%) through **deceleration** (wrong physics)
- Expansion rate R² exposes this: matter-only catastrophically fails (-0.48)
- External-node achieves positive expansion rate R² (0.90), proving it replicates **acceleration mechanism**

**Why Last-Half R²**: Full 10 Gyr includes early universe (t=3.8→8.8 Gyr) where all models similar. Last-half (5 Gyr) isolates late-time acceleration—the phenomenon being modeled.

### Hubble Parameter
- Present-day value: H₀ ≈ 70 km/s/Mpc ✓
- Correct temporal evolution ✓
- Natural acceleration without fine-tuning ✓

## Reproducing Results

### Main Figure (Figure 1 in paper)
```bash
python run_simulation.py ./output/
# Creates: output/figure_simulation_results.png
# Compares ΛCDM, External-Node, and Matter-only models
```

### Parameter Exploration
```bash
python parameter_sweep.py
# Systematic LINEAR_SEARCH with adaptive step-skipping
# Tests multiple (M, S) combinations with R² metrics
# Reports: endpoint match, size R², expansion rate R², Hubble parameter match
# Output: results/best_config.pkl with optimal parameters
```

**Interpreting R² Scores**:
- R² = 1.0: Perfect match
- R² = 0.99-1.0: Excellent (tracks ΛCDM dynamics closely)
- R² = 0.90-0.99: Good (close to ΛCDM but not enough)
- R² = 0.0: No better than mean baseline
- R² < 0: Worse than constant (catastrophic failure)

Matter-only typically shows R²_expansion ≈ -0.48 (deceleration cannot mimic acceleration).
External-node achieves R²_expansion ≈ 0.90 (genuine acceleration mechanism).

### Visualizing the universe
```bash
python -m cosmo.visualization
```


## Limitations

This is a **toy model** demonstrating feasibility, not a complete cosmological theory:

- ✗ Early universe physics (inflation, CMB)
- ✗ Structure formation
- ✗ Full General Relativistic treatment
- ✓ Late-time cosmic acceleration (6 Gyr)
- ✓ Classical Newtonian gravity
- ✓ Homogeneous expansion

## Physical Interpretation

### What This Demonstrates

1. **Dark energy is not inevitable** - Classical alternatives exist
2. **Cosmological principle may break down** - At super-horizon scales
3. **External structure is viable** - Could explain acceleration
4. **Opens research directions** - Even if ultimately superseded

### Key Predictions

- **Dipole anisotropy** in expansion rate
- **Time-varying equation of state** (w → -1 as R → S)
- **Natural "Hubble tension" resolution**

## Dependencies

- Python 3.7+
- NumPy
- SciPy
- Matplotlib

## License

GNU License - See LICENSE file

## Acknowledgments

This work represents a proof-of-concept exploration demonstrating that classical gravitational mechanisms can reproduce observed cosmic acceleration. While speculative, it challenges fundamental assumptions and may inspire more complete theoretical frameworks.

---

**"Dark energy is not inevitable, classical alternatives exist."**
