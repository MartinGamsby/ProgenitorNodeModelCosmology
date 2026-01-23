# External-Node Cosmology: Replacing Dark Energy with Classical Gravity

A proof-of-concept toy model demonstrating that classical gravitational forces from trans-observable structure can reproduce ΛCDM cosmic acceleration without requiring vacuum energy.

### Key Result

**99.4% agreement with ΛCDM expansion** over 6 billion years using only Newtonian gravity from external massive nodes.

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
├── parameter_sweep.py             # Parameter exploration
│
└── cosmo/                         # Core simulation code
    ├── constants.py               # Physical constants and parameters
    ├── particles.py               # Particle initialization
    ├── integrator.py              # Leapfrog integration
    ├── simulation.py              # Main simulation class
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

### Optimal Parameters

- Node mass: M = 800 × M_observable ≈ 8×10⁵⁵ kg
- Grid spacing: S = 24 Gpc
- Initial size: R₀ = 11.59 Gpc (same as ΛCDM)
- Effective Ω_Λ: 2.555

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

### Validation

Unit tests enforce critical physics constraints:
- Matter-only **never** exceeds ΛCDM expansion (no acceleration source)
- Early-time (<1 Gyr) behavior matches ΛCDM within 1% before divergence
- Smooth evolution from t=0 (no initialization artifacts)
- All models start with identical initial size (fair comparison)

See `tests/test_early_time_behavior.py` for enforcement.

## Results

### Quantitative Match
- ΛCDM final radius: 17.78 Gpc
- External-node final radius: 17.89 Gpc
- **Agreement: 99.4%**

### Hubble Parameter
- Present-day value: H₀ ≈ 70 km/s/Mpc ✓
- Correct temporal evolution ✓
- Natural acceleration without fine-tuning ✓

## Reproducing Results

### Main Figure (Figure 1 in paper)
```bash
python run_simulation.py ./output/
# Creates: output/figure_simulation_results.png
```

### Parameter Exploration
```bash
python parameter_sweep.py
# Tests multiple M and S combinations
```

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
