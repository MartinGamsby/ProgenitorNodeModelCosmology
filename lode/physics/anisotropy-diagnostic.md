# Anisotropy Diagnostic (Deliverable B)

Related: [theoretical-framework.md](./theoretical-framework.md),
[hubble-diagram-nbody.md](./hubble-diagram-nbody.md)

## Purpose

Measures directional expansion anisotropy in an evolved N-body snapshot.
The isotropic mu(z) chi^2 is barely sensitive to node-mass variation
(mean-preserving → isotropic background fixed). The SHEAR + HUBBLE DIPOLE
are the discriminating signal: per-node mass variation breaks the traceless
HMEA tidal tensor symmetry and produces measurable stretch / dipole aligned
with the mass-heavy face of the lattice.

## Module: `cosmo/anisotropy.py`

Pure functions (no I/O, no plotting, no sim imports). All operate on (N, 3)
numpy arrays of positions and velocities (SI units). Mirror of
`cosmo/sim_distance.py` design.

### API

```python
from cosmo.anisotropy import (
    shape_tensor,        # S_ij = mean(x_i x_j); eigenvalues, principal axis, shear_index
    axis_rms,            # per-axis RMS extent; max/min ratio
    hubble_dipole,       # H_plus / H_minus split; dipole = (H+ - H-)/H_mean; best_axis
    expansion_anisotropy,# rms_final / rms_initial per axis; spread
    anisotropy_summary,  # convenience wrapper returning all four sub-dicts
)
```

### Key quantities

| Quantity | Symbol | 0 = isotropic |
|---|---|---|
| `shear_index` | (λ_max − λ_min) / mean(λ) | 0 |
| `max_min_ratio` | RMS_x_max / RMS_x_min | 1 |
| `dipole` | (H_plus − H_minus) / H_mean | 0 |
| `expansion.spread` | max/min of per-axis growth factors | 1 |

### Edge cases

All four functions return a `degenerate=True` dict with zero values when
N < 2 (or N < 4 for hubble_dipole), all-zero positions, or collinear inputs.
No randomness anywhere — fully deterministic.

**Starved-hemisphere honesty (hubble_dipole)**: when the PROBE axis splits the
cloud so one hemisphere has < 2 particles, the Hubble slope there cannot be fit.
`hubble_dipole` then returns `degenerate=True` (honest "could-not-measure")
instead of a confident `dipole=0`. It does NOT substitute `H_global` for the
starved side — that would force `H_plus == H_minus` and collapse the dipole to
an exactly-zero FALSE ISOTROPY, masking real anisotropy (e.g. a lopsided
projection that starves one hemisphere). In the best-axis search, a candidate
axis that starves a hemisphere is skipped (never allowed to win). Well-populated
hemispheres (the intended N ≥ 2000 runs) are UNCHANGED: `degenerate=False` with
a confident dipole. Regression test:
`tests/test_anisotropy.py::TestHubbleDipole::test_starved_hemisphere_marks_degenerate_not_zero`.

### COM correction

`hubble_dipole` re-subtracts COM position AND COM velocity from the evolved
snapshot before fitting (defensive; the sim already removes COM velocity at
init in `particles.py:165-175`).

## Script: `anisotropy_report.py`

Thin CLI that runs two External-Node sims:
- UNIFORM: `node_mass_amplitude=0`
- ANISOTROPIC: user-specified seed and amplitude

Prints a comparison table:

```
metric           | uniform  | anisotropic | delta
shear_index      | 0.0xxx   | 0.xxxx      | +...
max_min_ratio    | 1.0xxx   | x.xxxx      | +...
hubble_dipole    | 0.0xxx   | 0.xxxx      | +...
...
```

Also reports the best-dipole direction vs the shape principal axis (they
should be roughly co-aligned when amplitude is non-trivial).

Usage:
```bash
python anisotropy_report.py --node-mass-seed 42 --node-mass-amplitude 0.5 --particles 80
```

Note: dipole estimates are noisy for N < 2000; prefer N ≥ 2000 for
publication-quality numbers.

## Observed signal (synthetic arrays, from unit tests)

- **Isotropic sphere (N=2000)**: shear_index < 0.2, best_dipole < 0.25, expansion spread < 1.15
- **3x-stretched anisotropic cloud (N=500)**: shear_index > 1.0, dipole > 0.10, expansion spread > 2.0

The HMEA node-mass variation (section 3, `node_mass_amplitude > 0`) rotates
the tidal stress tensor from traceless-isotropic into a shear+dipole
configuration. The seed selects the ORIENTATION; the amplitude controls the
MAGNITUDE. The paper should cite:

> Per-node mass variation with `node_mass_amplitude σ` produces a shear index
> and Hubble dipole that grow with σ while the isotropic expansion history
> (and hence mu(z) chi^2) is mean-preserving-invariant. The distinguishing
> observable is the anisotropy tensor, measurable via `cosmo.anisotropy`.

## Invariants

- `node_mass_amplitude=0` (uniform masses) → shear and dipole consistent with
  noise; expansion spread ≈ 1 for an isotropic initial condition.
- Same `(positions, velocities)` → identical diagnostic output (no internal RNG).
- Read-only: the diagnostic never modifies the simulation or any existing output.
