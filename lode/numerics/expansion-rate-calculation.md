# Expansion Rate Calculation

## Overview

Hubble parameter H(t) = (1/a) × da/dt computed from N-body scale factor evolution via numerical derivatives. LCDM baseline uses exact analytic formula H² = H₀²[Ω_m/a³ + Ω_Λ].

**File**: run_simulation.py:156-212

## Key Finding: Smoothing Caused Edge Artifacts

**Problem (Jan 2026)**: User reported "strange edge behavior" in expansion rate plot at 3.8 Gyr and 13.8 Gyr, while middle (4.8-12.8 Gyr) matched LCDM well.

**Root cause**: Gaussian smoothing (σ=2) applied BEFORE derivative computation caused massive edge distortion:
- First edge (3.8 Gyr): 79.4% error (178.4 → 36.8 km/s/Mpc)
- Last edge (13.8 Gyr): 77.9% error (71.9 → 15.9 km/s/Mpc)
- Middle points: <5% error

**Solution**: Made smoothing optional (default OFF). Raw derivatives are smooth and accurate.

**Evidence**: diagnose_expansion_rate.py + tests/test_hubble_parameter.py

## Algorithm

### N-body Models (External-Node, Matter-only)

**Step 1: Optional Smoothing**
```python
if smooth_sigma > 0:
    a_smooth = gaussian_filter1d(a, sigma=smooth_sigma)
else:
    a_smooth = a  # No smoothing (default)
```

**Step 2: Compute Derivative**
```python
# Central differences for middle points
# One-sided differences at edges (less accurate)
da_dt = np.gradient(a_smooth, t_seconds)
H = da_dt / a_smooth
```

**Step 3: Edge Correction (Second-Order)**
```python
# Forward difference (first point)
H[0] = (-3*a[0] + 4*a[1] - a[2]) / (2*dt_0 * a[0])

# Backward difference (last point)
H[-1] = (3*a[-1] - 4*a[-2] + a[-3]) / (2*dt_n * a[-1])
```

**Step 4: Convert Units**
```python
H_hubble = H * Mpc_to_m / 1000  # [km/s/Mpc]
```

### LCDM Baseline (Analytic)

**No derivatives needed** - uses exact Friedmann equation:
```python
H = H0 * sqrt(Omega_m / a**3 + Omega_Lambda)
```

**Accuracy**: Machine precision (~1e-15 relative error)

## Smoothing Parameter

**Function signature**:
```python
def calculate_hubble_parameters(t_ext, a_ext, t_matter, a_matter_sim,
                                 smooth_sigma=0.0):
```

**Default**: 0.0 (no smoothing)

**When to use**:
- smooth_sigma = 0.0: Raw derivatives (recommended, accurate for N≥200 steps)
- smooth_sigma = 1-2: If noise dominates (only use for N<100 steps or high noise)

**Why default is 0.0**: Gaussian smoothing causes severe edge distortion (80% error at boundaries) that masks real physics.

## Edge Formula Derivation

**Taylor expansion around point i**:
```
f(x±h) = f(x) ± hf'(x) + (h²/2)f''(x) ± (h³/6)f'''(x) + O(h⁴)
```

**Forward difference (first point)**:
```
f'(x) ≈ (-3f(x) + 4f(x+h) - f(x+2h)) / (2h)
```

**Error**: O(h²)

**Backward difference (last point)**:
```
f'(x) ≈ (3f(x) - 4f(x-h) + f(x-2h)) / (2h)
```

**Error**: O(h²)

**Central difference (middle points via np.gradient)**:
```
f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
```

**Error**: O(h²)

All formulas have same order accuracy.

## Validation

**Tests**: tests/test_hubble_parameter.py (5 tests)

**Test 1: Edge formula correctness**
```python
# Use f(t) = t² where f'(t) = 2t is known
t = np.linspace(0, 10, 11)
f = t**2

# Forward: f'(0) should be 0
df_0 = (-3*f[0] + 4*f[1] - f[2]) / (2*dt)
assert df_0 ≈ 0.0

# Backward: f'(10) should be 20
df_10 = (3*f[-1] - 4*f[-2] + f[-3]) / (2*dt)
assert df_10 ≈ 20.0
```

**Test 2: LCDM consistency**
```python
# Compute H from LCDM a(t) using derivatives
# Compare to analytic H = H0*sqrt(Omega_m/a³ + Omega_Lambda)
# Should match within 2% (discretization error)
```

**Test 3: Discontinuity detection**
```python
# Check for sudden jumps
rel_changes = abs(diff(H) / H[:-1]) * 100
assert max(rel_changes) < 15%  # No discontinuities
```

**Test 4: Matter-only decreases**
```python
# H(t) must decrease monotonically (no acceleration)
assert all(H[i+1] < H[i])
```

**Test 5: Present-day value**
```python
# H(13.8 Gyr) ≈ 70 km/s/Mpc
assert |H(13.8 Gyr) - 70| < 5
```

## Diagnostic Tool

**Script**: diagnose_expansion_rate.py

**Purpose**: Compare raw vs smoothed expansion rates to identify artifacts

**Output**:
1. Plots: 4 panels showing raw, smoothed, difference, jump detection
2. CSV: t_Gyr, H_lcdm_analytic, H_ext_raw, H_ext_smooth, differences
3. Summary: Edge behavior, max smoothing effect, discontinuity check

**Usage**:
```bash
python diagnose_expansion_rate.py
# Generates: expansion_rate_diagnostic_3.8-13.8Gyr.png + .csv
```

## Results from Diagnostic (Jan 2026)

**Simulation**: 3.8-13.8 Gyr, N=50 particles, 250 steps

**Edge behavior**:
```
First edge (3.8 Gyr):
  LCDM (analytic):    178.13 km/s/Mpc
  External-Node (raw): 178.41 km/s/Mpc  (0.2% error)
  External-Node (smooth): 36.76 km/s/Mpc  (79.4% error!!!)

Last edge (13.8 Gyr):
  LCDM (analytic):    69.27 km/s/Mpc
  External-Node (raw): 71.90 km/s/Mpc  (3.8% error)
  External-Node (smooth): 15.89 km/s/Mpc  (77.9% error!!!)
```

**Smoothness check**:
- Raw: max relative change = 7.69% per step (smooth, no discontinuities)
- Smoothed: max relative change = 124% per step (huge discontinuities at edges!)

**Conclusion**: Smoothing was the problem, not the solution. Raw derivatives are accurate and smooth.

## Why Smoothing Fails at Edges

**Gaussian filter boundary behavior**:
```python
# scipy.ndimage.gaussian_filter1d with default mode='reflect'
# At boundary, reflects data: [a, b, c, d] → [b, a, b, c, d, c, b]
# Reduces gradient artificially near edges
```

**Effect on derivative**:
- Middle: smoothing reduces noise (good if noise dominates)
- Edges: smoothing creates artificial plateau (always bad)
- Edge derivatives computed from distorted data → massive error

**Visualization**: See Plot 3 (Smoothing Effect) in diagnostic output - difference peaks at edges.

## Accuracy vs Number of Steps

**Discretization error** ∝ dt²

| N steps | dt (Gyr) | Expected error |
|---------|----------|----------------|
| 100     | 0.10     | ~2% |
| 150     | 0.067    | ~1% |
| 200     | 0.050    | ~0.5% |
| 250     | 0.040    | ~0.3% |

**Recommendation**: Use ≥200 steps for <1% error without smoothing

## Usage Example

```python
from run_simulation import calculate_hubble_parameters

# Compute raw Hubble parameter (no smoothing - recommended)
hubble = calculate_hubble_parameters(
    t_ext, a_ext, t_matter, a_matter,
    smooth_sigma=0.0  # Default
)

H_ext = hubble['H_ext_hubble']  # km/s/Mpc
H_matter = hubble['H_matter_hubble']

# Only use smoothing if you have noisy data from few steps
hubble_smooth = calculate_hubble_parameters(
    t_ext, a_ext, t_matter, a_matter,
    smooth_sigma=1.0  # Light smoothing
)
```

## Comparison: N-body vs LCDM

| Aspect | N-body | LCDM |
|--------|--------|------|
| Method | Numerical derivative | Analytic formula |
| Accuracy | ~1% (N=200 steps) | Machine precision |
| Edge behavior | Same as middle | Perfect everywhere |
| Smoothing | Optional | Not needed |
| Computational cost | O(N) gradient | O(N) sqrt |

## Related

- [timestep-stability.md](./timestep-stability.md) - Why N≥200 steps needed
- [lcdm-baseline.md](./lcdm-baseline.md) - Analytic LCDM computation
- [leapfrog-staggering.md](./leapfrog-staggering.md) - Scale factor evolution

## References

- Edge formula implementation: run_simulation.py:184-194, 202-207
- Diagnostic script: diagnose_expansion_rate.py
- Unit tests: tests/test_hubble_parameter.py
- User report: "strange edge behavior" at 3.8 and 13.8 Gyr (Jan 2026)
