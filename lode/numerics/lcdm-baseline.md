# LCDM Baseline Computation

Standardized method for computing ΛCDM expansion histories used across all analysis scripts.

## Core Principle

All ΛCDM baselines use `solve_friedmann_equation()` (analytic ODE solver), NOT N-body `.run()` method.

**Rationale:**
- Eliminates Leapfrog discretization artifacts
- Provides exact ΛCDM predictions for comparison against External-Node models
- Ensures consistency between `run_simulation.py` and `parameter_sweep.py`
- N-body integration introduces numerical noise inappropriate for reference curves

## Reference Values

Standard ΛCDM parameters (Ω_m=0.3, Ω_Λ=0.7, H₀=70 km/s/Mpc):

```
t=13.8 Gyr: size ≈ 14.5 Gpc  (present day, "today")
t=10.8 Gyr: size ≈ 11.6 Gpc  (default simulation start for run_simulation.py)
t=3.8 Gyr:  size ≈ 5.3 Gpc   (parameter sweep start)
```

The 14.5 Gpc reference represents the RMS radius of the observable universe today, approximately equal to the Hubble radius c/H₀.

## Usage Pattern

### In run_simulation.py (lines 54-66)

```python
lcdm_solution = solve_friedmann_equation(
    sim_params.t_start_Gyr,
    sim_params.t_end_Gyr,
    Omega_Lambda=lcdm_params.Omega_Lambda  # 0.7
)
t_lcdm = lcdm_solution['t_Gyr'] - sim_params.t_start_Gyr
a_lcdm = lcdm_solution['a']
size_lcdm = lcdm_initial_size * (a_lcdm / a_start)  # Scale from initial conditions
```

### In parameter_sweep.py (lines 67-79)

```python
lcdm_solution = solve_friedmann_equation(
    T_START_GYR,
    T_START_GYR + T_DURATION_GYR,
    Omega_Lambda=None  # Uses default 0.7
)
a_lcdm_array = lcdm_solution['a']
size_lcdm_curve = BOX_SIZE * (a_lcdm_array / A_START)
```

Both scripts now use identical LCDM computation method.

## Size Calculation

Sizes are computed via scale factor ratios:

```
size(t) = box_size_initial × [a(t) / a_start]
```

Where:
- `box_size_initial`: Initial box size [Gpc] from `calculate_initial_conditions(t_start)`
- `a(t)`: Scale factor at time t from `solve_friedmann_equation()`
- `a_start`: Scale factor at simulation start time

This ensures proper normalization: if simulation runs from t=3.8 Gyr to t=13.8 Gyr, final size equals 14.5 Gpc.

## Initial Conditions

`calculate_initial_conditions(t_start_Gyr)` computes starting box size and scale factor:

```python
# Solves from t=0 to max(t_start, 13.8)+1 Gyr to get both a_start and a_today
solution = solve_friedmann_equation(0.0, t_end_solve, n_points=400)

a_start = a[t_start]
a_today = a[13.8 Gyr]

box_size_Gpc = 14.5 × (a_start / a_today)
```

**Critical fix (2026-01-20):** Must solve to AT LEAST 13.8 Gyr to get accurate a_today. Previous bug solved only to t_start, causing incorrect normalization when t_start < 13.8 Gyr.

## Validation

Unit tests in `tests/test_simulation_baseline.py` verify:
- LCDM reaches 14.5 Gpc at t=13.8 Gyr (present day)
- Scale factor increases monotonically
- Cosmic acceleration (d²a/dt² > 0) in dark energy era
- LCDM expands faster than matter-only cosmology
- Size evolution matches expected behavior

All 8 tests passing confirms correct baseline computation.

## Historical Context

**Before 2026-01-20:**
- `parameter_sweep.py` used `CosmologicalSimulation.run()` for LCDM baseline
- This ran LCDM through Leapfrog integrator with N tracer particles
- Introduced numerical artifacts from particle-based RMS radius calculation
- Timestep discretization caused small deviations from analytic LCDM
- Comment on line 73: "THIS IS DIFFERENT THAN RUN_SIMULATION.PY!!!!"

**After fix:**
- Both scripts use `solve_friedmann_equation()` consistently
- Clean analytic LCDM baseline for parameter optimization
- Bug fix in `calculate_initial_conditions` ensures correct normalization
- Sizes now correctly reach 14.5 Gpc at t=13.8 Gyr

## Related Files

- `cosmo/analysis.py:46-106` - `solve_friedmann_equation()` implementation
- `cosmo/analysis.py:109-158` - `calculate_initial_conditions()` (fixed bug on line 138)
- `run_simulation.py:54-94` - LCDM baseline computation
- `parameter_sweep.py:67-79` - LCDM baseline computation (refactored)
- `tests/test_simulation_baseline.py` - Validation tests
