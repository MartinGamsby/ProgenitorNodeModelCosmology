# Unit Naming Conventions

SI units (meters, seconds, kilograms) used internally. Variable names include explicit unit suffixes for clarity and safety.

## Suffixes

**Distance**:
- `_m`: meters (r_m, diameter_m, rms_radius_m, softening_m, separation_m)
- `_Gpc`: Gigaparsecs (user-facing only, converted at I/O boundaries)

**Time**:
- `_s`: seconds (time_s, dt_s, t_start_s, t_end_s, age_s)
- `_Gyr`: Gigayears (user-facing only, converted at I/O boundaries)

**Mass**:
- `_kg`: kilograms (mass_kg, M_observable_kg, M_ext_kg, total_mass_kg, m_particle_kg)

**Rate**:
- `_si`: Hubble parameter in s^-1 (H0_si, H_si, hubble_rate_si)

**Acceleration**:
- `_mps2`: m/s² (a_internal_mps2, a_external_mps2, a_tidal_mps2, acc_mps2)

## Critical: Radius vs Diameter

**IMPORTANT DISTINCTION**:
- `rms_radius_m`: RMS distance from center of mass = sqrt(mean(r²))
- `diameter_m`: Full width = 2 × rms_radius_m

**Expansion history semantics**:
```python
expansion_history = {
    'time_s': [...],          # Time in seconds
    'diameter_m': [...],      # DIAMETER (2×RMS), not radius!
    'scale_factor': [...]     # a(t) / a(t_start)
}
```

**Why diameter?** Observable universe size is conventionally reported as diameter (e.g., "28 Gpc observable universe"). Storing diameter directly matches user expectations and avoids factor-of-2 confusion in comparisons.

**Internal calculations** use `rms_radius_m` for physics (e.g., force calculations, energy), but convert to `diameter_m` for storage/output.

## Examples

**Factory functions**:
```python
def create_default_external_node_model(
    total_mass_kg: float,
    lattice_spacing_m: float,
    t_start_s: float,
    t_end_s: float,
    dt_s: float
) -> dict:
    ...
```

**Force calculations**:
```python
def calculate_internal_forces(
    positions_m: np.ndarray,
    masses_kg: np.ndarray,
    softening_m: float
) -> np.ndarray:  # Returns accelerations in m/s²
    ...
```

**Integration**:
```python
def evolve(self, dt_s: float) -> None:
    # Leapfrog kick-drift-kick
    a_internal_mps2 = self.calculate_internal_forces()
    a_external_mps2 = self.calculate_external_forces()
    a_total_mps2 = a_internal_mps2 + a_external_mps2
    ...
```

**Analysis**:
```python
def compute_expansion_metrics(particles: ParticleCollection) -> dict:
    positions_m = particles.get_positions()
    rms_radius_m = compute_rms_radius(positions_m)
    diameter_m = 2.0 * rms_radius_m

    return {
        'rms_radius_m': rms_radius_m,
        'diameter_m': diameter_m,  # Stored for history
        'com_m': compute_center_of_mass(particles)
    }
```

## Conversion Constants

Defined in `cosmo/constants.py`:
```python
# Distance
GPC_TO_METERS = 3.0857e25        # 1 Gpc = 3.0857e25 m
METERS_TO_GPC = 1 / GPC_TO_METERS

# Time
GYR_TO_SECONDS = 3.15576e16      # 1 Gyr = 3.15576e16 s
SECONDS_TO_GYR = 1 / GYR_TO_SECONDS

# Hubble
H0_KM_S_MPC = 70.0               # User-facing: 70 km/s/Mpc
H0_SI = 2.268e-18                # Internal: s^-1
```

**I/O boundary pattern**:
```python
# User provides Gpc/Gyr, convert to SI immediately
lattice_spacing_m = lattice_spacing_Gpc * GPC_TO_METERS
t_end_s = t_end_Gyr * GYR_TO_SECONDS

# Internal calculations use SI
expansion_history = run_simulation(t_end_s, dt_s, ...)

# Output converts back to user-facing units
diameter_Gpc = expansion_history['diameter_m'] * METERS_TO_GPC
print(f"Final size: {diameter_Gpc:.2f} Gpc")
```

## Benefits

**Safety**: Type-like checking at the variable name level. Harder to mix units accidentally.

**Clarity**: Code is self-documenting. `softening_m` is unambiguous, `softening` is not.

**Debugging**: Unit mismatches become obvious in variable names before becoming bugs.

**Review**: Easier to verify physical correctness when units are explicit.

## Enforcement

**New code MUST**:
- Use unit suffixes for all physical quantities
- Document units in function signatures (type hints + docstrings)
- Convert at I/O boundaries, not mid-calculation

**Legacy code**:
- Refactor opportunistically during modifications
- Prioritize high-risk areas (force calculations, integrators)
- Tests validate unit handling (see test_units_validation.py)

## Related Files

- cosmo/constants.py - Conversion constants and SI values
- tests/test_units_validation.py - Unit handling validation
- tests/test_radius_diameter_semantics.py - Radius vs diameter correctness
- lode/practices.md - General coding practices
