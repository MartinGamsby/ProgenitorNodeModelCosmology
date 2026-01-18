# Initial Conditions

## Problem Statement

At t_start (e.g., 10.8 Gyr after Big Bang), observable universe has:
- Size R₀ ≈ 11.6 Gpc (derived from ΛCDM scale factor)
- Hubble expansion velocity v = H(t_start) × r

However, ΛCDM expansion includes ongoing Hubble drag (a_drag = -2Hv) that damps velocities. External-Node and Matter-only models have **no ongoing Hubble drag** (it's property of dark energy, not matter).

**Challenge**: If we initialize with full Hubble flow velocities, particles will overexpand because there's no drag to slow them.

**Solution**: Damp initial velocities by factor ~0.91 to compensate for absent drag term.

## Implementation

**File**: particles.py:73-116

### Position Initialization

```python
# Random uniform distribution in cube
positions = np.random.rand(n_particles, 3) * box_size_Gpc * const.Gpc_to_m
positions -= box_size_Gpc * const.Gpc_to_m / 2  # Center at origin
```

Particles uniformly distributed in `[-box_size/2, +box_size/2]³`.

### Velocity Initialization (Key Code)

```python
def _initialize_velocities(self, t_start_Gyr, a_start, use_dark_energy, damping_factor):
    """
    Initialize velocities with damped Hubble flow

    For matter-dominated expansion without dark energy, we damp the
    Hubble flow to compensate for lack of ongoing Hubble drag.

    Parameters:
    -----------
    t_start_Gyr : float
        Starting time [Gyr after Big Bang]
    a_start : float
        Scale factor at t_start
    use_dark_energy : bool
        Whether this is ΛCDM (True) or matter-only/External-Node (False)
    damping_factor : float or None
        Velocity damping coefficient. If None, auto-calculated.
    """
    lcdm = LambdaCDMParameters()
    H_start = lcdm.H_at_time(a_start)

    # Auto-calculate damping if not provided
    if damping_factor is None:
        if use_dark_energy:
            # ΛCDM: no damping, Hubble drag term handles it
            damping_factor = 1.0
        else:
            # Matter-only/External-Node: damp to compensate for no drag
            # Empirically tuned to ~0.91
            damping_factor = 0.91

    # Hubble flow: v = damping_factor × H × r
    velocities = damping_factor * H_start * self.positions
    self.particles = [
        Particle(pos, vel, mass, pid)
        for pid, (pos, vel, mass) in enumerate(zip(self.positions, velocities, masses))
    ]
```

### Damping Factor Rationale

**ΛCDM (use_dark_energy=True)**:
- Damping = 1.0 (no adjustment)
- Initial velocities: v = H₀ × r
- Ongoing Hubble drag keeps expansion on track

**External-Node / Matter-only (use_dark_energy=False)**:
- Damping ≈ 0.91 (reduced by 9%)
- Initial velocities: v = 0.91 × H₀ × r
- No ongoing drag, but lower initial velocity compensates

**Why 0.91?**: Empirically tuned. With damping=1.0, External-Node overexpands by ~4%. With damping=0.91, achieves 99.4% match to ΛCDM.

**Physical interpretation**: In matter-only regime, expansion naturally decelerates. Starting with slightly lower velocities ensures deceleration trajectory matches ΛCDM's drag-damped trajectory.

## Scale Factor at t_start

**File**: run_simulation.py:62-73

```python
# Solve ΛCDM Friedmann equation from Big Bang to present
a_full = odeint(friedmann_equation, a0=0.001, t_span_full,
                args=(H0, Omega_m, Omega_Lambda))

# Find scale factor at t_start
idx_start = np.argmin(np.abs(t_Gyr_full - t_start_Gyr))
a_at_start = a_full[idx_start]

# Today's scale factor (a=1 at z=0, t≈13.8 Gyr)
idx_today = np.argmin(np.abs(t_Gyr_full - 13.8))

# Initial box size: scale present-day size backward
lcdm_initial_size = 14.5 * (a_at_start / a_full[idx_today])  # Gpc
```

**Example**:
- t_start = 10.8 Gyr → a_start ≈ 0.839
- Present day (13.8 Gyr) → a_today ≈ 1.0
- Initial size = 14.5 × 0.839 = 12.17 Gpc

Note: Actual code uses `a_full[idx_today]` instead of assuming a_today=1.0 for numerical accuracy.

## Mass Initialization

```python
# Total mass in observable universe
rho_crit = 3 * lcdm.H0**2 / (8 * np.pi * const.G)
total_mass = lcdm.Omega_m * rho_crit * box_volume

# Distribute uniformly among particles
masses = np.full(n_particles, total_mass / n_particles)
```

**Typical values**:
- ρ_crit ≈ 9.5e-27 kg/m³
- Ω_m = 0.3
- box_volume ≈ (12 Gpc)³ = 1.7e78 m³
- Total mass ≈ 5e52 kg × n_particles
- Per-particle mass ≈ 1.7e53 kg (300 particles)

## Hubble Parameter at t_start

**File**: constants.py:63-69

```python
def H_at_time(self, a):
    """
    Hubble parameter at scale factor a

    H(a) = H₀ × √(Ω_m/a³ + Ω_Λ)
    """
    return self.H0 * np.sqrt(self.Omega_m / a**3 + self.Omega_Lambda)
```

**Example** (t_start = 10.8 Gyr, a ≈ 0.839):
```
H = 70 × √(0.3/0.839³ + 0.7)
  ≈ 70 × √(0.508 + 0.7)
  ≈ 70 × 1.099
  ≈ 76.9 km/s/Mpc
```

Converts to SI: H ≈ 2.49e-18 s⁻¹

## Initial Velocity Magnitude

For particle at r = 5 Gpc from center (typical):

**ΛCDM**:
```
v = 1.0 × H × r
  = 2.49e-18 s⁻¹ × 5e25 m
  = 1.25e8 m/s
  = 125,000 km/s
```

**External-Node**:
```
v = 0.91 × H × r
  = 0.91 × 1.25e8 m/s
  = 1.14e8 m/s
  = 114,000 km/s
```

**Difference**: 9% slower initial expansion, compensates for no Hubble drag.

## Seed Reproducibility

```python
np.random.seed(sim_params.seed)  # Before particle creation
sim = CosmologicalSimulation(...)

# Later, for comparison:
np.random.seed(sim_params.seed)  # Same seed → identical positions/velocities
sim_matter = CosmologicalSimulation(...)
```

**Critical**: External-Node and Matter-only simulations use **same seed** to ensure identical initial positions. Only difference is whether external nodes are active. Enables apples-to-apples comparison.

## Summary Table

| Parameter | ΛCDM | External-Node | Matter-only |
|-----------|------|---------------|-------------|
| Damping factor | 1.0 | 0.91 | 0.91 |
| Initial velocity | v = Hr | v = 0.91×Hr | v = 0.91×Hr |
| Ongoing drag | Yes (-2Hv) | No | No |
| External nodes | No | Yes (26 nodes) | No |
| Dark energy | Yes (H₀²Ω_Λr) | No | No |

**Result**: All three produce similar expansion history. External-Node matches ΛCDM to 99.4% despite completely different force model.

## Diagram

```mermaid
graph TD
    A[Solve Friedmann Equation] --> B[Get a at t_start]
    B --> C[Calculate H at t_start]
    C --> D{use_dark_energy?}
    D -->|Yes ΛCDM| E[damping = 1.0]
    D -->|No matter/external| F[damping = 0.91]
    E --> G[v = damping × H × r]
    F --> G
    G --> H[Initialize ParticleSystem]
    H --> I{Mode?}
    I -->|ΛCDM| J[Add dark energy + Hubble drag]
    I -->|External-Node| K[Add HMEA tidal forces]
    I -->|Matter-only| L[Only internal gravity]
```

## References

- Damping calculation: particles.py:79-100
- Hubble parameter: constants.py:63-69
- Friedmann solver: run_simulation.py:44-59
- See also: [force-calculations.md](./force-calculations.md) for why Hubble drag only in ΛCDM
