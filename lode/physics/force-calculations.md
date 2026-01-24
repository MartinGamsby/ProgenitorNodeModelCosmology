# Force Calculations

## Overview

Four force types contribute to particle acceleration:
1. **Internal gravity**: Pairwise attraction between particles
2. **External tidal forces**: Pull from 26 HMEA nodes
3. **Dark energy**: Cosmological constant repulsion (ΛCDM only)
4. **Hubble drag**: Friction from cosmic expansion (ΛCDM only)

## 1. Internal Gravity

**File**: integrator.py:71-112

**Formula**:
```
a_ij = -G × m_j × (r_i - r_j) / (r² + ε²)^(3/2)
```

### Two Implementation Methods

#### Method 1: NumPy Direct (Default)
**Complexity**: O(N²) - Direct pairwise summation
**File**: integrator.py:71-112
**Speed**: Baseline (10.6ms for N=300)

```python
def calculate_internal_forces(self):
    """
    O(N²) vectorized pairwise gravity calculation.

    Highly optimized NumPy/BLAS operations. Default method.
    """
    positions_m = self.particles.get_positions()  # (N, 3)
    masses_kg = self.particles.get_masses()       # (N,)

    # Broadcast to (N, N, 3) - all pairwise displacement vectors
    r_vec_m = positions_m[np.newaxis, :, :] - positions_m[:, np.newaxis, :]

    # Distance matrix with softening
    r_m = np.linalg.norm(r_vec_m, axis=2)
    r_soft_m = np.sqrt(r_m**2 + self.softening_m**2)

    # Acceleration magnitudes (N, N)
    with np.errstate(divide='ignore', invalid='ignore'):
        a_mag_mps2 = self.const.G * masses_kg[np.newaxis, :] / r_soft_m**2
        a_mag_mps2[np.isnan(a_mag_mps2)] = 0

    # Vectorized direction calculation (N, N, 3)
    r_hat = r_vec_m / r_soft_m[:, :, np.newaxis]
    a_vec_mps2 = a_mag_mps2[:, :, np.newaxis] * r_hat

    # Sum over all j for each i
    return np.sum(a_vec_mps2, axis=1)  # (N, 3)
```

#### Method 2: Numba Barnes-Hut (Optimized)
**Complexity**: O(N log N) - Hierarchical approximation
**File**: barnes_hut_numba.py
**Speed**: 14-17x faster (0.7ms for N=300)
**Accuracy**: ~1e-16 relative error (virtually identical)

```python
from cosmo.barnes_hut_numba import NumbaBarnesHutTree

# In integrator __init__:
self.force_method = 'barnes_hut'  # or 'direct'
self.barnes_hut_theta = 0.5       # opening angle

def calculate_internal_forces_barnes_hut(self):
    """
    O(N log N) Barnes-Hut approximation with Numba JIT compilation.

    14-17x faster than direct method with virtually identical results.
    """
    positions = self.particles.get_positions()
    masses = self.particles.get_masses()

    tree = NumbaBarnesHutTree(
        theta=self.barnes_hut_theta,
        softening_m=self.softening_m,
        G=self.const.G
    )
    tree.build_tree(positions, masses)
    return tree.calculate_all_accelerations()
```

**Performance comparison** (N=300 particles):

| Method | Time | Speedup | Accuracy |
|--------|------|---------|----------|
| NumPy Direct | 10.6 ms | baseline | exact |
| Numba Barnes-Hut | 0.7 ms | 14.4x | ~1e-16 error |

**Usage**:
```python
# Default: NumPy direct
integrator = LeapfrogIntegrator(particles)

# Optimized: Numba Barnes-Hut
integrator = LeapfrogIntegrator(particles,
                                force_method='barnes_hut',
                                barnes_hut_theta=0.5)
```

**Parameters**:
- `G = 6.674e-11 m³/(kg·s²)` - Gravitational constant
- `softening_m = 1e21 m ≈ 1 Mpc` - Prevents singularities at r→0
- `mass_kg ≈ 1e53 kg` - Particle mass (galaxy cluster)
- `theta = 0.5` - Barnes-Hut opening angle (smaller = more accurate)

**Typical magnitude**: ~1e-11 m/s² at 1 Gpc separation

**Diagram**:
```mermaid
graph LR
    P1[Particle i] -->|r_vec| P2[Particle j]
    P2 -->|F = Gm_j/r²| P1
    P1 -.->|softening ε| P1
```

## 2. External Tidal Forces

**File**: particles.py:238-272

**Formula**:
```
a_tidal = Σ_nodes [G × M_ext × (r - r_node) / |r - r_node|³]
```

**Implementation**:
```python
def calculate_tidal_acceleration_batch(self, positions_m):
    """
    Vectorized tidal force calculation

    positions_m: (N, 3) array of particle positions in meters
    returns: (N, 3) array of accelerations in m/s²
    """
    N = positions_m.shape[0]
    accelerations_mps2 = np.zeros((N, 3))

    for M_ext_kg, r_node_m in self.nodes:
        # Displacement vectors from node to particles (vectorized)
        displacement_m = positions_m - r_node_m  # (N, 3)

        # Distance from node to each particle
        r_m = np.linalg.norm(displacement_m, axis=1, keepdims=True)  # (N, 1)

        # Tidal acceleration (pointing away from node)
        a_tidal_mps2 = self.const.G * M_ext_kg * displacement_m / r_m**3  # (N, 3)

        accelerations_mps2 += a_tidal_mps2

    return accelerations_mps2
```

**Parameters**:
- `M_ext_kg = 800 × M_observable_kg ≈ 8e55 kg` - External node mass
- `r_node_m ≈ 24 Gpc × 3.0857e25 m/Gpc` - Distance to nearest nodes
- 26 nodes total in 3×3×3 lattice

**Key insight**: Tidal forces are *gradients* of gravitational potential. Node at (+S, 0, 0) pulls more on particles near (+box/2, 0, 0) than particles near (-box/2, 0, 0), creating net expansion.

**Typical magnitude**: ~1e-10 m/s² (dominates internal gravity)

**Effective dark energy**:
```
Ω_Λ_eff = G × M_ext / (S³ × H₀²) ≈ 2.555
```
With 26 nodes, symmetry causes cancellations → effective Ω_Λ ≈ 0.7

**Diagram**:
```mermaid
graph TD
    N1[Node -S,-S,-S] -.tidal.-> O[Observable<br/>Universe]
    N2[Node +S,0,0] -.tidal.-> O
    N3[Node 0,+S,0] -.tidal.-> O
    N4[Node 0,0,+S] -.tidal.-> O
    N5[...22 more nodes] -.tidal.-> O
    O -->|Net effect:<br/>expansion| O
```

## 3. Dark Energy (ΛCDM only)

**File**: integrator.py:104-125

**Formula**:
```
a_Λ = H₀² × Ω_Λ × r
```

**Implementation**:
```python
def calculate_dark_energy_forces(self):
    if not self.use_dark_energy:
        return np.zeros((len(self.particles), 3))

    positions_m = self.particles.get_positions()
    H0_si = self.lcdm.H0  # s^-1
    a_Lambda_mps2 = H0_si**2 * self.lcdm.Omega_Lambda * positions_m

    return a_Lambda_mps2
```

**Parameters**:
- `H0_si = 2.268e-18 s⁻¹` (70 km/s/Mpc in SI)
- `Ω_Λ = 0.7` - Dark energy density parameter

**Physical meaning**: Cosmological constant creates repulsive force proportional to distance. Farther particles accelerate faster (exponential expansion).

**Typical magnitude**: ~1e-10 m/s² at 10 Gpc

**Only active when**: `use_dark_energy=True` (ΛCDM mode)

## 4. Hubble Drag (ΛCDM only)

**File**: integrator.py:127-165 (calculation), integrator.py:265-270 (application)

**Formula**:
```
v(t+dt) = v(t) × exp(-2H₀ × dt)
```

**Implementation**:
```python
def calculate_hubble_drag(self):
    """Returns drag acceleration (legacy, now unused in leapfrog)"""
    if not self.use_dark_energy:
        return np.zeros((len(self.particles), 3))

    velocities = self.particles.get_velocities()
    H_current = self.lcdm.H0
    a_drag = -2.0 * H_current * velocities

    return a_drag
```

**Actual application** (integrator.py:265-270):
```python
# After leapfrog kicks, apply Hubble drag implicitly
if self.use_dark_energy:
    H0_si = self.lcdm.H0  # s^-1
    gamma_si = 2.0 * H0_si
    damping_factor = np.exp(-gamma_si * dt_s)
    for particle in self.particles.particles:
        particle.vel *= damping_factor
```

**Parameters**:
- `H0_si ≈ 2.268e-18 s⁻¹`
- `gamma_si = 2H0_si ≈ 4.537e-18 s⁻¹`

**Physical meaning**: In expanding universe, particles experience friction from cosmic expansion. Prevents runaway velocities from dark energy repulsion.

**CRITICAL UPDATE**: Hubble drag is **NOT APPLIED** in proper-coordinate simulations!

In proper coordinates with explicit dark energy, applying Hubble drag causes OVER-DAMPING:
- Dark energy provides: +H²Ω_Λr acceleration (outward)
- Hubble drag would provide: -2Hv deceleration (inward)
- With full Hubble flow v ≈ Hr, drag is ~3x stronger than dark energy
- Result: ΛCDM decelerates instead of accelerates!

Hubble drag (a_drag = -2Hv) is only appropriate for **comoving coordinates** where background expansion is implicit. In **proper coordinates**, dark energy acceleration alone handles expansion correctly.

**Typical damping** (over full timestep):
- dt_s=1e15 s (0.03 Gyr): 0.5% velocity reduction
- dt_s=1e16 s (0.32 Gyr): 4.4% velocity reduction
- dt_s=1e17 s (3.17 Gyr): 36.5% velocity reduction

**Only active when**: `use_dark_energy=True` (ΛCDM mode)

**Why not in External-Node/Matter-only?**: Hubble drag is property of cosmic expansion driven by dark energy. In matter-dominated regime, expansion decelerates naturally from gravity. External-Node model uses *damped initial conditions* instead of ongoing drag (see [initial-conditions.md](./initial-conditions.md)).

## Force Composition by Mode

| Mode | Internal | External | Dark Energy | Hubble Drag |
|------|----------|----------|-------------|-------------|
| ΛCDM | ❌ (negligible) | ❌ | ✅ | ✅ |
| External-Node | ✅ | ✅ | ❌ | ❌ |
| Matter-only | ✅ | ❌ | ❌ | ❌ |

**Note**: Internal gravity included in all modes but negligible compared to external/dark energy forces at Gpc scales.

## Total Force Calculation

**File**: integrator.py:159-179

```python
def calculate_total_forces(self):
    a_internal_mps2 = self.calculate_internal_forces()
    a_external_mps2 = self.calculate_external_forces()
    a_dark_energy_mps2 = self.calculate_dark_energy_forces()
    a_hubble_drag_mps2 = self.calculate_hubble_drag()

    return a_internal_mps2 + a_external_mps2 + a_dark_energy_mps2 + a_hubble_drag_mps2
```

Summed vectorially. Total acceleration determines time evolution via Leapfrog integrator.

## Relative Magnitudes

At 10 Gpc, v ~ 6e5 m/s:

| Force | Magnitude | Contribution |
|-------|-----------|--------------|
| Internal gravity | ~1e-11 m/s² | 10% |
| External tidal | ~1e-10 m/s² | 90% (External-Node) |
| Dark energy | ~1e-10 m/s² | 90% (ΛCDM) |
| Hubble drag | ~1e-11 m/s² | 10% (ΛCDM) |

External tidal ≈ Dark energy (by design, this is the key result!)
