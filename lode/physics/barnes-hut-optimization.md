# Force Calculation Methods

## Overview

Three internal force calculation methods available, selected via `force_method` parameter:

| Method | File | Complexity | Accuracy | Speedup |
|--------|------|------------|----------|---------|
| `'direct'` | integrator.py | O(N²) | Exact | baseline |
| `'numba_direct'` | numba_direct.py | O(N²) | Exact | 14-17x |
| `'barnes_hut'` | barnes_hut_numba.py | O(N log N) | ~5% (θ=0.5) | Varies |

**Default (auto mode)**: Uses `barnes_hut` for N≥1000, `numba_direct` for N≥100, `direct` for N<100.

## numba_direct (Recommended)

**File**: cosmo/numba_direct.py
**Class**: NumbaDirectSolver

Numba JIT-compiled O(N²) direct summation. Same accuracy as numpy vectorized but 14-17x faster through compiled loops.

```python
# Auto mode (default): uses numba_direct for N>=100
integrator = LeapfrogIntegrator(particles)

# Force numba_direct explicitly
integrator = LeapfrogIntegrator(particles, force_method='numba_direct')
```

N=300 particles benchmark:

| Method | Time | Speedup | Accuracy |
|--------|------|---------|----------|
| direct (NumPy) | 10.6 ms | baseline | exact |
| numba_direct (Numba) | 0.7 ms | 14.4x | exact |

## barnes_hut (Real Octree)

**File**: cosmo/barnes_hut_numba.py
**Class**: NumbaBarnesHutTree

Real Barnes-Hut octree algorithm with Numba JIT compilation. O(N log N) using opening angle criterion.

```python
# Barnes-Hut with default θ=0.5
integrator = LeapfrogIntegrator(particles, force_method='barnes_hut')

# Adjust opening angle for accuracy/speed tradeoff
integrator = LeapfrogIntegrator(particles, force_method='barnes_hut', barnes_hut_theta=0.3)
```

**Opening angle θ controls accuracy**:
- θ=0.3: ~2% error, slower
- θ=0.5: ~5% error, balanced
- θ=0.7: ~10% error, faster

**When to use**: Only for N>1000 when O(N²) becomes prohibitive. For typical simulations (N=200-500), numba_direct is faster and exact.

## Implementation Details

### numba_direct.py

```python
@jit(nopython=True, cache=True)
def calculate_forces_direct_numba(positions, masses, softening, G):
    """Pairwise O(N²) with softening, JIT-compiled."""
    N = len(positions)
    accelerations = np.zeros((N, 3))
    for i in range(N):
        for j in range(N):
            if i == j: continue
            r_soft = sqrt(r2 + softening**2)
            f = G * masses[j] / (r_soft**3)
            accelerations[i] += f * r_vec
    return accelerations
```

### barnes_hut_numba.py

Octree structure with iterative particle insertion:
1. `build_octree()`: Insert particles, split nodes when needed (max_depth=60)
2. `calculate_forces_barnes_hut()`: Stack-based tree traversal per particle
3. Opening angle: if s/r < θ (node_size/distance), use COM approximation

```python
# Opening angle criterion
if node_is_leaf or node_size/distance < theta:
    # Use center-of-mass approximation
else:
    # Traverse children
```

## Testing

**tests/test_barnes_hut.py**:
- `TestNumbaDirectPerformance`: Exact match to numpy, >5x speedup
- `TestBarnesHutOctree`: <5% error with θ=0.5, accuracy improves with smaller θ

**tests/test_integrator.py**: Parametrized tests for all three methods

```bash
pytest tests/test_barnes_hut.py -v
pytest tests/test_integrator.py -v
```
