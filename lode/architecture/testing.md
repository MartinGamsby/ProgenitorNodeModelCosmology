# Testing

## Current State

Unit tests created in `tests/` directory. Small, physics-based tests validating fundamental calculations before complex simulations.

## Test Files

### tests/test_constants.py ✅

**Status**: 21/21 passing

**Coverage**:
- `CosmologicalConstants`: G, c, unit conversions (Gpc↔m, Gyr↔s), M_observable
- `LambdaCDMParameters`: H₀ SI units, Ω_m + Ω_Λ = 1, H(a) calculation at different epochs
- `ExternalNodeParameters`: M_ext, S, Ω_Λ_eff calculation, `calculate_required_spacing()`
- `SimulationParameters`: Default values, custom parameters, t_end calculation

**Key validations**:
- H₀ ≈ 2.3×10⁻¹⁸ s⁻¹ (70 km/s/Mpc in SI)
- Flat universe: Ω_m + Ω_Λ = 1.0
- H(a=1) = H₀ (present day)
- H(a<1) > H₀ (early universe, matter dominated)
- Ω_Λ_eff = G×M_ext/(S³×H₀²)

### tests/test_forces.py ⚠️

**Status**: 0/12 passing (API mismatches)

**Intended coverage**:
- Gravitational forces: F=GMm/r², softening, Newton's 3rd law
- Tidal forces: Direction, linear approximation (a ∝ R for R≪S), symmetric grid cancellation
- Dark energy: a_Λ = H₀²Ω_Λr, radial direction
- Hubble drag: a_drag = -2Hv, opposes velocity, disabled without dark energy

**Why failing**:
Tests written against expected physics-based API, but actual implementation uses different signatures:
- Test expects: `ParticleSystem(box_size_Gpc=1.0)`
- Actual API: `ParticleSystem(box_size=<meters>)`
- Test expects: `HMEAGrid(M_ext=..., S=...)`
- Actual API: `HMEAGrid(node_params=ExternalNodeParameters(...))`

**Decision**: Keep tests as-is. They document expected physics-based API and may guide refactoring.

## Test Philosophy

**Physics-first**: Tests validate fundamental equations (F=GMm/r², a=H₀²Ω_Λr) not implementation details.

**Start small**: Constants → Forces → Integration → Full simulation. Build confidence incrementally.

**Failing tests are features**: If test fails, check if it exposes real issue. Keep failing tests that document expected behavior.

**No time estimates**: Tests either pass or fail. No predictions about "how long" fixes take.

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Specific file
python -m pytest tests/test_constants.py -v

# Watch for changes (if pytest-watch installed)
ptw tests/
```

## Test Results Summary

| Test Suite | Status | Count | Notes |
|------------|--------|-------|-------|
| test_constants.py | ✅ Pass | 21/21 | Physical constants, parameters validated |
| test_forces.py | ⚠️ Fail | 0/12 | API mismatches (tests correct, API needs review) |

## Missing Test Coverage

Not yet implemented:
- Integration tests (Leapfrog timestep correctness, energy conservation)
- ParticleSystem initialization (damped Hubble flow velocities)
- HMEAGrid construction (26-node cubic lattice, irregularity)
- Simulation end-to-end (expansion history extraction, RMS radius)
- Regression tests (compare against known good runs for M=800, S=24)
- Parameter sweep validation

## Test Structure

```python
# Example: test_constants.py structure
class TestCosmologicalConstants(unittest.TestCase):
    def setUp(self):
        self.const = CosmologicalConstants()

    def test_gravitational_constant(self):
        """G should be 6.674e-11 m^3/(kg*s^2)"""
        self.assertAlmostEqual(self.const.G, 6.674e-11, places=13)
```

Pattern:
1. setUp() creates objects under test
2. test_* methods validate single physics property
3. Docstrings explain what physics is being tested
4. Assertions check against known values or theoretical predictions

## Code Examples

**Testing H(a) calculation**:
```python
def test_hubble_at_present(self):
    """H(a=1) should equal H0"""
    H_present = self.lcdm.H_at_time(1.0)
    self.assertAlmostEqual(H_present, self.lcdm.H0, places=15)
```

**Testing tidal force direction**:
```python
def test_single_external_node_force_direction(self):
    """Particle should be pulled toward external node"""
    grid = HMEAGrid(node_params=ExternalNodeParameters(...))
    position = np.array([[0.0, 0.0, 0.0]])
    acceleration = grid.calculate_tidal_acceleration_batch(position)

    # Should pull in +x direction (toward node at +x)
    self.assertGreater(acceleration[0, 0], 0)
```

## Integration with Lode Coding

Tests are permanent learnings → documented in lode. Tests directory tracked in git. Test results inform practices (e.g., "always validate constants before running expensive simulations").

## Future Testing Plans

When implementing:
1. Fix API mismatches in test_forces.py (either update tests or refactor code)
2. Add integration tests for Leapfrog algorithm
3. Add regression tests comparing against known M=800, S=24 run
4. Add property-based tests (e.g., energy conservation for matter-only should hold within 1%)
5. Add performance benchmarks (N-body scaling, force calculation timing)

## Lessons Learned

- Writing tests exposed API inconsistencies (box_size_Gpc vs box_size)
- Physics-based tests are self-documenting (docstrings explain F=GMm/r²)
- Testing constants first caught unit conversion bugs early
- Failing tests provide value when they document expected behavior
