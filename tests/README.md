# Unit Tests

## Overview

Small, physics-based unit tests for validating fundamental calculations.

## Test Status

### ✅ Passing: test_constants.py (21/21)
- Physical constants (G, c, conversions)
- ΛCDM parameters (H₀, Ω_m, Ω_Λ)
- ExternalNodeParameters calculations
- SimulationParameters setup

### ⚠️  Failing: test_forces.py (0/12)

**Root cause**: Tests expose API mismatches. Tests were written against expected physics API, but actual implementation uses different signatures.

**Specific issues**:
1. `ParticleSystem(box_size_Gpc=...)` → should be `ParticleSystem(box_size=... * Gpc_to_m)`
2. `HMEAGrid(M_ext=..., S=...)` → should be `HMEAGrid(node_params=ExternalNodeParameters(...))`

**Decision**: Keep tests as-is (they test correct physics). Fix implementation or update tests in future work.

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Specific file
python -m pytest tests/test_constants.py -v

# Specific test
python -m pytest tests/test_constants.py::TestLambdaCDMParameters::test_hubble_at_present -v
```

## Test Philosophy

- **Start small**: Test individual physics calculations before complex simulations
- **Prefer correct tests**: If test fails, check if physics is wrong first (may expose bugs)
- **Keep failing tests**: Don't delete tests that expose issues - they document expected behavior
- **Physics-first**: Tests based on fundamental equations (F=GMm/r², a=H₀²Ω_Λr, etc.)

## Test Coverage

### Constants & Parameters ✅
- Unit conversions (Gpc↔m, Gyr↔s)
- Hubble parameter H(a)
- Ω_Λ_eff calculation
- Parameter validation

### Force Calculations ⚠️ (needs API fixes)
- Gravitational attraction (Newton's law)
- Tidal forces from external nodes
- Dark energy acceleration
- Hubble drag

### Integration (todo)
- Leapfrog timestep
- Energy conservation
- Position/velocity updates

### Simulation (todo)
- Expansion history extraction
- RMS radius calculation
- Snapshot saving/loading

## Future Work

1. Fix API mismatches in test_forces.py
2. Add integration tests (Leapfrog algorithm)
3. Add end-to-end simulation tests
4. Add regression tests (compare against known good runs)
5. Add parameter sweep validation
