# Testing

## Structure

Tests in `tests/`. Physics-first: validate equations (F=GMm/r², a=H₀²Ω_Λr) not implementation.

## Files

**test_constants.py**: Constants, ΛCDM params, External-Node params. All 21 tests passing.

**test_forces.py**: Gravitational, tidal, dark energy, Hubble drag forces. 6/12 passing. Failures expose physics bugs:
- Gravity magnitude wrong (damping factor interferes?)
- Tidal forces: irregular node positions cause directional drift, linear scaling fails
- Dark energy: acceleration direction wrong (points inward not outward)
- Numerical precision: tiny non-zero components (4e-8, 3e-13) in zero-expected directions

## Running

```bash
pytest tests/test_constants.py -v
pytest tests/test_forces.py -v
```

## Missing

Integration tests (Leapfrog correctness), particle initialization (damped Hubble flow), full simulation validation.
