# Testing

## Structure

Tests in `tests/`. Physics-first: validate equations (F=GMm/r², a=H₀²Ω_Λr) not implementation.

## Files

**test_constants.py**: Constants, ΛCDM params, External-Node params. Run before expensive simulations to catch unit bugs.

**test_forces.py**: Gravitational, tidal, dark energy, Hubble drag forces. Currently has API mismatches (tests expect `ParticleSystem(box_size_Gpc=...)`, code uses `box_size=<meters>`). Tests document expected physics API.

## Running

```bash
pytest tests/test_constants.py -v
```

## Missing

Integration tests (Leapfrog correctness), particle initialization (damped Hubble flow), full simulation validation.
