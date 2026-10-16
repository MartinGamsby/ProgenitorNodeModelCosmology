# Tests

Physics-first unit and integration tests for the simulator, scoring and sweep machinery.

## Running

```bash
# Everything except the heavy N-body tests (~1.5 min)
python -m pytest -q

# Only the heavy tests marked @pytest.mark.slow
python -m pytest -m slow -q

# One file / one test
python -m pytest tests/test_force_reuse.py -q
python -m pytest tests/test_constants.py::TestLambdaCDMParameters -q
```

Slow tests are skipped by default (see the root `conftest.py`); passing any `-m` expression
selects markers explicitly.

## Data

- `fixtures/pantheon_synthetic.dat` — 11 synthetic supernovae, used by every test that does
  not depend on the real data.
- The real Pantheon+SH0ES file is not committed (see `data/pantheon_plus/README.md`); tests
  that assert real-data values skip when it is absent.

## Philosophy

- Test the physics (e.g. M_ext = 0 reproduces Einstein-de Sitter, node masses are
  mean-preserving) and the contracts (every sweep knob reaches both the cache key and the
  simulation), not implementation details.
- A refactor that must not change results gets an oracle test proving bit-identical output.
- A failing test is first a question about the physics: check whether it exposes a real bug
  before changing it.
