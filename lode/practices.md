# Practices

## Code Organization

**Module separation**: `constants.py` (parameters) → `particles.py` (structures) → `integrator.py` (physics) → `simulation.py` (orchestration). Clear dependency hierarchy.

**SI units everywhere**: All internal calculations use meters, seconds, kilograms. Convert only at I/O boundaries. Variable names use explicit unit suffixes (_m, _s, _kg, _si, _mps2) for clarity. See [practices/unit-conventions.md](./practices/unit-conventions.md).

**Reproducibility**: Always set `np.random.seed(sim_params.seed)` before particle initialization.

## Docstring Style

Functions use concise docstrings relying on type hints and explicit variable naming.

**Pattern**:
- One-line summary describing WHAT/WHY, not parameter lists
- Type hints for all parameters and returns
- Unit suffixes in variable names (_m, _s, _Gyr, _kg, _si, _mps2)
- Brief notes for edge cases, physics context, or gotchas
- NO Parameters/Returns sections (redundant with type hints + naming)

**Preserve detailed docs only for**:
- Empirical data (e.g., stability thresholds)
- Non-obvious physics (formulas, derivations)
- Complex algorithms (e.g., leapfrog staggering)

**Example**:
```python
def calculate_hubble_rate(a: float, Omega_m: float, Omega_Lambda: float) -> float:
    """Hubble parameter H(a) = H0 * sqrt(Ω_m/a³ + Ω_Λ)"""
    return H0_si * np.sqrt(Omega_m / a**3 + Omega_Lambda)
```

**Special case - validation functions**:
```python
def _validate_timestep(self, t_duration_Gyr: float, n_steps: int) -> None:
    """
    Validate timestep for leapfrog numerical stability.

    Empirical testing shows dt < 0.05 Gyr required for stability:
    - 150 steps over 20 Gyr (dt=0.133 Gyr): UNSTABLE (1600% energy drift)
    - 500 steps over 20 Gyr (dt=0.040 Gyr): STABLE

    Recommended: dt < 0.04 Gyr for safety margin.

    Raises SystemExit if timestep too large.
    """
```

## Physics Implementation

**Initial conditions consistency**: Same seed for External-Node and Matter-only simulations ensures identical starting state for fair comparison.

**Damping factor calculation**: Auto-calculated from deceleration parameter q (formula: d=0.4-0.25×q, clipped to 0.1-0.7) unless explicitly overridden. Override with 0.91 for best empirical ΛCDM match, or 0.0 for test isolation.

**Force separation**: Keep internal gravity, external tidal, dark energy, and Hubble drag in separate methods. Aids debugging and A/B testing.

**Matter-only baseline**: Always run matter-only simulation alongside External-Node for sanity check. Matter-only should underexpand vs ΛCDM.

## Integration

**Leapfrog only**: Don't use Euler or RK4. Leapfrog preserves phase space volume (symplectic).

**Timestep sizing**: dt ≈ 40 Myr typical. Too large → energy drift. Too small → computation waste.

**Snapshot frequency**: Default save_interval=10 gives ~15 snapshots. Sufficient for smooth plots without bloat.

## Validation

**ΛCDM ground truth**: Always solve Friedmann equation analytically first. N-body is cross-check, not source of truth for ΛCDM.

**Three-way comparison**: Plot ΛCDM (analytic) vs External-Node (N-body) vs Matter-only (N-body). External-Node should track ΛCDM, Matter-only should fall below.

**Curve comparison metric**: Use R² (coefficient of determination) to compare expansion curves. R² = 1 - (SS_res/SS_tot) measures fraction of ΛCDM variance explained by model. R² > 0.9999 is excellent (corresponds to ~99.4% match), R² > 0.999 is good. For backward compatibility, percentage match still available via `use_r_squared=False`. This is exploratory research—no configuration is "optimal", just testing mechanism viability.

**Parameter exploration**: Use `parameter_sweep.py` for systematic grid search over M and S. Goal: understand parameter space, not find single "optimal" point.

## Output

**Filename convention**: Include timestamp, parameters (M, S, particles, timesteps, t_range) in output filename. Enables unambiguous identification.

**Pickle simulation object**: Save full `CosmologicalSimulation` instance, not just plots. Enables post-hoc analysis without re-running.

**4-panel standard plot**: Scale factor, Hubble parameter, relative expansion, physical size. This is the canonical comparison view.

## Performance

**N² scaling accepted**: Direct pairwise gravity for N~300 is fine (O(N²)). Only optimize if N>1000 needed.

**Numba JIT**: Internal gravity (barnes_hut_numba.py) and tidal forces (tidal_forces_numba.py) use Numba JIT compilation for 14-17x speedup. Auto-selected for N≥100.

**Vectorization**: External forces use batch calculation over all 26 nodes. Never loop over nodes in Python (except in Numba-compiled code where explicit loops are faster).

**Progress bars**: Use `tqdm` for user feedback during multi-minute runs.

## Git

**Worktree workflow**: Use Claude Code worktrees for parallel experimentation. Branch naming: descriptive (e.g., `eager-hermann`, `sweet-mclean`).

**Don't commit results**: `results/` in `.gitignore`. Only commit code and lode documentation.

## Testing

**Physics-first tests**: Validate fundamental equations (F=GMm/r², a=H₀²Ω_Λr) not implementation details.

**Start small**: Constants → Forces → Integration → Full simulation. Build confidence incrementally.

**Failing tests document expected behavior**: If test fails, check if it exposes real issue. Keep tests that document expected API or physics.

**Run before expensive work**: `pytest tests/test_constants.py` before long simulations. Catches unit conversion bugs early.

## Research Context

**Draft paper**: docs/paper.tex is working document, not published. Code generates data for ongoing refinement.

**Purpose**: Validate proof-of-concept mechanism, explore parameter space, test theoretical claims before publication.

**Communication**: Describe results as "testing", "exploring", "hypothesized", not "proven" or "optimal".
