# Lode Map

## Core Files
- [summary.md](./summary.md) - One-paragraph project overview
- [terminology.md](./terminology.md) - Domain vocabulary (HMEA, Ω_Λ_eff, damping factor, etc.)
- [practices.md](./practices.md) - Development patterns and conventions
- [practices/unit-conventions.md](./practices/unit-conventions.md) - Unit naming conventions (_m, _s, _kg, _si, _mps2 suffixes)
- [paper-reference.md](./paper-reference.md) - docs/paper.tex ground truth: what code tests, scope, success criteria

## Architecture
- [architecture/module-structure.md](./architecture/module-structure.md) - Code organization and dependencies
- [architecture/data-flow.md](./architecture/data-flow.md) - How data flows through run_simulation.py
- [architecture/testing.md](./architecture/testing.md) - Unit test structure, status, philosophy

## Physics
- [physics/theoretical-framework.md](./physics/theoretical-framework.md) - External-Node Model, Progenitor Hypothesis, predictions, scope/limitations
- [physics/force-calculations.md](./physics/force-calculations.md) - Internal gravity, tidal forces, dark energy, Hubble drag
- [physics/barnes-hut-optimization.md](./physics/barnes-hut-optimization.md) - Numba JIT internal forces (O(N²) direct, not tree); speedup from compilation
- [physics/initial-conditions.md](./physics/initial-conditions.md) - Damped Hubble flow setup and rationale
- [physics/integration.md](./physics/integration.md) - Leapfrog algorithm implementation

## Numerics
- [numerics/timestep-stability.md](./numerics/timestep-stability.md) - Timestep requirements, instability symptoms, energy monitoring
- [numerics/lcdm-baseline.md](./numerics/lcdm-baseline.md) - ΛCDM baseline computation standardization, reference values, bug fixes
- [numerics/leapfrog-staggering.md](./numerics/leapfrog-staggering.md) - Velocity staggering, pre-kick fix, initial bump elimination
- [numerics/expansion-rate-calculation.md](./numerics/expansion-rate-calculation.md) - Hubble parameter H(t) from numerical derivatives, edge artifacts from smoothing, diagnostic tools

## Scripts
- [scripts/parameter-sweep.md](./scripts/parameter-sweep.md) - Grid search methodology, match metrics, best configurations
- [scripts/visualization.md](./scripts/visualization.md) - 3D visualization pipeline, comparison mode, animation

## Plans
- [plans/](./plans/) - Future enhancements and TODOs

## Temporary
- [tmp/](./tmp/) - Session scraps (git-ignored)
