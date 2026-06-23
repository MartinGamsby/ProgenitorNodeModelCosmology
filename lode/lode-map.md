# Lode Map

## Core Files
- [summary.md](./summary.md) - One-paragraph project overview
- [terminology.md](./terminology.md) - Domain vocabulary (HMEA, Ω_Λ_eff, damping factor, etc.)
- [practices.md](./practices.md) - Development patterns and conventions
- [practices/unit-conventions.md](./practices/unit-conventions.md) - Unit naming conventions (_m, _s, _kg, _si, _mps2 suffixes)
- [paper-reference.md](./paper-reference.md) - docs/VirializedMetaStructure.tex ground truth: what code tests, scope, success criteria

## Architecture
- [architecture/module-structure.md](./architecture/module-structure.md) - Code organization and dependencies
- [architecture/data-flow.md](./architecture/data-flow.md) - How data flows through run_simulation.py
- [architecture/testing.md](./architecture/testing.md) - Unit test structure, status, philosophy

## Physics
- [physics/theoretical-framework.md](./physics/theoretical-framework.md) - External-Node Model, Progenitor Hypothesis, predictions, scope/limitations
- [physics/force-calculations.md](./physics/force-calculations.md) - Internal gravity, tidal forces, dark energy, Hubble drag
- [physics/barnes-hut-optimization.md](./physics/barnes-hut-optimization.md) - Force methods: direct, numba_direct (O(N²) JIT), barnes_hut (real octree O(N log N))
- [physics/initial-conditions.md](./physics/initial-conditions.md) - Damped Hubble flow setup, velocity calibration, validated t_start range (safe floor 2.9 Gyr, z_max~2.23)
- [physics/integration.md](./physics/integration.md) - Leapfrog algorithm implementation
- [physics/hubble-diagram.md](./physics/hubble-diagram.md) - Data-anchored mu(z) Hubble-diagram test vs real Pantheon+ SNe (semi-analytic H(z), offset marginalization); open M,S discrepancy

## Numerics
- [numerics/timestep-stability.md](./numerics/timestep-stability.md) - Timestep requirements, instability symptoms, energy monitoring
- [numerics/lcdm-baseline.md](./numerics/lcdm-baseline.md) - ΛCDM baseline computation standardization, reference values, bug fixes
- [numerics/leapfrog-staggering.md](./numerics/leapfrog-staggering.md) - Velocity staggering, pre-kick fix, initial bump elimination
- [numerics/expansion-rate-calculation.md](./numerics/expansion-rate-calculation.md) - Hubble parameter H(t) from numerical derivatives, edge artifacts from smoothing, diagnostic tools

## Scripts
- [scripts/parameter-sweep.md](./scripts/parameter-sweep.md) - Grid search methodology, match metrics, best configurations
- [scripts/visualization.md](./scripts/visualization.md) - 3D visualization pipeline, comparison mode, animation
- hubble_diagram.py - Standalone Hubble-diagram-vs-Pantheon+ script (documented in [physics/hubble-diagram.md](./physics/hubble-diagram.md))

## Plans
- [plans/](./plans/) - Future enhancements and TODOs
- [plans/hubble-diagram-followups.md](./plans/hubble-diagram-followups.md) - Reconcile (M,S) discrepancy, optional paper edit, full-covariance chi^2, N-body-derived d_L

## Temporary
- [tmp/](./tmp/) - Session scraps (git-ignored)
