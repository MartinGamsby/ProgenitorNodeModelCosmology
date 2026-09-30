# Lode map

Read first: [summary.md](./summary.md), [terminology.md](./terminology.md), this map. Then the
files for the task at hand. [README.md](./README.md) = what Lode coding is.

## Core
- [summary.md](./summary.md) — one-paragraph snapshot of model, evidence, status
- [terminology.md](./terminology.md) — domain vocabulary, one line per term
- [practices.md](./practices.md) — honest reporting, invariants, adding knobs, sweeps, style
- [practices/unit-conventions.md](./practices/unit-conventions.md) — SI + suffixes, constants, the 365 vs 365.25 d year issue
- [paper-reference.md](./paper-reference.md) — draft paper section map, what is current vs legacy

## Findings (research record; PF IDs are cited in code)
- [findings/summary.md](./findings/summary.md) — headline, do-not-claim list, PF index
- [findings/fit.md](./findings/fit.md) — Pantheon+ fit: PF1, 3–6, 10, 12, 15–16, 19, 21, 24, 28
- [findings/structure.md](./findings/structure.md) — anisotropy, knot, universe-fit: PF2, 18, 20, 22–23, 25–28
- [findings/methodology.md](./findings/methodology.md) — measurement traps: PF7–9, 11, 13, 17, 22, keyed==run bugs

## Physics
- [physics/theoretical-framework.md](./physics/theoretical-framework.md) — mechanism, Progenitor Hypothesis, predictions, scope
- [physics/initial-conditions.md](./physics/initial-conditions.md) — EdS-consistent ICs, samplers (uniform/grf/grfmass), start size, pre-start boost
- [physics/centerm-outer-mass.md](./physics/centerm-outer-mass.md) — centerM in both epochs, outer shell, observable mask, vcm coupling
- [physics/node-geometry.md](./physics/node-geometry.md) — cube26/lattices/virialized; medium relaxation; contracts
- [physics/forces.md](./physics/forces.md) — internal gravity methods + softening, tidal laws, perturbation knobs, dark energy
- [physics/integration.md](./physics/integration.md) — KDK with one force evaluation per step, sub-steps, timestep guard
- [physics/hubble-diagram.md](./physics/hubble-diagram.md) — a(t) → μ(z), χ² engine, growth anchor, references
- [physics/observers.md](./physics/observers.md) — per-particle observers, best-observer scoring
- [physics/anisotropy-diagnostic.md](./physics/anisotropy-diagnostic.md) — shear, Hubble dipole, axis RMS

## Numerics
- [numerics/lcdm-baseline.md](./numerics/lcdm-baseline.md) — analytic ΛCDM at snapshot times, initial box
- [numerics/expansion-rate-calculation.md](./numerics/expansion-rate-calculation.md) — H(t) from a(t); never smooth first
- [numerics/memory-and-parallelism.md](./numerics/memory-and-parallelism.md) — RAM-bound pools, PE byte budget, pool-driver rules

## Architecture
- [architecture/module-structure.md](./architecture/module-structure.md) — what each module owns, import graph
- [architecture/data-flow.md](./architecture/data-flow.md) — one sim and one sweep cell, stage contracts
- [architecture/testing.md](./architecture/testing.md) — running tests, slow marker, map by area, test rules

## Scripts
- [scripts/sweep.md](./scripts/sweep.md) — `sweep.py`: config, grid rules, S search, knot guard, outputs, launching
- [scripts/sweep-cache.md](./scripts/sweep-cache.md) — cache key anatomy, PHYSICS_CACHE_VERSION, concurrency
- [scripts/visualization.md](./scripts/visualization.md) — paper figures and their sources, study scripts

## Plans
- [plans/roadmap.md](./plans/roadmap.md) — ALL open work, pending decisions, housekeeping
- [plans/scale-out-gpu-hf.md](./plans/scale-out-gpu-hf.md) — self-hosted GPU (`cuda_direct`) + HF Dataset/Space

## Temporary
- `tmp/` — session scraps and handovers (gitignored)
