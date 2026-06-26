# WS4 — centerM Reconception (design rationale)

STATUS: **IMPLEMENTED**. The mechanism as it exists in code is described in
[../physics/observable-mask-and-outer-mass.md](../physics/observable-mask-and-outer-mass.md).
This file keeps the DESIGN RATIONALE (why centerM was repurposed and what the
constraint was). The honest result is in
[pinned-findings.md](./pinned-findings.md) (PF6).

Back to [deeper-exploration-roadmap.md](./deeper-exploration-roadmap.md). Phase 2.

## Why the OLD centerM was conceptually wrong

Pre-WS4, under `eds_consistent=True`, the cloud mass was OVERRIDDEN to the EdS
critical mass, so `center_node_mass` (centerM) only set the softening scale — the
observed chi2 shift with centerM (1→10: 0.85→0.63) was a softening/resolution
ARTIFACT, not added self-gravity. Worse, the naive intent (inject more central
mass) would CRAM more mass into the observable sphere → raise its density →
decelerate MORE → push AWAY from LCDM. Wrong direction.

**Fixed constraint:** the observable universe — the inner sphere compared to
Pantheon — is FIXED. Its mass/density must NOT change. Increasing the simulated mass
must not raise the observable sphere's density.

## The reconception that was implemented

"More simulated mass" means EXTENDING the simulated region BEYOND the observable
sphere: adding particles OUTSIDE the inner observable sphere at the SAME density (the
Big-Bang matter that exists outside our observable universe), keeping the inner region
UNCHANGED, and using ONLY the inner region's a(t) for the Pantheon comparison.

```mermaid
graph TD
    subgraph SIM["Simulated sphere radius R_sim = R_obs * centerM^(1/3)"]
        OUT["outer shell particles<br/>same density + per-particle mass<br/>(mass outside observable universe)"]
        subgraph OBS["Inner observable sub-region radius R_obs (FIXED)"]
            INNER["observable particles<br/>density unchanged<br/>a(t) here -> Pantheon mu(z)"]
        end
    end
    OUT -.gravity + tides.-> INNER
    INNER --> CMP[comparison reads INNER a(t) only]
```

The user's FINAL naming decision was to REUSE `centerM` literally (repurpose
`center_node_mass`), NOT introduce a new param. The CLI flag, CSV column, and cache
slug keep their names; the MEANING changed to the outer-mass multiplier. See the
physics file for the exact semantics (linear N, R_sim cube-root, frozen softening,
v3 cache, slug fix, density ceiling).

**HARD INVARIANT (the central correctness gate):** a(t)/H(z)/μ(z) and the growth
anchor are computed on the OBSERVABLE (inner) sub-region ONLY, never on the enlarged
simulated size. Implemented as a single observable-index mask applied at the a(t)
measurement seam (`_calculate_expansion_history`); the integrator is untouched so
outer particles still exert gravity.

## Density constraint (the "couldn't go that high" note)

The outer region is added at SIMILAR density to the inner (≈ EdS critical). The outer
density cannot be pushed arbitrarily high — that re-introduces over-dense / extra-
deceleration. Implemented as `outer_density_ceiling` (default 1.0, capped at
`MAX_OUTER_DENSITY_CEILING = 2.0` with a warning).

## The hypothesis that was tested (small-M + centerM>1 + small-S)

The user suspected the model might fit best NOT at large node masses (M~855–3000) but
in a small-M (M≈1–2) + centerM>1 + small-S corner. RESULT (PF6): the small-M corner
produced NO anchor_ok rows, and outer matter helped only marginally (best centerM=2.0
chi2/dof 0.6801 vs centerM=1.0 0.6921, both at M=20/S=39). Outer matter alone did not
move the isotropic fit toward LCDM. NOTE the reduced grid omitted the prior M~50/S~20
corner (~0.52), so this is the best of the small-M grid, not the global best — a
fuller sweep is still open. Honest outcome recorded either way.

## Files this workstream touched (implemented)

- `cosmo/particles.py` — outer-shell sampler + observable mask.
- `cosmo/simulation.py` — mask applied in `_calculate_expansion_history`; frozen softening.
- `cosmo/constants.py` — repurposed `center_node_mass` + `outer_density_ceiling` + clip/warn.
- `cosmo/parameter_sweep.py` — v3 cache version, centerM float slug, ceiling slug.
- `sweep.py`, `cosmo/cli.py`, `hubble_diagram_nbody.py` — centerM/ceiling wiring.
- `_generate_ws4_figs.py` + `sweeps/ws4_centerm.json` — sweep + figures.
- Tests: `tests/test_observable_mask.py`, `tests/test_ws4_cache_slug.py`,
  `tests/test_matter_only_consistency.py`.
