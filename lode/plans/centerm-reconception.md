# WS4 — centerM Reconception (extend the sim sphere OUTSIDE the observable region)

Back to [deeper-exploration-roadmap.md](./deeper-exploration-roadmap.md). Phase 2.
Depends on WS1 (sweep) + WS3 (geometry) being in place; figures via
[graphs-from-scripts.md](./graphs-from-scripts.md).

## Why the current centerM is conceptually WRONG (user was emphatic)

Today, under `eds_consistent=True`, the cloud mass is OVERRIDDEN to the EdS critical
mass, so `center_node_mass` (centerM) only sets the softening scale — the observed chi2
shift with centerM (1→10: 0.85→0.63) is a softening/resolution artifact, not added
self-gravity (see [../physics/initial-conditions.md](../physics/initial-conditions.md)
"centerM > 1"). Worse, the naive intent (inject more central mass) would CRAM more mass
into the observable sphere → raise its density → make M_obs bigger / decelerate MORE →
push AWAY from LCDM. That is the wrong direction.

**Fixed constraint:** the observable universe — the inner sphere we compare to Pantheon
— is FIXED by the current equations. Its mass/density must NOT change. Increasing the
simulated mass must not raise the observable sphere's density.

## Correct reconception: a larger sim sphere with an inner observable sub-region

"More simulated mass" should mean EXTENDING THE SIMULATED REGION BEYOND the observable
sphere — adding MORE PARTICLES OUTSIDE the initial observable sphere at SIMILAR density
(the mass from the Big Bang that exists outside our observable universe), while keeping
the inner observable region UNCHANGED, and using ONLY that inner region's a(t) for the
Pantheon comparison.

```mermaid
graph TD
    subgraph SIM["Simulated sphere radius R_sim (large)"]
        OUT["outer shell particles<br/>same density as inner<br/>(mass outside observable universe)"]
        subgraph OBS["Inner observable sub-region radius R_obs (FIXED)"]
            INNER["observable particles<br/>density unchanged<br/>a(t) here -> Pantheon mu(z)"]
        end
    end
    OUT -.gravity + tides.-> INNER
    INNER --> CMP[comparison reads INNER a(t) only]
```

- The outer particles are extra Big-Bang matter at the SAME critical-ish density (not a
  denser core). They add self-gravity / boundary structure felt by the inner region
  WITHOUT changing the inner region's own density.
- The comparison metric (`sim_to_distance_modulus`) reads the INNER observable sub-
  region's a(t) only — NOT the whole sim's RMS radius. This is the key change: today a(t)
  is the global RMS; here it must be computed on the inner sub-region.
- This ties to what the GRF "outside" structure represents (WS5): the clustered field
  beyond the observable patch is exactly this outer matter.

## Density constraint (the "couldn't go that high" note)

The outer region is added at SIMILAR density to the inner observable region (≈ EdS
critical). You cannot push the outer density arbitrarily high — that would re-introduce
the over-dense / extra-deceleration problem this reconception is meant to avoid, and the
inner region's behavior must remain physical. Document and enforce a density ceiling;
record empirically how large R_sim / how many outer particles can be added before the
inner a(t) is disturbed beyond tolerance.

## Implementation concept (to design, not implement here)

- A new param: `sim_sphere_factor` (R_sim / R_obs) or `n_outer_particles` — the sim is
  initialized on a sphere of radius R_sim, with the inner R_obs flagged as observable.
- Initialization seeds outer particles at the same density profile as inner (EdS
  critical), so the inner region is byte-identical to today when sim_sphere_factor=1.
  This is the backward-compat invariant: factor=1 ⇒ current behavior ⇒ M=0==EdS (PF1).
- a(t) / growth / mu(z) and the anisotropy diagnostic all operate on the INNER subset
  (an "observable mask" of particle indices). The comparison kernel takes that subset.
- Likely retire or repurpose `center_node_mass` for the eds_consistent path (it no
  longer means "more observable mass"); keep it only as the softening knob it currently
  is, clearly documented, OR fold it into the new scheme.
- Edge / boundary care: outer particles need enough shell thickness that the inner
  region's tidal environment is well-sampled but the outer EDGE artifacts (a hard
  particle boundary) do not leak into the inner a(t). Use the inner sub-region radius
  comfortably inside R_sim.

## Files this workstream touches

- `cosmo/particles.py` / `ParticleSystem._initialize_particles` — seed outer particles
  at inner density; tag an observable index mask.
- `cosmo/simulation.py` — carry the observable mask; ensure pre-start boost + COM
  removal handle the larger cloud correctly.
- `cosmo/sim_distance.py` / `hubble_diagram_nbody.py` — compute a(t) on the INNER subset.
- `cosmo/anisotropy.py` callers — measure shear/dipole on the inner subset.
- `cosmo/constants.py` — `sim_sphere_factor` / observable-mask params; centerM
  semantics clarified or folded in.
- WS1 sweep — add sim_sphere_factor as an axis (modest values; density-capped).
- Tests: factor=1 reproduces current inner a(t) exactly; M=0==EdS preserved; inner
  density unchanged as outer particles are added; density-ceiling enforcement.

## Deliverables

- A sim with a larger outer region + an inner observable sub-region whose a(t) drives
  the Pantheon comparison.
- A figure: inner-region a(t) / chi2 vs sim_sphere_factor, showing the inner region is
  undisturbed (factor=1 baseline) and where the density ceiling bites.
- An honest answer: does adding realistic outside matter change the inner observable
  expansion, and in which direction?
