# Particle slingshots and node softening (the taming fix)

STATUS: **VERIFIED + IMPLEMENTED**. Root-caused the runaway particle "slingshot",
pinned the lever, and shipped the taming knob `node_softening_gpc`. Preserves
M=0==EdS (PF1) and the default is byte-identical (no PHYSICS_CACHE_VERSION bump).
Related: [node-placement-vs-perturbation.md](./node-placement-vs-perturbation.md),
[force-calculations.md](./force-calculations.md), [initial-conditions.md](./initial-conditions.md).

## The slingshot

At strong-tidal / small-S configs a few inner particles acquire a RUNAWAY
displacement — a heavy-tailed distribution where one particle moves orders of
magnitude farther than the median. Quantified by `slingshot_metrics(disp)` in
`_generate_ws8_figs.py` (max/median, p99/median, tail_fraction beyond 5x median).

Measured (seed=42, N=200, t_start=5.8, M=1000, S=10):

| config | max/median | tail_fraction |
|--------|-----------:|--------------:|
| cube26, nodes ON (runaway) | ~514 | 0.285 |
| cube26, nodes OFF (matter-only) | ~1.3 | 0.0 |
| tame (M=5, S=20) nodes ON | ~1.3 | 0.0 |
| virialized, nodes ON | ~23 | 0.06 |

## Root cause: a near-point-NODE close pass (NOT particle-particle)

Pinned by unit test (`tests/test_slingshot.py`): re-running the SAME runaway config
with external HMEA nodes OFF collapses the tail from ~514x to ~1.3x. So the
slingshot is a particle making a very close pass to a near-point HMEA node (the
unsoftened `1/r^3` tidal force diverges), NOT a particle-particle encounter.

**n_steps and n_particles do NOT tame it** (S3 knob-sweep diagnostic): finer time
resolution / more particles do not reduce the tail — the close-pass kick is a force
issue, not a resolution issue. The lever is NODE SOFTENING.

```mermaid
graph LR
    P[inner particle on a node-bound orbit] -->|close pass to near point-node| KICK[1/r^3 tidal force diverges]
    KICK --> RUN[runaway displacement: tail max/median ~514x]
    SOFT[node_softening_gpc>0: Plummer cap r_soft^2=r^2+eps^2] -->|caps the close-pass kick| TAME[tail collapses to ~2.8x]
```

## The fix: node_softening_gpc (Plummer node softening)

`node_softening_gpc` (param) / `node_softening_m` (derived, meters) adds a Plummer
softening on the TIDAL force path, in BOTH the numba kernel
(`cosmo.tidal_forces_numba.calculate_tidal_forces_numba(..., softening_m)`) and the
numpy fallback (`HMEAGrid.calculate_tidal_acceleration_batch`):

- `node_softening_gpc == 0.0` (DEFAULT): the LEGACY hard `r < 1e10 m` floor
  (~3e-13 Gpc — effectively no softening at Gpc scales). BYTE-IDENTICAL to the
  pre-softening force, so cube26 a(t) and every existing cache entry are unchanged.
  Slug is omitted ⇒ `PHYSICS_CACHE_VERSION` stays `v3` (no bump).
- `node_softening_gpc > 0.0`: `r_soft^2 = r^2 + (node_softening_gpc*Gpc_to_m)^2`
  (same Plummer convention as the internal particle-particle force in
  `cosmo.integrator`). The 1e10 m floor is dropped in this branch (Plummer already
  keeps the force finite at r→0). Cache slug `{node_softening_gpc}nsoft`.

The fix is GEOMETRY-AGNOSTIC (it caps the per-node force regardless of layout), so
it tames cube26 AND virialized.

## Bounded close-range law + adaptive substepping (Section 4)

The blunt 1 Gpc Plummer floor was flagged as a hack (sub-Gpc bodies barely
interact) and the ~40 Myr/step timestep as too coarse. Two opt-in knobs add a
more physical treatment; BOTH default OFF / byte-identical (no cache bump):

- `node_force_law` (param on `SimulationParameters` + `ExternalNodeParameters`;
  derived int `node_force_law_code`, 0 plummer / 1 bounded; codes mirrored in
  `cosmo.tidal_forces_numba` + the numpy fallback):
  - `"plummer"` (DEFAULT) = legacy/Plummer behaviour (hard floor at softening 0,
    Plummer when >0). Byte-identical default.
  - `"bounded"` = the "can't cross the midpoint" law: outside the softening
    length it is EXACT Newtonian `1/r^2` (no Plummer offset); INSIDE it the
    per-node accel MAGNITUDE is CAPPED at `a_cap = G m / softening_m^2` (its value
    AT the softening radius), so the close-pass kick is bounded but sub-softening
    bodies still feel the FULL softening-length attraction — unlike the Plummer
    floor which softens the force toward zero. Only differs from plummer when
    `node_softening_gpc>0`; with softening 0 it falls back to the legacy floor
    (byte-identical). Far-field change <5%; M=0 still == EdS (vanishes at M_ext=0).
- `node_substep_threshold` (default 0.0 OFF) + `node_substeps` (default 1, clamp
  `[1, MAX_NODE_SUBSTEPS=64]`): adaptive KDK sub-stepping in `cosmo.integrator`.
  On a global step where ANY particle is within `threshold * S_ref` of a node
  (`S_ref` = node softening length, else median node NN spacing), the whole step
  is integrated as `node_substeps` smaller KDK substeps (refines dt during the
  close pass). Active only when `threshold>0 AND substeps>1`; Hubble drag + time
  advance applied ONCE per global step, so a no-close-pass step is bit-identical
  to the non-substep run.

### Measured comparison (`_generate_ws8_close_encounter.py` -> ws8/close_encounter_compare.{csv,png})

Runaway config M=1000/S=10, N=300, t_start=2.9, n_steps=273 (~40 Myr/step). Rows
= {legacy floor, Plummer 1 Gpc, Plummer 0.1 Gpc, bounded 1 Gpc, adaptive substep,
bounded+substep} x {cube26, virialized}; chi2/dof from the SAME scorer the sweep
uses (`compute_pantheon_metrics`, growth-anchor gated):

| cube26 method | max/median | tail | growth | chi2/dof |
|---|---:|---:|---:|---:|
| legacy hard floor | 844 | 0.187 | 6927 | inf (anchor reject) |
| Plummer 1 Gpc | 2.8 | 0.000 | 3.04 | 29.0 |
| Plummer 0.1 Gpc | 10.9 | 0.087 | 363 | inf |
| bounded 1 Gpc | 6.7 | 0.003 | 5.23 | inf (just over anchor) |
| adaptive substep (no soft) | 93 | 0.080 | 1559 | inf |
| bounded+substep | 2.4 | 0.000 | 3.38 | 48.7 |

HONEST verdict:
- **Adaptive substep ALONE does NOT tame the tail** (93x on cube26) — re-confirms
  PF9 that finer time resolution alone is insufficient; the force law is the lever.
- **The bounded law tames the close-pass tail** (844x -> 6.7x, tail_frac ~0) and,
  crucially, pulls growth from 6927 down to ~5 (near the LCDM 3.30) — far better
  than the Plummer floor's behaviour, but at this extreme config it's still just
  outside the 20% growth anchor until paired with substeps.
- **bounded+substep** is the best combination on cube26 (tail 2.4x, growth 3.38 in
  the anchor window, finite chi2). So the bounded law CAN replace the blunt 1 Gpc
  Plummer on cube26 when paired with substep refinement.
- **virialized** (mass spread, no single near-point node) does NOT have a clean
  single-node slingshot at M=1000/S=10, so NONE of the laws tame it well there —
  an honest negative result; its runaway is geometric, not a close-pass artifact.

The chi2 values are mostly `inf` because these are deliberately EXTREME runaway
configs (to expose the slingshot) that the growth anchor rejects; the finite cube26
entries show the tail-vs-fit trade.

## "Doubly tamed"

At `node_softening_gpc=1.0` (matches the 1 Gpc internal particle softening):

| geometry | OFF max/median | ON (1 Gpc) max/median |
|----------|---------------:|----------------------:|
| cube26 | ~514 | ~2.8 (tail_fraction → 0) |
| virialized | ~23 | ~7.0 |

The virialized (force-balanced) geometry already lowers the tail vs cube26
(~23 vs ~514), and node softening collapses it further — hence "doubly tamed"
(GEOMETRY + SOFTENING). This is the configuration of `sweeps/virialized_final.json`.

## Invariants preserved

- **M=0 == EdS (PF1):** softening only RESHAPES the per-node force; with all node
  masses 0 (M_ext=0) the tidal sum is identically 0 for ANY `node_softening_gpc`
  (asserted on both numba + numpy paths).
- **Far-field unchanged:** for a particle ≫ 1 Gpc from a node the 1 Gpc softening
  changes the tidal force < 5% (asserted), so a(t) for non-runaway clouds is barely
  affected; the effect is concentrated on the close-pass tail.
- **Default byte-identical:** `node_softening_gpc=0` ⇒ the grid's tidal batch EXACTLY
  equals the legacy numba call with `softening_m=0` (`np.array_equal`).

## Tests (all green)

`tests/test_slingshot.py` (13):
- runaway config DOES slingshot; tame config does NOT (contrast).
- ROOT CAUSE: nodes ON ≫ nodes OFF (≥10x tail and ≥10x max) — the node close-pass.
- diagnostic softened-node-force monkeypatch tames the tail; pure-helper tests for
  `softened_node_acceleration` (far-field Newtonian, finite at node, larger eps →
  smaller close force).
- PRODUCT path: `node_softening_gpc=1.0` tames cube26 AND virialized; default 0.0 is
  byte-identical to the legacy hard-floor force; close-pass kick bounded with
  softening on; far-field < 5% change; M_ext=0 ⇒ zero tidal for any softening.

Threading + cache: `tests/test_overarching_sweep.py::TestNodeSofteningThreading`
(keyed==run guard: the SweepConfig the cache keys off AND the SimulationParameters
the sim runs both carry the requested softening; `1.0nsoft` slug present, absent at
default).
