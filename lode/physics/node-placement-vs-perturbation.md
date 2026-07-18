# Node placement vs perturbation — why the virialized grid exists

STATUS: **VERIFIED FINDING**. Explains the physics motivation for the
`virialized` node geometry and records the HONEST section-1 result that the
per-node mass/position PERTURBATION machinery was never buggy — the issue is
node PLACEMENT, not perturbation. Geometry implementation lives in
[../plans/node-geometries.md](../plans/node-geometries.md) (WS3, "Virialized geometry").

## The 1/r³ near-field rule

Tidal stretch from a point node falls off as ~1/r³ (the differential of the
~1/r² acceleration across the cloud). So the NEAREST nodes dominate utterly and
far nodes are effectively inert. The observable cloud sits at the centre; the
characteristic horizon scale is ~14 Gpc.

## Multi-layer lattices park most nodes far beyond the horizon

The dense lattices (`fcc`, `bcc`) are enumerated by unit-cell layers, so at a
given spacing they place MOST of their nodes 4–6× beyond ~14 Gpc, where 1/r³
makes them contribute almost nothing:

| geometry | inner reach | outer reach (verified) |
|----------|-------------|------------------------|
| `cube26` | clean single 26-node shell at ~S | ~√3·S |
| `fcc` (n_shells=2) | ~S | ~86 Gpc out (most nodes far) |
| `bcc` (n_shells=2) | ~S | ~386 nodes out to ~83 Gpc |

Consequence: **a single number "S" is NOT comparable across geometries.** For a
multi-layer lattice the realized characteristic spacing and the effective near-field
are dominated by the few nearest nodes, not by the "S" that parametrized the lattice.
`cube26` is the clean single-shell case where S means what it says. This is why the
fair cross-geometry control is "per-node mass fixed + nearest node at S"
(`normalize_nearest`), NOT equal total mass (see node-geometries.md), and it is why
the cross-geometry sweep found no lattice beats the cube (WS3 result; PF2/PF3).

```mermaid
graph LR
    C[cube26: 26 nodes, one shell ~S] -->|S meaningful| OK[near-field well-defined]
    F[fcc/bcc: most nodes 4-6x beyond ~14 Gpc] -->|1/r^3 inert| FAR[far nodes contribute ~0]
    FAR --> NC["S NOT comparable across geometries"]
```

## Why the virialized grid

A physically-motivated alternative to "add more lattice layers": a RELAXED-CLUSTER
configuration where node mass and radius are COUPLED (mass segregation) and the node
count + radial RANGE are EXACT free parameters (`vir_n_nodes`, `vir_extent`) instead
of being dictated by lattice enumeration. The realized spacing is pinned to a target S
via `nearest_neighbour_spacing` so S is well-defined per a chosen metric
(median/mean). Smaller M can then pair with smaller S and a bigger radial multiplier.
This lets us probe whether a segregated, controlled-extent structure changes the
near-field / growth story that the far-parked lattice nodes cannot. Implementation +
parameters: [../plans/node-geometries.md](../plans/node-geometries.md).

## The virialization criterion — CENTER-ONLY metric, Option A (lattice) vs Option B (gradient)

The user's physical test: *"make a LARGE structure and check the forces only in the nodes
near the CENTER; the DEEP-INTERIOR nodes of a truly virialized grid should not move."*
The OUTER nodes obviously feel a net inward pull (they are at the boundary), so the metric
must isolate the deep interior of a BIG grid — not "inner 50% of a small grid." Made
measurable by two pure helpers in `cosmo/node_geometry.py`:

- `node_net_accelerations(positions, masses, *, center_mass_kg, G)` — the net
  gravitational accel on each node from all OTHER nodes + a central node of mass
  `center_mass_kg` (= centerM, 1 by default). Mirrors the sim's tidal law incl. the
  1e10 m floor.
- `virialization_residual(...)` → a DIMENSIONLESS per-node residual `|net_accel| / a_ref`
  (a_ref = one characteristic neighbour pull, `reference="mean_pairwise"`). A node is
  "virialized" when `max_residual <= VIRIALIZATION_TOL = 0.25`. The CENTER-ONLY selector
  is `center_k` (the K nodes closest to the centroid — SIZE-INDEPENDENT, so a bigger grid
  genuinely deepens the interior) or `center_frac`; the legacy `inner_frac=0.5` default is
  byte-identical. "0.25" is a dimensionless ratio (net force / one neighbour pull), NOT 25%
  of total force.

**Empirical result (center-only, deep interior; S=30 Gpc; both mass rules; n=26/100/500):**

| realization | center max residual | virialized? |
|-------------|---------------------|-------------|
| Option A (analytic lattice, `vir_relax_mode="lattice"`, DEFAULT) | ~1e-28..1e-31 at every size | YES |
| realistic un-relaxed (Fibonacci segregated) | 25..275, GROWS with grid size | no |
| Option B (TRUE relaxation, `vir_relax_mode="gradient"`) | radial n=500 275→**95**; massfunc n=500 96→**32** | no (bottoms ~O(100×) above TOL) |

**The physics finding (re-grounded, empirical — the old "irreducible monopole" hand-wave
is DROPPED):** Option B is a genuine iterative relaxation — it descends the force-residual
objective f=Σ|a_i|² (analytic gradient validated vs finite diff ~3e-6, monotone
backtracking; descending the POTENTIAL instead collapses the blob, which is why a naive
"+a" step diverges). It REDUCES the residual monotonically but does NOT reach center
force-balance: a realistic blob relaxed by position descent bottoms out at ~95/32 vs the
lattice's ~1e-29. So the analytic lattice is the ONE perfectly-balanced realization, and
the HMEA nodes are STATIC boundary conditions (a frozen virialized meta-structure), so the
symmetric lattice is the physically correct realization of "virialized". `vir_relax_mode`
selects A (lattice) vs B (gradient); in mode B `vir_relax_steps` is the gradient step
COUNT (a balance LEVEL in mode A).

```mermaid
graph TD
    REAL[realistic segregated blob] -->|center residual 25..275, grows with size| NOBAL[not virialized]
    GRAD[Option B: gradient relaxation<br/>descend f=sum a_i squared] -->|monotone descent, bottoms 95/32| PART[reduced but >> TOL]
    LAT[Option A: symmetric cubic-lattice ball<br/>node at origin, masses by shell] -->|opposing pulls cancel| BAL[center residual ~1e-29 << TOL=0.25]
    BC[HMEA nodes = STATIC boundary conditions] --> LAT
```

## HONEST section-1 finding: the perturbation machinery is CORRECT

When generalizing the unit tests from cube26-only to ALL geometries
(`tests/test_node_geometry_anisotropy.py`, 128 tests over
cube26/cube_dense/fcc/bcc with each geometry's actual N), the per-node
PERTURBATION knobs were verified correct on EVERY geometry:

- `node_mass_amplitude` (mass perturbation): mean-preserving for ANY N
  (mean == M_ext_kg, rtol 1e-12), strictly positive, genuinely spread,
  seeded-deterministic, independent of global np.random.
- `node_s_amplitude` (radial position perturbation): per-node UNIT direction
  unchanged (ray preserved generically, not just for cube26's axis-aligned rays),
  mean radial scale preserved (mean(r_pert/r_unpert) == 1.0), no node crosses the
  origin, seeded-deterministic.
- The two knobs are SEPARATE `default_rng` draws (cross-knob independence holds
  per geometry).

**There is NO multi-layer perturbation bug.** The amplitude machinery behaves
identically and correctly on cube_dense/fcc/bcc as on cube26. What differs between
geometries is the unperturbed PLACEMENT (where the nodes sit, and hence the
near-field that 1/r³ weights), not how the perturbation acts on them. The
discriminating lever remains anisotropy (PF2), and no volume-filling lattice beats
the cube on the isotropic fit (PF3 / node-geometries.md).

## References

- [../plans/node-geometries.md](../plans/node-geometries.md) — WS3 factory + virialized geometry + vir_relax_mode (lattice/gradient) (IMPLEMENTED)
- [../plans/pinned-findings.md](../plans/pinned-findings.md) — PF1 (M=0==EdS), PF2 (anisotropy), PF3 (M/S³), PF8 (center-only: only the lattice balances, Option B does not)
- [force-calculations.md](./force-calculations.md) — tidal 1/r³, per-node mass/position knobs
- [slingshot-and-softening.md](./slingshot-and-softening.md) — node close-pass slingshot + node_softening_gpc taming
- [observable-mask-and-outer-mass.md](./observable-mask-and-outer-mass.md) — WS4 outer-mass (related "horizon" framing)
- Tests: `tests/test_virialization_validation.py` (the metric + center-only criterion), `tests/test_virialized_grid.py` (balance level + Option B gradient), `tests/test_virialization_figs.py` (residual-vs-gridsize + Option-B descent figures)
