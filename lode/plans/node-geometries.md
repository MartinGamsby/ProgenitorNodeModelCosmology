# WS3 — Alternative Node Geometries

Back to [deeper-exploration-roadmap.md](./deeper-exploration-roadmap.md). Phase 1.
Feeds the sweep ([overarching-sweep.md](./overarching-sweep.md)) and the anisotropy
diagnostic; figures via [graphs-from-scripts.md](./graphs-from-scripts.md).

## Motivation

The current external structure is a 26-node 3×3×3−1 cubic lattice (`HMEAGrid._create_
grid` in `cosmo/particles.py`): perfectly symmetric, so the tidal forces CANCEL at the
origin (traceless). PF2 says this traceless cancellation is exactly why symmetry-
breaking can't move the isotropic fit. The question: does a DIFFERENT geometry — more
nodes or a denser lattice — yield a larger NET effect while staying honest about the
vacuum-traceless constraint?

**PHYSICS REQUIREMENT — geometries must be VOLUME-FILLING (virialized).** The HMEA nodes
represent a *virialized meta-structure*: relaxed mass distributed throughout a 3D volume.
The cube lattice is the simplest such approximation. **Hollow spherical SHELLS are
explicitly excluded** — concentrating all mass on a sphere surface with an empty interior
is the OPPOSITE of a virialized structure (and, by Birkhoff, a continuous shell exerts no
interior force at all; see [../physics/theoretical-framework.md](../physics/theoretical-framework.md)).
Valid geometries: `cube26` (default), `cube_dense`, `fcc`, `bcc`, `virialized` —
all volume-filling. `virialized` is a NEW coupled (positions, masses) mass-segregated
generator (see "Virialized geometry" section below).

**Honesty constraint up front:** for masses in vacuum the tidal tensor is traceless for
ANY arrangement, so no geometry will magically produce a large net isotropic
acceleration at the origin. The realistic gains are (a) a DIFFERENT shear/dipole pattern
(the PF2 signal), and (b) a different 2nd-order growth coupling at finite cloud size /
strong field. Judge geometries on those, not on a hoped-for isotropic-chi2 breakthrough.

## Plan: a node-geometry abstraction / factory

Replace the hard-coded `_create_grid` cube loop with a geometry FACTORY that returns
base node positions (before the existing per-node mass + radial-scale perturbations are
applied). HMEAGrid keeps applying `node_masses()` and `node_scale_factors()` on top, so
the symmetry-breaking knobs compose with ANY geometry.

```
def build_node_positions(geometry: str, S: float, **kwargs) -> np.ndarray:
    # returns (n_nodes, 3) base positions, all at characteristic scale ~S
```

### Geometries to support

All VOLUME-FILLING (virialized). Hollow shells are excluded (see Motivation).

| id | Description | Key kwargs |
|----|-------------|-----------|
| `cube26` | current 3×3×3−1 cubic lattice (DEFAULT, backward-compatible) | — |
| `cube_dense` | denser cubic lattice, e.g. 5×5×5−1 (124 nodes) | `n_per_side` |
| `fcc` / `bcc` | denser close-packed lattices | `n_shells` |

### Parametrization that threads cleanly

- `ExternalNodeParameters` / `SimulationParameters` gain a `node_geometry: str` (default
  `"cube26"`) + `geometry_kwargs: dict`. n_nodes becomes geometry-derived, not the fixed
  26, so `node_masses(n)` / `node_scale_factors(n)` already take `n` and still apply.
- `HMEAGrid._create_grid` calls `build_node_positions(...)` for base positions, then
  applies scale factors + masses exactly as now (order preserved so position/scale/mass
  vectors stay aligned).
- Numba tidal force path (`calculate_tidal_forces_numba`) already takes arbitrary
  `node_positions` / `node_masses` arrays — geometry is transparent to it.
- WS1 sweep adds geometry as an axis; cache key gains a geometry slug (see
  [overarching-sweep.md](./overarching-sweep.md)).
- Anisotropy diagnostic (`anisotropy_report.py`) runs per geometry so F7/F8/F12 show how
  shear/dipole differ by geometry.

```mermaid
graph TD
    GEO[node_geometry + geometry_kwargs] --> FAC[build_node_positions]
    FAC --> BASE[base node positions n x 3]
    BASE --> SCALE[apply node_scale_factors mean=1]
    SCALE --> POS[node positions]
    NM[node_masses mean-preserving] --> GRID[HMEAGrid.nodes]
    POS --> GRID
    GRID --> TID[tidal force numba path]
```

## Invariants to preserve (per PF1/PF2)

- M=0 still == EdS for EVERY geometry (the geometry only sets node positions; at M=0 the
  tidal sum is zero regardless).
- Mean-preserving knobs stay mean-preserving for any n_nodes.
- A symmetric geometry (e.g. cube26) must keep shear ≈ noise at amplitude=0
  (sanity check: the geometry alone, with uniform masses, should not create spurious
  anisotropy beyond discreteness).
- Fair-comparison normalization (DECIDED — see Mass bookkeeping below): hold PER-NODE mass
  fixed AND put each geometry's NEAREST node at the same S (`normalize_nearest`). Equal-
  TOTAL-mass rescaling is REJECTED: tidal stretch ~1/d³ is near-field dominated, so far
  nodes barely matter and dividing per-node mass by n/26 would wrongly weaken the near
  nodes that do the work.

## Implementation status (COMPLETE as of WS3 coding pass)

### Files added / modified
- **NEW `cosmo/node_geometry.py`** — `build_node_positions(geometry, S, **kwargs)` factory
  + `list_geometries()` + `effective_M_ext_kg()` helper. Registry-based; four
  volume-filling geometries (hollow `shell`/`shell_multi` were removed — not virialized).
- **`cosmo/particles.py`** — `HMEAGrid._create_grid` replaced hard-coded cube loop with
  `build_node_positions` call; `n_nodes` set from actual geometry count.
- **`cosmo/constants.py`** — `node_geometry: str = "cube26"` + `geometry_kwargs: dict = {}`
  added to both `ExternalNodeParameters` and `SimulationParameters`; threaded into
  `external_params` in `_calculate_derived`.
- **`cosmo/parameter_sweep.py`** — `SweepConfig` gains `node_geometry` / `geometry_kwargs`;
  `build_cache_name` appends `{geo}geo` slug only when geometry != `"cube26"`.
- **`cosmo/cli.py`** — `--node-geometry` argument + threaded in `args_to_sim_params`.
- **NEW `tests/test_node_geometry.py`** — node-count, byte-identical-cube26, volume-filling,
  mass-bookkeeping, cache-slug, threading, and shells-excluded tests; all pass.
- **`visualize_geometries.py`** — 3D scatter of all geometries →
  `results/figures/ws3/node_geometries.png`.

### Geometries implemented (all volume-filling / virialized)
| id | nodes | kwargs |
|----|-------|--------|
| `cube26` | 26 | — (DEFAULT, byte-identical to old loop) |
| `cube_dense` | (n³-1), default n=5 → 124 | `n_per_side` (odd ≥3) |
| `fcc` | ~86 (n_shells=2) | `n_shells` |
| `bcc` | ~386 (n_shells=2) | `n_shells` |
| `virialized` | exact `vir_n_nodes` (default 26) | `vir_*` (coupled positions+masses, mass-segregated — see Virialized section) |

`shell` / `shell_multi` were implemented then REMOVED: a hollow sphere of nodes is the
opposite of a virialized meta-structure (and Birkhoff → no interior force). Geometries
must fill the volume.

## Virialized geometry — COUPLED (positions, masses), mass-segregated (IMPLEMENTED)

`node_geometry="virialized"` is fundamentally different from the four lattices above:
it is the ONE geometry that returns POSITIONS AND MASSES TOGETHER, already paired
(mass i ↔ radius i), so it models a relaxed cluster where **more massive nodes sit
FURTHER from the centre** (mass segregation) and small nodes cluster near it. Every
other geometry gets positions from `build_node_positions` and masses INDEPENDENTLY from
`node_masses(n)`; virialized cannot use that path (it would lose the coupling), so:

- **`cosmo/node_geometry.py::build_virialized_grid(S, *, n_nodes, M_ext_kg, vir_extent,
  vir_mass_rule, vir_mass_spread, vir_segregation, vir_s_metric, seed) -> (positions
  (N,3), masses (N,))`** is the entry point. `build_node_positions("virialized", ...)`
  RAISES a `ValueError` pointing callers here (positions-only would be un-coupled).
- **`HMEAGrid._create_grid` branches** on `geometry == "virialized"`: it calls
  `params.build_virialized()` → uses the returned masses DIRECTLY (`node_masses()` /
  `node_mass_amplitude` are IGNORED on this branch; `vir_mass_spread` owns the mass
  distribution). `node_scale_factors()` (the `node_s_amplitude` radial jitter) STILL
  composes on top, mean-preserving, exactly as for every geometry.
- `list_geometries()` includes `"virialized"` (a sentinel builder registers the name).

### Parameters (on ExternalNodeParameters + SimulationParameters + SweepConfig + CLI)

| param | meaning | default |
|-------|---------|---------|
| `vir_n_nodes` | exact node count (NOT derived from lattice radius bounds) | 26 |
| `vir_extent` | continuous radial-RANGE multiplier: radii span `[0.5S, (0.5+extent)·S]` before the NN rescale, so range ratio = `1+2·extent` (extent=1→3, extent=2→5) | 1.0 |
| `vir_mass_rule` | `"radial"` (mass ~ f(r)) or `"massfunc"` (log-normal draw + segregate by rank) | `"radial"` |
| `vir_mass_spread` | amplitude of the node-mass distribution. **THE falsifiable knob**: 0 → uniform masses | 0.0 |
| `vir_segregation` | mass↔radius coupling strength [0,1]; 0 → decoupled | 1.0 |
| `vir_s_metric` | `"median"` or `"mean"` — which NN-spacing statistic the layout targets as S | `"median"` |
| seed | `np.random.default_rng(node_mass_seed)` (one-seed coherence, like node_s_amplitude) | node_mass_seed |

### nearest_neighbour_spacing — the S definition

`nearest_neighbour_spacing(positions, metric)` (in `node_geometry.py`) = for each node
the distance to its CLOSEST other node, reduced by `metric` ("median"/"mean"). The
generator builds the raw radial profile then applies a SINGLE global factor so the
realized `nearest_neighbour_spacing(positions, vir_s_metric) == S` exactly. So `S` is the
TARGET characteristic spacing per the chosen metric (a pure global factor would cancel —
which is WHY `vir_extent` widens the RANGE rather than scaling it).

### Mean-preservation + falsifiable reductions (the contracts)

- **Mean-preserving:** raw masses are normalized `masses *= M_ext_kg / masses.mean()` so
  `mean(masses) == M_ext_kg` exactly (rtol 1e-12) → total = `N·M_ext_kg`, Ω_Λ_eff
  comparable (same contract as `node_masses`). `effective_M_ext_kg` is available for
  26·M_ref total parity but not forced.
- **Falsifiable:** `vir_mass_spread == 0` → all masses == M_ext_kg (uniform, both rules);
  `vir_segregation == 0` → mass/radius DECOUPLED (~0 correlation). With both 0 the grid is
  a clean "uniform masses, isotropic-ish positions" null. With spread>0 + segregation>0
  the mass-radius correlation is strongly POSITIVE (bigger mass → larger radius).
- **Directions** are a deterministic Fibonacci sphere (volume-filling, ≥2 distinct radii —
  never a hollow shell).
- **RNG isolation / determinism:** uses `default_rng(seed)` only; independent of global
  np.random and of the particle realization. Same (seed, params) → identical positions AND
  masses; different seed differs.

### M=0 == EdS (PF1) for virialized

Holds: geometry only sets node positions/masses, and at `M_ext_kg=0` all node masses are 0
(mean-preserving of 0), so the tidal sum is identically zero → pure-matter EdS. Asserted at
the force-path level in `tests/test_virialized_grid.py` (zero tidal acceleration on a test
cloud, both numba + numpy paths, both rules).

### Cache slug (NO PHYSICS_CACHE_VERSION bump)

`build_cache_name` appends `virializedgeo` (via the existing `!= "cube26"` branch) PLUS
virialized-only sub-slugs (`{vir_n_nodes}vn`, `{vir_extent}vx`, `{rule}vr`,
`{spread}vsp`, `{seg}vsg`, `{metric}vsm`). These are appended ONLY for virialized, so every
existing cube26/cube_dense/fcc/bcc key is UNTOUCHED and virialized lives at a brand-new key
→ `PHYSICS_CACHE_VERSION` stays `v3`.

### Tests (all green)

- `tests/test_virialized_grid.py` (70) — generator shapes/dtype/count, mean-preservation
  (both rules, several N), falsifiable reductions, positive segregation correlation,
  vir_extent range scaling, median/mean NN metric, determinism + global-RNG isolation,
  volume-filling, positions-only raises, NN-helper sanity, HMEAGrid coupled-branch
  threading (count, mean-preserving masses, masses-not-from-node_masses, node_s composition,
  M=0 zero-tidal EdS invariant), cube26 byte-identical opt-in, SimulationParameters /
  SweepConfig threading, cache-slug distinctness + non-virialized-key regression.
- `tests/test_node_geometry_anisotropy.py` (128) — generalizes node POSITIONS + node MASSES
  + node_mass_amplitude + node_s_amplitude invariants (mean-preservation for ANY N, ray
  preservation, seeded determinism, separate-RNG-draw cross-knob independence) across
  cube26/cube_dense/fcc/bcc with each geometry's ACTUAL N, tightly checking the
  AFTER-amplitude node state (the previous tests were cube26-only).
- WS8 figures: `_generate_ws8_figs.py` + `tests/test_ws8_figs.py` (18) — see
  [../scripts/visualization.md](../scripts/visualization.md#ws8--virialized-grid-figures-_generate_ws8_figspy).

### Fair-comparison normalization (DECIDED: per-node mass fixed + nearest node at S)
`M_ext_kg` is the per-node mass; it is held FIXED across geometries (NOT rescaled to equal
total mass). To match the dominant near-field, `build_node_positions(..., normalize_nearest=
True)` rescales each geometry so its NEAREST node sits at S (cube26 is already there → no-op;
cube_dense ×2, fcc rescaled, bcc already at S). Pass it via `geometry_kwargs={"normalize_
nearest": True}` (threads through HMEAGrid + the sweep). Rationale: tidal stretch ~1/d³ is
near-field dominated, so the right control is "same per-node mass + same nearest distance",
NOT same total mass. `effective_M_ext_kg()` still exists but is NOT the chosen normalization
(kept only for an equal-total-mass view if ever wanted).

### Cross-geometry result (HONEST — sweeps/geometry_compare.json, 400p, isotropic, normalize_nearest)
**No geometry beats the cube.** F12 figure `results/figures/ws3/geometry_comparison_chi2.png`:
- `cube26` ≈ `cube_dense` everywhere (the denser cube's extra nodes barely change the fit).
- `fcc` is consistently the WORST (highest chi2); `bcc` between cube and fcc.
- Geometry only matters in the STRONG-field regime (small S): at M=1500/S=30 cube reaches
  chi2/dof≈0.52 (nearest LCDM 0.436) vs fcc 0.68, bcc 0.57; at S=40/50 all converge to
  ~0.66–0.69 (geom range <0.03) — far nodes don't matter, confirming the near-field argument.
- Growth factor is nearly geometry-independent (2.835–2.880); M/S set it, not geometry.
Conclusion: the cube (simplest virialized approximation) is also the best-fitting; alternative
volume-filling lattices give no isotropic improvement. The anisotropy (shear/dipole) per
geometry is a separate, still-open question (F7/F8 per geometry).

### Remaining
- F12 isotropic geometry comparison: DONE (above). Open: shear/dipole PER geometry (the PF2
  signal) — run anisotropy_report.py per geometry to see if any geometry changes the
  anisotropic signature even though it doesn't change the isotropic fit.
- Honest verdict on geometry vs cube26 on shear/dipole signal.

## Deliverables

- A geometry factory + registry, threaded through HMEAGrid + sweep + anisotropy.
- F12 geometry-comparison figures (net effect + shear per geometry).
- An honest verdict: does any geometry beat cube26 on the anisotropy signal / net
  effect, given the vacuum-traceless constraint?
