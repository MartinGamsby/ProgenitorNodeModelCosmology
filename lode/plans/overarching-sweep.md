# WS1 — Overarching Multi-Parameter Sweep Tool

Back to [deeper-exploration-roadmap.md](./deeper-exploration-roadmap.md).
Phase 1. Depends on the geometry factory ([node-geometries.md](./node-geometries.md))
for the geometry axis; emits figures via [graphs-from-scripts.md](./graphs-from-scripts.md).

## Status: IMPLEMENTED

`sweep.py` is the single config-driven entrypoint. `pantheon_knob_sweep.py` is
retained as a legacy script for the 2-knob era; the new tool supersedes it.

## Architecture

```mermaid
graph TD
    CFG[JSON config or defaults] --> EXP[expand_grid<br/>collapse amp=0 to 1 seed]
    EXP --> LOOP[for each cell]
    LOOP --> COFIT{S = co-fit?}
    COFIT -->|yes| LIN[linear_search_S on pantheon objective<br/>warm-start per group]
    COFIT -->|no| FIXED[for each listed S]
    LIN --> WORST[worst_callback + cache]
    FIXED --> WORST
    WORST --> ANCHOR[growth anchor check]
    ANCHOR --> CSV[ws1_sweep_<tag>.csv + sweep_results_pantheon_<tag>.csv]
    CSV --> FIG[cosmo/plots.py: F1+F2+F3+F4+F6+F9]
```

## Parameters searched (all axes)

| Axis | Config key | Notes |
|------|-----------|-------|
| M_value | `M_values` | list of mass factors |
| S_value (Gpc) | `S_values` | `"co-fit"` => per-M linear search; or list of ints |
| node_mass_amplitude | `node_mass_amplitudes` | amp=0 collapse: single run per M |
| node_s_amplitude | `node_s_amplitudes` | per-node position anisotropy |
| node_mass_seed | `node_mass_seeds` | orientation seed |
| init_distribution | `init_distributions` | `"uniform_sphere"` or `"grf"` |
| particle count | `particle_count` | pinned via `_FixedSweepConfig` |
| node_geometry | `node_geometries` | `"cube26"` default; others add slug to cache key |

## CLI

```bash
python sweep.py                                     # built-in defaults
python sweep.py --config sweeps/coarse.json         # JSON config override
python sweep.py --probe-only                        # time 5 sims then exit
python sweep.py --plots-only results/ws1_sweep.csv  # regenerate figures, no sims
python sweep.py --tag my_run                        # custom CSV/figure prefix
```

## Config shape (JSON, all keys optional)

```json
{
  "M_values":            [100, 500, 1000, 5000],
  "S_values":            "co-fit",
  "s_min_gpc":           20,
  "s_max_gpc":           80,
  "node_mass_amplitudes":[0.0, 0.5],
  "node_s_amplitudes":   [0.0],
  "node_mass_seeds":     [42],
  "init_distributions":  ["uniform_sphere"],
  "node_geometries":     ["cube26"],
  "geometry_kwargs":     {},
  "s_cofit_method":      "linear",
  "particle_count":      400,
  "n_steps":             273,
  "t_start_Gyr":         2.9,
  "centerM":             1,
  "results_dir":         "results",
  "tag":                 "ws1"
}
```

## CSV columns

`results/ws1_sweep_<tag>.csv` — full factorial including all knob rows:
```
M_factor, S_gpc, centerM,
node_mass_amplitude, node_s_amplitude, node_mass_seed, init_distribution, node_geometry,
chi2_dof, chi2, chi2_lcdm, chi2_eds,
R2, n_sne_used, growth_factor, growth_target, anchor_ok, runaway,
match_avg_pct, diff_pct
```

`results/sweep_results_pantheon_<tag>.csv` — best-isotropic (amp=0) subset,
`load_best_config`-compatible (`hubble_diagram_nbody.py --from-best-config` works).

`chi2_lcdm` and `chi2_eds` are analytic reference values stamped on every row
(same value per run; used by the heatmap comparisons F2/F3).

## Figures emitted (ws1)

`results/figures/ws1/`:
- `ms_chi2_dof_heatmap_<tag>.png` — F1: chi2/dof vs Pantheon+ heatmap
- `ms_chi2_lcdm_<tag>.png` — F2: LCDM reference chi2/dof
- `ms_chi2_eds_<tag>.png` — F3: EdS null chi2/dof
- `growth_map_<tag>.png` — F4: growth factor over (M, S)
- `runaway_boundary_<tag>.png` — F9: bound/runaway frontier
- `mu_z_panel_M<M>_S<S>_<tag>.png` — F6: best-config mu(z) panel

## Key implementation notes

- `_FixedSweepConfig` subclass pins `particle_count` and `n_steps` so the
  `build_cache_name` slug always matches the actual sim.
- Grid groups cells by `(geometry, init, s_amplitude, amplitude, nm_seed)` and
  iterates M descending within each group, so the linear S co-fit warm-starts
  from the previous M's best S (fewer evaluations for closely-spaced M values).
- `chi2_lcdm` / `chi2_eds` are computed once analytically (no sim) from
  `cosmo.distances.model_distance_modulus` + `cosmo.hubble_diagram.evaluate_precomputed`.
- Cache is fully reused: existing `data/metrics_400_s42.csv` entries from
  `pantheon_knob_sweep.py` runs are read by `worst_callback` unchanged.

## First-exploration results (first_exploration tag, 2026-06-25)

Config: M=[100..50000], S=co-fit [20..90], amp=[0, 0.5], 400p/273steps/t_start=2.9.

- **LCDM reference**: chi2/dof = 0.4360  (1580 SNe)
- **EdS null**: chi2/dof = 0.8430
- **Best isotropic (amp=0)**: M=50000, S=89, chi2/dof=0.6775 — FLAT across M
  (M/S^3 degeneracy confirmed: chi2/dof range 0.6775..0.6923 across 9 M values)
- **Growth**: realized growth 2.84 vs target 3.30 (below anchor by ~14%)
- **amp=0.5**: runaways at M≥1000/S=20; M=200/S=20 gives 1.16, M=100/S=20 gives 0.83
- **Runaway cells**: 7/18 (all amp=0.5 at M≥1000 fell to s_min and failed anchor)

HONEST OBSERVATION: chi2/dof at S=80-90 sits at ~0.69 (far from near-LCDM),
because the M/S^3 degeneracy band at S>80 is NOT the near-LCDM band.
Near-LCDM requires smaller S (see targeted_near_lcdm sweep below).
The ~0.50 chi2 in older cache entries (metrics_400_s42.csv) came from a different
code/key format (those entries have n_sne_used=1339, not 1425); they do NOT
correspond to the current amp=0 flat-key. This is a lode correction.

## Targeted near-LCDM results (targeted_near_lcdm tag, 2026-06-25)

Config: M=[50, 200, 855, 1500, 3000], S=[20..60] explicit list,
amp=[0.0, 0.5], 400p/273steps/t_start=2.9. sweeps/targeted_near_lcdm.json.

- **LCDM reference**: chi2/dof = 0.4360  (1580 SNe)
- **EdS null**: chi2/dof = 0.8430
- **Growth target**: 3.304

**Best overall (all knobs, anchor_ok)**:
M=1500, S=55, amp=0.5, chi2/dof=0.4866 — near-LCDM band confirmed!
growth=2.990, anchor_ok=True.

**Best isotropic (amp=0, anchor_ok)**:
M=200, S=20: chi2/dof=0.5195, growth=2.88
M=1500, S=30: chi2/dof=0.5217, growth=2.88
M=3000, S=35: chi2/dof=0.5300, growth=2.87
M=855, S=25: chi2/dof=0.5322, growth=2.91
(M/S^3 degeneracy confirmed: very flat across M at matched S; best band 0.52-0.54)

**amp=0.5 results**:
M=855/S=45: chi2/dof=0.4887, M=1500/S=55: 0.4866 (best). These amplitude
configs sit between LCDM (0.436) and the isotropic floor (0.52). The amplitude
lever is a growth nudge per PF2/PF3 — it does NOT represent a clean independent
fit knob (anisotropy is the primary effect).

**Runaway cells**: 17/90
- M=855 at S=20, S=25: amp=0.5 runaways
- M=1500 at S=20, S=25: both amp=0 and amp=0.5 runaways
- M=3000 at S=20..45 (all amp variants): heavy runaways
- Runaway boundary: S_crit ∝ M^(1/3), confirmed (larger M → larger minimum S needed)

**Near-LCDM band at amp=0**: chi2/dof 0.52–0.54 for M=200-3000 at S=20-35.
Band is at lower S than first-exploration searched (S=20-35 vs S=80-90).

**Particle-count sensitivity (at best isotropic cells, amp=0)**:
M=200/S=20: 400p→0.5195, 1000p→0.5319, 2000p→0.5269 (within ~0.01, stable)
M=1500/S=30: 400p→0.5217, 1000p→0.5350, 2000p→0.5293 (within ~0.01, stable)
CONCLUSION: chi2/dof is stable across N=400..2000 at the low-S best configs.
The previous note "400p gives 0.69 vs 2000p gives 0.48" was a CONFIGURATION
artifact (that comparison used different M/S; the high-S region IS sensitive to N
but the low-S near-LCDM band is NOT). No large particle-count shift here.

**Geometry thread-through**: the earlier "shell vs cube26" comparison here was INVALID
on two counts and has been removed: (1) a bug had `sweep.py` keying cells by geometry
while always running `cube26`, so the "shell" row was actually cube26 physics (fixed —
see [../scripts/parameter-sweep.md](../scripts/parameter-sweep.md)); and (2) hollow
`shell`/`shell_multi` geometries were removed entirely (not virialized — see
[node-geometries.md](./node-geometries.md)). A fair CROSS-GEOMETRY comparison (now over
cube26 / cube_dense / fcc / bcc) still requires `effective_M_ext_kg()` normalization so
total external mass (hence Ω_Λ_eff) is equal across geometries — DEFERRED until that is
wired into the geometry sweep.

**Figures** (results/figures/ws1/):
- ms_chi2_dof_heatmap_targeted_near_lcdm.png
- ms_chi2_lcdm_targeted_near_lcdm.png
- ms_chi2_eds_targeted_near_lcdm.png
- growth_map_targeted_near_lcdm.png
- runaway_boundary_targeted_near_lcdm.png
- mu_z_panel_M1500_S55_targeted_near_lcdm.png  (best all-knobs)
- mu_z_panel_M200_S20_targeted_near_lcdm_iso_400p.png  (best isotropic 400p)
- mu_z_panel_M200_S20_targeted_near_lcdm_iso_2000p.png  (best isotropic 2000p)

## Tests

`tests/test_overarching_sweep.py` — 33 fast unit tests (no sims):
- Config loading + JSON override
- Grid expansion: amp=0 collapse, total count, required keys
- CSV column contract: SWEEP_CSV_COLS ⊇ BEST_ISO_COLS
- Extra WS1 columns: chi2_lcdm, chi2_eds, growth_target, runaway
- `_FixedSweepConfig`: particle_count and n_steps pinned
- Cache-key uniqueness across geometry/init/amplitude/seed
- S co-fit vs explicit list selection
- --plots-only wiring
- load_best_config compatibility

## Next steps / known limitations

- The linear_search_S early-stop threshold (~0.025% match change) is tuned for
  the LCDM objective; on the pantheon flat landscape it can stop too early at
  large S. For a finer sweep use `--config sweeps/fine_grid.json` with explicit
  `"S_values": [30,35,40,...,70]` instead of "co-fit".
- Re-pin chi2/dof at 2000p to reconcile with Stage-3 numbers (WS6 is the
  high-N convergence workstream).
- Add `--refine` mode: coarse pass → auto-narrow S range → fine pass.
