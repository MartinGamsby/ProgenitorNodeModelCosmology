# 3D Visualization

Creates 3D visualizations of cosmological simulations showing particles and HMEA nodes.

## CLI Arguments

Uses shared CLI from `cosmo.cli`:
```bash
python visualize_3d.py [sim_file] [output_dir] [--M 855] [--S 25] [--particles 300] ...
python visualize_3d.py --compare [output_dir] [--M 855] [--S 25] ...
```

**All arguments**:
- `sim_file`: Path to .pkl file (optional positional)
- `output_dir`: Output directory (optional positional, default: `.`)
- `--M`: External mass parameter (default: 855)
- `--S`: Node separation in Gpc (default: 25.0)
- `--particles`: Number of particles (default: 300)
- `--seed`: Random seed (default: 42)
- `--t-start`: Start time in Gyr (default: 5.8)
- `--t-duration`: Duration in Gyr (default: 10.0)
- `--n-steps`: Number of timesteps (default: 150)
- `--damping`: Velocity damping (default: auto)
- `--center-node-mass`: Central node mass as multiple of M_obs (default: 1.0)
- `--compare`: Enable 3-way comparison mode

## Modes

### Single Simulation Mode (default)
```bash
python visualize_3d.py [sim_file.pkl] [output_dir]
```
- If no .pkl provided, runs simulation with CLI parameters
- Generates: multipanel evolution, snapshots, animation

### Comparison Mode
```bash
python visualize_3d.py --compare [output_dir] --M 800 --S 24
```
- Runs External-Node, Matter-Only, ΛCDM side-by-side
- Generates: 3-way comparison panel, separate animations

## Output Files

**Single mode**:
- `3d_evolution_multipanel.png` - 6-panel time series
- `3d_snapshot_t*.png` - Individual time snapshots
- `3d_animation.gif` - Rotating expansion animation

**Comparison mode**:
- `3d_comparison_multipanel.png` - 6×3 panel (times × models)
- `3d_animation_external_node.gif`
- `3d_animation_matter_only.gif`
- `3d_animation_lcdm.gif`

## Key Functions

### Visualization Helpers (cosmo/visualization.py)
- `get_node_positions(S_Gpc)`: 26 HMEA node positions in 3×3×3 grid
- `draw_universe_sphere(ax, radius, center_Gpc, ...)`: Wireframe sphere
- `draw_cube_edges(ax, half_size, ...)`: Grid boundary cube
- `setup_3d_axes(ax, lim, title, ...)`: Configure 3D plot

### Main Functions (visualize_3d.py)
- `parse_visualize_arguments()`: Parse CLI args using `cosmo.cli.add_common_arguments()`
- `load_or_run_simulation(sim_params, sim_file, output_dir)`: Load .pkl or run with given params
- `create_3d_snapshot(sim_data, idx, start_time, output_dir)`: Single time visualization
- `create_multi_panel_evolution(sim_data, start_time, output_dir)`: 6-panel time series
- `create_animation(sim_data, start_time, output_dir, fps)`: GIF with rotating view
- `run_comparison_simulations(sim_params, output_dir)`: Run all 3 models
- `create_comparison_multipanel(comparison_data, start_time, output_dir)`: 6×3 comparison grid
- `create_comparison_animations(comparison_data, start_time, output_dir, fps)`: Separate GIFs per model

### ΛCDM Reference Generation
For comparison mode, ΛCDM doesn't run N-body. Instead:
1. Solve Friedmann at exact snapshot times
2. Generate sphere positions at each a(t): `generate_sphere_positions(radius_max, n_points)`
3. Create fake "snapshots" for consistent visualization

## Sphere Drawing

Universe boundaries centered on COM:
- **Outer (red, α=0.03)**: max_particle_distance (shows outliers)
- **Inner (cyan, α=0.08)**: diameter/2 = RMS radius (typical distribution)

RMS→sphere conversion:
```python
# For uniform sphere of radius R, RMS radius = R * sqrt(3/5)
radius_max = size_m / 2 / np.sqrt(3/5)
```

## Animation Details

- Frame rate: 10 fps (configurable)
- Rotation: +2° azimuth per frame
- View: elev=20°, starting azim=45°

## WS8 — Virialized-grid figures (`_generate_ws8_figs.py`)

Standalone regeneration script (mirrors `_generate_ws4_figs.py`: `matplotlib.use("Agg")`,
`configure_utf8_stdout()`, argparse, one function per figure). Output PNGs go to
`results/figures/ws8/` (GITIGNORED) via `cosmo.plots.figure_path("ws8", name)`.

Figures:
1. `node_geometries_by_mass.png` — static 3D node positions for cube26/cube_dense/fcc/bcc
   plus virialized variants (extent 1/2 × rule radial/massfunc), colored by NODE MASS so
   mass segregation is visible.
2. `particle_motion_slingshot.png` — DIAGNOSTIC: per-particle initial→final displacement
   for a SHORT sim, cube26 vs virialized, with a slingshot-tail metric.
3. `massrule_a_vs_b.png` — "radial" vs "massfunc" mass↔position rules: side-by-side mass
   scatter + mass-vs-radius, with printed/annotated stats (Pearson/Spearman correlation,
   median & mean NN spacing, mass min/max/mean/std, mean-preservation |Δ|/M).
4. Virialization-residual + slingshot-taming figures (Fig4 / Fig6 "doubly tamed"):
   the inner-node force residual vs grid size / balance level (the virialization
   criterion), and the slingshot tail OFF vs node_softening_gpc ON for cube26 AND
   virialized (the TAMING fix is now IN scope and shipped — see
   [../physics/slingshot-and-softening.md](../physics/slingshot-and-softening.md)).

CLI: `--no-sim` (skip the sim figures → fast), `--n-particles`, `--n-steps`, `--seed`,
`--M`, `--S`, `--t-start`, `--vir-extent`, `--vir-n-nodes`, and the taming/balance knobs.
All seeded so figures reproduce. Prints each saved path at the end.

PURE helpers (no sim, no I/O) are unit-tested in `tests/test_ws8_figs.py` (33):
`resolve_n_steps` (keeps dt < 0.05 Gyr ceiling), `displacement_magnitudes`,
`slingshot_metrics`, `softened_node_acceleration` (diagnostic Plummer node force),
the virialization-residual curve, `mass_radius_stats`, `_pearson`, `_spearman`. See
[../plans/node-geometries.md](../plans/node-geometries.md#virialized-geometry--coupled-positions-masses-mass-segregated-implemented).

## Paper figures for the BEST config (`_generate_paper_figs.py` + `_generate_campaign_figs.py`)

The candidate figure set for the `docs/VirializedMetaStructure.tex` UPDATE (we are updating the
existing draft, not writing a new paper). Output → `results/figures/paper/` (gitignored).

`_generate_paper_figs.py` — runs the BEST config (built via the sweep's own `_build_sim_params`
so the figure config == the swept cell: virialized lattice 300 nodes, vir_mass_spread=6, M=300,
S=20, GRF sphere, Plummer 1 Gpc, 2000p/1092, t_start=2.9) PLUS Matter-only(EdS) and the analytic
LCDM baseline, then emits + a summary JSON:
- `size_vs_time.png` / `rate_vs_time.png` — RMS diameter a(t) and H(t) vs time, External vs LCDM
  vs Matter, with R² annotations. (R²_size 0.915, R²_rate 0.932 External-vs-LCDM.)
- `hubble_diagram.png` — μ(z) vs REAL Pantheon+ (External/LCDM/EdS) + Δμ residual + χ²/dof per
  model. HEADLINE NUMBERS: External-Node centre χ²/dof **0.442** ≈ LCDM **0.436**, EdS null
  **0.843** (matter-only sim 0.816); R²_μ=0.997, n=1580 SNe.
- `particles_3d.png` / `particles_3way.png` — the particle cloud over time (External alone; and
  External vs Matter vs LCDM at 4 times). Axis frame = RMS-radius based (diameter/2/√(3/5)·1.15),
  COMMON per time column for the 3-way, so the bulk cloud fills the frame and sizes are
  comparable (NOT the 99th-percentile, which let slingshot outliers shrink the bulk to a dot).
- `observer_distribution.png` — per-observer χ²/dof histogram (centre/best/median + LCDM/EdS
  refs + frac_below). 93% of observers below EdS; ≈0 below LCDM at 1092 steps (matches LCDM).

`_generate_campaign_figs.py` — CSV-only (no sims, fast), the MULTIPLICITY story:
- `multiplicity_map.png` (best-observer χ² over (M,S) per σ, 85 cells, many ≤0.45 ≈LCDM),
  `step_convergence.png`, `seed_robustness.png`, `chi2_ladder.png` (External vs LCDM vs EdS).

Both reuse `cosmo.plots.figure_path`, `cosmo.factories`, `cosmo.observer_distance`,
`cosmo.sim_distance`/`distances`/`pantheon`/`hubble_diagram`. See PF16/PF17 + the campaign
headline in [../plans/pinned-findings.md](../plans/pinned-findings.md).

## Dependencies

- matplotlib (3D projection, animation)
- numpy
- cosmo.* modules
