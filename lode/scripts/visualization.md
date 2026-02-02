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

## Dependencies

- matplotlib (3D projection, animation)
- numpy
- cosmo.* modules
