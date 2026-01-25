# 3D Visualization

Creates 3D visualizations of cosmological simulations showing particles and HMEA nodes.

## Modes

### Single Simulation Mode (default)
```bash
python visualize_3d.py [sim_file.pkl] [output_dir]
```
- If no .pkl provided, runs quick simulation first
- Generates: multipanel evolution, snapshots, animation

### Comparison Mode
```bash
python visualize_3d.py --compare [output_dir]
```
- Runs External-Node, Matter-Only, ΛCDM side-by-side
- Generates: 3-way comparison panel, separate animations

## Default Parameters

```python
sim_params = SimulationParameters(
    M_value=644,
    S_value=21,
    n_particles=1400,
    seed=42,
    t_start_Gyr=3.8,
    t_duration_Gyr=12.5,  # 10.0 * 5/4
    n_steps=1000,
    damping_factor=0.92
)
```

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
- `load_or_run_simulation()`: Load .pkl or run new sim
- `create_3d_snapshot()`: Single time visualization
- `create_multi_panel_evolution()`: 6-panel time series
- `create_animation()`: GIF with rotating view
- `run_comparison_simulations()`: Run all 3 models
- `create_comparison_multipanel()`: 6×3 comparison grid
- `create_comparison_animations()`: Separate GIFs per model

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
