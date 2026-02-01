# Module Structure

## Dependency Hierarchy

```mermaid
graph TD
    constants[constants.py<br/>Constants & Parameters]
    cli[cli.py<br/>CLI utilities]
    particles[particles.py<br/>Particle, ParticleSystem, HMEAGrid]
    integrator[integrator.py<br/>Integrator, LeapfrogIntegrator]
    simulation[simulation.py<br/>CosmologicalSimulation]
    analysis[analysis.py<br/>Shared analysis utilities]
    viz[visualization.py<br/>Shared plotting utilities]
    factories[factories.py<br/>Simulation utilities]
    numba_direct[numba_direct.py<br/>Numba JIT O(N²) direct]
    bh_numba[barnes_hut_numba.py<br/>Barnes-Hut O(N log N) octree]
    tidal_numba[tidal_forces_numba.py<br/>Numba JIT tidal forces]
    param_sweep[cosmo/parameter_sweep.py<br/>Search algorithms]
    run[run_simulation.py<br/>Main entry point]
    sweep[parameter_sweep.py<br/>Grid search script]
    viz3d[visualize_3d.py<br/>3D visualizations]

    cli --> constants
    param_sweep --> analysis
    particles --> constants
    integrator --> constants
    integrator --> particles
    integrator --> numba_direct
    integrator --> bh_numba
    integrator --> tidal_numba
    simulation --> constants
    simulation --> particles
    simulation --> integrator
    analysis --> constants
    viz --> constants
    factories --> simulation
    factories --> constants
    factories --> analysis
    run --> cli
    run --> simulation
    run --> constants
    run --> analysis
    run --> viz
    sweep --> param_sweep
    sweep --> simulation
    sweep --> constants
    sweep --> analysis
    viz3d --> cli
    viz3d --> simulation
    viz3d --> constants
    viz3d --> analysis
    viz3d --> viz
```

## Module Responsibilities

### `cosmo/constants.py`
**Purpose**: Physical constants and parameter configuration classes.

**Classes**:
- `CosmologicalConstants`: G, c, Mpc_to_m, Gpc_to_m, Gyr_to_s, M_observable, etc.
- `LambdaCDMParameters`: H₀, Ω_m, Ω_Λ, method `H_at_time(a)`
- `ExternalNodeParameters`: M_ext, S, Ω_Λ_eff, method `calculate_required_spacing()`
- `SimulationParameters`: Unified config (M_value, S_value, n_particles, seed, t_start_Gyr, t_duration_Gyr, n_steps, damping_factor, center_node_mass, mass_randomize)

**Exports**: All four classes.

**Key feature**: SimulationParameters auto-calculates derived quantities (M_ext, S in SI units, t_end_Gyr, center_node_mass_kg, external_params) in `_calculate_derived()`.

### `cosmo/cli.py`
**Purpose**: Command-line interface utilities shared across scripts.

**Functions**:
- `add_common_arguments(parser)`: Add shared simulation args (--M, --S, --particles, --seed, --t-start, --t-duration, --n-steps, --damping, --center-node-mass, --mass-randomize, --compare)
- `parse_arguments(description, add_output_dir)`: Create parser, add common args, parse CLI
- `args_to_sim_params(args)`: Convert parsed args to SimulationParameters

**Used by**: `run_simulation.py`, `visualize_3d.py`

**Exports**: All three functions.

### `cosmo/particles.py`
**Purpose**: Physical structures (particles and external nodes).

**Classes**:
- `Particle`: Single entity with position, velocity, mass, acceleration, id
- `ParticleSystem`: N particles with update methods, energy calculations
- `HMEAGrid`: 26-node cubic lattice (3×3×3-1), vectorized tidal force calculation

**Key methods**:
- `ParticleSystem._initialize_particles()`: Sets up particles with Hubble flow + peculiar velocities (particles.py:60-175)
  - Velocity uses model-appropriate H (H_lcdm for dark energy, H_matter for non-LCDM)
  - `mass_randomize` parameter (0.0=equal masses, 1.0=masses from 0 to 2x mean). Default 0.5. Total mass preserved via normalization.
  - No damping applied here; damping is applied at sim.run() via velocity calibration
- `HMEAGrid.calculate_tidal_acceleration_batch()`: Vectorized tidal forces across all 26 nodes

**Exports**: All three classes.

### `cosmo/integrator.py`
**Purpose**: N-body physics and time evolution.

**Classes**:
- `Integrator`: Base class calculating forces (internal, external, dark energy, Hubble drag)
- `LeapfrogIntegrator`: Kick-Drift-Kick time stepping

**Key methods**:
- `calculate_internal_forces()`: Direct O(N²) gravity with softening (integrator.py:52-87)
- `calculate_external_forces()`: Delegates to HMEAGrid (integrator.py:89-102)
- `calculate_dark_energy_forces()`: a_Λ = H₀² Ω_Λ r (integrator.py:104-125)
- `calculate_hubble_drag()`: a_drag = -2Hr (ΛCDM only) (integrator.py:127-157)
- `LeapfrogIntegrator.step(dt)`: Single timestep (integrator.py:229-252)
- `LeapfrogIntegrator.evolve()`: Full simulation loop (integrator.py:254-298)

**Exports**: `Integrator`, `LeapfrogIntegrator`.

### `cosmo/numba_direct.py`
**Purpose**: Fast O(N²) gravity via Numba JIT compilation.

**Class**: `NumbaDirectSolver`

**Functions**:
- `calculate_forces_direct_numba()`: JIT-compiled pairwise gravity with softening (exact, 14-17x speedup)

**Used by**: `integrator.py` for N≥100 particles (auto mode default)

### `cosmo/barnes_hut_numba.py`
**Purpose**: Real Barnes-Hut octree O(N log N) gravity with Numba JIT.

**Class**: `NumbaBarnesHutTree`

**Functions**:
- `build_octree()`: Iterative particle insertion into octree (max_depth=60)
- `calculate_forces_barnes_hut()`: Stack-based tree traversal with opening angle criterion

**Parameters**: `theta` controls accuracy/speed tradeoff (0.3=accurate, 0.5=balanced, 0.7=fast)

**Used by**: `integrator.py` when `force_method='barnes_hut'`

### `cosmo/tidal_forces_numba.py`
**Purpose**: Numba JIT-compiled tidal force calculation from external HMEA nodes.

**Functions**:
- `calculate_tidal_forces_numba(particle_positions, node_positions, node_masses, G)`: Returns (N, 3) accelerations in m/s²

**Algorithm**: Double loop over N particles × M nodes (M=26), computing attractive acceleration toward each node. Singularity protection at r < 1e10 m.

**Used by**: `integrator.py` for external force calculations

### `cosmo/factories.py`
**Purpose**: Utility functions for running simulations and extracting results.

**Functions**:
- `run_and_extract_results(sim, t_duration_Gyr, n_steps, save_interval, damping=None)`: Runs simulation with damping parameter, returns dict with t_Gyr, a, diameter_Gpc, max_radius_Gpc, sim

**Used by**: Scripts needing streamlined simulation execution

### `cosmo/simulation.py`
**Purpose**: High-level simulation orchestration.

**Classes**:
- `CosmologicalSimulation`: Combines ParticleSystem + Integrator, tracks expansion history

**Key methods**:
- `__init__()`: Sets up particles, HMEA grid, integrator based on mode flags
- `run(t_end_Gyr, n_steps, save_interval, damping=None)`: Executes integration, calculates expansion metrics. For non-LCDM models, applies velocity calibration at start based on damping parameter (auto-calculated from t_start if None).
- `_calibrate_velocity_for_lcdm_match()`: Scales initial velocities so non-LCDM models never exceed LCDM
- `save(filename)`, `load(filename)`: Pickle persistence

**Mode flags**:
- `use_external_nodes=True, use_dark_energy=False`: External-Node Model
- `use_external_nodes=False, use_dark_energy=True`: ΛCDM
- `use_external_nodes=False, use_dark_energy=False`: Matter-only

**Exports**: `CosmologicalSimulation`.

### `cosmo/analysis.py`
**Purpose**: Shared analysis utilities for cosmological calculations.

**Functions**:
- `friedmann_equation(a, t, H0, Omega_m, Omega_Lambda)`: ODE for scale factor evolution
- `solve_friedmann_equation(t_start, t_end, Omega_Lambda, n_points)`: Solve ΛCDM/matter-only evolution
- `calculate_initial_conditions(t_start, reference_size)`: Compute a_start, box_size from t_start
- `normalize_to_initial_size(a_array, initial_size)`: Convert scale factors to physical sizes
- `compare_expansion_histories(size_ext, size_lcdm, return_array=False)`: Calculate match percentage. Returns array if return_array=True and inputs are arrays; otherwise returns scalar averaged match.
- `detect_runaway_particles(max_distance, rms_size, threshold)`: Detect numerical instability
- `calculate_today_marker(t_start, t_duration, today)`: Position of "today" in simulation time

**Used by**: `run_simulation.py`, `parameter_sweep.py`, `visualize_3d.py`

**Exports**: All functions listed above.

### `cosmo/visualization.py`
**Purpose**: Shared visualization utilities for plots and 3D graphics.

**Functions**:
- `get_node_positions(S_Gpc)`: Calculate 26 HMEA node positions in 3×3×3 grid
- `draw_universe_sphere(ax, radius, alpha, color, resolution)`: Draw sphere on 3D axes
- `draw_cube_edges(ax, half_size, color, alpha, linewidth)`: Draw cube outline
- `setup_3d_axes(ax, lim, title, elev, azim)`: Configure 3D plot axes
- `generate_output_filename(base_name, sim_params, extension, output_dir, include_timestamp)`: Standardized filenames with parameters
- `format_simulation_title(sim_params, include_particles)`: Standardized plot titles

**Used by**: `run_simulation.py`, `visualize_3d.py`

**Exports**: All functions listed above.

### `run_simulation.py`
**Purpose**: Main script orchestrating full comparison workflow.

**Key functions**:
- `run_simulation(output_dir, sim_params)`: Runs 3 simulations (ΛCDM analytic, External-Node N-body, Matter-only N-body), generates 4-panel plot

**CLI**: Uses `cosmo.cli.parse_arguments()` and `cosmo.cli.args_to_sim_params()` for argument handling. Supports --M, --S, --particles, --seed, --t-start, --t-duration, --n-steps, --damping, --center-node-mass, --compare, --output-dir.

**Workflow**:
1. Calculate initial conditions using `analysis.calculate_initial_conditions()`
2. Solve Friedmann equation for ΛCDM and Matter-only using `analysis.solve_friedmann_equation()`
3. Run External-Node N-body simulation
4. Run Matter-only N-body simulation
5. Compare using `analysis.compare_expansion_histories()`, detect runaways with `analysis.detect_runaway_particles()`
6. Generate plot using `visualization.format_simulation_title()`
7. Save using `visualization.generate_output_filename()`

**Entry point**: `if __name__ == "__main__"`

### `cosmo/parameter_sweep.py`
**Purpose**: Reusable parameter sweep infrastructure with search algorithms and dataclasses.

**Classes**:
- `SearchMethod`: Enum (BRUTE_FORCE, TERNARY_SEARCH, LINEAR_SEARCH)
- `SweepConfig`: Configuration dataclass (quick_search, many_search, search_center_mass, etc.)
- `MatchWeights`: Match metric weights (hubble_curve, size_curve, endpoint, max_radius)
- `SimResult`: Raw simulation output dataclass (size_curve_Gpc, hubble_curve, etc.)
- `LCDMBaseline`: Precomputed LCDM reference data

**Functions**:
- `build_m_list(many_search)`: Build M value list (descending)
- `build_s_list(s_min, s_max)`: Build S value list
- `build_center_mass_list(search_center_mass, many_search)`: Build centerM value list
- `compute_match_metrics(sim_result, baseline, weights)`: Compute R^2-based match metrics
- `ternary_search_S(...)`: Ternary search for optimal S
- `linear_search_S(...)`: Linear search with early stopping
- `brute_force_search(...)`: Exhaustive grid search
- `run_sweep(config, method, callback, baseline, weights)`: Main entry point

**Exports**: All classes and functions.

### `parameter_sweep.py`
**Purpose**: Script wiring simulation callback to sweep module, handling output formatting.

**Workflow**:
1. Create SweepConfig and LCDMBaseline from analysis utilities
2. Define sim_callback that runs real simulations, returns SimResult
3. Call run_sweep() from cosmo.parameter_sweep module
4. Format and display results, save best config

**Uses**: `cosmo.parameter_sweep.run_sweep()` for search algorithms.

## File Locations

| File | Lines | Purpose |
|------|-------|---------|
| `cosmo/constants.py` | 175 | Parameter definitions |
| `cosmo/cli.py` | 95 | CLI argument parsing |
| `cosmo/particles.py` | 340 | Physical structures |
| `cosmo/integrator.py` | 316 | Force calculations + integration |
| `cosmo/simulation.py` | 218 | High-level runner |
| `cosmo/analysis.py` | 382 | Shared analysis utilities |
| `cosmo/visualization.py` | 213 | Shared plotting utilities |
| `cosmo/numba_direct.py` | 82 | Numba JIT O(N²) direct |
| `cosmo/barnes_hut_numba.py` | 200 | Barnes-Hut O(N log N) octree |
| `cosmo/tidal_forces_numba.py` | 60 | Numba JIT tidal forces |
| `cosmo/factories.py` | 39 | Simulation utilities |
| `cosmo/parameter_sweep.py` | 350 | Search algorithms, dataclasses |
| `run_simulation.py` | 280 | Main comparison script |
| `parameter_sweep.py` | 210 | Parameter exploration script |
| `visualize_3d.py` | 765 | 3D visualization |

## Import Pattern

All scripts import from `cosmo` package:
```python
from cosmo.constants import CosmologicalConstants, LambdaCDMParameters, SimulationParameters
from cosmo.simulation import CosmologicalSimulation
from cosmo.cli import parse_arguments, args_to_sim_params
```

No circular dependencies. Linear dependency chain: constants → cli → particles → integrator → simulation → scripts.
