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
    param_sweep[cosmo/parameter_sweep.py<br/>Search algorithms LIBRARY]
    run[run_simulation.py<br/>Main entry point]
    sweep[sweep.py<br/>THE config-driven sweep driver]
    viz3d[visualize_3d.py<br/>3D visualizations]
    distances[cosmo/distances.py<br/>Distance kernel: H z, d_L, mu]
    pantheon[cosmo/pantheon.py<br/>Pantheon+ loader]
    hd_engine[cosmo/hubble_diagram.py<br/>Offset fit + chi2/R2]
    hd_script[hubble_diagram.py<br/>Hubble-diagram script]

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
    distances --> constants
    hd_engine --> analysis
    hd_engine --> constants
    hd_engine --> distances
    hd_script --> cli
    hd_script --> distances
    hd_script --> hd_engine
    hd_script --> pantheon
    hd_script --> viz
```

The Hubble-diagram modules (distances, pantheon, hd_engine, hd_script) form an
independent additive branch: they do NOT touch the
particles -> integrator -> simulation N-body chain.

## Module Responsibilities

### `cosmo/constants.py`
**Purpose**: Physical constants and parameter configuration classes.

**Classes**:
- `CosmologicalConstants`: G, c, Mpc_to_m, Gpc_to_m, Gyr_to_s, M_observable, etc.
- `LambdaCDMParameters`: H₀, Ω_m, Ω_Λ, method `H_at_time(a)`
- `ExternalNodeParameters`: M_ext, S, Ω_Λ_eff, method `calculate_required_spacing()`
- `SimulationParameters`: Unified config (M_value, S_value, n_particles, seed, t_start_Gyr, t_duration_Gyr, n_steps, damping_factor, center_node_mass, mass_randomize, node_geometry + vir_* virialized knobs incl. vir_relax_steps balance level, node_softening_gpc, start_size_scale). Validates start_size_scale > 0; derives node_softening_m and external_params (incl. vir_*) in `_calculate_derived()`.

**Exports**: All four classes.

**Key feature**: SimulationParameters auto-calculates derived quantities (M_ext, S in SI units, t_end_Gyr, center_node_mass_kg, external_params) in `_calculate_derived()`.

### `cosmo/cli.py`
**Purpose**: Command-line interface utilities shared across scripts.

**Functions**:
- `add_common_arguments(parser)`: Add shared simulation args (--M, --S, --particles, --seed, --t-start, --t-duration, --n-steps, --damping, --center-node-mass, --mass-randomize, --compare, --node-geometry, --vir-n-nodes/--vir-extent/--vir-mass-rule/--vir-mass-spread/--vir-segregation/--vir-s-metric/--vir-relax-steps, --node-softening-gpc, --start-size-scale)
- `parse_arguments(description, add_output_dir)`: Create parser, add common args, parse CLI
- `args_to_sim_params(args)`: Convert parsed args to SimulationParameters

**Used by**: `run_simulation.py`, `visualize_3d.py`

**Exports**: All three functions.

### `cosmo/particles.py`
**Purpose**: Physical structures (particles and external nodes).

**Classes**:
- `Particle`: Single entity with position, velocity, mass, acceleration, id
- `ParticleSystem`: N particles with update methods, energy calculations
- `HMEAGrid`: geometry-driven node grid (`build_node_positions` / coupled `build_virialized_grid`), vectorized tidal force calculation with optional Plummer node softening (numba + numpy paths)

**Key methods**:
- `ParticleSystem._initialize_particles()`: Sets up particles with Hubble flow + peculiar velocities (particles.py:60-175)
  - Velocity uses model-appropriate H (H_lcdm for dark energy, H_matter for non-LCDM)
  - `mass_randomize` parameter (0.0=equal masses, 1.0=masses from 0 to 2x mean). Default 0.0. Total mass preserved via normalization.
  - No damping applied here; damping is applied at sim.run() via velocity calibration
- `ParticleSystem.set_positions(positions)`: Set positions for all particles (used by velocity calibration state restore)
- `ParticleSystem.set_velocities(velocities)`: Set velocities for all particles
- `HMEAGrid.calculate_tidal_acceleration_batch()`: Vectorized tidal forces across all nodes; applies `params.node_softening_m` (0.0 = legacy hard floor, byte-identical; >0 = Plummer) on both numba and numpy paths

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
- `calculate_tidal_forces_numba(particle_positions, node_positions, node_masses, G, softening_m=0.0)`: Returns (N, 3) accelerations in m/s²

**Algorithm**: Double loop over N particles × M nodes, computing attractive acceleration toward each node. Singularity handling: `softening_m==0.0` (DEFAULT) uses the LEGACY hard `r<1e10 m` floor (byte-identical to pre-softening); `softening_m>0.0` uses Plummer softening `r_soft^2 = r^2 + softening_m^2` (the slingshot taming knob `node_softening_gpc`; the 1e10 floor is dropped in that branch). See [../physics/slingshot-and-softening.md](../physics/slingshot-and-softening.md).

**Used by**: `HMEAGrid.calculate_tidal_acceleration_batch` (numba path; the numpy fallback mirrors the same softening) → `integrator.py` external forces

### `cosmo/factories.py`
**Purpose**: Shared simulation functions used by both run_simulation.py and sweep.py for consistency. Single source of truth for simulation execution.

**Functions**:
- `run_and_extract_results(sim, t_duration_Gyr, n_steps, save_interval, damping=None)`: Runs simulation, returns dict with t_Gyr, a, diameter_Gpc, max_radius_Gpc, H_hubble, sim
- `solve_lcdm_baseline(sim_params, box_size_Gpc, a_start, save_interval)`: Compute ΛCDM analytic baseline at N-body snapshot times
- `run_external_node_simulation(sim_params, box_size_Gpc, a_start, save_interval)`: Run External-Node N-body simulation
- `run_matter_only_simulation(sim_params, box_size_Gpc, a_start, save_interval)`: Run matter-only N-body simulation
- `setup_simulation_context(t_start_Gyr, t_duration_Gyr, n_steps, save_interval)`: Combined initial conditions + LCDM baseline setup, returns (box_size, a_start, baseline_dict)
- `results_to_sim_result(ext_results, sim_params)`: Convert factory results dict to SimResult for parameter_sweep

**Used by**: `run_simulation.py`, `sweep.py`

### `cosmo/cache.py`
**Purpose**: Two-level key-value disk cache with JSON, CSV, and Pickle format support.

**Classes**:
- `EnhancedJSONEncoder`: Custom JSON encoder that serializes dataclasses via `dataclasses.asdict()`
- `CacheType`: Enum — VELOCITY, METRICS, RESULTS
- `CacheFormat`: Enum — JSON, CSV, PICKLE (default: CSV)
- `CacheLock`: File-based lock with PID staleness detection
- `Cache`: Two-level `{key: {data_type: value}}` store persisted to `data/<name>.<ext>`

**Constructor**: `Cache(name, format=CacheFormat.CSV, _data_dir="data")`

**Concurrency**: `CacheLock` creates `<filepath>.lock` containing the owning PID. Uses atomic `os.open(O_CREAT|O_EXCL)`. Lock held for entire Cache lifetime (acquired in `__init__`, released in `close()`/`__del__`). `close()` registered via `atexit` for Ctrl+C cleanup; idempotent (`_closed` flag). Three conflict scenarios:
- **Own PID**: prints BUG warning (duplicate Cache or crash leftover), prompts `[D/Y/n]` — D=delete lock and retry, Y=read-only, n=abort
- **Other live PID**: prompts `[Y/n/kill]` — Y=read-only, n=abort, kill=terminate owner
- **Dead PID**: auto-broken silently
PID liveness: `ctypes`+`OpenProcess`/`GetExitCodeProcess` on Windows, `os.kill(pid, 0)` on Unix. Kill: `taskkill /F` on Windows, `SIGTERM` on Unix.

**Key methods**:
- `_load_from_disk()`: Loads primary format; falls back to other formats. Locked.
- `_save_to_disk()`: Saves in configured format. Locked.
- `get_cached_value(key, data_type)`: Two-level lookup, returns None if missing
- `add_cached_value(key, data_type, value, save_interval_s=5)`: Set + time-based save
- `__del__()`: Saves on garbage collection

**CSV format**: One row per cache key. Cache keys are split on `_` into `key.{i}_{suffix}` columns (e.g. `key.0_p`, `key.1_Gyr`, `key.2_M`). Scalar values get a column named after data_type (e.g. `velocity`). Dict values are flattened: each field becomes `data_type.field` (e.g. `metrics.match_avg_pct`, `results.size_final_Gpc`). Nested dicts within fields are JSON-encoded per cell.

**Used by**: `simulation.py` (velocity cache), `sweep.py` / `cosmo/parameter_sweep.py` (metrics/results cache)

### `cosmo/simulation.py`
**Purpose**: High-level simulation orchestration.

**Classes**:
- `CosmologicalSimulation`: Combines ParticleSystem + Integrator, tracks expansion history

**Key methods**:
- `__init__()`: Sets up particles, HMEA grid, integrator based on mode flags
- `run(t_end_Gyr, n_steps, save_interval, damping=None)`: Executes integration, calculates expansion metrics. For non-LCDM models, applies velocity calibration at start.
- `_calibrate_velocity_for_lcdm_match()`: Runs ~2 Gyr N-body test to measure expansion vs LCDM, scales initial velocity to match LCDM in early phase. Includes 1% safety margin to ensure matter-only never exceeds LCDM. Temporarily disables external nodes during calibration test.
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

**Used by**: `run_simulation.py`, `sweep.py`, `visualize_3d.py`

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
**Purpose**: Main script orchestrating full comparison workflow. Uses shared functions from factories.py for consistency with sweep.py.

**Key functions**:
- `run_simulation(output_dir, sim_params)`: Runs 3 simulations (ΛCDM analytic, External-Node N-body, Matter-only N-body), generates 4-panel plot

**CLI**: Uses `cosmo.cli.parse_arguments()` and `cosmo.cli.args_to_sim_params()` for argument handling. Supports --M, --S, --particles, --seed, --t-start, --t-duration, --n-steps, --damping, --center-node-mass, --mass-randomize, --compare, --output-dir.

**Default values** (aligned with the sweep quick_search mode):
- n_steps=250
- particles=200
- mass_randomize=0.0 (deterministic)
- save_interval=10

**Workflow**:
1. Calculate initial conditions using `analysis.calculate_initial_conditions()`
2. Solve ΛCDM baseline using `factories.solve_lcdm_baseline()` (shared with sweep.py)
3. Run External-Node N-body using `factories.run_external_node_simulation()`
4. Run Matter-only N-body using `factories.run_matter_only_simulation()`
5. Compare using `analysis.compare_expansion_histories()`, detect runaways with `analysis.detect_runaway_particles()`
6. Generate plot using `visualization.format_simulation_title()`
7. Save using `visualization.generate_output_filename()`

**Entry point**: `if __name__ == "__main__"`

### `cosmo/parameter_sweep.py` (LIBRARY — kept)
**Purpose**: Reusable parameter sweep infrastructure with search algorithms,
dataclasses, scoring, and `build_cache_name`. Consumed by `sweep.py`.

**Classes**:
- `SearchMethod`: Enum (BRUTE_FORCE, TERNARY_SEARCH, LINEAR_SEARCH)
- `SweepConfig`: Configuration dataclass (objective, node_geometry, vir_*,
  node_softening_gpc, start_size_scale, etc.)
- `MatchWeights`: Match metric weights (curve, curve_r2, end, ...)
- `SimResult`: Raw simulation output dataclass (size_curve_Gpc, hubble_curve, a_curve)
- `LCDMBaseline`: Precomputed LCDM reference data

**Functions**:
- `build_m_list` / `build_s_list` / `build_center_mass_list`: parameter list builders
- `compute_match_metrics` (lcdm) / `compute_pantheon_metrics` (pantheon): scorers
- `ternary_search_S` / `linear_search_S` / `brute_force_search`: S searches
- `run_sweep(config, method, callback, baseline, weights, pantheon_data)`: entry point
- `build_cache_name(config, M, S, centerM, seeds)`: physics-versioned cache key
  (PHYSICS_CACHE_VERSION="v3"; non-default-only slugs incl. nsoft / ssz / vir_*)

**Exports**: All classes and functions.

### `sweep.py` (THE single sweep driver)
**Purpose**: The one config-driven, resumable, cached sweep over ALL parameter axes
(M, S, amplitudes, seed, init, particles, geometry, vir_*, node_softening, start-size).
Replaces the DELETED root `parameter_sweep.py` (LCDM grid → `sweeps/lcdm_example.json`)
and `pantheon_knob_sweep.py` (GRF factorial → `sweeps/knob_grf.json`).

**Key functions**: `load_config`, `expand_grid`, `_make_sweep_config_for_cell`,
`_make_sim_callback`, `_select_best_row`, `_compute_reference_chi2`, `run_plots_only`.

**Workflow**: load JSON config → `expand_grid` (amp=0 collapse) → per cell:
co-fit S (linear/ternary) or explicit list on the configured objective → cached
`worst_callback` → growth-anchor check → resumable per-cell CSV checkpoint → figures.

**Uses**: `cosmo.parameter_sweep` (search + cache key), `cosmo.factories` (sim),
`cosmo.plots` (figures). Documented in [../plans/overarching-sweep.md](../plans/overarching-sweep.md).

### `cosmo/distances.py`
**Purpose**: Pure-function cosmological distance kernel for the Hubble-diagram test. No I/O, no plotting.

**Functions**:
- `hubble_z(z, Omega_m, Omega_de, H0)`: H(z)=H0*sqrt(Omega_m(1+z)^3+Omega_k(1+z)^2+Omega_de); Omega_k derived (no flat assumption); raises ValueError on E^2<0 (turnaround in closed models)
- `comoving_distance`, `transverse_comoving_distance` (sinh/flat/sin curvature branches), `luminosity_distance`, `distance_modulus`
- `model_distance_modulus(z, model, sim_params, H0=70)`: builds mu(z) for `lcdm` / `matter_only` / `external_node` (latter uses `sim_params.external_params.Omega_Lambda_eff`)

**Used by**: `cosmo/hubble_diagram.py`, `hubble_diagram.py`

### `cosmo/pantheon.py`
**Purpose**: Loader for the vendored Pantheon+SH0ES SN Ia compilation. No network access.

**Functions**:
- `load_pantheon(path=DEFAULT_PATH, z_min=0.01, exclude_calibrators=True)`: reads `data/pantheon_plus/Pantheon+SH0ES.dat` by COLUMN NAME (zHD, MU_SH0ES, MU_SH0ES_ERR_DIAG, IS_CALIBRATOR); applies z_min and calibrator cuts; returns dict {z, mu, sigma, n} sorted by z. Raises FileNotFoundError pointing to data/pantheon_plus/README.md if absent.
- `bin_for_plot(z, mu, sigma, n_bins=20)`: inverse-variance log-z binning for plot overlays only (statistics use unbinned data)

**Used by**: `hubble_diagram.py`. Real data file NOT committed; tests use `tests/fixtures/pantheon_synthetic.dat`.

### `cosmo/hubble_diagram.py`
**Purpose**: Offset-marginalized chi^2/R^2 comparison engine. Pure numpy.

**Functions**:
- `fit_offset(mu_obs, mu_model, sigma)`: analytic inverse-variance-weighted additive offset DeltaM (closed form)
- `evaluate_model(z, mu_obs, sigma, model, sim_params, H0=70)`: computes mu_model, fits DeltaM, returns {model, DeltaM, chi2, dof=n-1, chi2_dof, R2, residuals, mu_fit}; surfaces turnaround ValueError descriptively
- `compare_all_models(data, sim_params, H0=70)`: runs all three models, returns dict keyed by model name

**Uses**: `cosmo.analysis.calculate_r_squared`, `cosmo.distances.model_distance_modulus`.

### `hubble_diagram.py` (top-level script)
**Purpose**: Standalone Hubble-diagram-vs-Pantheon+ runner. INDEPENDENT of run_simulation.py — semi-analytic, never calls CosmologicalSimulation.run().

**Workflow**: load_pantheon -> compare_all_models -> print per-model chi^2/dof/R^2 table -> save 2-panel PNG (data+curves / residuals) via `visualization.generate_output_filename`.

**CLI**: `--M`/`--S` (via `cosmo.cli.add_common_arguments`, defaulted to M=855,S=37.8 -> Omega_Lambda_eff~=0.70), `--pantheon-path`, `--z-min`, `--n-bins`, `--output-dir`. Reconfigures stdout/stderr to UTF-8 so Greek labels print on Windows cp1252.

See [../physics/hubble-diagram.md](../physics/hubble-diagram.md) for the physics, offset rationale, and the open (M,S) discrepancy.

## File Locations

| File | Lines | Purpose |
|------|-------|---------|
| `cosmo/cache.py` | 125 | JSON/CSV disk cache |
| `cosmo/constants.py` | 175 | Parameter definitions |
| `cosmo/cli.py` | 95 | CLI argument parsing |
| `cosmo/particles.py` | 340 | Physical structures |
| `cosmo/integrator.py` | 316 | Force calculations + integration |
| `cosmo/simulation.py` | 218 | High-level runner |
| `cosmo/analysis.py` | 382 | Shared analysis utilities |
| `cosmo/visualization.py` | 213 | Shared plotting utilities |
| `cosmo/numba_direct.py` | 82 | Numba JIT O(N²) direct |
| `cosmo/barnes_hut_numba.py` | 200 | Barnes-Hut O(N log N) octree |
| `cosmo/tidal_forces_numba.py` | ~83 | Numba JIT tidal forces (+ node Plummer softening) |
| `cosmo/factories.py` | 120 | Shared simulation functions |
| `cosmo/parameter_sweep.py` | ~760 | Search algorithms, dataclasses, build_cache_name (LIBRARY) |
| `cosmo/distances.py` | 309 | Cosmological distance kernel (H z, d_L, mu) |
| `cosmo/pantheon.py` | 207 | Pantheon+SH0ES loader + plot binning |
| `cosmo/hubble_diagram.py` | 238 | Offset-marginalized chi^2/R^2 engine |
| `run_simulation.py` | 280 | Main comparison script |
| `sweep.py` | ~700 | THE config-driven sweep driver (resumable, cached) |
| `hubble_diagram.py` | 345 | Hubble-diagram vs Pantheon+ script |
| `visualize_3d.py` | 765 | 3D visualization |

## Import Pattern

All scripts import from `cosmo` package:
```python
from cosmo.constants import CosmologicalConstants, LambdaCDMParameters, SimulationParameters
from cosmo.simulation import CosmologicalSimulation
from cosmo.cli import parse_arguments, args_to_sim_params
```

No circular dependencies. Linear dependency chain: constants → cli → particles → integrator → simulation → scripts.
