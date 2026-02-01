# Parameter Sweep

Grid search over M (mass factor), S (spacing), and centerM (center node mass) to find optimal LCDM match.

## Architecture

Logic split between:
- `cosmo/parameter_sweep.py` - reusable search algorithms, dataclasses, parameter builders
- `cosmo/factories.py` - shared simulation functions (solve_lcdm_baseline, run_external_node_simulation)
- `parameter_sweep.py` - script handling callback wiring, output formatting
- `run_simulation.py` - single-run CLI using same shared functions

Both `parameter_sweep.py` and `run_simulation.py` use shared functions from `cosmo/factories.py` to ensure identical simulation behavior.

## Core Types (cosmo/parameter_sweep.py)

```python
class SearchMethod(Enum):
    BRUTE_FORCE = 1
    TERNARY_SEARCH = 2
    LINEAR_SEARCH = 3

@dataclass
class SweepConfig:
    quick_search: bool = False      # 150 particles, 250 steps
    many_search: bool = False       # 1000 particles (vs 10000)
    search_center_mass: bool = True # search 3D: M x S x centerM
    t_start_Gyr: float = 3.8
    t_duration_Gyr: float = 10.0
    damping_factor: float = None
    s_min_gpc: int = 15
    s_max_gpc: int = 60             # 100 if many_search
    save_interval: int = 10

@dataclass
class MatchWeights:
    hubble_half_curve: float = 0.025
    hubble_curve: float = 0.025
    size_half_curve: float = 0.4
    size_curve: float = 0.4
    endpoint: float = 0.1
    max_radius: float = 0.05

@dataclass
class SimResult:
    """Raw simulation output passed from callback"""
    size_curve_Gpc: np.ndarray
    hubble_curve: np.ndarray
    size_final_Gpc: float
    radius_max_Gpc: float
    a_final: float
    t_Gyr: np.ndarray
    params: Any

@dataclass
class LCDMBaseline:
    """Precomputed LCDM reference data"""
    t_Gyr: np.ndarray
    size_Gpc: np.ndarray
    H_hubble: np.ndarray
    size_final_Gpc: float
    radius_max_Gpc: float
    a_final: float

SimCallback = Callable[[int, int, int, int], SimResult]  # (M, S, centerM, seed)
```

## Parameter Space

**M (mass factor)**: 25 -> 25,000 x M_obs (100,000 if many_search)
- Fine increments near low M, coarse at high M
- List built descending (high->low) for optimization

**S (spacing)**: 15 -> 60 Gpc (100 if many_search)
- Integer increments

**centerM (center node mass)**: 1 -> 100 x M_obs
- Searched when search_center_mass=True

## Search Methods

### LINEAR_SEARCH (default)
For each M (descending), sweep S from previous best downward:
1. Start at S_max = prev_best_S (or s_max_gpc initially)
2. Evaluate S, S-1, S-2... until match decreases
3. Skip S values when match change <0.002% (adaptive stepping)
4. Stop M search when best S reaches minimum

Optimization: ~10-50x fewer evaluations than brute force

### TERNARY_SEARCH
Assumes unimodal match quality over S. Warm-starts from previous best S.
Faster than linear but may miss local optima.

### BRUTE_FORCE
Exhaustive grid: all M x all S x all centerM. Most thorough, slowest.

## Match Metric

Weighted average using both full curves and half curves (last 5 Gyr):
```python
match_avg = (hubble_half_curve * 0.025) + (hubble_curve * 0.025) +
            (size_half_curve * 0.4) + (size_curve * 0.4) +
            (endpoint * 0.1) + (max_radius * 0.05)
```

Metrics returned by `compute_match_metrics(sim_result, baseline, weights)`:
- `match_curve_pct`: Full size curve R² match
- `match_half_curve_pct`: Second-half size curve R² match
- `match_hubble_curve_pct`: Full Hubble curve R² match
- `match_hubble_half_curve_pct`: Second-half Hubble curve R² match
- `match_end_pct`: Endpoint size match
- `match_max_pct`: Max radius match
- `match_avg_pct`: Weighted average
- `diff_pct`: 100 - match_avg_pct

## Workflow

```mermaid
graph TD
    A[calculate_initial_conditions] --> B[solve_friedmann_at_times for LCDM]
    B --> C[Create LCDMBaseline]
    C --> D[run_sweep with sim_callback]
    D --> E[For each centerM]
    E --> F[For each M descending]
    F --> G[Search S space via method]
    G --> H[sim_callback: run simulation]
    H --> I[compute_match_metrics]
    I --> J{Early stopping?}
    J -->|No| G
    J -->|Yes| F
    F --> K[Collect all results]
    K --> L[Sort, display, save best_config.pkl]
```

## Key Functions

**cosmo/parameter_sweep.py:**
- `build_m_list(many_search)` - returns descending M values
- `build_s_list(s_min, s_max)` - returns S range
- `build_center_mass_list(search_center_mass, many_search)` - returns centerM values
- `compute_match_metrics(sim_result, baseline, weights)` - returns match dict
- `ternary_search_S(...)` - ternary search for optimal S
- `linear_search_S(...)` - linear search with early stopping
- `brute_force_search(...)` - exhaustive evaluation
- `run_sweep(config, method, callback, baseline, weights)` - main entry

**parameter_sweep.py:**
- `sim_callback(M, S, centerM, seed)` - runs real simulation, returns SimResult

## Testing

`tests/test_parameter_sweep.py` - 37 tests using dummy callbacks:
- Parameter space builders
- Match metric computation
- Search algorithm correctness with unimodal callbacks
- Early stopping, adaptive skipping, boundary handling

Dummy callbacks create SimResult with predictable quality based on distance from optimal point, enabling search algorithm testing without real simulations.

## Output

**Console**: Progress updates, match percentages per config

**Files**:
- `results/sweep_results.csv` - all evaluated configurations
- `results/sweep_best_per_S.csv` - best (M, centerM) for each S value

CSV columns:
- M_factor, S_gpc, centerM
- match_avg_pct, diff_pct, match_curve_pct, match_half_curve_pct, match_end_pct, match_max_pct
- match_hubble_curve_pct, match_hubble_half_curve_pct
- a_ext, size_ext, desc

## Usage

```bash
python parameter_sweep.py
```

Edit script constants (SEARCH_METHOD, QUICK_SEARCH, MANY_SEARCH, SEARCH_CENTER_MASS) to change behavior.

## Best Known Configurations

| M x M_obs | S (Gpc) | Match% | Notes |
|-----------|---------|--------|-------|
| 855 | 25 | 99.4% | R^2>0.89 expansion rate |
| 97000 | 65 | 99%+ | High-mass solution |
| 69 | 15 | 99%+ | Low-mass solution |
