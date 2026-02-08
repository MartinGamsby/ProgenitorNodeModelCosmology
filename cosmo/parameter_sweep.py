"""
Parameter sweep infrastructure for cosmological simulations.

Provides reusable search algorithms and configuration for finding optimal
External-Node parameters (M, S, centerM) that match LCDM expansion.
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Any, Optional, Tuple
from .visualization import generate_output_filename
from .cache import Cache, CacheType
from cosmo.constants import SimulationParameters
import numpy as np

from .analysis import compare_expansion_histories, compare_expansion_history

# Canonical list of individual match metric keys (excludes derived match_avg_pct / diff_pct).
# Used by compute_match_metrics return dict, early-stop checks, and CSV columns.
MATCH_METRIC_KEYS = (
    'match_curve_pct',
    'match_curve_r2',
    'match_curve_rmse_pct',
    'match_half_curve_pct',
    'match_half_rmse_pct',

    'match_max_pct',
    'match_curve_error_pct',
    'match_curve_error_max',

    'match_hubble_curve_pct',
    'match_hubble_curve_r2',
    'match_hubble_rmse_pct',
    'match_hubble_half_curve_pct',
    'match_hubble_half_rmse_pct',

    'match_end_pct',
    'match_hubble_end_pct',    
)

USED_MATCH_METRIC_KEYS = (
    #'match_half_curve_pct',
    'match_half_rmse_pct',
    'match_half_rmse_pct',
    'match_half_rmse_pct',
    'match_end_pct',
    'match_end_pct',
    'match_end_pct',
    #'match_hubble_half_curve_pct',
    #'match_hubble_half_rmse_pct',
    #'match_hubble_curve_r2',
    #'match_hubble_end_pct',
    'match_curve_error_pct',
    'match_curve_error_max',
)
#USED_MATCH_METRIC_KEYS = USED_MATCH_METRIC_KEYS

CSV_COLUMNS = (
    ['M_factor', 'S_gpc', 'centerM', 'match_avg_pct', 'diff_pct']
    + list(MATCH_METRIC_KEYS)
    + ['a_ext', 'size_ext', 'desc']
)

CACHE = None
SKIP_CACHE = False


class SearchMethod(Enum):
    """Search algorithm selection for parameter sweep."""
    BRUTE_FORCE = 1
    TERNARY_SEARCH = 2
    LINEAR_SEARCH = 3


@dataclass
class SweepConfig:
    """Configuration for parameter sweep."""
    quick_search: bool = False
    many_search: int = 3
    search_center_mass: bool = True
    t_start_Gyr: float = 5.8
    t_duration_Gyr: float = 8.0
    damping_factor: float = None
    s_min_gpc: int = 15
    s_max_gpc: int = 60
    save_interval: int = 10

    @property
    def particle_count(self) -> int:
        if self.quick_search:
            return 200
        return 2000

    @property
    def n_steps(self) -> int:
        return 250 if self.quick_search else 300


@dataclass
class MatchWeights:
    """Weights for computing weighted average match metric."""
    hubble_half_curve: float = 0.025
    hubble_curve: float = 0.025
    size_half_curve: float = 0.25
    size_curve: float = 0.2
    endpoint: float = 0.4
    max_radius: float = 0.1

@dataclass
class SimSimpleResult:
    size_final_Gpc: float
    radius_max_Gpc: float
    a_final: float

@dataclass
class SimResult:
    """Raw simulation output (no computed match metrics)."""
    size_curve_Gpc: np.ndarray
    hubble_curve: np.ndarray
    t_Gyr: np.ndarray
    params: Any  # ExternalNodeParameters
    results: SimSimpleResult


@dataclass
class LCDMBaseline:
    """Precomputed LCDM reference data for comparison."""
    t_Gyr: np.ndarray
    size_Gpc: np.ndarray
    H_hubble: np.ndarray
    size_final_Gpc: float
    radius_max_Gpc: float
    a_final: float


# Type alias for simulation callback
# Signature: (M_factor, S_gpc, centerM, seed) -> SimResult
SimCallback = Callable[[int, int, int, List[int]], List[SimResult]]

def generate_increments(max_value, terms_per_decade=5, min_value=1):
    """
    Generates a sequence of increasing "nice" round numbers using a geometric progression
    based on the Renard series concept. The terms_per_decade parameter controls how many
    increments there are roughly per order of magnitude (higher value means more increments,
    slower growth). The min_value parameter sets the starting minimum value—only values >= min_value are included.
    
    This is a mathematical algorithm without hardcoded segment lists—just three parameters.
    It produces sequences with round, human-friendly numbers, growing more gradually than
    Fibonacci or simple powers.
    """
    seq = []
    k = 0
    added_something = True
    while added_something:
        added_something = False
        for i in range(terms_per_decade):
            mantissa = 10 ** (i / terms_per_decade)
            value = round(mantissa) * (10 ** k)
            if value > max_value:
                break
            elif value >= min_value and value not in seq and value > 0:
                seq.append(int(value))
                added_something = True
            elif value < min_value:
                added_something = True
        k += 1
    return sorted(seq)  # Ensure sorted, though usually already is


def add_mid_values(input_list: list):
    result = []
    for i in range(len(input_list)):
        result.append(input_list[i])
        
        # If there's a next element, calculate and add the midpoint
        if i < len(input_list) - 1:
            midpoint = (input_list[i] + input_list[i + 1]) // 2
            result.append(midpoint)
    return sorted(list(set(result)))

def build_m_list(many_search: int = 3, multiplier=1) -> List[int]:
    """
    Build list of M values (external node mass factors) to search.

    Returns descending list for optimization (high M searched first).
    Fine increments when many_search=True, coarse otherwise.
    """
    m_list = generate_increments(25000*multiplier, terms_per_decade=many_search, min_value=20)
    for i in range(10):
        if many_search > 10*i:
            m_list = add_mid_values(m_list)
    m_list.reverse()  # Search high M first
    return m_list


def build_s_list(s_min: int, s_max: int) -> List[int]:
    """Build list of S values (grid spacing in Gpc) to search."""
    return list(range(s_min, s_max + 1))


def build_center_mass_list(search_center_mass: bool = True, many_search: int = 3) -> List[int]:
    """
    Build list of center node mass values to search.

    Returns [1] if search_center_mass=False.
    Otherwise returns list with fine/coarse increments based on many_search.
    """
    if not search_center_mass:
        return [1]

    center_masses = generate_increments(1000, terms_per_decade=many_search, min_value=1)
    for i in range(10):
        if many_search > 10*i:
            center_masses = add_mid_values(center_masses)
    return center_masses

def compute_avg(metrics):
    # Multiplicative aggregate: product of all metric values (clamped to [0,1])
    match_avg_pct = 1.0
    for key in USED_MATCH_METRIC_KEYS:
        match_avg_pct *= max(0.0, min(1.0, metrics[key] / 100))
    match_avg_pct *= 100
    return match_avg_pct

def compute_match_metrics(
    sim_result: SimResult,
    baseline: LCDMBaseline,
    weights: MatchWeights
) -> Dict[str, float]:
    """
    Compute match metrics between simulation result and LCDM baseline.

    Computes both full curve and half curve (last 5 Gyr) metrics.

    Returns dict with:
        - match_curve_pct: full size curve match (R^2 * 100)
        - match_half_curve_pct: second-half size curve match
        - match_end_pct: endpoint size match
        - match_max_pct: max radius match
        - match_hubble_curve_pct: full Hubble parameter curve match
        - match_hubble_half_curve_pct: second-half Hubble curve match
        - match_avg_pct: weighted average of all metrics
        - diff_pct: 100 - match_avg_pct
    """
    
    half_point = len(baseline.size_Gpc) // 2

    # Full curve comparisons
    match_curve_diagnostics = compare_expansion_histories(
        sim_result.size_curve_Gpc,
        baseline.size_Gpc,
        return_diagnostics=True
    )
    match_hubble_diagnostics = compare_expansion_histories(
        sim_result.hubble_curve,
        baseline.H_hubble,
        return_diagnostics=True
    )

    # Half curve comparisons (second half only, late-time acceleration)
    match_half_curve_diagnostics = compare_expansion_histories(
        sim_result.size_curve_Gpc[half_point:],
        baseline.size_Gpc[half_point:],
        return_diagnostics=True
    )
    match_hubble_half_curve_diagnostics = compare_expansion_histories(
        sim_result.hubble_curve[half_point:],
        baseline.H_hubble[half_point:],
        return_diagnostics=True
    )

    # Endpoint comparisons
    match_end_pct = compare_expansion_history(
        sim_result.results.size_final_Gpc,
        baseline.size_final_Gpc
    )
    match_hubble_end_pct = compare_expansion_history(
        sim_result.hubble_curve[-1],
        baseline.H_hubble[-1]
    )

    # TODO: Do something better: (Right now: 5% buffer)
    match_end_pct = match_end_pct if (baseline.size_final_Gpc > sim_result.results.size_final_Gpc) else min(100.0, match_end_pct+5)
    match_max_pct = compare_expansion_history(
        sim_result.results.radius_max_Gpc,
        baseline.radius_max_Gpc
    )

    # Build metrics dict from MATCH_METRIC_KEYS
    metrics = {
        'match_curve_pct': match_curve_diagnostics['match_pct'],
        'match_curve_rmse_pct': 100 - match_curve_diagnostics['rmse_pct'],
        'match_curve_r2': match_curve_diagnostics['r_squared'],
        'match_half_curve_pct': match_half_curve_diagnostics['match_pct'],
        'match_half_rmse_pct': 100 - match_half_curve_diagnostics['rmse_pct'],
        'match_hubble_curve_pct': match_hubble_diagnostics['match_pct'],
        'match_hubble_curve_r2': match_hubble_diagnostics['r_squared'],
        'match_hubble_rmse_pct': 100 - match_hubble_diagnostics['rmse_pct'],
        'match_hubble_half_curve_pct': match_hubble_half_curve_diagnostics['match_pct'],
        'match_hubble_half_rmse_pct': 100 - match_hubble_half_curve_diagnostics['rmse_pct'],
        'match_end_pct': match_end_pct,
        'match_hubble_end_pct': match_hubble_end_pct,
        'match_max_pct': match_max_pct,
        'match_curve_error_pct': 100 - match_curve_diagnostics['mean_error_pct'],
        'match_curve_error_max': 100 - match_curve_diagnostics['max_error_pct'],
    }

    match_avg_pct = compute_avg(metrics)

    metrics['match_avg_pct'] = match_avg_pct
    metrics['diff_pct'] = 100 - match_avg_pct
    return metrics


def _build_result_dict(
    M_factor: int,
    S_gpc: int,
    centerM: int,
    sim_result: SimResult,
    metrics: Dict[str, float]
) -> Dict[str, Any]:
    """Build full result dictionary combining simulation output and metrics."""
    return {
        'M_factor': M_factor,
        'S_gpc': S_gpc,
        'centerM': centerM,
        'desc': f"M={M_factor}, S={S_gpc}, centerM={centerM}",
        'a_ext': sim_result.results.a_final,
        'size_ext': sim_result.results.size_final_Gpc,
        'params': sim_result.params,
        **metrics
    }


def worst_callback(sim_callback, config, M_factor, S_val, centerM, seeds, baseline, weights):
    
    parts = []
    parts.append(f"{config.particle_count}p")
    parts.append(f"{config.t_start_Gyr}-{config.t_duration_Gyr+config.t_start_Gyr}Gyr")
    parts.append(f"{M_factor}M")
    parts.append(f"{int(centerM)}centerM")
    parts.append(f"{S_val}S")
    parts.append(f"{config.n_steps}steps")
    parts.append(f"{'_'.join([str(seed) for seed in seeds])}seeds")
    # No mass randomize??
    if config.damping_factor:
        parts.append(f"{config.damping_factor}d")
    cache_name =  "_".join(parts)

    cache_filename = f"metrics_{config.particle_count}"
    global CACHE
    if not not SKIP_CACHE:
        if not CACHE or CACHE.name != cache_filename:
            CACHE = Cache(cache_filename)

        cached_metrics = CACHE.get_cached_value(cache_name, CacheType.METRICS)
        if cached_metrics:
            has_all_keys = True
            for key in USED_MATCH_METRIC_KEYS:
                if not key in cached_metrics:
                    has_all_keys = False
                    break
            if has_all_keys:
                cached_results = CACHE.get_cached_value(cache_name, CacheType.RESULTS)
                print(f"Using cache for {cache_name}")

                new_avg = compute_avg(cached_metrics)
                if cached_metrics['match_avg_pct'] != new_avg:
                    print(f"Updating avg: from {cached_metrics['match_avg_pct']} to {new_avg}")
                    cached_metrics['match_avg_pct'] = new_avg
                    CACHE.add_cached_value(cache_name, CacheType.METRICS, cached_metrics, save_interval_s=100)
                # cached_results may be a dict (from JSON) or SimSimpleResult (in-memory)
                if isinstance(cached_results, dict):
                    results = SimSimpleResult(
                        size_final_Gpc=cached_results['size_final_Gpc'],
                        radius_max_Gpc=cached_results['radius_max_Gpc'],
                        a_final=cached_results['a_final'],
                    )
                else:
                    results = cached_results
                return SimResult(
                    size_curve_Gpc=None,
                    hubble_curve=None,
                    t_Gyr=None,
                    params=None,
                    results=results,
                ), cached_metrics


            
        
    sim_results = sim_callback(M_factor, S_val, centerM, seeds)

    worst_result = None
    worst_metrics = None
    for result in sim_results:
        metrics = compute_match_metrics(result, baseline, weights)
        if not worst_result:
            worst_result = result
            worst_metrics = metrics
        elif metrics['match_avg_pct'] < worst_metrics['match_avg_pct']:
            worst_result = result
            worst_metrics = metrics

    if not not SKIP_CACHE:
        CACHE.add_cached_value(cache_name, CacheType.RESULTS, worst_result.results, save_interval_s=100)
        CACHE.add_cached_value(cache_name, CacheType.METRICS, worst_metrics)
    return worst_result, worst_metrics

def ternary_search_S(
    config: SweepConfig,
    M_factor: int,
    centerM: int,
    sim_callback: SimCallback,
    baseline: LCDMBaseline,
    weights: MatchWeights,
    s_min: int,
    s_max: int,
    s_hint: Optional[int] = None,
    hint_window: int = 10,
    seeds: List[int] = [42]
) -> Tuple[int, float, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Ternary search for optimal S given fixed M.

    Assumes unimodal (bell curve) match quality over S space.

    Args:
        M_factor: External node mass factor
        centerM: Center node mass factor
        sim_callback: Callback to run simulation
        baseline: LCDM baseline for comparison
        weights: Match metric weights
        s_min: Minimum S value to search
        s_max: Maximum S value to search
        s_hint: Previous best S (warm start)
        hint_window: Search within +/- hint_window of s_hint first
        seed: Random seed for simulations

    Returns:
        (best_S, best_match_pct, best_result_dict, all_results)
    """
    evaluated: Dict[int, Tuple[SimResult, Dict[str, float]]] = {}
    all_results: List[Dict[str, Any]] = []

    def evaluate_S(S_val: int) -> float:
        """Evaluate and cache simulation result for given S."""
        S_val = round(S_val)
        if S_val not in evaluated:
            sim_result, metrics = worst_callback(sim_callback, config, M_factor, S_val, centerM, seeds, baseline, weights)

            evaluated[S_val] = (sim_result, metrics)
            result_dict = _build_result_dict(M_factor, S_val, centerM, sim_result, metrics)
            all_results.append(result_dict)
        return evaluated[S_val][1]['match_avg_pct']

    # Warm start: search locally around hint first
    if s_hint is not None:
        begin = max(s_min, s_hint - hint_window)
        end = min(s_max, s_hint + hint_window)
    else:
        begin = s_min
        end = s_max

    # Ternary search
    while end - begin > 3:
        low = (begin * 2 + end) // 3
        high = (begin + end * 2) // 3

        if evaluate_S(low) > evaluate_S(high):
            end = high - 1
        else:
            begin = low + 1

    # Exhaustively check remaining small range
    for S_val in range(begin, end + 1):
        evaluate_S(S_val)

    # Find best result
    best_S = max(evaluated.keys(), key=lambda s: evaluated[s][1]['match_avg_pct'])
    sim_result, metrics = evaluated[best_S]
    best_result = _build_result_dict(M_factor, best_S, centerM, sim_result, metrics)
    best_match = metrics['match_avg_pct']

    return best_S, best_match, best_result, all_results


def linear_search_S(
    config: SweepConfig,
    M_factor: int,
    centerM: int,
    sim_callback: SimCallback,
    baseline: LCDMBaseline,
    weights: MatchWeights,
    s_min: int,
    s_max: int,
    prev_best_S: Optional[int] = None,
    seeds: List[int] = [42]
) -> Tuple[int, Dict[str, Any], bool, List[Dict[str, Any]]]:
    """
    Linear search for optimal S given fixed M.

    Searches from prev_best_S (or s_max) downward with adaptive skipping.
    Stops early when match starts decreasing.

    Args:
        M_factor: External node mass factor
        centerM: Center node mass factor
        sim_callback: Callback to run simulation
        baseline: LCDM baseline for comparison
        weights: Match metric weights
        s_min: Minimum S value to search
        s_max: Maximum S value to search
        prev_best_S: Previous best S (search starts here, going down)
        seed: Random seed for simulations

    Returns:
        (best_S, best_result_dict, should_stop_M_search, all_results)
    """
    S_start = prev_best_S if prev_best_S else s_max
    current_evaluated: List[Tuple[int, Dict[str, Any]]] = []
    all_results: List[Dict[str, Any]] = []

    S_list = list(range(S_start, s_min - 1, -1))
    i = 0

    prev_result = None
    while i < len(S_list):
        S = S_list[i]

        # Run simulation
        sim_result, metrics = worst_callback(sim_callback, config, M_factor, S, centerM, seeds, baseline, weights)
        result = _build_result_dict(M_factor, S, centerM, sim_result, metrics)
        all_results.append(result)

        if current_evaluated:
            prev_result = current_evaluated[-1][1]

        print(f"\tMatch: {result['match_avg_pct']:.2f}% (curve {result['match_curve_pct']:.2f}%, half {result['match_half_curve_pct']:.2f}%, end {result['match_end_pct']:.2f}%, radius {result['match_max_pct']:.2f}%, Hubble {result['match_hubble_curve_pct']:.2f}%)")
        if prev_result:
            diff = result['match_half_curve_pct'] - prev_result['match_half_curve_pct']
            print(f"\tHalf Curve Match: {result['match_half_curve_pct']:.4f}, CHANGE: {diff:.4f}%")
        else:
            print(f"\tHalf Curve Match: {result['match_half_curve_pct']:.4f}")
            
        print("\n")
        if current_evaluated:
            # Check for negative match
            if result['match_avg_pct'] <= 0:
                print("\r\tMatch below 0%, stopping search for this M.")
                break

            all_worse = True
            for key in USED_MATCH_METRIC_KEYS:
                #print(key, prev_result[key] > result[key] * 1.00025, prev_result[key], result[key])
                if prev_result[key] < result[key] * 1.00025:
                    all_worse = False
                    break
            if all_worse:
                print("\r\tMatch decreasing > 0.025%, stopping search for this M.", end="")
                current_evaluated.append((S, result))
                break

                
            #exit(1)

            # Adaptive skipping based on match change
            if S > 40:  # Only skip if we have room
                if abs(diff) < 0.002:
                    print("\r\tMatch change < 0.002%, skipping S/10 S.", end="")
                    i += max(1, int(S / 10))
                elif diff > 0 and diff < 0.01:
                    print("\r\tMatch change < 0.01%, skipping 2 S.", end="")
                    i += 2
                elif diff > 0 and diff < 0.02:
                    print("\r\tMatch change < 0.02%, skipping 1 S.", end="")
                    i += 1
        elif result['match_avg_pct'] <= 0:
            # First result negative, skip ahead
            print("\r\tMatch below 0%, trying to find a better one, skipping S/10 S.", end="")
            i += max(1, int(S / 10 ))
            continue  # Don't add to evaluated

        current_evaluated.append((S, result))
        i += 1

    # Find best from evaluated
    if not current_evaluated:
        # No valid results, return placeholder
        return None, {'match_avg_pct': 0, 'diff_pct': 100}, True, all_results

    best_S, best_result = max(current_evaluated, key=lambda x: x[1]['match_avg_pct'])

    # Signal to stop M search if we've hit S minimum
    should_stop = (best_S == s_min)
    return best_S, best_result, should_stop, all_results


def brute_force_search(
    config: SweepConfig,
    many_search: int,
    s_list: List[int],
    center_masses: List[int],
    sim_callback: SimCallback,
    baseline: LCDMBaseline,
    weights: MatchWeights,
    seeds: List[int] = [42]
) -> List[Dict[str, Any]]:
    """
    Exhaustive search over all M x S x centerM combinations.

    Returns list of result dicts for all configurations.
    """
    results: List[Dict[str, Any]] = []

    for centerM in center_masses:
        m_list = build_m_list(many_search, multiplier=many_search)#centerM)
        for M in m_list:
            for S in s_list:
                sim_result, metrics = worst_callback(sim_callback, config, M, S, centerM, seeds, baseline, weights)
                result = _build_result_dict(M, S, centerM, sim_result, metrics)
                results.append(result)

    return results


def run_sweep(
    config: SweepConfig,
    search_method: SearchMethod,
    sim_callback: SimCallback,
    baseline: LCDMBaseline,
    weights: Optional[MatchWeights] = None,
    seeds = [42,123]
) -> List[Dict[str, Any]]:
    """
    Run parameter sweep using specified search method.

    Args:
        config: Sweep configuration
        search_method: Search algorithm to use
        sim_callback: Callback to run simulations
        baseline: LCDM baseline for comparison
        weights: Match metric weights (uses defaults if None)
        seed: Random seed for simulations

    Returns:
        List of result dicts for all evaluated configurations
    """
    if weights is None:
        weights = MatchWeights()

    s_list = build_s_list(config.s_min_gpc, config.s_max_gpc)
    center_masses = build_center_mass_list(config.search_center_mass, config.many_search)

    all_results: List[Dict[str, Any]] = []

    if search_method == SearchMethod.BRUTE_FORCE:
        all_results = brute_force_search(
            config, config.many_search, s_list, center_masses,
            sim_callback, baseline, weights, seeds
        )

    elif search_method == SearchMethod.TERNARY_SEARCH:
        for centerM in center_masses:
            prev_best_S = None            
            m_list = build_m_list(config.many_search, multiplier=config.many_search)#centerM)
            for M in m_list:
                S_best, _, _, results = ternary_search_S(
                    config, M, centerM, sim_callback, baseline, weights,
                    config.s_min_gpc,
                    prev_best_S if prev_best_S else config.s_max_gpc,
                    s_hint=prev_best_S,
                    hint_window=(prev_best_S // 4) if prev_best_S else (config.s_max_gpc // 4),
                    seeds=seeds
                )
                all_results.extend(results)
                prev_best_S = S_best

                if S_best == config.s_min_gpc or S_best == config.s_max_gpc:
                    if S_best == config.s_min_gpc:
                        break

    elif search_method == SearchMethod.LINEAR_SEARCH:
        for centerM in center_masses:
            prev_best_S = None
            m_list = build_m_list(config.many_search, multiplier=config.many_search)#centerM)
            for M in m_list:
                best_S, _, should_stop, results = linear_search_S(
                    config, M, centerM, sim_callback, baseline, weights,
                    config.s_min_gpc,
                    prev_best_S if prev_best_S else config.s_max_gpc,
                    prev_best_S=prev_best_S,
                    seeds=seeds
                )
                all_results.extend(results)

                if prev_best_S == config.s_min_gpc:
                    break
                prev_best_S = best_S
    global CACHE
    del CACHE
    CACHE = None
    return all_results
