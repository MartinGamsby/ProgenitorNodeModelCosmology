"""
Parameter sweep infrastructure for cosmological simulations.

Provides reusable search algorithms and configuration for finding optimal
External-Node parameters (M, S, centerM) that match LCDM expansion.
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Any, Optional, Tuple
import numpy as np

from .analysis import compare_expansion_histories, compare_expansion_history


class SearchMethod(Enum):
    """Search algorithm selection for parameter sweep."""
    BRUTE_FORCE = 1
    TERNARY_SEARCH = 2
    LINEAR_SEARCH = 3


@dataclass
class SweepConfig:
    """Configuration for parameter sweep."""
    quick_search: bool = False
    many_search: bool = False
    search_center_mass: bool = True
    t_start_Gyr: float = 3.8
    t_duration_Gyr: float = 10.0
    damping_factor: float = 0.98
    s_min_gpc: int = 15
    s_max_gpc: int = 60
    save_interval: int = 10

    @property
    def particle_count(self) -> int:
        if self.quick_search:
            return 150
        return 1000 if self.many_search else 10000

    @property
    def n_steps(self) -> int:
        return 250 if self.quick_search else 300


@dataclass
class MatchWeights:
    """Weights for computing weighted average match metric."""
    hubble_curve: float = 0.05
    size_curve: float = 0.8
    endpoint: float = 0.1
    max_radius: float = 0.05


@dataclass
class SimResult:
    """Raw simulation output (no computed match metrics)."""
    size_curve_Gpc: np.ndarray
    hubble_curve: np.ndarray
    size_final_Gpc: float
    radius_max_Gpc: float
    a_final: float
    t_Gyr: np.ndarray
    params: Any  # ExternalNodeParameters


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
SimCallback = Callable[[int, int, int, int], SimResult]


def build_m_list(many_search: bool = False) -> List[int]:
    """
    Build list of M values (external node mass factors) to search.

    Returns descending list for optimization (high M searched first).
    Fine increments when many_search=True, coarse otherwise.
    """
    m_list = []
    M = 25
    while M < 500:
        m_list.append(M)
        M += 1 if many_search else 25
    while M < 1000:
        m_list.append(M)
        M += 5 if many_search else 100
    while M < 2000:
        m_list.append(M)
        M += 10 if many_search else 500
    while M < 5000:
        m_list.append(M)
        M += 100 if many_search else 1000
    while M < (100001 if many_search else 25001):
        m_list.append(M)
        M += 1000 if many_search else 5000
    m_list.reverse()  # Search high M first
    return m_list


def build_s_list(s_min: int, s_max: int) -> List[int]:
    """Build list of S values (grid spacing in Gpc) to search."""
    return list(range(s_min, s_max + 1))


def build_center_mass_list(search_center_mass: bool = True, many_search: bool = False) -> List[int]:
    """
    Build list of center node mass values to search.

    Returns [1] if search_center_mass=False.
    Otherwise returns list with fine/coarse increments based on many_search.
    """
    if not search_center_mass:
        return [1]

    center_masses = []
    M = 1
    while M < 5:
        center_masses.append(M)
        M += 1
    while M < 20:
        center_masses.append(M)
        M += 2 if many_search else 5
    while M < 50:
        center_masses.append(M)
        M += 5 if many_search else 10
    while M < 101:
        center_masses.append(M)
        M += 10 if many_search else 25
    return center_masses


def compute_match_metrics(
    sim_result: SimResult,
    baseline: LCDMBaseline,
    weights: MatchWeights
) -> Dict[str, float]:
    """
    Compute match metrics between simulation result and LCDM baseline.

    Uses second half of curves (last 5 Gyr) to focus on late-time acceleration.

    Returns dict with:
        - match_curve_pct: size curve match (R^2 * 100)
        - match_end_pct: endpoint size match
        - match_max_pct: max radius match
        - match_hubble_curve_pct: Hubble parameter curve match
        - match_avg_pct: weighted average of all metrics
        - diff_pct: 100 - match_avg_pct
    """
    half_point = len(baseline.size_Gpc) // 2

    # Compare curves using second half only
    match_curve_pct = compare_expansion_histories(
        sim_result.size_curve_Gpc[half_point:],
        baseline.size_Gpc[half_point:]
    )
    match_end_pct = compare_expansion_history(
        sim_result.size_final_Gpc,
        baseline.size_final_Gpc
    )
    match_max_pct = compare_expansion_history(
        sim_result.radius_max_Gpc,
        baseline.radius_max_Gpc
    )
    match_hubble_curve_pct = compare_expansion_histories(
        sim_result.hubble_curve[half_point:],
        baseline.H_hubble[half_point:]
    )

    # Weighted average
    match_avg_pct = (
        weights.hubble_curve * match_hubble_curve_pct +
        weights.size_curve * match_curve_pct +
        weights.endpoint * match_end_pct +
        weights.max_radius * match_max_pct
    )

    return {
        'match_curve_pct': match_curve_pct,
        'match_end_pct': match_end_pct,
        'match_max_pct': match_max_pct,
        'match_hubble_curve_pct': match_hubble_curve_pct,
        'match_avg_pct': match_avg_pct,
        'diff_pct': 100 - match_avg_pct
    }


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
        'a_ext': sim_result.a_final,
        'size_ext': sim_result.size_final_Gpc,
        'params': sim_result.params,
        **metrics
    }


def ternary_search_S(
    M_factor: int,
    centerM: int,
    sim_callback: SimCallback,
    baseline: LCDMBaseline,
    weights: MatchWeights,
    s_min: int,
    s_max: int,
    s_hint: Optional[int] = None,
    hint_window: int = 10,
    seed: int = 42
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
            sim_result = sim_callback(M_factor, S_val, centerM, seed)
            metrics = compute_match_metrics(sim_result, baseline, weights)
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
    M_factor: int,
    centerM: int,
    sim_callback: SimCallback,
    baseline: LCDMBaseline,
    weights: MatchWeights,
    s_min: int,
    s_max: int,
    prev_best_S: Optional[int] = None,
    seed: int = 42
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

    while i < len(S_list):
        S = S_list[i]

        # Run simulation
        sim_result = sim_callback(M_factor, S, centerM, seed)
        metrics = compute_match_metrics(sim_result, baseline, weights)
        result = _build_result_dict(M_factor, S, centerM, sim_result, metrics)
        all_results.append(result)

        if current_evaluated:
            # Check for negative match
            if result['match_avg_pct'] <= 0:
                break

            # Check for decreasing match (early stopping)
            prev_result = current_evaluated[-1][1]
            if (prev_result['match_curve_pct'] > result['match_curve_pct'] * 1.0001 and
                prev_result['match_avg_pct'] > result['match_avg_pct'] * 1.00025):
                break

            # Adaptive skipping based on match change
            diff = result['match_curve_pct'] - prev_result['match_curve_pct']
            if S > 40:  # Only skip if we have room
                if abs(diff) < 0.002:
                    i += max(1, int(S / 10 * centerM))
                elif diff > 0 and diff < 0.01:
                    i += 2
                elif diff > 0 and diff < 0.02:
                    i += 1
        elif result['match_avg_pct'] <= 0:
            # First result negative, skip ahead
            i += max(1, int(S / 10 * centerM))
            continue  # Don't add to evaluated

        current_evaluated.append((S, result))
        i += 1

    # Find best from evaluated
    if not current_evaluated:
        # No valid results, return placeholder
        return s_min, {'match_avg_pct': 0, 'diff_pct': 100}, True, all_results

    best_S, best_result = max(current_evaluated, key=lambda x: x[1]['match_avg_pct'])

    # Signal to stop M search if we've hit S minimum
    should_stop = (best_S == s_min)

    return best_S, best_result, should_stop, all_results


def brute_force_search(
    m_list: List[int],
    s_list: List[int],
    center_masses: List[int],
    sim_callback: SimCallback,
    baseline: LCDMBaseline,
    weights: MatchWeights,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Exhaustive search over all M x S x centerM combinations.

    Returns list of result dicts for all configurations.
    """
    results: List[Dict[str, Any]] = []

    for centerM in center_masses:
        for M in m_list:
            for S in s_list:
                sim_result = sim_callback(M, S, centerM, seed)
                metrics = compute_match_metrics(sim_result, baseline, weights)
                result = _build_result_dict(M, S, centerM, sim_result, metrics)
                results.append(result)

    return results


def run_sweep(
    config: SweepConfig,
    search_method: SearchMethod,
    sim_callback: SimCallback,
    baseline: LCDMBaseline,
    weights: Optional[MatchWeights] = None,
    seed: int = 42
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

    m_list = build_m_list(config.many_search)
    s_list = build_s_list(config.s_min_gpc, config.s_max_gpc)
    center_masses = build_center_mass_list(config.search_center_mass, config.many_search)

    all_results: List[Dict[str, Any]] = []

    if search_method == SearchMethod.BRUTE_FORCE:
        all_results = brute_force_search(
            m_list, s_list, center_masses,
            sim_callback, baseline, weights, seed
        )

    elif search_method == SearchMethod.TERNARY_SEARCH:
        for centerM in center_masses:
            prev_best_S = None
            for M in m_list:
                S_best, _, _, results = ternary_search_S(
                    M, centerM, sim_callback, baseline, weights,
                    config.s_min_gpc,
                    prev_best_S if prev_best_S else config.s_max_gpc,
                    s_hint=prev_best_S,
                    hint_window=(prev_best_S // 4) if prev_best_S else (config.s_max_gpc // 4),
                    seed=seed
                )
                all_results.extend(results)
                prev_best_S = S_best

                if S_best == config.s_min_gpc or S_best == config.s_max_gpc:
                    if S_best == config.s_min_gpc:
                        break

    elif search_method == SearchMethod.LINEAR_SEARCH:
        for centerM in center_masses:
            prev_best_S = None
            for M in m_list:
                best_S, _, should_stop, results = linear_search_S(
                    M, centerM, sim_callback, baseline, weights,
                    config.s_min_gpc,
                    prev_best_S if prev_best_S else config.s_max_gpc,
                    prev_best_S=prev_best_S,
                    seed=seed
                )
                all_results.extend(results)

                if prev_best_S == config.s_min_gpc:
                    break
                prev_best_S = best_S

    return all_results
