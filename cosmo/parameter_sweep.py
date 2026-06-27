"""
Parameter sweep infrastructure for cosmological simulations.

Provides reusable search algorithms and configuration for finding optimal
External-Node parameters (M, S, centerM) that match LCDM expansion.
"""
from enum import Enum
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, List, Dict, Any, Optional, Tuple
from .visualization import generate_output_filename
from .cache import Cache, CacheType
from cosmo.constants import SimulationParameters
import numpy as np

from .analysis import (
    compare_expansion_histories, compare_expansion_history, solve_friedmann_at_times
)

# Tolerance on the physical expansion anchor (see expected_growth_factor). A from-sim
# config must reproduce the real total expansion a(today)/a(t_start) to within this
# fractional tolerance to be physically admissible; otherwise its renormalized shape
# could fit the SN window while predicting a nonsensical expansion history.
GROWTH_ANCHOR_TOL = 0.20


@lru_cache(maxsize=16)
def expected_growth_factor(t_start_Gyr: float, t_today_Gyr: float = 13.8) -> float:
    """Physical scale-factor growth a(today)/a(t_start) from the LCDM background.

    A from-sim External-Node model normalizes a=1 at t_start and is compared to the
    SNe only over the observed z-range; nothing in that comparison forces its TOTAL
    expansion to be physical. This anchor supplies the missing constraint: over
    [t_start, today] the universe expands by 1+z(t_start) (~3.2x for t_start=2.9 Gyr),
    so the model's a_curve[-1]/a_curve[0] must match that. Runaway configs (e.g. huge
    M expanding thousands-fold) violate it and are rejected.
    """
    res = solve_friedmann_at_times(np.array([float(t_start_Gyr), float(t_today_Gyr)]))
    a = res['a']
    return float(a[-1] / a[0])

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
    'match_curve_r2',
    'match_curve_rmse_pct',
    'match_end_pct',
    'match_hubble_curve_r2',
    'match_hubble_rmse_pct',
    #'match_curve_error_pct',
    #'match_curve_error_max',
)

#USED_MATCH_METRIC_KEYS = (
#    #'match_half_curve_pct',
#    'match_half_rmse_pct',
#    'match_half_rmse_pct',
#    'match_half_rmse_pct',
#    'match_half_rmse_pct',
#    'match_end_pct',
#    'match_end_pct',
#    'match_end_pct',
#    'match_end_pct',
#    'match_curve_error_pct',
#    'match_curve_error_max',
#)

USED_MATCH_METRIC_KEYS = (
    #'match_curve_pct',
    'match_curve_r2',
    'match_curve_rmse_pct',
    #'match_half_curve_pct',
    'match_half_rmse_pct',

    #'match_max_pct',
    #'match_curve_error_pct',
    #'match_curve_error_max',

    #'match_hubble_curve_pct',
    'match_hubble_curve_r2',
    'match_hubble_rmse_pct',
    #'match_hubble_half_curve_pct',
    'match_hubble_half_rmse_pct',

    'match_hubble_end_pct',
    'match_end_pct',

)
#USED_MATCH_METRIC_KEYS = (
#    #'match_curve_pct',
#    'match_curve_r2',
#    #'match_curve_rmse_pct',
#    #'match_half_rmse_pct',
#
#    'match_hubble_curve_r2',
#    #'match_hubble_rmse_pct',
#    #'match_hubble_half_rmse_pct',
#
#    'match_hubble_end_pct',
#    'match_end_pct',
#)
USED_MATCH_METRIC_KEYS = MATCH_METRIC_KEYS
#for i in USED_MATCH_METRIC_KEYS:
#    # Add to tuple: 'match_end_pct', 'match_hubble_end_pct' multiple times to increase their weight in the average
#    USED_MATCH_METRIC_KEYS += ('match_hubble_end_pct',)

#for i in range(5):
#    USED_MATCH_METRIC_KEYS += ('match_hubble_end_pct',)
#
#for i in range(10//5):
#    USED_MATCH_METRIC_KEYS += ('match_end_pct',)


CSV_COLUMNS = (
    ['M_factor', 'S_gpc', 'centerM', 'match_avg_pct', 'diff_pct']
    + list(MATCH_METRIC_KEYS)
    + ['a_ext', 'size_ext', 'desc']
)

CACHE = None
SKIP_CACHE = False

# ---------------------------------------------------------------------------
# Physics-version cache token
# ---------------------------------------------------------------------------
# The metrics/results cache is keyed on PARAMETERS only. That is unsafe across a
# physics change: if the simulation's a(t) is computed differently for the SAME
# (M, S, centerM, particles, steps, seeds, objective) tuple, an old cache entry
# would be silently reused and return STALE numbers.
#
# PHYSICS_CACHE_VERSION is a manually-bumped token that MUST be incremented
# whenever a change alters the from-sim a(t) for a fixed parameter tuple (initial
# conditions, force law, the pre-start boost, integrator, softening, etc.). It is
# appended to every cache key (via build_cache_name), so bumping it makes all
# pre-change entries unreachable — old and new physics can never collide.
#
# History (bump + one-line reason; keep newest last):
#   v1  legacy calibrated-velocity ICs (pre-EdS-consistent).
#   v2  EdS-consistent ICs (M=0 == EdS critical mass + H_EdS Hubble flow) and the
#       pre-t_start HMEA tidal velocity boost. Both change a(t) for fixed params,
#       so every v1 entry is stale under the current defaults.
#   v3  centerM repurposed as an outer-MASS multiplier (extra Big-Bang matter outside
#       the observable sphere) + softening frozen at the centerM=1 baseline; a(t) for
#       any centerM>1 differs from v2's softening-only centerM, so all v2 entries are
#       stale. centerM=1 a(t) is byte-identical but the version token bump retires all
#       old entries uniformly (safe).
PHYSICS_CACHE_VERSION = "v3"


def physics_cache_token(config) -> str:
    """Return the per-config physics-version slug appended to every cache key.

    Combines the manually-bumped PHYSICS_CACHE_VERSION with the physics-affecting
    flags that the sweep does NOT otherwise encode (eds_consistent,
    pre_start_tidal_boost). Those flags default to True (current physics) but a
    config may override them; encoding them means a legacy (eds_consistent=False)
    run gets a DISTINCT key from a current run with the same parameters, so the
    two never share a cache entry even within the same PHYSICS_CACHE_VERSION.

    Format: ``phys<VERSION>[<flags>]`` where <flags> is a compact suffix that only
    appears when a flag deviates from the current-physics default, e.g.
    ``physv2`` (defaults) or ``physv2noeds`` / ``physv2noboost``. The leading
    'phys' keeps the slug self-describing; the trailing digit/letter shape is
    chosen so cache._split_key round-trips it cleanly.
    """
    eds = bool(getattr(config, "eds_consistent", True))
    boost = bool(getattr(config, "pre_start_tidal_boost", True))
    suffix = ""
    if not eds:
        suffix += "noeds"
    if not boost:
        suffix += "noboost"
    # Token shape: "phys" + version (e.g. "v2") + optional deviation suffix.
    # build_cache_name appends it as one part; cache._split_key will split it at
    # the last digit ("physv2" -> value "physv2", or "physv2noeds" -> "physv2"
    # value + "noeds" suffix), which round-trips back to the same string.
    return f"phys{PHYSICS_CACHE_VERSION}{suffix}"


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
    leet_search: bool = False
    search_center_mass: bool = True
    t_start_Gyr: float = 5.8
    t_duration_Gyr: float = 8.0
    damping_factor: float = None
    s_min_gpc: int = 15
    s_max_gpc: int = 60
    save_interval: int = 10
    objective: str = "lcdm"  # "lcdm" or "pantheon"
    # Per-node mass anisotropy (Deliverable B). Defaults keep backward compatibility.
    node_mass_seed: int = 0
    node_mass_amplitude: float = 0.0
    # Per-node RADIAL position anisotropy (analogous to node_mass_amplitude, but
    # perturbs node POSITIONS). Reuses node_mass_seed. Default 0 = symmetric lattice.
    node_s_amplitude: float = 0.0
    # Particle initial-condition sampler. "uniform_sphere" keeps backward-compatible
    # cache keys; "grf" appends a slug to the cache key so the two never collide.
    init_distribution: str = "uniform_sphere"
    # GRF support geometry (consumed only when init_distribution == "grf"). The
    # sample_grf default changed box->sphere (WS5 §8 bug fix: confine the GRF cloud
    # to the uniform_sphere radius so the ONLY difference vs uniform is clustering).
    # That changed a(t) for the fixed tuple init_distribution="grf", so the cache
    # key MUST distinguish the two supports or a pre-fix box cache would be served
    # stale for the new sphere default. Default "sphere" mirrors the sampler default;
    # its sub-slug is appended ONLY for grf AND only for the NEW "sphere" support, so
    # the LEGACY "box" keeps the pre-existing bare "grfinit" token (any pre-fix
    # on-disk cache is correctly addressed AS box and is NOT reused for sphere). See
    # build_cache_name. Threaded into init_kwargs={"support": ...} for grf runs so
    # the value that keys the cache is the value the sampler actually uses (keyed==run).
    grf_support: str = "sphere"
    # Node geometry (WS3). "cube26" is the default/backward-compatible value and
    # does NOT add a slug to the cache key. Any other geometry appends a slug.
    node_geometry: str = "cube26"
    geometry_kwargs: dict = field(default_factory=dict)
    # Virialized COUPLED (positions, masses) mass-segregated grid params. Consumed
    # only when node_geometry == "virialized"; their cache sub-slugs are appended
    # ONLY for the virialized geometry (so all existing keys stay valid -> no
    # PHYSICS_CACHE_VERSION bump). Defaults match constants.SimulationParameters.
    vir_n_nodes: int = 26
    vir_extent: float = 1.0
    vir_mass_rule: str = "radial"
    vir_mass_spread: float = 0.0
    vir_segregation: float = 1.0
    vir_s_metric: str = "median"
    # Virialized BALANCE LEVEL: 0 -> realistic (not force-balanced) Fibonacci
    # layout; >= 1 (default) -> force-balanced cubic-lattice ball. Its cache
    # sub-slug is appended ONLY for the virialized geometry (see build_cache_name).
    vir_relax_steps: int = 1
    # Option A vs Option B selector (Section 2). "lattice" (default, Option A:
    # analytic balance level) is byte-identical, so its cache sub-slug is appended
    # ONLY for virialized AND only when != "lattice" (all existing keys untouched ->
    # NO PHYSICS_CACHE_VERSION bump). "gradient" (Option B) is the TRUE iterative
    # relaxation of a realistic blob; vir_relax_rate / vir_hold_outer_frac tune the
    # descent (used only in gradient mode). They also key the cache in gradient mode
    # so a distinct (rate, hold) tuple -> a distinct key (keyed == run).
    vir_relax_mode: str = "lattice"
    vir_relax_rate: float = 0.1
    vir_hold_outer_frac: float = 0.3
    # Item-10 coupling: when True, vir_extent DRIVES vir_n_nodes (density-preserving
    # N ~ extent^3), making vir_extent meaningful in the force-balanced lattice mode.
    # False (default) -> byte-identical; its cache sub-slug is appended ONLY for the
    # virialized geometry AND only when True (so default keys are untouched -> NO
    # PHYSICS_CACHE_VERSION bump). A no-op at the default vir_extent == 1.0.
    vir_extent_couples_nodes: bool = False
    # Outer-region density ceiling (WS4). Default 1.0 = outer density == inner density
    # (EdS critical). Clipped in SimulationParameters to MAX_OUTER_DENSITY_CEILING.
    outer_density_ceiling: float = 1.0
    # Node Plummer softening length in Gpc on the tidal force path (Section 4
    # slingshot taming knob). 0.0 (default) keeps the legacy hard 1e10 m floor ->
    # tidal force is byte-identical, so its cache sub-slug is appended ONLY when
    # != 0.0 (all existing keys untouched -> NO PHYSICS_CACHE_VERSION bump). > 0.0
    # tames the runaway slingshot for both cube26 and virialized.
    node_softening_gpc: float = 0.0
    # Close-range tidal force law (Section 4). "plummer" (default) is the legacy
    # law -> byte-identical, so its cache sub-slug is appended ONLY when != the
    # default (all existing keys untouched -> NO PHYSICS_CACHE_VERSION bump).
    # "bounded" is the regularized "can't cross the midpoint" law.
    node_force_law: str = "plummer"
    # Adaptive KDK sub-stepping (Section 4). threshold 0.0 (default) -> OFF
    # (byte-identical). Their cache sub-slugs are appended ONLY when sub-stepping
    # is actually active (threshold > 0 AND substeps > 1), so default runs keep
    # their existing keys (NO PHYSICS_CACHE_VERSION bump).
    node_substep_threshold: float = 0.0
    node_substeps: int = 1
    # Start-size lever (Section 6): multiplier on the LCDM-implied initial cloud
    # size. 1.0 (default) keeps the LCDM-implied size -> a(t) byte-identical, so
    # its cache sub-slug is appended ONLY when != 1.0 (all existing keys untouched
    # -> NO PHYSICS_CACHE_VERSION bump). != 1.0 changes a(t) (cloud spans a
    # different fraction of the fixed node spacing S), so it MUST key the cache.
    start_size_scale: float = 1.0
    # Physics-affecting IC flags (mirror SimulationParameters defaults). These are
    # NOT otherwise encoded in the cache key, so they are folded into the physics
    # token (physics_cache_token) — a legacy run with these turned off gets a
    # DISTINCT cache key from a current-physics run with the same parameters.
    eds_consistent: bool = True
    pre_start_tidal_boost: bool = True

    @property
    def particle_count(self) -> int:
        if self.leet_search:
            return 1337//2
        if self.quick_search:
            return 200
        return 2000

    @property
    def n_steps(self) -> int:
        return 250 if self.quick_search else 300

RMSE_WEIGHT = 1#25
R2_WEIGHT = 1
SIZE_WEIGHT_VS_HUBBLE = 250
MAX_WEIGHT = 0.25

@dataclass
class MatchWeights:
    """Weights for computing weighted average match metric."""
    curve: float = SIZE_WEIGHT_VS_HUBBLE
    curve_r2: float = SIZE_WEIGHT_VS_HUBBLE*R2_WEIGHT
    curve_rmse: float = RMSE_WEIGHT*SIZE_WEIGHT_VS_HUBBLE
    half_curve: float = SIZE_WEIGHT_VS_HUBBLE/2
    half_rmse: float = RMSE_WEIGHT*SIZE_WEIGHT_VS_HUBBLE/2

    max: float = MAX_WEIGHT*SIZE_WEIGHT_VS_HUBBLE
    curve_error: float = MAX_WEIGHT*SIZE_WEIGHT_VS_HUBBLE
    curve_error_max: float = MAX_WEIGHT*SIZE_WEIGHT_VS_HUBBLE
    
    hubble_curve: float = 1
    hubble_curve_r2: float = R2_WEIGHT
    hubble_rmse: float = RMSE_WEIGHT
    hubble_half_curve: float = 0.5
    hubble_half_rmse: float = 0.5*RMSE_WEIGHT

    end: float = 10*SIZE_WEIGHT_VS_HUBBLE
    hubble_end: float = 10


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
    a_curve: Optional[np.ndarray] = None  # Full scale-factor array; populated by results_to_sim_result


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
        
    result = sorted(seq) # Ensure sorted, though usually already is
    for i in range(10):
        if terms_per_decade > 10*i:
            result = add_mid_values(result)
    return result


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
    m_list.reverse()  # Search high M first
    return m_list


def build_s_list(s_min: int, s_max: int) -> List[int]:
    """Build list of S values (grid spacing in Gpc) to search."""
    return generate_increments(s_max, terms_per_decade=31, min_value=s_min)#list(range(s_min, s_max + 1))


def build_center_mass_list(search_center_mass: bool = True, many_search: int = 3) -> List[int]:
    """
    Build list of center node mass values to search.

    Returns [1] if search_center_mass=False.
    Otherwise returns list with fine/coarse increments based on many_search.
    """
    if not search_center_mass:
        return [1]

    center_masses = generate_increments(1000, terms_per_decade=many_search, min_value=1)
    return center_masses

def compute_avg(metrics, multiplicative=False):
    # Multiplicative aggregate: product of all metric values (clamped to [0,1])
    match_avg_pct = 1.0
    if multiplicative:
        for key in USED_MATCH_METRIC_KEYS:
            match_avg_pct *= max(0.0, min(1.0, metrics[key] / 100))
        match_avg_pct *= 100
    else:
        # Additive aggregate: weighted average of metric values
        total_weight = sum(getattr(MatchWeights(), key.replace('match_', '').replace('_pct', '')) for key in USED_MATCH_METRIC_KEYS)
        if total_weight == 0:
            return 0.0
        for key in USED_MATCH_METRIC_KEYS:
            weight = getattr(MatchWeights(), key.replace('match_', '').replace('_pct', ''))
            match_avg_pct += (metrics[key] * weight) / total_weight
        print(f"Total weight for average: {total_weight}, {list(getattr(MatchWeights(), key.replace('match_', '').replace('_pct', '')) for key in USED_MATCH_METRIC_KEYS)}, {match_avg_pct}")
        
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

    buffer_end_pct = 5#10
    buffer_hubble_end_pct = 0#5
    ## TODO: Do something better: (Right now: 5% buffer)
    match_end_pct = match_end_pct if (baseline.size_final_Gpc > sim_result.results.size_final_Gpc) else min(100.0, match_end_pct+buffer_end_pct)
    ## 5%
    match_hubble_end_pct = match_hubble_end_pct if (baseline.H_hubble[-1] > sim_result.hubble_curve[-1]) else min(100.0, match_hubble_end_pct+buffer_hubble_end_pct)
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


_PANTHEON_WORST_SCORE: Dict[str, float] = {
    'chi2': float('inf'),
    'chi2_dof': float('inf'),
    'R2': -float('inf'),
    'n_sne_used': 0,
    'match_avg_pct': 0.0,
    'diff_pct': 100.0,
}


def compute_pantheon_metrics(
    sim_result: "SimResult",
    pantheon_data: Dict[str, Any],
    t_start_Gyr: float,
) -> Dict[str, float]:
    """
    Score a SimResult against REAL Pantheon+ via its sim-derived mu(z).

    Scoring pipeline:
      1. mu = sim_to_distance_modulus(z_pantheon, a_curve, t_Gyr, t_start_Gyr)
         -> in-range subset of SNe.
      2. evaluate_precomputed(z_in, mu_obs_in, sigma_in, mu) -> chi2, chi2_dof, R2.
      3. match_avg_pct = 100 / (1 + chi2_dof)  — monotone-decreasing in chi2_dof
         so the existing max-by-match_avg_pct logic selects the best config.

    Edge cases handled gracefully (returns worst-case score, does NOT raise):
      - a_curve is None  (SimResult from cache without a_curve)
      - too few in-range SNe (< 2)
      - non-finite chi2 / ValueError from the distance kernel
      - empty in-range subset

    Returns dict with keys:
        chi2, chi2_dof, R2, n_sne_used, match_avg_pct, diff_pct
    Also contains all MATCH_METRIC_KEYS set to 0.0 for CSV compatibility.
    """
    from .sim_distance import sim_to_distance_modulus
    from .hubble_diagram import evaluate_precomputed

    # Build a worst-case return dict with zero-filled MATCH_METRIC_KEYS
    def worst_case():
        metrics = dict(_PANTHEON_WORST_SCORE)
        for k in MATCH_METRIC_KEYS:
            metrics.setdefault(k, 0.0)
        return metrics

    if sim_result.a_curve is None:
        return worst_case()

    # Physical expansion anchor: reject configs whose TOTAL expansion over
    # [t_start, today] is not the real ~1+z(t_start). Without this, a runaway config
    # (e.g. M huge, expanding thousands-fold) renormalizes a=1 at the last snapshot and
    # can fit the z<z_start window while predicting a nonsensical history. See
    # expected_growth_factor.
    a_curve = np.asarray(sim_result.a_curve, dtype=float)
    if a_curve.size < 2 or a_curve[0] <= 0.0:
        return worst_case()
    model_growth = float(a_curve[-1] / a_curve[0])
    target_growth = expected_growth_factor(t_start_Gyr)
    if not np.isfinite(model_growth) or abs(model_growth / target_growth - 1.0) > GROWTH_ANCHOR_TOL:
        metrics = worst_case()
        metrics['growth_factor'] = model_growth
        metrics['growth_target'] = target_growth
        return metrics

    try:
        dist = sim_to_distance_modulus(
            z_target=pantheon_data['z'],
            a=sim_result.a_curve,
            t_Gyr=sim_result.t_Gyr,
            t_start_Gyr=t_start_Gyr,
        )
    except ValueError:
        # The kernel raises ValueError for every documented bad-input case
        # (today_tol guard, non-positive a, all-out-of-range, length/shape
        # mismatch). Score those as worst-case so a long sweep keeps going.
        # Any other exception is a genuine bug and is allowed to propagate.
        return worst_case()

    in_range = dist['in_range']
    z_in = pantheon_data['z'][in_range]
    mu_obs_in = pantheon_data['mu'][in_range]
    sigma_in = pantheon_data['sigma'][in_range]
    mu_model = dist['mu']

    if len(z_in) < 2:
        return worst_case()

    try:
        ev = evaluate_precomputed(z_in, mu_obs_in, sigma_in, mu_model)
    except ValueError:
        # evaluate_precomputed raises ValueError for empty data / bad sigma;
        # score worst-case. Other exceptions surface as real bugs.
        return worst_case()

    chi2 = ev['chi2']
    chi2_dof = ev['chi2_dof']
    R2 = ev['R2']

    if not np.isfinite(chi2_dof):
        return worst_case()

    # Monotone-decreasing score: closer to 100 as chi2_dof -> 0
    match_avg_pct = 100.0 / (1.0 + chi2_dof)

    metrics = {
        'chi2': chi2,
        'chi2_dof': chi2_dof,
        'R2': R2,
        'n_sne_used': int(len(z_in)),
        'growth_factor': model_growth,
        'growth_target': target_growth,
        'match_avg_pct': match_avg_pct,
        'diff_pct': 100.0 - match_avg_pct,
    }
    for k in MATCH_METRIC_KEYS:
        metrics.setdefault(k, 0.0)
    return metrics


def _build_result_dict(
    M_factor: int,
    S_gpc: int,
    centerM: float,
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


def build_cache_name(config, M_factor, S_val, centerM, seeds) -> str:
    """Build the metrics/results cache slug for a single sweep configuration.

    The slug must be UNIQUE per distinct physical configuration so two configs
    never share a cache entry. Node-mass anisotropy slugs are appended ONLY when
    node_mass_amplitude != 0.0, so uniform runs keep their pre-existing cache
    keys (backward compatible) and distinct (seed, amplitude) anisotropic runs
    get distinct keys.

    Extracted from worst_callback so it is a single source of truth that tests
    can exercise directly (instead of re-implementing the format by hand).
    """
    seeds_slug = '_'.join([str(seed) for seed in seeds])
    # Read node-mass anisotropy from config (defaults to 0/0.0 — backward compatible)
    node_mass_seed = getattr(config, 'node_mass_seed', 0)
    node_mass_amplitude = getattr(config, 'node_mass_amplitude', 0.0)
    node_s_amplitude = getattr(config, 'node_s_amplitude', 0.0)
    init_distribution = getattr(config, 'init_distribution', 'uniform_sphere')
    node_geometry = getattr(config, 'node_geometry', 'cube26')

    parts = []
    parts.append(f"{config.particle_count}p")
    parts.append(f"{config.t_start_Gyr}-{config.t_duration_Gyr+config.t_start_Gyr}Gyr")
    parts.append(f"{M_factor}M")
    parts.append(f"{float(centerM)}centerM")
    parts.append(f"{S_val}S")
    parts.append(f"{config.n_steps}steps")
    parts.append(f"{seeds_slug}seeds")
    # Include objective so lcdm and pantheon caches never collide
    parts.append(f"{config.objective}obj")
    # No mass randomize??
    if config.damping_factor:
        parts.append(f"{config.damping_factor}d")
    # Node-mass anisotropy slugs: append only when amplitude > 0 so uniform runs
    # keep their existing cache keys and only anisotropic runs get distinct keys.
    if node_mass_amplitude != 0.0:
        parts.append(f"{node_mass_seed}nmseed")
        parts.append(f"{node_mass_amplitude}nmamp")
    # Node-position anisotropy slug: append only when amplitude > 0 so symmetric
    # runs keep their existing cache keys. node_s also depends on node_mass_seed,
    # so include the seed here too (guarded so an amp=0/s>0 run is still distinct
    # per seed even when no nmseed slug was added above).
    if node_s_amplitude != 0.0:
        if node_mass_amplitude == 0.0:
            parts.append(f"{node_mass_seed}nmseed")
        parts.append(f"{node_s_amplitude}nsamp")
    # init_distribution slug: append only when non-default so uniform_sphere runs
    # keep their existing cache keys and grf runs get distinct keys.
    if init_distribution != "uniform_sphere":
        parts.append(f"{init_distribution}init")
    # GRF support discriminator (WS5 §8): the sample_grf default changed box->sphere,
    # which changed a(t) for the fixed tuple init_distribution="grf". To avoid serving
    # a pre-fix box cache for the new sphere default WITHOUT a PHYSICS_CACHE_VERSION
    # bump (which would needlessly invalidate the byte-identical uniform_sphere "v3"
    # caches), encode support ONLY for grf runs, and ONLY for the NEW "sphere" support:
    #   - support="box"    -> NO extra token: key stays the pre-existing bare
    #                         "..._grfinit_..." so any on-disk pre-fix cache (computed
    #                         with box support) remains addressed AS box (NOT reused).
    #   - support="sphere" -> append "sphsup": the new default gets a DISTINCT key and
    #                         MUST recompute rather than reuse the old box cache.
    # Non-grf runs (uniform_sphere) are completely untouched. Suffix is purely
    # alphabetic ("sphsup") so cache._split_key round-trips it. The value here is the
    # SAME grf_support threaded into init_kwargs={"support": ...} (keyed == run).
    if init_distribution == "grf":
        grf_support = getattr(config, "grf_support", "sphere")
        if grf_support == "sphere":
            parts.append("sphsup")
    # node_geometry slug: append only when non-default so cube26 runs keep their
    # existing cache keys and alternative geometries get distinct keys.
    if node_geometry != "cube26":
        parts.append(f"{node_geometry}geo")
    # Virialized sub-slugs: append the vir_* params ONLY for the virialized geometry
    # so every existing (cube26/cube_dense/fcc/bcc) key is untouched (NO
    # PHYSICS_CACHE_VERSION bump). Each distinct vir_* tuple -> a distinct key.
    if node_geometry == "virialized":
        parts.append(f"{getattr(config, 'vir_n_nodes', 26)}vn")
        parts.append(f"{getattr(config, 'vir_extent', 1.0)}vx")
        parts.append(f"{getattr(config, 'vir_mass_rule', 'radial')}vr")
        parts.append(f"{getattr(config, 'vir_mass_spread', 0.0)}vsp")
        parts.append(f"{getattr(config, 'vir_segregation', 1.0)}vsg")
        parts.append(f"{getattr(config, 'vir_s_metric', 'median')}vsm")
        parts.append(f"{getattr(config, 'vir_relax_steps', 1)}vrx")
        # Geometry-seed slug (B3a bug fix): for the virialized massfunc rule with a
        # NON-ZERO mass spread, node_mass_seed drives the log-normal mass draw AND the
        # segregation permutation (cosmo/node_geometry.py:583-586), so the seed
        # MATERIALLY changes the realized grid -> two seeds MUST get distinct cache
        # keys. Before this fix the seed only entered the key when an anisotropy
        # amplitude was non-zero (node_mass_amplitude/node_s_amplitude), so amp=0
        # virialized runs that differed ONLY by seed COLLIDED on one cache entry and
        # silently returned the same a(t) (run-but-not-keyed inversion). Gated tightly
        # so the radial rule (deterministic, no RNG) and spread==0 (raw_masses=ones,
        # no RNG) keys stay BYTE-IDENTICAL -> no existing key changes, no
        # PHYSICS_CACHE_VERSION bump. Suffix is purely alphabetic ("virseed") so
        # cache._split_key round-trips it. The value keyed here is the SAME
        # node_mass_seed threaded into SimulationParameters and forwarded to
        # build_virialized_grid (keyed == run).
        vir_mass_rule = getattr(config, 'vir_mass_rule', 'radial')
        vir_mass_spread = getattr(config, 'vir_mass_spread', 0.0)
        if vir_mass_rule == "massfunc" and vir_mass_spread > 0.0:
            parts.append(f"{node_mass_seed}virseed")
        # Extent->node-count coupling slug (item 10): append ONLY when the coupling
        # is ON (and only for virialized), so default virialized runs keep their
        # existing keys (NO PHYSICS_CACHE_VERSION bump) and a coupled run gets a
        # distinct key. Emitted as "1vxcouple" (value "1", purely-alphabetic suffix
        # "vxcouple") so cache._split_key round-trips it. The coupling changes the
        # built grid (effective node count = round(vir_n_nodes*vir_extent^3)), so it
        # is keyed == run, not just keyed.
        if getattr(config, 'vir_extent_couples_nodes', False):
            parts.append("1vxcouple")
        # Relaxation-mode slug (Section 2 Option A/B): append ONLY when the mode is
        # non-default ("gradient", Option B) so every existing virialized key (all
        # built in the default "lattice" mode) stays valid -> NO PHYSICS_CACHE_VERSION
        # bump. In gradient mode the descent tuning (rate, hold_outer_frac) changes the
        # built grid, so key those too -> a distinct (mode, rate, hold) tuple maps to a
        # distinct key (keyed == run). Suffixes are purely alphabetic so cache._split_key
        # round-trips them ("gradientvrm", "0.1vrr", "0.3vho").
        vir_relax_mode = getattr(config, 'vir_relax_mode', 'lattice')
        if vir_relax_mode != 'lattice':
            parts.append(f"{vir_relax_mode}vrm")
            parts.append(f"{getattr(config, 'vir_relax_rate', 0.1)}vrr")
            parts.append(f"{getattr(config, 'vir_hold_outer_frac', 0.3)}vho")
    # outer_density_ceiling slug: append only when != 1.0 so the default (no outer
    # over-density) keeps its existing cache key and non-default ceilings get distinct keys.
    outer_density_ceiling = getattr(config, 'outer_density_ceiling', 1.0)
    if outer_density_ceiling != 1.0:
        parts.append(f"{outer_density_ceiling}ceil")
    # Node-softening slug: append ONLY when != 0.0 so the default (legacy hard
    # 1e10 m floor, byte-identical tidal force) keeps its existing cache key and
    # NO PHYSICS_CACHE_VERSION bump is needed. Non-zero softening (the slingshot
    # taming knob) gets a distinct key per value. Suffix is purely alphabetic so
    # cache._split_key round-trips it ("1.0nsoft" -> value "1.0", suffix "nsoft").
    node_softening_gpc = getattr(config, 'node_softening_gpc', 0.0)
    if node_softening_gpc != 0.0:
        parts.append(f"{node_softening_gpc}nsoft")
    # Node force-law slug: append ONLY when non-default ("plummer") so legacy runs
    # keep their existing cache key (NO PHYSICS_CACHE_VERSION bump) and the bounded
    # law gets a distinct key. Suffix is purely alphabetic for cache._split_key.
    node_force_law = getattr(config, 'node_force_law', 'plummer')
    if node_force_law != 'plummer':
        parts.append(f"{node_force_law}nlaw")
    # Adaptive-substep slugs: append ONLY when sub-stepping is actually active
    # (threshold > 0 AND substeps > 1) so default runs keep their existing keys.
    node_substep_threshold = getattr(config, 'node_substep_threshold', 0.0)
    node_substeps = getattr(config, 'node_substeps', 1)
    if node_substep_threshold > 0.0 and node_substeps > 1:
        parts.append(f"{node_substep_threshold}nsubth")
        parts.append(f"{node_substeps}nsub")
    # Start-size slug: append ONLY when != 1.0 so the default (LCDM-implied size,
    # byte-identical a(t)) keeps its existing cache key and NO PHYSICS_CACHE_VERSION
    # bump is needed. A non-default size changes a(t), so each distinct value gets a
    # distinct key. Suffix is purely alphabetic so cache._split_key round-trips it
    # ("1.2ssz" -> value "1.2", suffix "ssz").
    start_size_scale = getattr(config, 'start_size_scale', 1.0)
    if start_size_scale != 1.0:
        parts.append(f"{start_size_scale}ssz")
    # Physics-version token (ALWAYS appended): invalidates entries computed under a
    # different simulation-physics version (e.g. pre-EdS-ICs / pre-boost), so a
    # physics change can never silently reuse a stale parameter-only cache entry.
    parts.append(physics_cache_token(config))
    return "_".join(parts)


def worst_callback(
    sim_callback, config, M_factor, S_val, centerM, seeds, baseline, weights,
    pantheon_data=None,
):
    seeds_slug = '_'.join([str(seed) for seed in seeds])
    cache_name = build_cache_name(config, M_factor, S_val, centerM, seeds)

    cache_filename = f"metrics_{config.particle_count}_s{seeds_slug}"
    global CACHE
    if not SKIP_CACHE:
        if not CACHE or CACHE.name != cache_filename:
            CACHE = Cache(cache_filename)

        cached_metrics = CACHE.get_cached_value(cache_name, CacheType.METRICS)
        if cached_metrics:
            has_all_keys = True
            check_keys = USED_MATCH_METRIC_KEYS if config.objective == "lcdm" else ('match_avg_pct',)
            for key in check_keys:
                if not key in cached_metrics:
                    has_all_keys = False
                    break
            if has_all_keys:
                cached_results = CACHE.get_cached_value(cache_name, CacheType.RESULTS)
                print(f"Using cache for {cache_name}")

                if config.objective == "lcdm":
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
                    # a_curve is not cached; pantheon scorer handles None gracefully
                ), cached_metrics

    sim_results = sim_callback(M_factor, S_val, centerM, seeds)

    worst_result = None
    worst_metrics = None
    for result in sim_results:
        if config.objective == "pantheon":
            metrics = compute_pantheon_metrics(result, pantheon_data, config.t_start_Gyr)
        else:
            metrics = compute_match_metrics(result, baseline, weights)
        if not worst_result:
            worst_result = result
            worst_metrics = metrics
        elif metrics['match_avg_pct'] < worst_metrics['match_avg_pct']:
            worst_result = result
            worst_metrics = metrics

    if not SKIP_CACHE:
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
    seeds: List[int] = [42],
    pantheon_data: Optional[Dict] = None,
) -> Tuple[int, float, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Ternary search for optimal S given fixed M.

    Assumes unimodal (bell curve) match quality over S space.

    Args:
        M_factor: External node mass factor
        centerM: Outer-mass multiplier (total sim mass / inner observable mass, >=1.0)
        sim_callback: Callback to run simulation
        baseline: LCDM baseline for comparison
        weights: Match metric weights
        s_min: Minimum S value to search
        s_max: Maximum S value to search
        s_hint: Previous best S (warm start)
        hint_window: Search within +/- hint_window of s_hint first
        seed: Random seed for simulations
        pantheon_data: Loaded Pantheon+ dict; required when config.objective=="pantheon".

    Returns:
        (best_S, best_match_pct, best_result_dict, all_results)
    """
    evaluated: Dict[int, Tuple[SimResult, Dict[str, float]]] = {}
    all_results: List[Dict[str, Any]] = []

    def evaluate_S(S_val: int) -> float:
        """Evaluate and cache simulation result for given S."""
        S_val = round(S_val)
        if S_val not in evaluated:
            sim_result, metrics = worst_callback(
                sim_callback, config, M_factor, S_val, centerM, seeds, baseline, weights,
                pantheon_data=pantheon_data,
            )
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
    seeds: List[int] = [42],
    pantheon_data: Optional[Dict] = None,
) -> Tuple[int, Dict[str, Any], bool, List[Dict[str, Any]]]:
    """
    Linear search for optimal S given fixed M.

    Searches from prev_best_S (or s_max) downward with adaptive skipping.
    Stops early when match starts decreasing.

    Args:
        M_factor: External node mass factor
        centerM: Outer-mass multiplier (total sim mass / inner observable mass, >=1.0)
        sim_callback: Callback to run simulation
        baseline: LCDM baseline for comparison
        weights: Match metric weights
        s_min: Minimum S value to search
        s_max: Maximum S value to search
        prev_best_S: Previous best S (search starts here, going down)
        seed: Random seed for simulations
        pantheon_data: Loaded Pantheon+ dict; required when config.objective=="pantheon".

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
        sim_result, metrics = worst_callback(
            sim_callback, config, M_factor, S, centerM, seeds, baseline, weights,
            pantheon_data=pantheon_data,
        )
        result = _build_result_dict(M_factor, S, centerM, sim_result, metrics)
        all_results.append(result)

        if current_evaluated:
            prev_result = current_evaluated[-1][1]

        print(f"\tMatch:", end="")
        for key in USED_MATCH_METRIC_KEYS:
            simple_key = key.replace('match_', '').replace('curve_', ''). replace('_pct', '').replace('_', ' ')
            if prev_result:
                diff = result[key] - prev_result[key]
                print(f" {simple_key}: {result[key]:.2f}% ({diff:.2f}%), ", end="")
            else:
                print(f" {simple_key}: {result[key]:.2f}, ", end="")
        if prev_result:
            diff = result['match_avg_pct'] - prev_result['match_avg_pct']
            print(f"\n\t Avg Match: {result['match_avg_pct']:.4f}%, CHANGE: {diff:.4f}%")
            if diff < 0 and abs(diff) > result['match_avg_pct']//4:
                print("\r\tMatch decreased significantly, stopping search for this M.\n")
                current_evaluated.append((S, result))
                break

        else:
            print(f"\n\t Avg Match: {result['match_avg_pct']:.4f}%")
            
        print("\n")
        if current_evaluated:
            # Check for negative match
            if result['match_avg_pct'] <= 0:
                print("\r\tMatch below 0%, stopping search for this M.")
                break

            # Early-stop "all worse" decision must key on the metric the co-fit is
            # ACTUALLY optimizing for the active objective. For objective=="pantheon",
            # compute_pantheon_metrics zero-fills USED_MATCH_METRIC_KEYS (the LCDM keys),
            # so iterating them here would compare 0.0 vs 0.0 and trip the stop on the
            # 2nd S evaluated -> linear pins near s_max with an inflated chi2. Use the
            # objective's scalar score (match_avg_pct == 100/(1+chi2_dof), the same value
            # the co-fit ranks candidates by) instead. LCDM keeps its multi-key behavior
            # byte-identical.
            if config.objective == "lcdm":
                early_stop_keys = USED_MATCH_METRIC_KEYS
            else:
                early_stop_keys = ('match_avg_pct',)
            all_worse = True
            for key in early_stop_keys:
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
    seeds: List[int] = [42],
    pantheon_data: Optional[Dict] = None,
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
                sim_result, metrics = worst_callback(
                    sim_callback, config, M, S, centerM, seeds, baseline, weights,
                    pantheon_data=pantheon_data,
                )
                result = _build_result_dict(M, S, centerM, sim_result, metrics)
                results.append(result)

    return results


def run_sweep(
    config: SweepConfig,
    search_method: SearchMethod,
    sim_callback: SimCallback,
    baseline: Optional[LCDMBaseline],
    weights: Optional[MatchWeights] = None,
    seeds = [42,123],
    pantheon_data: Optional[Dict] = None,
) -> List[Dict[str, Any]]:
    """
    Run parameter sweep using specified search method.

    Args:
        config: Sweep configuration
        search_method: Search algorithm to use
        sim_callback: Callback to run simulations
        baseline: LCDM baseline for comparison (may be None when objective="pantheon")
        weights: Match metric weights (uses defaults if None)
        seeds: Random seeds for simulations
        pantheon_data: Loaded Pantheon+ dict; required when config.objective=="pantheon".

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
            sim_callback, baseline, weights, seeds,
            pantheon_data=pantheon_data,
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
                    seeds=seeds,
                    pantheon_data=pantheon_data,
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
                    seeds=seeds,
                    pantheon_data=pantheon_data,
                )
                all_results.extend(results)

                if prev_best_S == config.s_min_gpc:
                    break
                prev_best_S = best_S
    global CACHE
    del CACHE
    CACHE = None
    return all_results
