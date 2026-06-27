"""
Node Geometry Factory (WS3)

Provides ``build_node_positions(geometry, S, **kwargs)`` which returns an (N, 3)
array of BASE node positions (in meters, before per-node radial-scale-factor and
mass perturbations are applied).  HMEAGrid._create_grid calls this factory instead
of its previous hard-coded 3×3×3-1 cube loop.

Mass bookkeeping convention
---------------------------
TOTAL external mass is held fixed across geometries:
    total_mass = n_nodes * M_ext_kg   (where M_ext_kg is the per-node mean mass)
When a geometry has n_nodes != 26, the caller (ExternalNodeParameters.node_masses)
still returns per-node masses that sum to n_nodes * M_ext_kg — no change required
there.  To make cross-geometry comparisons *apples-to-apples on Omega_Lambda_eff*,
use the ``effective_M_ext_kg`` helper to rescale M_ext_kg so the total equals that
of the cube26 reference (26 * M_ext_kg_ref).

Geometries
----------
All geometries MUST be VOLUME-FILLING — a 3D lattice approximating a *virialized*
meta-structure (relaxed mass distributed throughout the volume).  Hollow shells are
NOT valid: concentrating all mass on a sphere surface with an empty interior is the
opposite of a virialized structure, so spherical-shell geometries are deliberately
excluded.

cube26      3×3×3 − 1 cubic lattice (26 nodes). DEFAULT, byte-identical to the
            legacy HMEAGrid loop; the simplest virialized-grid approximation.
cube_dense  n×n×n − 1 cubic lattice (default n=5 → 124 nodes).
fcc         Face-centred cubic — corners + face centres per unit cell, iterated
            outward for n_shells unit-cell layers (default 2).
bcc         Body-centred cubic — corners + body centre per unit cell, n_shells
            unit-cell layers (default 2).
virialized  COUPLED (positions, masses) generator with MASS SEGREGATION: bigger
            nodes sit FURTHER from the centre, smaller nodes cluster near it
            (like a relaxed cluster). UNLIKE every other geometry this returns
            BOTH positions AND masses already paired (mass↔radius coupled), so it
            is NOT reachable through build_node_positions (positions-only); call
            build_virialized_grid(...) instead. HMEAGrid._create_grid branches on
            geometry=="virialized" and consumes the coupled masses directly.

Invariants
----------
- cube26 produces EXACTLY the same (N=26, 3) array as the old HMEAGrid loop (same
  loop order, same dtype=float).
- All nodes are at characteristic radius ~ S (outside the cloud).
- build_node_positions never touches the global numpy RNG (pure deterministic math).
"""

from __future__ import annotations

import math
import numpy as np
from typing import Dict, Any

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

#: Registry mapping geometry name -> callable(S, **kwargs) -> (N,3) array
_REGISTRY: Dict[str, Any] = {}


def register(name: str):
    """Decorator to register a geometry builder."""
    def _decorator(fn):
        _REGISTRY[name] = fn
        return fn
    return _decorator


def build_node_positions(geometry: str, S: float, **kwargs) -> np.ndarray:
    """Return base node positions as an (N, 3) float64 array (units: meters).

    Args:
        geometry: Geometry identifier (see module docstring).  Default "cube26".
        S:        Characteristic node scale in meters (the lattice unit / sphere radius).
        **kwargs: Geometry-specific keyword arguments (see each builder for details).
                  Plus the cross-geometry option ``normalize_nearest`` (bool, default
                  False): rescale the whole geometry so the NEAREST node sits at radius S.

    Returns:
        (N, 3) float64 array; all nodes at characteristic distance ~S from origin.

    Raises:
        ValueError: If ``geometry`` is not in the registry.

    Note on ``normalize_nearest`` (fair cross-geometry comparison):
        The tidal stretch falls off as ~1/d^3, so the NEAREST nodes dominate and far
        nodes barely matter.  The correct way to compare geometries is therefore to
        hold the per-node mass fixed AND put the nearest node of every geometry at the
        same distance S (so the dominant near-field is equivalent).  Equal-TOTAL-mass
        rescaling would be WRONG here: it would weaken the near nodes (which do the
        work) just to compensate for far nodes (which don't).  cube26 already has its
        nearest node at S, so normalize_nearest is a no-op for it.
    """
    if geometry not in _REGISTRY:
        valid = sorted(_REGISTRY.keys())
        raise ValueError(
            f"Unknown node geometry {geometry!r}. Valid choices: {valid}"
        )
    normalize_nearest = kwargs.pop("normalize_nearest", False)
    positions = np.asarray(_REGISTRY[geometry](S, **kwargs), dtype=np.float64)
    if normalize_nearest:
        radii = np.linalg.norm(positions, axis=1)
        r_min = float(radii.min())
        if r_min > 0:
            positions = positions * (S / r_min)
    return positions


def list_geometries() -> list[str]:
    """Return sorted list of registered geometry names."""
    return sorted(_REGISTRY.keys())


def effective_M_ext_kg(M_ext_kg: float, n_nodes: int, ref_nodes: int = 26) -> float:
    """Per-node mean mass that preserves the cube26 total external mass.

    When comparing geometries on Omega_Lambda_eff, keep the TOTAL external mass
    (= n_nodes * M_per_node) equal to the cube26 reference (26 * M_ext_kg_ref):

        M_per_node = M_ext_kg * ref_nodes / n_nodes

    Use this helper to rescale M_ext_kg when constructing ExternalNodeParameters
    for a non-default geometry so chi2 comparisons across geometries are fair.

    Args:
        M_ext_kg:  Per-node mass for the reference geometry (cube26).
        n_nodes:   Node count of the target geometry.
        ref_nodes: Reference node count (default 26 = cube26).

    Returns:
        Rescaled per-node mass for the target geometry.
    """
    return M_ext_kg * ref_nodes / n_nodes


# ---------------------------------------------------------------------------
# Virialized-grid helpers (coupled positions+masses)
# ---------------------------------------------------------------------------

def nearest_neighbour_spacing(positions: np.ndarray, metric: str = "median") -> float:
    """Characteristic nearest-neighbour (NN) spacing of a node set.

    For each node, find the distance to its CLOSEST other node, then reduce the
    per-node NN distances with ``metric`` ("median" or "mean"). This is the
    derived statistic the virialized generator targets as its spacing ``S``.

    Args:
        positions: (N, 3) node positions (N >= 2).
        metric: "median" (default) or "mean" — both return finite, positive
                values for any non-degenerate set; they differ in general.

    Returns:
        The median/mean per-node NN distance (float).

    Raises:
        ValueError: if N < 2 or metric is unknown.
    """
    pos = np.asarray(positions, dtype=np.float64)
    n = pos.shape[0]
    if n < 2:
        raise ValueError("nearest_neighbour_spacing needs at least 2 nodes.")
    # Pairwise distances; mask the diagonal so a node is not its own neighbour.
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    nn = dist.min(axis=1)
    if metric == "median":
        return float(np.median(nn))
    if metric == "mean":
        return float(np.mean(nn))
    raise ValueError(f"Unknown vir_s_metric {metric!r}; use 'median' or 'mean'.")


def _fibonacci_sphere(n: int) -> np.ndarray:
    """n roughly-isotropic unit direction vectors (Fibonacci sphere).

    Deterministic (no RNG): used as the default node directions so the virialized
    grid is volume-filling rather than a hollow shell or a clumped patch.

    Returns:
        (n, 3) float64 unit vectors.
    """
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float64)
    i = np.arange(n, dtype=np.float64)
    # z evenly spaced in (-1, 1) avoiding the exact poles; golden-angle azimuth.
    z = 1.0 - (2.0 * i + 1.0) / n
    radius_xy = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    theta = golden_angle * i
    x = radius_xy * np.cos(theta)
    y = radius_xy * np.sin(theta)
    return np.stack([x, y, z], axis=1).astype(np.float64)


def build_virialized_grid(
    S: float,
    *,
    n_nodes: int = 26,
    M_ext_kg: float = 1.0,
    vir_extent: float = 1.0,
    vir_mass_rule: str = "radial",
    vir_mass_spread: float = 0.0,
    vir_segregation: float = 1.0,
    vir_s_metric: str = "median",
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """COUPLED virialized node grid: (positions (N,3), masses (N,)), mass-segregated.

    A physically-motivated *relaxed cluster*: smaller nodes cluster near the
    centre, more massive nodes are pushed FURTHER out (mass segregation). UNLIKE
    every positions-only geometry this returns BOTH arrays already paired, so mass
    i determines radius i. Reachable ONLY via this function (build_node_positions
    raises for "virialized").

    Spacing contract: the radial layout is built out to ``vir_extent * S`` then
    rescaled by a single factor so the realized nearest-neighbour spacing
    (``nearest_neighbour_spacing(positions, vir_s_metric)``) equals ``S`` exactly.
    Thus ``S`` is the TARGET characteristic spacing per ``vir_s_metric``.

    Mean-preservation contract: ``mean(masses) == M_ext_kg`` EXACTLY (rtol 1e-12),
    so total external mass == N * M_ext_kg and Omega_Lambda_eff stays comparable.

    Falsifiable knobs:
      * ``vir_mass_spread == 0`` -> all masses == M_ext_kg (uniform), both rules.
      * ``vir_segregation == 0`` -> mass and radius are DECOUPLED (no segregation).

    Args:
        S: Target characteristic NN spacing in meters (see spacing contract).
        n_nodes: Number of nodes (>= 1; segregation/spacing need >= 2).
        M_ext_kg: Per-node MEAN mass in kg (mean-preserving target).
        vir_extent: Continuous RADIAL-RANGE multiplier (>= 0). Nodes span radii in
            [0.5*S, (0.5+vir_extent)*S] before the global NN-spacing rescale. Because
            that rescale fixes the spacing to S, a PURE global factor would cancel;
            vir_extent therefore widens the radial RANGE (extent=1 -> [0.5S,1.5S];
            extent=2 -> [0.5S,2.5S]), reaching further relative to the spacing.
        vir_mass_rule: "radial" (deterministic mass ~ f(r), default) or "massfunc"
            (log-normal mass-function draw + spatial segregation by mass rank).
        vir_mass_spread: Amplitude of the node-mass distribution about the mean.
            0.0 (default) -> uniform masses (THE falsifiable knob).
        vir_segregation: Segregation strength in [0, 1+]; 0 -> mass/radius decoupled.
        vir_s_metric: "median" (default) or "mean" NN-spacing definition to target.
        seed: RNG seed for virialized draws (np.random.default_rng(seed)); INDEPENDENT
            of the global np.random state and of the particle/simulation RNG.

    Returns:
        (positions, masses): positions (N,3) float64 (mass-segregated radii),
        masses (N,) float64 with mean == M_ext_kg exactly.

    Raises:
        ValueError: for invalid n_nodes / vir_mass_rule / vir_s_metric.
    """
    if n_nodes < 1:
        raise ValueError(f"vir_n_nodes must be >= 1, got {n_nodes}")
    if vir_mass_rule not in ("radial", "massfunc"):
        raise ValueError(
            f"Unknown vir_mass_rule {vir_mass_rule!r}; use 'radial' or 'massfunc'."
        )
    if vir_s_metric not in ("median", "mean"):
        raise ValueError(
            f"Unknown vir_s_metric {vir_s_metric!r}; use 'median' or 'mean'."
        )

    n = int(n_nodes)
    rng = np.random.default_rng(seed)

    # ---- Directions: roughly isotropic, volume-filling (NOT a hollow shell) ----
    directions = _fibonacci_sphere(n)

    # ---- Raw radial profile: nodes span [r_inner, r_outer] (NOT a hollow shell) ----
    # A PURE global radius factor would cancel under the later NN-spacing rescale, so
    # vir_extent must widen the radial RANGE, not just scale it. We anchor an inner
    # floor at 0.5*S and set the outer reach to (0.5 + vir_extent)*S, so:
    #   extent=1 -> radii in [0.5*S, 1.5*S] (ratio 3); extent=2 -> [0.5*S, 2.5*S]
    # (ratio 5). The RANGE grows with vir_extent and there is always a real spread
    # (>= 2 distinct radii) even at extent=1 — never a degenerate shell.
    # slot fractions in [0, 1]; slot 0 = innermost, slot n-1 = outermost.
    if n == 1:
        slot = np.array([0.0], dtype=np.float64)
    else:
        slot = np.arange(n, dtype=np.float64) / (n - 1)
    r_inner = 0.5 * float(S)
    r_outer = (0.5 + max(float(vir_extent), 0.0)) * float(S)
    radius_levels = r_inner + slot * (r_outer - r_inner)
    # Centred radius proxy in ~[-0.5, 0.5] for the deterministic mass rule.
    rank_frac = slot

    if vir_mass_rule == "radial":
        # (b) Deterministic mass ~ f(r): place nodes evenly in radius from floor to
        # outer reach; mass INCREASES monotonically with radius (scaled by spread &
        # segregation). segregation=0 -> flat -> uniform (after normalization).
        radii_raw = radius_levels
        x = rank_frac - rank_frac.mean()  # centred radius proxy in ~[-0.5, 0.5]
        raw_masses = 1.0 + float(vir_mass_spread) * float(vir_segregation) * x
        raw_masses = np.maximum(raw_masses, 1e-12)
    else:  # "massfunc"
        # (a) Draw N masses from a log-normal mass function (many small, few large).
        if float(vir_mass_spread) == 0.0:
            raw_masses = np.ones(n, dtype=np.float64)
        else:
            g = rng.standard_normal(n)
            raw_masses = np.exp(float(vir_mass_spread) * g)
        # Radii on the deterministic floor->outer profile (radius_levels above);
        # ASSIGN by mass rank so bigger mass -> larger radius (segregation).
        # vir_segregation blends the mass-sorted order with a random order:
        # seg=1 -> fully sorted, seg=0 -> mass/radius decoupled.
        order_by_mass = np.argsort(raw_masses, kind="stable")  # ascending mass
        order_random = rng.permutation(n)
        seg = float(np.clip(vir_segregation, 0.0, 1.0))
        # Blend two RANKINGS: a continuous score interpolating sorted vs random rank.
        rank_sorted = np.empty(n, dtype=np.float64)
        rank_sorted[order_by_mass] = np.arange(n, dtype=np.float64)
        rank_random = np.empty(n, dtype=np.float64)
        rank_random[order_random] = np.arange(n, dtype=np.float64)
        blended = seg * rank_sorted + (1.0 - seg) * rank_random
        assign = np.argsort(blended, kind="stable")  # node index -> radius slot
        radii_raw = np.empty(n, dtype=np.float64)
        radii_raw[assign] = radius_levels

    # ---- Mean-preserving normalization: mean(masses) == M_ext_kg exactly ----
    raw_masses = np.asarray(raw_masses, dtype=np.float64)
    if raw_masses.mean() == 0.0:
        masses = np.full(n, float(M_ext_kg), dtype=np.float64)
    else:
        masses = float(M_ext_kg) * raw_masses / raw_masses.mean()

    # ---- Compose positions; rescale so realized NN spacing (metric) == S ----
    positions = directions * radii_raw[:, None]
    if n >= 2:
        realized = nearest_neighbour_spacing(positions, vir_s_metric)
        if realized > 0.0:
            positions = positions * (float(S) / realized)

    return positions.astype(np.float64), masses.astype(np.float64)


# ---------------------------------------------------------------------------
# Geometry builders
# ---------------------------------------------------------------------------

@register("cube26")
def _cube26(S: float) -> np.ndarray:
    """3×3×3 − 1 cubic lattice (26 nodes).

    Traversal order: i in [-1,0,1], j in [-1,0,1], k in [-1,0,1], skip (0,0,0).
    This is byte-identical to the original HMEAGrid._create_grid() loop.
    """
    positions = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                if i == 0 and j == 0 and k == 0:
                    continue
                positions.append(np.array([i, j, k], dtype=float) * S)
    return np.array(positions, dtype=np.float64)


@register("cube_dense")
def _cube_dense(S: float, n_per_side: int = 5) -> np.ndarray:
    """n×n×n − 1 cubic lattice, default n=5 (124 nodes).

    Grid indices run from -(n//2) to +(n//2), skipping (0,0,0).
    For n=3 this is equivalent to cube26.  For n=5 (the default) the spacing
    between adjacent nodes is S / 2 so the outermost nodes sit at sqrt(3)*S
    from the origin — well outside the cloud.

    Args:
        n_per_side: Grid size (must be odd ≥ 3).  Default 5.
    """
    if n_per_side < 3 or n_per_side % 2 == 0:
        raise ValueError(f"n_per_side must be an odd integer >= 3, got {n_per_side}")
    half = n_per_side // 2
    # Scale so outermost nodes are at distance sqrt(3)*S (same as cube26 outer nodes).
    # For cube26 half=1, step=1.  For n=5 half=2, step=0.5 => same outer radius.
    step = S / half
    positions = []
    for i in range(-half, half + 1):
        for j in range(-half, half + 1):
            for k in range(-half, half + 1):
                if i == 0 and j == 0 and k == 0:
                    continue
                positions.append(np.array([i, j, k], dtype=float) * step)
    return np.array(positions, dtype=np.float64)


@register("fcc")
def _fcc(S: float, n_shells: int = 2) -> np.ndarray:
    """Face-centred cubic lattice, n_shells unit-cell layers.

    The FCC unit cell has side 2*S (so nearest-neighbour distance is S*sqrt(2)).
    Basis vectors (fractional):
        corners:       (±1, ±1, ±1) × S   — 8 corners / 8 cells each
        face centres:  permutations of (±1, 0, 0) × S × sqrt(2)
    We enumerate all integer multiples of the FCC primitive vectors within a
    distance bound and keep nodes at radius in [S/2, n_shells*S*sqrt(3)*1.05].

    For n_shells=1: 12 nodes (nearest-neighbour shell of an FCC lattice).
    For n_shells=2: extends to a second shell, typically 54 nodes.

    Args:
        n_shells: Number of shells (default 2).
    """
    # FCC primitive vectors (conventional cell side = 2S)
    a = S  # nearest-neighbour / sqrt(2) => conventional cell = 2S
    # FCC basis in a cubic unit cell of side `a`:
    # corners at (0,0,0),(a,0,0),(0,a,0),(0,0,a),(a,a,0),(a,0,a),(0,a,a),(a,a,a)
    # face centres: already included as midpoints of cube edges and faces
    # Enumerate via integer lattice with FCC primitive vectors
    #   a1 = a*(1,1,0), a2 = a*(1,0,1), a3 = a*(0,1,1)
    a1 = np.array([1.0, 1.0, 0.0]) * a
    a2 = np.array([1.0, 0.0, 1.0]) * a
    a3 = np.array([0.0, 1.0, 1.0]) * a
    r_max = n_shells * a * math.sqrt(3) * 1.05
    r_min = a * 0.5  # exclude any near-origin point
    bound = n_shells + 2
    seen = set()
    positions = []
    for i in range(-bound, bound + 1):
        for j in range(-bound, bound + 1):
            for k in range(-bound, bound + 1):
                pt = i * a1 + j * a2 + k * a3
                r = float(np.linalg.norm(pt))
                if r < r_min or r > r_max:
                    continue
                key = (round(pt[0] / a * 1e6), round(pt[1] / a * 1e6),
                       round(pt[2] / a * 1e6))
                if key in seen:
                    continue
                seen.add(key)
                positions.append(pt.copy())
    if not positions:
        raise RuntimeError(f"FCC geometry produced 0 nodes for n_shells={n_shells}")
    return np.array(positions, dtype=np.float64)


@register("bcc")
def _bcc(S: float, n_shells: int = 2) -> np.ndarray:
    """Body-centred cubic lattice, n_shells unit-cell layers.

    BCC has two atoms per conventional cell: corners (0,0,0) and body centre (½,½,½).
    Primitive vectors: a1=a(-1,1,1), a2=a(1,-1,1), a3=a(1,1,-1) with a=S/sqrt(3)*2.
    We enumerate all integer multiples within a radius bound.

    For n_shells=1: 8 nodes (nearest-neighbour shell, the cube body-diagonals).
    For n_shells=2: 386 nodes — a volume-FILLING lattice ball out to r~4*S, NOT
        ~26. (cube26 by contrast is a single 3x3x3-1 shell of 26 nodes.) Node
        count grows ~n_shells**3; pass n_shells via geometry_kwargs to control it.

    Args:
        n_shells: Number of shells (default 2).
    """
    # BCC primitive vectors; conventional cell side = 2*a where a = S/sqrt(3)
    a = S / math.sqrt(3.0)
    b1 = np.array([-1.0, 1.0, 1.0]) * a
    b2 = np.array([1.0, -1.0, 1.0]) * a
    b3 = np.array([1.0, 1.0, -1.0]) * a
    r_max = n_shells * 2 * a * math.sqrt(3) * 1.05
    r_min = a * 0.5
    bound = n_shells + 2
    seen = set()
    positions = []
    for i in range(-bound, bound + 1):
        for j in range(-bound, bound + 1):
            for k in range(-bound, bound + 1):
                pt = i * b1 + j * b2 + k * b3
                r = float(np.linalg.norm(pt))
                if r < r_min or r > r_max:
                    continue
                key = (round(pt[0] / a * 1e6), round(pt[1] / a * 1e6),
                       round(pt[2] / a * 1e6))
                if key in seen:
                    continue
                seen.add(key)
                positions.append(pt.copy())
    if not positions:
        raise RuntimeError(f"BCC geometry produced 0 nodes for n_shells={n_shells}")
    return np.array(positions, dtype=np.float64)


@register("virialized")
def _virialized_positions_only(S: float, **kwargs) -> np.ndarray:
    """Sentinel so list_geometries() includes "virialized".

    The virialized geometry returns COUPLED (positions, masses) — it cannot be
    served through the positions-only build_node_positions path (that would yield
    an UN-COUPLED virialized run, losing mass segregation). Always raise with a
    pointer to the coupled entry point.
    """
    raise ValueError(
        "node_geometry='virialized' returns COUPLED (positions, masses) and is not "
        "available via build_node_positions(); call build_virialized_grid(S, "
        "n_nodes=..., M_ext_kg=..., vir_*=...) instead, or build an HMEAGrid whose "
        "ExternalNodeParameters.node_geometry=='virialized' (HMEAGrid._create_grid "
        "takes the coupled path automatically)."
    )
