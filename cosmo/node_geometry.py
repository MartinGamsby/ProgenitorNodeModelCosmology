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


def extent_coupled_n_nodes(base_n_nodes: int, vir_extent: float) -> int:
    """Node count that holds node DENSITY constant as ``vir_extent`` grows.

    Item-10 coupling: the user wants a larger radial extent to imply MORE nodes
    (like centerM implies more particles), keeping the node *density* of the
    virialized ball roughly fixed rather than thinning out. The virialized grid is
    VOLUME-FILLING (a 3D ball, see module docstring), so its density is

        rho = N / V,   V = (4/3) pi R^3.

    The realized ball radius R scales ~LINEARLY with ``vir_extent`` relative to the
    FIXED nearest-neighbour spacing S (the radial RANGE ratio is ``1 + 2*extent`` and
    the NN spacing is rescaled to S, so reach grows ~linearly with extent). Holding
    ``rho`` constant under ``R ~ extent`` therefore needs ``N ~ R^3 ~ extent^3``:

        n_eff = round(base_n_nodes * (vir_extent / EXTENT_DEFAULT)^3),   EXTENT_DEFAULT = 1.0.

    The reference is the DEFAULT extent (1.0), so at ``vir_extent == 1.0`` the factor
    is exactly 1.0 and ``n_eff == base_n_nodes`` (the default node count is reproduced
    EXACTLY -> byte-identical when coupling is enabled at default extent). A bigger
    extent raises the count by extent^3 (extent=2 -> ~8x), a smaller extent lowers it,
    keeping ``N / extent^3`` (i.e. the density) ~constant. The result is clamped to
    ``>= 1`` so a tiny extent never yields a degenerate empty grid.

    This is a PURE count derivation (no positions/masses, no RNG): callers pass the
    returned count straight to ``build_virialized_grid(n_nodes=...)``.

    Args:
        base_n_nodes: The configured ``vir_n_nodes`` (the count at extent == 1.0).
        vir_extent:   The radial-range multiplier (>= 0); 1.0 -> unchanged count.

    Returns:
        The density-preserving effective node count (int, >= 1).
    """
    EXTENT_DEFAULT = 1.0
    ext = max(float(vir_extent), 0.0)
    factor = (ext / EXTENT_DEFAULT) ** 3
    return max(1, int(round(float(base_n_nodes) * factor)))


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


def node_net_accelerations(
    positions: np.ndarray,
    masses: np.ndarray,
    *,
    center_mass_kg: float | None = None,
    G: float | None = None,
) -> np.ndarray:
    """Net gravitational acceleration on each node from all OTHER nodes + a centre.

    Mirrors the simulation's tidal law (``calculate_tidal_forces_numba``) EXACTLY,
    including the ``|r| < 1e10 m -> 1e10`` singularity floor, so the residual this
    feeds is consistent with the integrator. Pure numpy node-node sum (no particle
    cloud, no RNG, no I/O).

    For node ``i`` the acceleration from source ``j`` (every other node + one
    central node at the origin) is the attractive
        ``a_i += G * m_j * (x_j - x_i) / |x_j - x_i|^3``.
    A central node of mass ``center_mass_kg`` is added at the origin (the user's
    "centerM being a node"); its default is ``mean(masses)`` (one unit node).

    Args:
        positions: (N, 3) node positions in meters.
        masses:    (N,) node masses in kg (same order as positions).
        center_mass_kg: Mass of the extra central node at the origin. Defaults to
            ``mean(masses)`` (the natural "one unit node" central mass).
        G: Gravitational constant. Defaults to ``CosmologicalConstants.G``.

    Returns:
        (N, 3) net accelerations in m/s^2 (one row per input node; the central
        node is a SOURCE only and has no output row).

    Raises:
        ValueError: if N < 1 or positions/masses lengths disagree.
    """
    pos = np.asarray(positions, dtype=np.float64)
    m = np.asarray(masses, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions must be (N,3), got {pos.shape}")
    n = pos.shape[0]
    if n < 1:
        raise ValueError("node_net_accelerations needs at least 1 node.")
    if m.shape != (n,):
        raise ValueError(
            f"masses shape {m.shape} does not match {n} positions."
        )

    if G is None:
        from .constants import CosmologicalConstants
        G = CosmologicalConstants.G
    if center_mass_kg is None:
        center_mass_kg = float(np.mean(m))

    # Source set = the N nodes + one central node at the origin.
    src_pos = np.vstack([pos, np.zeros((1, 3), dtype=np.float64)])
    src_mass = np.concatenate([m, np.array([float(center_mass_kg)], dtype=np.float64)])

    # Vector from target i to source j: (N, S, 3). Diagonal (i==j among nodes)
    # is excluded by the 1e10 floor making self-pull ~0 AND by zeroing it below.
    diff = src_pos[None, :, :] - pos[:, None, :]
    r = np.sqrt(np.sum(diff * diff, axis=2))  # (N, S)
    # Singularity floor — identical to the numba kernel.
    r = np.where(r < 1e10, 1e10, r)
    inv_r3 = 1.0 / (r * r * r)  # (N, S)
    # Exclude a node's pull on itself (i==j for the first N sources).
    eye = np.eye(n, dtype=bool)
    self_mask = np.zeros((n, n + 1), dtype=bool)
    self_mask[:, :n] = eye
    inv_r3 = np.where(self_mask, 0.0, inv_r3)
    # a_i = G * sum_j m_j * diff_ij / r_ij^3
    weights = (G * src_mass)[None, :, None] * inv_r3[:, :, None]
    accel = np.sum(weights * diff, axis=1)
    return accel.astype(np.float64)


def virialization_residual(
    positions: np.ndarray,
    masses: np.ndarray,
    *,
    inner_frac: float = 0.5,
    center_frac: float | None = None,
    center_k: int | None = None,
    center_mass_kg: float | None = None,
    G: float | None = None,
    reference: str = "mean_pairwise",
) -> dict:
    """Dimensionless per-INNER-node force-balance residual for a node grid.

    Physical criterion (the user's): the DEEP-INTERIOR nodes of a *truly virialized*
    grid feel ~NET-ZERO gravity from all the other nodes plus a central node, so they
    would not move. This returns, per selected node, the DIMENSIONLESS residual
        ``residual_i = |net_accel_i| / a_ref``
    where ``a_ref`` is a characteristic single-neighbour pull (see ``reference``).
    A well-virialized interior node has residual << 1 (its directional pulls cancel);
    an UN-balanced grid has residual ~O(1) or larger.

    What "0.25" means (the user asked: "Not 25%?")
    ----------------------------------------------
    The tolerance is a DIMENSIONLESS RATIO, not a percentage of anything physical:
        ``residual = |net node acceleration| / a_ref``,
    where ``a_ref`` is the characteristic magnitude of a SINGLE neighbour's pull on
    a node (see ``reference``). residual=1 means "the leftover net force is as big as
    one typical neighbour pull" (clearly un-balanced); residual=0.25 means "the
    directional pulls cancel to within a quarter of one neighbour pull" — i.e. the
    node is ~4× closer to equilibrium than a single un-cancelled neighbour would
    leave it. It is NOT "25% of the gravitational force is unbalanced relative to the
    total"; the denominator is one neighbour, not the total. 0.25 is a deliberately
    GENEROUS bar (a strict crystal reaches ~1e-30); the figures report the actual
    numbers at several thresholds so the cut can be judged, not assumed.

    Node SELECTION (which nodes the criterion is applied to)
    --------------------------------------------------------
    Edge nodes of a finite grid ALWAYS feel a net inward pull (expected, NOT a
    failure), so the criterion is only meaningful on the genuinely deep interior of
    a LARGE grid. Three mutually-exclusive selectors (checked in priority order):

      * ``center_k`` (int) — the K nodes CLOSEST to the centroid, independent of
        ``r_max``. This is the SIZE-INDEPENDENT deep-interior selector: on a 2000-node
        grid ``center_k=20`` isolates the 20 most-buried nodes regardless of overall
        radius, so growing the grid genuinely deepens the interior being tested.
      * ``center_frac`` (float in (0,1]) — the innermost ``center_frac`` FRACTION of
        nodes by distance-to-centroid (``ceil(center_frac*N)`` nodes). Scales the
        interior population with grid size.
      * ``inner_frac`` (float, default 0.5) — LEGACY radius-threshold selector:
        nodes with ``radius < inner_frac * r_max`` (measured from the ORIGIN, not the
        centroid). Kept default so existing callers/tests are byte-identical; it does
        NOT isolate the deep interior of a large grid (it is "inner half of the ball
        by radius"), which is why ``center_k`` / ``center_frac`` were added.

    At least one node is always returned (the single closest-to-centroid node if a
    selector would otherwise be empty).

    Args:
        positions: (N, 3) node positions in meters.
        masses:    (N,) node masses in kg.
        inner_frac: LEGACY radius fraction in (0, 1]; used ONLY when neither
            center_k nor center_frac is given (default 0.5).
        center_frac: If given, select the innermost ``center_frac`` fraction of
            nodes by distance to the CENTROID (overrides inner_frac).
        center_k: If given, select the ``center_k`` nodes closest to the CENTROID
            (overrides both center_frac and inner_frac). The size-independent
            deep-interior selector.
        center_mass_kg: Central-node mass (see node_net_accelerations); defaults
            to ``mean(masses)`` (one unit node).
        G: Gravitational constant (defaults to CosmologicalConstants.G).
        reference: Normalization scale for the residual:
            "mean_pairwise" (default) — a_ref = mean over ALL nodes i of the mean
                single-source pull magnitude ``mean_j G m_j / |x_j - x_i|^2``
                (incl. centre): "how big a typical single-neighbour pull is".
            "max_pairwise" — a_ref = mean over selected nodes of the STRONGEST single
                -source pull on that node (the nearest/heaviest neighbour).

    Returns:
        dict with:
          'residual_per_node' (array over selected nodes, dimensionless),
          'inner_idx'   (indices into the input arrays of the selected nodes),
          'max_residual', 'median_residual' (floats),
          'a_ref'       (the normalization scale, m/s^2),
          'n_inner'     (number of selected nodes),
          'selector'    (str describing which selector was used).

    Raises:
        ValueError: for N < 2 or unknown ``reference``.
    """
    pos = np.asarray(positions, dtype=np.float64)
    m = np.asarray(masses, dtype=np.float64)
    n = pos.shape[0]
    if n < 2:
        raise ValueError("virialization_residual needs at least 2 nodes.")
    if reference not in ("mean_pairwise", "max_pairwise"):
        raise ValueError(
            f"Unknown reference {reference!r}; use 'mean_pairwise' or 'max_pairwise'."
        )
    if G is None:
        from .constants import CosmologicalConstants
        G = CosmologicalConstants.G
    if center_mass_kg is None:
        center_mass_kg = float(np.mean(m))

    radius = np.linalg.norm(pos, axis=1)
    r_max = float(radius.max())
    # ---- Node selection (center-only takes priority over the legacy inner_frac) ----
    if center_k is not None:
        # K nodes closest to the CENTROID (size-independent deep interior).
        centroid = pos.mean(axis=0)
        d_centroid = np.linalg.norm(pos - centroid[None, :], axis=1)
        k = max(1, min(int(center_k), n))
        inner_idx = np.argsort(d_centroid, kind="stable")[:k]
        selector = f"center_k={k}"
    elif center_frac is not None:
        centroid = pos.mean(axis=0)
        d_centroid = np.linalg.norm(pos - centroid[None, :], axis=1)
        k = max(1, int(math.ceil(float(center_frac) * n)))
        inner_idx = np.argsort(d_centroid, kind="stable")[:k]
        selector = f"center_frac={float(center_frac)}"
    else:
        inner_idx = np.nonzero(radius < float(inner_frac) * r_max)[0]
        if inner_idx.size == 0:
            # Fall back to the single innermost node.
            inner_idx = np.array([int(np.argmin(radius))])
        selector = f"inner_frac={float(inner_frac)}"

    accel = node_net_accelerations(
        pos, m, center_mass_kg=center_mass_kg, G=G
    )
    net_mag = np.linalg.norm(accel, axis=1)  # (N,)

    # ---- Characteristic single-source pull magnitudes (reference scale) ----
    src_pos = np.vstack([pos, np.zeros((1, 3), dtype=np.float64)])
    src_mass = np.concatenate([m, np.array([float(center_mass_kg)], dtype=np.float64)])
    diff = src_pos[None, :, :] - pos[:, None, :]
    r = np.sqrt(np.sum(diff * diff, axis=2))
    r = np.where(r < 1e10, 1e10, r)
    pull = (G * src_mass)[None, :] / (r * r)  # (N, S) single-source |a|
    # Mask self-pull (i==j among the first N sources).
    self_mask = np.zeros((n, n + 1), dtype=bool)
    self_mask[:, :n] = np.eye(n, dtype=bool)
    pull = np.where(self_mask, np.nan, pull)

    if reference == "mean_pairwise":
        # Mean single-source pull per node, then mean over ALL nodes.
        a_ref = float(np.nanmean(np.nanmean(pull, axis=1)))
    else:  # "max_pairwise"
        # Strongest single-source pull on each inner node, averaged.
        a_ref = float(np.nanmean(np.nanmax(pull[inner_idx], axis=1)))

    if not np.isfinite(a_ref) or a_ref <= 0.0:
        a_ref = 1.0  # degenerate guard; residuals then equal raw |net accel|

    residual_per_node = net_mag[inner_idx] / a_ref
    return {
        "residual_per_node": residual_per_node,
        "inner_idx": inner_idx,
        "max_residual": float(np.max(residual_per_node)),
        "median_residual": float(np.median(residual_per_node)),
        "a_ref": a_ref,
        "n_inner": int(inner_idx.size),
        "selector": selector,
    }


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


def _lattice_ball_positions(n_nodes: int) -> np.ndarray:
    """The ``n_nodes`` integer cubic-lattice points CLOSEST to the origin.

    Grows a cube [-h, h]^3 until it contains at least ``n_nodes`` points, sorts
    them by radius (the ORIGIN first), and returns the n closest. The origin
    (0,0,0) is ALWAYS included as the first node — it is the symmetric centre that
    makes the configuration force-balanced (see build_virialized_grid).

    Antipodal symmetry: for every point (i,j,k) at radius r the cubic lattice also
    contains (-i,-j,-k) at the SAME radius. Including a node at the origin therefore
    yields a point-symmetric ball: each node's pull from its antipode is exactly
    opposed, so the inner-node net force cancels (the force-balance the residual
    metric checks). Deterministic, no RNG.

    Returns:
        (n_nodes, 3) float64 integer-valued lattice positions, origin first then by
        increasing radius (ties broken stably by the meshgrid traversal order).
    """
    half = 1
    while True:
        coords = np.arange(-half, half + 1)
        gx, gy, gz = np.meshgrid(coords, coords, coords, indexing="ij")
        pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float64)
        if pts.shape[0] >= n_nodes:
            r = np.linalg.norm(pts, axis=1)
            order = np.argsort(r, kind="stable")  # origin (r=0) first
            return pts[order[:n_nodes]]
        half += 1


def _build_force_balanced_grid(
    S: float,
    *,
    n_nodes: int,
    M_ext_kg: float,
    vir_mass_rule: str,
    vir_mass_spread: float,
    vir_segregation: float,
    vir_s_metric: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Force-balanced virialized grid: an exact cubic-lattice ball, origin-centred.

    This is the ``vir_relax_steps >= 1`` mode of build_virialized_grid (Option A).
    The realistic Fibonacci-sphere mode (vir_relax_steps == 0) is NOT force-balanced
    (its inner nodes feel an O(10-70) net pull); a CONTINUOUS position relaxation
    cannot reach force balance on a finite canvas (the central monopole pull is
    irreducible by position moves). The ONLY configuration whose inner nodes feel
    ~zero net force is an EXACT cubic-lattice ball WITH a node at the origin and
    masses assigned by radius shell (antipodal pairs share a mass) — then every
    pull is opposed by an equal antipodal pull and cancels (max_residual ~ 1e-30).

    TRADE-OFF (documented for honesty): this gains EXACT force balance at the cost
    of the realistic mode's randomly-oriented local neighbour density. A real
    cubic lattice has uniform neighbour counts (a perfect crystal), unlike a
    relaxed random blob; we accept that because the user's criterion is "a
    virialized grid's inner nodes should not move", which only the lattice satisfies.

    Preserved contracts (identical to the realistic mode):
      * mean(masses) == M_ext_kg EXACTLY (mean-preserving).
      * Mass segregation: bigger node -> larger radius (positive mass-radius corr).
        vir_segregation/vir_mass_spread keep their meaning; spread=0 -> uniform.
      * Realized nearest-neighbour spacing (vir_s_metric) == S EXACTLY.
      * >= 2 distinct radii (a volume-filling ball, never a hollow shell).
      * Deterministic & seeded; independent of the global/particle RNG.

    Returns:
        (positions (N,3), masses (N,)) float64, mass-segregated, force-balanced.
    """
    n = int(n_nodes)
    rng = np.random.default_rng(seed)

    # ---- Exact cubic-lattice ball with a node at the origin (the centre) ----
    pos = _lattice_ball_positions(n)

    # ---- Re-target nearest-neighbour spacing to S (a pure global scale) ----
    if n >= 2:
        realized = nearest_neighbour_spacing(pos, vir_s_metric)
        if realized > 0.0:
            pos = pos * (float(S) / realized)

    # ---- Radius SHELLS: nodes at the same radius (e.g. antipodes) share a shell,
    # so they get an IDENTICAL mass -> antipodal pulls cancel exactly (force
    # balance). Shell index increases with radius (innermost = 0). ----
    r = np.linalg.norm(pos, axis=1)
    r_max = float(r.max()) if r.size else 0.0
    r_key = np.round(r / r_max, 9) if r_max > 0.0 else np.zeros(n)
    uniq, inv = np.unique(r_key, return_inverse=True)  # inv: node -> shell index
    n_shells = uniq.size
    shell_frac = inv.astype(np.float64) / max(n_shells - 1, 1)

    if vir_mass_rule == "radial":
        # Deterministic mass ~ f(shell): increases monotonically with radius,
        # scaled by spread & segregation. spread=0 -> uniform; seg=0 -> flat.
        x = shell_frac - shell_frac.mean()
        raw_masses = 1.0 + float(vir_mass_spread) * float(vir_segregation) * x
        raw_masses = np.maximum(raw_masses, 1e-12)
    else:  # "massfunc"
        # One log-normal mass per SHELL (so all nodes in a shell share a mass),
        # then assign sorted shell masses to shells by ascending radius
        # (bigger mass -> larger radius). seg blends sorted vs random shell order.
        if float(vir_mass_spread) == 0.0:
            raw_masses = np.ones(n, dtype=np.float64)
        else:
            g = rng.standard_normal(n_shells)
            shell_mass = np.exp(float(vir_mass_spread) * g)
            shell_sorted = np.sort(shell_mass)  # ascending
            order_random = rng.permutation(n_shells)
            seg = float(np.clip(vir_segregation, 0.0, 1.0))
            rank_sorted = np.arange(n_shells, dtype=np.float64)  # shell radius rank
            rank_random = np.empty(n_shells, dtype=np.float64)
            rank_random[order_random] = np.arange(n_shells, dtype=np.float64)
            blended = seg * rank_sorted + (1.0 - seg) * rank_random
            # blended[k] = target slot for shell k's mass; assign by sorted blend.
            slot_of_shell = np.argsort(np.argsort(blended, kind="stable"), kind="stable")
            shell_assigned = shell_sorted[slot_of_shell]
            raw_masses = shell_assigned[inv]

    # ---- Mean-preserving normalization: mean(masses) == M_ext_kg exactly ----
    raw_masses = np.asarray(raw_masses, dtype=np.float64)
    if raw_masses.mean() == 0.0:
        masses = np.full(n, float(M_ext_kg), dtype=np.float64)
    else:
        masses = float(M_ext_kg) * raw_masses / raw_masses.mean()

    return pos.astype(np.float64), masses.astype(np.float64)


def _force_residual_objective_and_grad(
    pos: np.ndarray, masses: np.ndarray, center_mass: float, G: float
) -> tuple[float, np.ndarray]:
    """Force-residual objective ``f = sum_i |a_i|^2`` and its position gradient.

    ``a_i`` is the net acceleration on node i from all OTHER nodes + a central node
    at the origin (node_net_accelerations, same 1e10 m floor). Minimising f drives
    every node toward force balance (a_i -> 0). Returns (f, grad) where grad has the
    same shape as pos; the analytic gradient is validated against finite differences
    in the tests. This is the objective Option B descends — NOT the potential energy
    (whose minimum is a collapse), so it has genuine force-balance equilibria.

    Cost is O(N^2) per call (one Python loop over source nodes), so a full relaxation
    is O(n_steps * N^2) — tractable up to a few thousand nodes.
    """
    pos = np.asarray(pos, dtype=np.float64)
    m = np.asarray(masses, dtype=np.float64)
    n = pos.shape[0]
    src = np.vstack([pos, np.zeros((1, 3))])
    sm = np.concatenate([m, np.array([center_mass], dtype=np.float64)])
    diff = src[None, :, :] - pos[:, None, :]            # r_ij = x_j - x_i, (N,S,3)
    r = np.sqrt(np.sum(diff * diff, axis=2))
    r = np.where(r < 1e10, 1e10, r)
    inv_r3 = 1.0 / (r * r * r)
    inv_r5 = inv_r3 / (r * r)
    self_mask = np.zeros((n, n + 1), dtype=bool)
    self_mask[:, :n] = np.eye(n, dtype=bool)
    inv_r3 = np.where(self_mask, 0.0, inv_r3)
    inv_r5 = np.where(self_mask, 0.0, inv_r5)
    a = np.sum((G * sm)[None, :, None] * inv_r3[:, :, None] * diff, axis=1)  # (N,3)
    f = float(np.sum(a * a))

    grad = np.zeros((n, 3), dtype=np.float64)
    for k in range(n):
        # Source = node k acting on all targets i (column k of diff/inv_r*).
        d3 = inv_r3[:, k]; d5 = inv_r5[:, k]; dvec = diff[:, k, :]  # x_k - x_i
        coef = G * sm[k]
        adotd = np.sum(a * dvec, axis=1)
        contrib = coef * (a * d3[:, None] - 3.0 * adotd[:, None] * dvec * d5[:, None])
        contrib[k] = 0.0
        gk = 2.0 * np.sum(contrib, axis=0)
        # Self term: a_k depends on x_k through every source j (row k).
        d3k = inv_r3[k, :]; d5k = inv_r5[k, :]; dvk = diff[k, :, :]  # x_j - x_k
        coefj = G * sm
        adotk = np.sum(a[k] * dvk, axis=1)
        j_action = coefj[:, None] * (
            a[k][None, :] * d3k[:, None] - 3.0 * adotk[:, None] * dvk * d5k[:, None])
        gk += 2.0 * (-np.sum(j_action, axis=0))
        grad[k] = gk
    return f, grad


def _gradient_relax_positions(
    positions: np.ndarray,
    masses: np.ndarray,
    *,
    n_steps: int,
    rate: float,
    hold_outer_frac: float,
    G: float | None = None,
) -> np.ndarray:
    """OPTION B: drive a node blob toward force balance by gradient descent.

    True iterative relaxation: starting from the given (realistic, segregated)
    positions, repeatedly step every INTERIOR node DOWN the gradient of the force-
    residual objective ``f = sum_i |a_i|^2`` (``_force_residual_objective_and_grad``)
    toward the equilibrium where each node's directional pulls cancel. The masses are
    held fixed (only positions relax, like a real cluster settling); the OUTER shell
    is pinned (``hold_outer_frac``) so the interior relaxes inside a fixed boundary —
    an outer node CANNOT be force-free on a finite canvas, and freeing it would just
    let the blob collapse onto its heaviest members.

    IMPORTANT — why descend ``|a|^2`` and not the potential: stepping a node ALONG its
    net acceleration descends the POTENTIAL, whose minimum is a gravitational COLLAPSE
    (every node falls onto the nearest heavy mass), which INCREASES the residual. The
    force-balance objective is ``|net force| -> 0``, so Option B descends ``sum|a_i|^2``;
    that surface has real interior equilibria.

    Per step: a fixed-fraction-of-spacing step
        ``x <- x - rate * S_char * grad / max_i|grad_i|``
    with a HALVING backtracking line search (up to a few halvings) so a step is
    accepted only if it lowers f — this keeps the descent monotone and prevents the
    overshoot a fixed rate causes near the basin. Normalising by ``max|grad|`` makes
    ``rate`` a spacing fraction, consistent across grid sizes and mass scales.

    Args:
        positions:       (N, 3) starting node positions in meters.
        masses:          (N,) node masses in kg (held fixed during relaxation).
        n_steps:         Number of relaxation iterations (>= 1).
        rate:            Initial step as a fraction of the characteristic NN spacing.
        hold_outer_frac: Fraction of nodes (largest centroid distance) pinned each
            step (0 -> none pinned; 0.3 -> outer 30% fixed).
        G:               Gravitational constant (defaults to CosmologicalConstants.G).

    Returns:
        (N, 3) relaxed positions (float64). Deterministic (no RNG): a pure descent.
    """
    if G is None:
        from .constants import CosmologicalConstants
        G = CosmologicalConstants.G
    pos = np.asarray(positions, dtype=np.float64).copy()
    m = np.asarray(masses, dtype=np.float64)
    n = pos.shape[0]
    if n < 2 or int(n_steps) < 1:
        return pos

    center_mass = float(np.mean(m))
    # Pin the OUTERMOST nodes (by centroid distance) as a fixed boundary.
    centroid = pos.mean(axis=0)
    d0 = np.linalg.norm(pos - centroid[None, :], axis=1)
    n_hold = int(round(max(0.0, min(1.0, float(hold_outer_frac))) * n))
    free_mask = np.ones(n, dtype=bool)
    if n_hold > 0:
        free_mask[np.argsort(d0, kind="stable")[-n_hold:]] = False

    f_cur, _ = _force_residual_objective_and_grad(pos, m, center_mass, G)
    for _ in range(int(n_steps)):
        f_cur, grad = _force_residual_objective_and_grad(pos, m, center_mass, G)
        grad[~free_mask] = 0.0
        g_max = float(np.max(np.linalg.norm(grad, axis=1)))
        if g_max <= 0.0:
            break  # at a stationary point
        s_char = nearest_neighbour_spacing(pos, "median")
        # Backtracking line search: accept the first step that lowers f.
        step_rate = float(rate)
        accepted = False
        for _bt in range(8):
            trial = pos - step_rate * s_char * grad / g_max
            f_trial, _ = _force_residual_objective_and_grad(
                trial, m, center_mass, G)
            if f_trial < f_cur:
                pos = trial
                f_cur = f_trial
                accepted = True
                break
            step_rate *= 0.5
        if not accepted:
            break  # no downhill step found -> converged to a local minimum

    return pos.astype(np.float64)


def _build_relaxed_grid(
    S: float,
    *,
    n_nodes: int,
    M_ext_kg: float,
    vir_extent: float,
    vir_mass_rule: str,
    vir_mass_spread: float,
    vir_segregation: float,
    vir_s_metric: str,
    vir_relax_steps: int,
    vir_relax_rate: float,
    hold_outer_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """OPTION B: a realistic segregated blob iteratively relaxed toward balance.

    Builds the realistic Fibonacci-sphere segregated layout (the ``vir_relax_steps==0``
    / lattice-mode-off configuration), then runs ``vir_relax_steps`` gradient-descent
    relaxation iterations (``_gradient_relax_positions``) to drive the INTERIOR nodes
    toward force balance while pinning the outer boundary. Finally re-targets the NN
    spacing to S so the spacing contract still holds. Masses are the realistic mode's
    segregated masses (UNCHANGED by relaxation), so all mass contracts are preserved.

    This is the counterpart to Option A (``_build_force_balanced_grid``, an analytic
    crystal): Option B never becomes a perfect lattice, so its DEEP-CENTER residual is
    the real test of whether a realistic relaxed structure can virialize at its core.

    Returns:
        (positions (N,3), masses (N,)) float64, segregated, interior-relaxed.
    """
    # Start from the realistic (un-balanced) segregated layout.
    pos, masses = build_virialized_grid(
        S, n_nodes=int(n_nodes), M_ext_kg=M_ext_kg, vir_extent=vir_extent,
        vir_mass_rule=vir_mass_rule, vir_mass_spread=vir_mass_spread,
        vir_segregation=vir_segregation, vir_s_metric=vir_s_metric,
        vir_relax_steps=0, vir_relax_mode="lattice", seed=seed,
    )
    if int(n_nodes) >= 2 and int(vir_relax_steps) >= 1:
        pos = _gradient_relax_positions(
            pos, masses, n_steps=int(vir_relax_steps), rate=float(vir_relax_rate),
            hold_outer_frac=float(hold_outer_frac),
        )
        # Re-target the NN spacing to S (a pure global factor; preserves balance).
        realized = nearest_neighbour_spacing(pos, vir_s_metric)
        if realized > 0.0:
            pos = pos * (float(S) / realized)
    return pos.astype(np.float64), masses.astype(np.float64)


# ---------------------------------------------------------------------------
# The INFINITE virialized meta-structure as the node geometry (vir_relax_mode="medium").
#
# Uses cosmo.virialized_medium.relax_virialized_medium to relax a PERIODIC self-gravitating medium
# to VIRIAL EQUILIBRIUM (2K/|U| ~ 1): homogeneous on average, disordered, locally clumpy -- the
# honest "virialized in an infinite universe" structure. (A crystal or glass is STATIC and
# force-balanced, which is NOT virialized: a virialized system is held up by velocity dispersion,
# its members orbit. This is why the old force-balance criterion was the wrong test.)
#
# "Us" / the progenitor and centerM: node 0 is our observable universe -- the progenitor node --
# at the origin, part of the pre-Big-Bang equilibrium. Per the Progenitor Hypothesis that node
# destabilized and BECAME the Big Bang, so in the SIMULATION (post-Big-Bang) "us" is the particle
# CLOUD (+ centerM outer mass), NOT an HMEA. center_node_mass (centerM) and the central-node mass
# are the SAME progenitor mass in different epochs. Therefore we relax the medium WITH the central
# node (it belongs in the equilibrium) but DROP node 0 from the returned HMEA set -- the cloud is
# it. (center_mass_frac is us's mass in units of the mean HMEA node mass; physically ~ centerM /
# M_value, i.e. small, but since node 0 is dropped its exact value only perturbs the equilibrium.)
#
# The relaxed structure is disk-cached under data/vir_medium/ by (n_nodes, sigma, seed,
# center_mass_frac); every S/M reuses it via the NN-spacing rescale + mean-preserving mass (the
# same contract as the other modes). Deterministic. See cosmo/virialized_medium.py +
# tests/test_virialized_medium.py.
# ---------------------------------------------------------------------------

def _medium_cache_dir() -> str:
    """Directory for cached relaxed virialized-medium structures (repo data/vir_medium)."""
    import os
    return os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "vir_medium"))


def _relax_medium_structure(n_total: int, sigma: float, seed: int, center_mass_frac: float):
    """Relax the periodic virialized medium (node 0 == "us"/progenitor at the origin); disk-cached.

    Returns (pos (n_total,3) centred on "us", weights (n_total,) node masses with mean ~ 1). n_total
    is n_HMEA + 1 (the extra node is "us"). Deterministic in (n_total, sigma, seed, center_mass_frac).
    """
    import os
    from .virialized_medium import relax_virialized_medium
    cdir = _medium_cache_dir()
    cache = os.path.join(cdir, f"medium_N{int(n_total)}_sig{float(sigma):.4f}_"
                               f"seed{int(seed)}_cm{float(center_mass_frac):.4f}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["pos"], d["weights"]
    r = relax_virialized_medium(n_nodes=int(n_total), sigma=float(sigma), seed=int(seed),
                                center_mass_frac=float(center_mass_frac))
    pos, w = np.asarray(r["pos"], dtype=np.float64), np.asarray(r["mass"], dtype=np.float64)
    try:
        os.makedirs(cdir, exist_ok=True)
        np.savez(cache, pos=pos, weights=w)
    except OSError:
        pass                                            # cache is best-effort
    return pos, w


def _build_medium_virialized_grid(S, *, n_nodes, M_ext_kg, vir_mass_spread, vir_s_metric, seed,
                                  center_mass_frac=1.0):
    """Assemble the HMEA grid from the virialized medium: relax n_nodes+1 nodes (node 0 == "us"),
    DROP "us" (it is the particle cloud in the sim, NOT an HMEA), rescale so the HMEAs' realized NN
    spacing (metric) == S, and mean-preserve the HMEA masses (mean == M_ext_kg). vir_mass_spread is
    the sigma of the log-normal node-mass function."""
    pos, w = _relax_medium_structure(int(n_nodes) + 1, float(vir_mass_spread), int(seed),
                                     float(center_mass_frac))
    hmea_pos, hmea_w = pos[1:], w[1:]                        # DROP node 0 ("us" / progenitor)
    masses = float(M_ext_kg) * hmea_w / hmea_w.mean()       # mean(HMEA masses) == M_ext_kg
    if len(hmea_pos) >= 2:
        realized = nearest_neighbour_spacing(hmea_pos, vir_s_metric)
        if realized > 0.0:
            hmea_pos = hmea_pos * (float(S) / realized)
    return hmea_pos.astype(np.float64), masses.astype(np.float64)


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
    vir_relax_steps: int = 1,
    vir_relax_mode: str = "lattice",
    vir_relax_rate: float = 0.1,
    vir_hold_outer_frac: float = 0.3,
    vir_extent_couples_nodes: bool = False,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """COUPLED virialized node grid: (positions (N,3), masses (N,)), mass-segregated.

    A physically-motivated *relaxed cluster*: smaller nodes cluster near the
    centre, more massive nodes are pushed FURTHER out (mass segregation). UNLIKE
    every positions-only geometry this returns BOTH arrays already paired, so mass
    i determines radius i. Reachable ONLY via this function (build_node_positions
    raises for "virialized").

    Modes, selected by ``vir_relax_mode`` x ``vir_relax_steps``:
      * ``vir_relax_mode="lattice"`` (DEFAULT) — the analytic Option-A path, where
        ``vir_relax_steps`` is a BALANCE LEVEL (not an iteration count):
          - ``vir_relax_steps == 0`` -> the REALISTIC Fibonacci-sphere layout (the
            rest of this docstring). Mass-segregated and volume-filling but NOT
            force-balanced: its inner nodes feel an O(10-70) net pull.
          - ``vir_relax_steps >= 1`` (the DEFAULT) -> a FORCE-BALANCED cubic-lattice
            ball with a node at the origin and masses assigned by radius shell. Its
            inner nodes feel ~ZERO net force (max_residual ~ 1e-30 for both rules)
            because antipodal pulls cancel exactly. See _build_force_balanced_grid.
            This is an ANALYTIC crystal, not an iterative relaxation.
      * ``vir_relax_mode="gradient"`` (OPTION B, opt-in) — a TRUE iterative relaxation:
        start from the realistic segregated blob and run ``vir_relax_steps`` gradient-
        descent steps (rate ``vir_relax_rate``) moving each interior node down its
        net-force gradient toward equilibrium, pinning the outer boundary
        (``vir_hold_outer_frac``). The masses stay segregated; the structure NEVER
        becomes a perfect lattice. Use the center-only ``virialization_residual``
        (``center_k`` / ``center_frac``) to test whether the DEEP CENTER reaches
        balance — the realistic-virialization question Option A sidesteps with a crystal.

    The DEFAULT (lattice, steps=1) is byte-identical to the prior force-balanced
    behaviour; gradient mode is fully opt-in.

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
        vir_relax_steps: In ``vir_relax_mode="lattice"`` a BALANCE LEVEL (int,
            default 1): 0 -> realistic Fibonacci layout (byte-identical to the legacy
            generator, NOT force-balanced); >= 1 -> the FORCE-BALANCED cubic-lattice
            ball (inner max_residual ~ 1e-30). In ``vir_relax_mode="gradient"`` it is
            the literal NUMBER of gradient-descent relaxation iterations (Option B).
        vir_relax_mode: "lattice" (default, Option A: analytic balance level) or
            "gradient" (Option B: true iterative relaxation of a realistic blob).
        vir_relax_rate: Gradient-descent step size as a fraction of the NN spacing
            (only used by vir_relax_mode="gradient", default 0.1).
        vir_hold_outer_frac: Fraction of outermost nodes pinned during gradient
            relaxation (only used by vir_relax_mode="gradient", default 0.3); a fixed
            boundary so the interior relaxes without the blob collapsing.
        vir_extent_couples_nodes: When False (DEFAULT) ``n_nodes`` is used as given
            (byte-identical). When True, ``vir_extent`` DRIVES the node count to hold
            the virialized ball's DENSITY constant: the effective count becomes
            ``round(n_nodes * vir_extent^3)`` (see ``extent_coupled_n_nodes``). This
            is the item-10 coupling ("a higher extent should imply MORE nodes, like
            centerM"); it ALSO makes ``vir_extent`` meaningful in the force-balanced
            lattice mode (where it was otherwise a no-op, since the lattice ball's
            radius derives from the node count, not from the radial-range knob). At
            the default ``vir_extent == 1.0`` the factor is exactly 1.0, so enabling
            the coupling at default extent leaves the node count (and the whole grid)
            unchanged.
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
    # Item-10 coupling: a larger radial extent auto-raises the node count to hold the
    # ball's DENSITY constant (N ~ extent^3). OFF by default (byte-identical); even
    # when ON, vir_extent == 1.0 is a no-op (factor 1.0). Applied here so EVERY mode
    # (force-balanced lattice, realistic Fibonacci, gradient Option B) sees the same
    # density-preserving count -> this is what makes vir_extent matter in the
    # force-balanced lattice mode, where the knob was otherwise inert.
    if vir_extent_couples_nodes:
        n_nodes = extent_coupled_n_nodes(int(n_nodes), vir_extent)
    if vir_mass_rule not in ("radial", "massfunc"):
        raise ValueError(
            f"Unknown vir_mass_rule {vir_mass_rule!r}; use 'radial' or 'massfunc'."
        )
    if vir_s_metric not in ("median", "mean"):
        raise ValueError(
            f"Unknown vir_s_metric {vir_s_metric!r}; use 'median' or 'mean'."
        )
    if vir_relax_mode not in ("lattice", "gradient", "medium"):
        raise ValueError(
            f"Unknown vir_relax_mode {vir_relax_mode!r}; use 'lattice', 'gradient', or 'medium'."
        )

    # "medium": the INFINITE virialized meta-structure -- a periodic self-gravitating medium relaxed
    # to VIRIAL EQUILIBRIUM (2K/|U| ~ 1): homogeneous on average, disordered, locally clumpy. Node 0
    # ("us"/the progenitor) is relaxed WITH the medium but DROPPED from the HMEAs (in the sim the
    # cloud is us). See _build_medium_virialized_grid + cosmo/virialized_medium.py.
    if vir_relax_mode == "medium" and int(n_nodes) >= 2:
        return _build_medium_virialized_grid(
            S,
            n_nodes=int(n_nodes),
            M_ext_kg=M_ext_kg,
            vir_mass_spread=vir_mass_spread,
            vir_s_metric=vir_s_metric,
            seed=seed,
            center_mass_frac=1.0,
        )

    # Option B (gradient): true iterative relaxation of a realistic segregated blob.
    if vir_relax_mode == "gradient" and int(n_nodes) >= 2:
        return _build_relaxed_grid(
            S,
            n_nodes=int(n_nodes),
            M_ext_kg=M_ext_kg,
            vir_extent=vir_extent,
            vir_mass_rule=vir_mass_rule,
            vir_mass_spread=vir_mass_spread,
            vir_segregation=vir_segregation,
            vir_s_metric=vir_s_metric,
            vir_relax_steps=int(vir_relax_steps),
            vir_relax_rate=float(vir_relax_rate),
            hold_outer_frac=float(vir_hold_outer_frac),
            seed=seed,
        )

    # Option A (lattice): vir_relax_steps as a BALANCE LEVEL: >= 1 -> analytic force-
    # balanced lattice ball; == 0 -> the legacy realistic Fibonacci layout below.
    if int(vir_relax_steps) >= 1 and int(n_nodes) >= 2:
        return _build_force_balanced_grid(
            S,
            n_nodes=int(n_nodes),
            M_ext_kg=M_ext_kg,
            vir_mass_rule=vir_mass_rule,
            vir_mass_spread=vir_mass_spread,
            vir_segregation=vir_segregation,
            vir_s_metric=vir_s_metric,
            seed=seed,
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
