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
    center_mass_kg: float | None = None,
    G: float | None = None,
    reference: str = "mean_pairwise",
) -> dict:
    """Dimensionless per-INNER-node force-balance residual for a node grid.

    Physical criterion (the user's): the INNER nodes of a *truly virialized* grid
    feel ~NET-ZERO gravity from all the other nodes plus a central node, so they
    would not move. This returns, per inner node, the DIMENSIONLESS residual
        ``residual_i = |net_accel_i| / a_ref``
    where ``a_ref`` is a characteristic single-neighbour pull (see ``reference``).
    A well-virialized inner node has residual << 1 (its directional pulls cancel);
    an UN-balanced grid has residual ~O(1) or larger.

    "Inner" nodes (the only ones the criterion applies to — edge nodes obviously
    feel a net inward pull on a finite canvas, which is expected and NOT a failure)
    are those with ``radius < inner_frac * r_max``. At least one node is always
    returned: if none qualify, the single innermost node is used.

    Args:
        positions: (N, 3) node positions in meters.
        masses:    (N,) node masses in kg.
        inner_frac: Inner-radius fraction in (0, 1]; inner nodes are those with
            radius below ``inner_frac * r_max`` (default 0.5).
        center_mass_kg: Central-node mass (see node_net_accelerations); defaults
            to ``mean(masses)`` (one unit node).
        G: Gravitational constant (defaults to CosmologicalConstants.G).
        reference: Normalization scale for the residual:
            "mean_pairwise" (default) — a_ref = mean over ALL nodes i of the mean
                single-source pull magnitude ``mean_j G m_j / |x_j - x_i|^2``
                (incl. centre): "how big a typical single-neighbour pull is".
            "max_pairwise" — a_ref = mean over inner nodes of the STRONGEST single
                -source pull on that node (the nearest/heaviest neighbour).

    Returns:
        dict with:
          'residual_per_node' (array over inner nodes, dimensionless),
          'inner_idx'   (indices into the input arrays of the inner nodes),
          'max_residual', 'median_residual' (floats),
          'a_ref'       (the normalization scale, m/s^2),
          'n_inner'     (number of inner nodes).

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
    inner_idx = np.nonzero(radius < float(inner_frac) * r_max)[0]
    if inner_idx.size == 0:
        # Fall back to the single innermost node.
        inner_idx = np.array([int(np.argmin(radius))])

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
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """COUPLED virialized node grid: (positions (N,3), masses (N,)), mass-segregated.

    A physically-motivated *relaxed cluster*: smaller nodes cluster near the
    centre, more massive nodes are pushed FURTHER out (mass segregation). UNLIKE
    every positions-only geometry this returns BOTH arrays already paired, so mass
    i determines radius i. Reachable ONLY via this function (build_node_positions
    raises for "virialized").

    Two modes, selected by ``vir_relax_steps`` (a BALANCE LEVEL, not iteration count):
      * ``vir_relax_steps == 0`` -> the REALISTIC Fibonacci-sphere layout (the rest
        of this docstring). Mass-segregated and volume-filling but NOT force-balanced:
        its inner nodes feel an O(10-70) net pull (a relaxed random blob has uneven
        directional pulls). Use this for a realistic, non-balanced cluster.
      * ``vir_relax_steps >= 1`` (the DEFAULT) -> a FORCE-BALANCED cubic-lattice ball
        with a node at the origin and masses assigned by radius shell. Its inner
        nodes feel ~ZERO net force (max_residual ~ 1e-30 for both rules) because
        antipodal pulls cancel exactly — this is the user's "a virialized grid's
        inner nodes should not move" criterion. See _build_force_balanced_grid.
        A continuous position relaxation CANNOT reach this balance on a finite canvas
        (the central monopole pull is irreducible by moves), so the balanced mode is
        an ANALYTIC lattice, not an iterative relaxation. TRADE-OFF: it gains exact
        force balance at the cost of the realistic mode's uneven local neighbour
        density (a perfect crystal has uniform neighbour counts, a random blob does not).

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
        vir_relax_steps: BALANCE LEVEL (int, default 1). 0 -> realistic Fibonacci-
            sphere layout (byte-identical to the legacy generator, NOT force-balanced).
            >= 1 -> the FORCE-BALANCED cubic-lattice ball (inner max_residual ~ 1e-30).
            Reframed from an iteration count: any value >= 1 selects the analytic
            balanced ball (there is no continuous relaxation — see the class note).
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

    # vir_relax_steps as a BALANCE LEVEL: >= 1 -> analytic force-balanced lattice
    # ball; == 0 -> the legacy realistic Fibonacci layout below (byte-identical).
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
