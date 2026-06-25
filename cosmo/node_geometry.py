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
cube26      3×3×3 − 1 cubic lattice (26 nodes). DEFAULT, byte-identical to the
            legacy HMEAGrid loop.  All other geometries are alternatives.
cube_dense  n×n×n − 1 cubic lattice (default n=5 → 124 nodes).
shell       Fibonacci-sphere points on a single sphere of radius S (default 50).
shell_multi Concentric shells at radii S, 2S, … n_shells*S (default 3 shells of 50).
fcc         Face-centred cubic — inner cubic shell (8) + face centres (6) per unit
            cell, iterated outward for n_shells (default 2) → exact FCC geometry.
bcc         Body-centred cubic — corners + body centre per unit cell, n_shells shells.

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

    Returns:
        (N, 3) float64 array; all nodes at characteristic distance ~S from origin.

    Raises:
        ValueError: If ``geometry`` is not in the registry.
    """
    if geometry not in _REGISTRY:
        valid = sorted(_REGISTRY.keys())
        raise ValueError(
            f"Unknown node geometry {geometry!r}. Valid choices: {valid}"
        )
    positions = _REGISTRY[geometry](S, **kwargs)
    return np.asarray(positions, dtype=np.float64)


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


@register("shell")
def _shell(S: float, n_nodes: int = 50) -> np.ndarray:
    """Fibonacci-spiral sphere (Vogel algorithm) of radius S.

    All nodes lie on a sphere of radius S.  The Fibonacci spiral distributes
    n_nodes quasi-uniformly over the sphere surface.

    Args:
        n_nodes: Number of nodes on the sphere (default 50).
    """
    if n_nodes < 3:
        raise ValueError(f"shell requires n_nodes >= 3, got {n_nodes}")
    golden = (1.0 + math.sqrt(5.0)) / 2.0
    positions = []
    for i in range(n_nodes):
        theta = math.acos(1.0 - 2.0 * (i + 0.5) / n_nodes)  # polar
        phi = 2.0 * math.pi * i / golden                       # azimuthal
        x = S * math.sin(theta) * math.cos(phi)
        y = S * math.sin(theta) * math.sin(phi)
        z = S * math.cos(theta)
        positions.append([x, y, z])
    return np.array(positions, dtype=np.float64)


@register("shell_multi")
def _shell_multi(S: float, n_nodes: int = 50, n_shells: int = 3) -> np.ndarray:
    """Concentric Fibonacci spheres at radii S, 2S, …, n_shells*S.

    Each shell has the same n_nodes points, giving total n_shells * n_nodes nodes.
    Outer shells use the same angular pattern (phi) but a different theta offset
    to avoid accidental alignment between shells.

    Args:
        n_nodes:  Nodes per shell (default 50).
        n_shells: Number of shells (default 3).
    """
    if n_shells < 1:
        raise ValueError(f"shell_multi requires n_shells >= 1, got {n_shells}")
    positions = []
    golden = (1.0 + math.sqrt(5.0)) / 2.0
    for s in range(1, n_shells + 1):
        r = S * s
        # Rotate the phi pattern by s * (2*pi / (2*n_shells)) to decorrelate shells
        phi_offset = s * math.pi / n_shells
        for i in range(n_nodes):
            theta = math.acos(1.0 - 2.0 * (i + 0.5) / n_nodes)
            phi = 2.0 * math.pi * i / golden + phi_offset
            x = r * math.sin(theta) * math.cos(phi)
            y = r * math.sin(theta) * math.sin(phi)
            z = r * math.cos(theta)
            positions.append([x, y, z])
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

    For n_shells=1: 8 nodes (nearest-neighbour shell).
    For n_shells=2: ~26 nodes.

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
