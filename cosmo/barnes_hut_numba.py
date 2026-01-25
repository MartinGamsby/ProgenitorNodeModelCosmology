"""
Numba JIT-compiled Barnes-Hut octree for O(N log N) gravitational force calculation.

Real octree implementation: recursively subdivides space into octants,
groups distant particles into single center-of-mass nodes, and uses
the opening angle theta to decide when approximation is valid.

For theta=0 -> exact (equivalent to direct), theta=0.5 -> typical, theta=1.0 -> aggressive.
"""

import numpy as np
from numba import jit, float64, int32, boolean


@jit(nopython=True, cache=True)
def _get_octant(pos, center):
    """Return octant index (0-7) for position relative to center."""
    ix = 1 if pos[0] > center[0] else 0
    iy = 1 if pos[1] > center[1] else 0
    iz = 1 if pos[2] > center[2] else 0
    return ix * 4 + iy * 2 + iz


@jit(nopython=True, cache=True)
def _octant_center(parent_center, parent_half_size, octant):
    """Compute center of child octant."""
    cx = parent_center[0] + parent_half_size * (0.5 if octant // 4 == 1 else -0.5)
    cy = parent_center[1] + parent_half_size * (0.5 if (octant // 2) % 2 == 1 else -0.5)
    cz = parent_center[2] + parent_half_size * (0.5 if octant % 2 == 1 else -0.5)
    return np.array([cx, cy, cz])


@jit(nopython=True, cache=True)
def build_octree(positions, masses, max_nodes=50000):
    """
    Build Barnes-Hut octree using iterative insertion.

    Each node stores:
        - center_of_mass (3,): mass-weighted position
        - total_mass: sum of contained masses
        - center (3,): geometric center of cell
        - half_size: half the cell width
        - children (8,): indices of child nodes (-1 if empty)
        - is_leaf: whether node contains a single particle
        - particle_idx: index of particle if leaf (-1 otherwise)
        - n_particles: number of particles in subtree

    Returns flat arrays for all node properties.
    """
    N = len(positions)

    # Allocate node storage
    node_com = np.zeros((max_nodes, 3), dtype=float64)
    node_mass = np.zeros(max_nodes, dtype=float64)
    node_center = np.zeros((max_nodes, 3), dtype=float64)
    node_half_size = np.zeros(max_nodes, dtype=float64)
    node_children = np.full((max_nodes, 8), -1, dtype=int32)
    node_is_leaf = np.zeros(max_nodes, dtype=boolean)
    node_particle = np.full(max_nodes, -1, dtype=int32)
    node_n_particles = np.zeros(max_nodes, dtype=int32)

    # Root node: bounding box
    min_pos = np.empty(3)
    max_pos = np.empty(3)
    for d in range(3):
        min_pos[d] = positions[0, d]
        max_pos[d] = positions[0, d]
        for i in range(1, N):
            if positions[i, d] < min_pos[d]:
                min_pos[d] = positions[i, d]
            if positions[i, d] > max_pos[d]:
                max_pos[d] = positions[i, d]

    # Add small padding
    padding = 1e15  # ~0.03 pc padding
    for d in range(3):
        min_pos[d] -= padding
        max_pos[d] += padding

    # Root center and half-size
    root_center = (min_pos + max_pos) / 2.0
    half_sizes = (max_pos - min_pos) / 2.0
    root_half = max(half_sizes[0], max(half_sizes[1], half_sizes[2]))

    node_center[0] = root_center
    node_half_size[0] = root_half
    n_nodes = 1

    # Insert particles one by one
    for p in range(N):
        pos = positions[p]
        m = masses[p]

        # Walk tree from root
        current = 0

        depth = 0
        max_depth = 60  # Prevent infinite loops from coincident particles

        while depth < max_depth:
            depth += 1

            if node_n_particles[current] == 0:
                # Empty node: insert particle as leaf
                node_com[current] = pos.copy()
                node_mass[current] = m
                node_is_leaf[current] = True
                node_particle[current] = p
                node_n_particles[current] = 1
                break

            elif node_is_leaf[current]:
                # Leaf with existing particle: split into children
                old_p = node_particle[current]
                old_pos = positions[old_p]
                old_m = masses[old_p]

                # Clear leaf status
                node_is_leaf[current] = False
                node_particle[current] = -1

                # Update current node COM for both particles
                total = old_m + m
                node_com[current] = (old_pos * old_m + pos * m) / total
                node_mass[current] = total
                node_n_particles[current] = 2

                # Re-insert old particle into child
                oct_old = _get_octant(old_pos, node_center[current])
                if node_children[current, oct_old] == -1:
                    child_idx = n_nodes
                    n_nodes += 1
                    if n_nodes >= max_nodes:
                        break
                    node_children[current, oct_old] = child_idx
                    child_center = _octant_center(node_center[current], node_half_size[current], oct_old)
                    node_center[child_idx] = child_center
                    node_half_size[child_idx] = node_half_size[current] / 2.0

                child = node_children[current, oct_old]
                node_com[child] = old_pos.copy()
                node_mass[child] = old_m
                node_is_leaf[child] = True
                node_particle[child] = old_p
                node_n_particles[child] = 1

                # Now descend with new particle into correct octant
                oct_new = _get_octant(pos, node_center[current])
                if node_children[current, oct_new] == -1:
                    child_idx = n_nodes
                    n_nodes += 1
                    if n_nodes >= max_nodes:
                        break
                    node_children[current, oct_new] = child_idx
                    child_center = _octant_center(node_center[current], node_half_size[current], oct_new)
                    node_center[child_idx] = child_center
                    node_half_size[child_idx] = node_half_size[current] / 2.0

                current = node_children[current, oct_new]
                # Continue loop to insert new particle into child

            else:
                # Internal node: update COM and descend
                old_total = node_mass[current]
                new_total = old_total + m
                node_com[current] = (node_com[current] * old_total + pos * m) / new_total
                node_mass[current] = new_total
                node_n_particles[current] += 1

                # Descend into correct octant
                oct = _get_octant(pos, node_center[current])
                if node_children[current, oct] == -1:
                    child_idx = n_nodes
                    n_nodes += 1
                    if n_nodes >= max_nodes:
                        break
                    node_children[current, oct] = child_idx
                    child_center = _octant_center(node_center[current], node_half_size[current], oct)
                    node_center[child_idx] = child_center
                    node_half_size[child_idx] = node_half_size[current] / 2.0

                current = node_children[current, oct]

    return (node_com, node_mass, node_center, node_half_size,
            node_children, node_is_leaf, node_particle, node_n_particles, n_nodes)


@jit(nopython=True, cache=True)
def calculate_forces_barnes_hut(
    positions, masses, softening, theta, G,
    node_com, node_mass, node_center, node_half_size,
    node_children, node_is_leaf, node_particle, n_nodes
):
    """
    Calculate gravitational accelerations using Barnes-Hut tree traversal.

    For each particle, walk the tree. At each node:
    - If node is far enough (size/distance < theta), use COM approximation
    - Otherwise, recurse into children

    Args:
        positions: (N, 3) particle positions in meters
        masses: (N,) particle masses in kg
        softening: gravitational softening length in meters
        theta: opening angle (0=exact, 0.5=typical, 1.0=aggressive)
        G: gravitational constant
        node_*: octree arrays from build_octree()

    Returns:
        (N, 3) accelerations in m/s²
    """
    N = len(positions)
    accelerations = np.zeros((N, 3), dtype=float64)

    # Stack for iterative tree traversal (avoid recursion in Numba)
    stack = np.zeros(2000, dtype=int32)

    for i in range(N):
        px = positions[i, 0]
        py = positions[i, 1]
        pz = positions[i, 2]

        # Reset stack
        stack_top = 0
        stack[stack_top] = 0  # Start at root
        stack_top += 1

        while stack_top > 0:
            stack_top -= 1
            node = stack[stack_top]

            if node < 0 or node >= n_nodes:
                continue

            # Skip self
            if node_is_leaf[node] and node_particle[node] == i:
                continue

            # Skip empty nodes
            if node_mass[node] == 0.0:
                continue

            # Distance from particle to node COM
            dx = node_com[node, 0] - px
            dy = node_com[node, 1] - py
            dz = node_com[node, 2] - pz
            r2 = dx*dx + dy*dy + dz*dz
            r = np.sqrt(r2)

            # Opening angle criterion: s/d < theta
            s = 2.0 * node_half_size[node]  # Cell size

            if node_is_leaf[node] or (r > 0 and s / r < theta):
                # Use COM approximation (or it's a single particle)
                r_soft = np.sqrt(r2 + softening * softening)
                r_soft3 = r_soft * r_soft * r_soft

                f = G * node_mass[node] / r_soft3

                accelerations[i, 0] += f * dx
                accelerations[i, 1] += f * dy
                accelerations[i, 2] += f * dz
            else:
                # Open node: push children onto stack
                for c in range(8):
                    child = node_children[node, c]
                    if child >= 0 and stack_top < 1999:
                        stack[stack_top] = child
                        stack_top += 1

    return accelerations


class NumbaBarnesHutTree:
    """
    Barnes-Hut octree solver using Numba JIT compilation.

    O(N log N) force calculation with configurable accuracy via opening angle theta.
    theta=0 -> exact (equivalent to O(N^2) direct)
    theta=0.5 -> standard accuracy
    theta=1.0 -> fast/approximate
    """

    def __init__(self, theta: float = 0.5, softening_m: float = 1e24, G: float = 6.674e-11):
        self.theta = theta
        self.softening_m = softening_m
        self.G = G

        self._tree = None
        self.positions_m = None
        self.masses_kg = None

    def build_tree(self, positions_m: np.ndarray, masses_kg: np.ndarray) -> None:
        """Build octree from particle positions and masses."""
        self.positions_m = np.ascontiguousarray(positions_m, dtype=np.float64)
        self.masses_kg = np.ascontiguousarray(masses_kg, dtype=np.float64)

        # Scale max_nodes with particle count (typically ~8N nodes needed)
        max_nodes = max(10000, len(positions_m) * 10)
        self._tree = build_octree(self.positions_m, self.masses_kg, max_nodes)

    def calculate_all_accelerations(self) -> np.ndarray:
        """
        Calculate accelerations using Barnes-Hut tree traversal.

        Returns:
            (N, 3) accelerations in m/s²
        """
        (node_com, node_mass, node_center, node_half_size,
         node_children, node_is_leaf, node_particle, node_n_particles, n_nodes) = self._tree

        return calculate_forces_barnes_hut(
            self.positions_m, self.masses_kg,
            self.softening_m, self.theta, self.G,
            node_com, node_mass, node_center, node_half_size,
            node_children, node_is_leaf, node_particle, n_nodes
        )
