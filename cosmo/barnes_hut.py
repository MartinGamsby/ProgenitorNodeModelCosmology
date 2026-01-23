"""
Barnes-Hut octree implementation for O(N log N) gravitational force calculation.

The Barnes-Hut algorithm uses a hierarchical octree to approximate distant
particle groups as single point masses (center of mass), reducing O(N²)
pairwise interactions to O(N log N).

Key idea: If a cubic cell's width divided by distance to particle is small
(< theta), treat all particles in that cell as a single mass at the center
of mass position. Otherwise, recursively descend into child cells.
"""

import numpy as np
from typing import Optional, Tuple, List
from cosmo.constants import CosmologicalConstants


class OctreeNode:
    """
    Single node in Barnes-Hut octree representing a cubic region of space.

    Each node either:
    - Is a leaf containing a single particle (particle_idx is set)
    - Is internal with 8 child octants (children is set)
    - Is empty (both None)

    Attributes:
        center_of_mass_m: COM position (3D) in meters
        total_mass_kg: Total mass of all particles in this node
        bounds: (min_corner, max_corner) defining cubic region
        children: List of 8 child nodes (internal) or None (leaf/empty)
        particle_idx: Index of particle if leaf node, else None
        size_m: Width of cubic cell in meters
    """

    def __init__(self, bounds: Tuple[np.ndarray, np.ndarray]):
        """
        Initialize octree node with spatial bounds.

        Args:
            bounds: (min_corner, max_corner) arrays of shape (3,)
        """
        self.bounds = bounds
        self.min_corner_m = bounds[0]
        self.max_corner_m = bounds[1]
        self.size_m = np.linalg.norm(self.max_corner_m - self.min_corner_m)

        # COM and mass (computed during tree construction)
        self.center_of_mass_m: Optional[np.ndarray] = None
        self.total_mass_kg: float = 0.0

        # Either children OR particle_idx is set (not both)
        self.children: Optional[List['OctreeNode']] = None
        self.particle_idx: Optional[int] = None

    def is_leaf(self) -> bool:
        """Check if this is a leaf node (contains single particle)."""
        return self.particle_idx is not None

    def is_empty(self) -> bool:
        """Check if this node contains no particles."""
        return self.total_mass_kg == 0.0

    def get_octant(self, position_m: np.ndarray) -> int:
        """
        Determine which of 8 octants a position falls into.

        Octant numbering:
        0: (-, -, -)  1: (+, -, -)  2: (-, +, -)  3: (+, +, -)
        4: (-, -, +)  5: (+, -, +)  6: (-, +, +)  7: (+, +, +)

        Args:
            position_m: 3D position in meters

        Returns:
            Octant index 0-7
        """
        center_m = (self.min_corner_m + self.max_corner_m) / 2.0
        octant = 0
        if position_m[0] >= center_m[0]:
            octant |= 1
        if position_m[1] >= center_m[1]:
            octant |= 2
        if position_m[2] >= center_m[2]:
            octant |= 4
        return octant

    def get_octant_bounds(self, octant: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get spatial bounds for a specific octant.

        Args:
            octant: Octant index 0-7

        Returns:
            (min_corner, max_corner) for the octant
        """
        center_m = (self.min_corner_m + self.max_corner_m) / 2.0

        min_corner = np.copy(self.min_corner_m)
        max_corner = np.copy(self.max_corner_m)

        # x dimension
        if octant & 1:
            min_corner[0] = center_m[0]
        else:
            max_corner[0] = center_m[0]

        # y dimension
        if octant & 2:
            min_corner[1] = center_m[1]
        else:
            max_corner[1] = center_m[1]

        # z dimension
        if octant & 4:
            min_corner[2] = center_m[2]
        else:
            max_corner[2] = center_m[2]

        return (min_corner, max_corner)


class BarnesHutTree:
    """
    Barnes-Hut octree for O(N log N) gravitational force approximation.

    Usage:
        tree = BarnesHutTree(theta=0.5, softening_m=1e24)
        tree.build_tree(positions_m, masses_kg)
        accelerations = tree.calculate_all_accelerations()

    Attributes:
        theta: Opening angle criterion (width/distance threshold)
        softening_m: Gravitational softening length in meters
        const: Gravitational constant G
        root: Root node of octree
        positions_m: Particle positions (N, 3)
        masses_kg: Particle masses (N,)
    """

    def __init__(self, theta: float = 0.5, softening_m: float = 1e24):
        """
        Initialize Barnes-Hut tree.

        Args:
            theta: Opening angle criterion (0.0 = exact, 1.0 = fast/approximate)
                   Typical values: 0.3 (accurate), 0.5 (standard), 0.7 (fast)
            softening_m: Gravitational softening length in meters
        """
        self.theta = theta
        self.softening_m = softening_m
        self.const = CosmologicalConstants()

        self.root: Optional[OctreeNode] = None
        self.positions_m: Optional[np.ndarray] = None
        self.masses_kg: Optional[np.ndarray] = None

    def build_tree(self, positions_m: np.ndarray, masses_kg: np.ndarray) -> None:
        """
        Construct octree from particle positions and masses.

        Args:
            positions_m: Particle positions, shape (N, 3)
            masses_kg: Particle masses, shape (N,)
        """
        self.positions_m = positions_m
        self.masses_kg = masses_kg

        # Determine bounding box for all particles
        min_corner = np.min(positions_m, axis=0)
        max_corner = np.max(positions_m, axis=0)

        # Expand slightly to ensure all particles are strictly inside
        epsilon = 1e-10 * np.linalg.norm(max_corner - min_corner)
        min_corner -= epsilon
        max_corner += epsilon

        # Create root node spanning all particles
        self.root = OctreeNode((min_corner, max_corner))

        # Insert each particle into tree
        for i in range(len(positions_m)):
            self._insert_particle(self.root, i)

    def _insert_particle(self, node: OctreeNode, particle_idx: int) -> None:
        """
        Recursively insert particle into octree, updating COM along the way.

        Args:
            node: Current node to insert into
            particle_idx: Index of particle to insert
        """
        pos_m = self.positions_m[particle_idx]
        mass_kg = self.masses_kg[particle_idx]

        # Update this node's COM and total mass
        if node.center_of_mass_m is None:
            # First particle in this node
            node.center_of_mass_m = np.copy(pos_m)
            node.total_mass_kg = mass_kg
            node.particle_idx = particle_idx
        else:
            # Update COM: r_com_new = (m1*r1 + m2*r2) / (m1 + m2)
            total_mass_new = node.total_mass_kg + mass_kg
            node.center_of_mass_m = (
                node.center_of_mass_m * node.total_mass_kg + pos_m * mass_kg
            ) / total_mass_new
            node.total_mass_kg = total_mass_new

            # If node was a leaf, need to subdivide
            if node.is_leaf():
                # Node contained one particle, now adding second
                old_particle_idx = node.particle_idx
                node.particle_idx = None  # No longer a leaf

                # Create 8 children
                node.children = [
                    OctreeNode(node.get_octant_bounds(i)) for i in range(8)
                ]

                # Re-insert old particle into appropriate child
                old_pos_m = self.positions_m[old_particle_idx]
                octant_old = node.get_octant(old_pos_m)
                self._insert_particle(node.children[octant_old], old_particle_idx)

                # Insert new particle into appropriate child
                octant_new = node.get_octant(pos_m)
                self._insert_particle(node.children[octant_new], particle_idx)
            else:
                # Node already has children, recurse into correct octant
                octant = node.get_octant(pos_m)
                self._insert_particle(node.children[octant], particle_idx)

    def calculate_acceleration(self, particle_idx: int) -> np.ndarray:
        """
        Calculate gravitational acceleration on a single particle.

        Args:
            particle_idx: Index of particle to calculate force on

        Returns:
            Acceleration vector (3,) in m/s²
        """
        pos_m = self.positions_m[particle_idx]
        acceleration_mps2 = np.zeros(3)

        # Recursively calculate force from tree
        self._calculate_node_force(
            self.root, pos_m, particle_idx, acceleration_mps2
        )

        return acceleration_mps2

    def _calculate_node_force(
        self,
        node: OctreeNode,
        particle_pos_m: np.ndarray,
        particle_idx: int,
        acceleration_mps2: np.ndarray
    ) -> None:
        """
        Recursively calculate force contribution from a node.

        Args:
            node: Current octree node
            particle_pos_m: Position of particle we're calculating force on
            particle_idx: Index of particle (to avoid self-interaction)
            acceleration_mps2: Accumulator for acceleration (modified in-place)
        """
        if node.is_empty():
            return

        # Don't calculate self-interaction
        if node.is_leaf() and node.particle_idx == particle_idx:
            return

        # Vector from particle to node's COM
        r_vec_m = node.center_of_mass_m - particle_pos_m
        r_m = np.linalg.norm(r_vec_m)

        # Opening angle criterion: width/distance < theta?
        if node.is_leaf() or (node.size_m / r_m < self.theta):
            # Use COM approximation for this node
            r_soft_m = np.sqrt(r_m**2 + self.softening_m**2)

            # a = G * M / r_soft²  in direction of r_vec
            a_mag_mps2 = self.const.G * node.total_mass_kg / r_soft_m**2
            acceleration_mps2[:] += a_mag_mps2 * (r_vec_m / r_soft_m)
        else:
            # Node too close, recurse into children
            for child in node.children:
                self._calculate_node_force(
                    child, particle_pos_m, particle_idx, acceleration_mps2
                )

    def calculate_all_accelerations(self) -> np.ndarray:
        """
        Calculate gravitational accelerations on all particles.

        Returns:
            Accelerations array, shape (N, 3) in m/s²
        """
        N = len(self.positions_m)
        accelerations_mps2 = np.zeros((N, 3))

        for i in range(N):
            accelerations_mps2[i] = self.calculate_acceleration(i)

        return accelerations_mps2
