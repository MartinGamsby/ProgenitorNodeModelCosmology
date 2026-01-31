"""
Particle and Grid Structures
Represents galaxies/clusters as particles and HMEA nodes as boundary conditions
"""

from typing import Optional, Tuple
import numpy as np
from .constants import CosmologicalConstants, ExternalNodeParameters, LambdaCDMParameters


class Particle:
    """Represents a single galaxy cluster or tracer particle"""
    
    def __init__(self, position, velocity, mass_kg: float, particle_id: int = 0):
        """Initialize a particle with position/velocity in meters and m/s."""
        self.pos = np.array(position, dtype=np.float64)
        self.vel = np.array(velocity, dtype=np.float64)
        self.mass_kg = float(mass_kg)
        self.id = particle_id
        self.acc = np.zeros(3, dtype=np.float64)  # Acceleration

    def __repr__(self):
        return f"Particle(id={self.id}, mass_kg={self.mass_kg:.2e} kg, pos={self.pos})"


class ParticleSystem:
    """Collection of particles representing the observable universe"""
    
    def __init__(self, n_particles: int = 1000, box_size_m: Optional[float] = None,
                 total_mass_kg: Optional[float] = None, a_start: float = 1.0,
                 use_dark_energy: bool = True, damping_factor_override: float = None,
                 mass_randomize: float = 0.5):
        """
        Initialize particle system with damped Hubble flow initial conditions.

        Args:
            n_particles: Number of particles
            box_size_m: Box size in meters (default: Hubble radius)
            total_mass_kg: Total mass in kg (default: M_observable)
            a_start: Initial scale factor
            use_dark_energy: Whether dark energy is enabled
            damping_factor_override: Damping factor for initial Hubble flow
            mass_randomize: Mass distribution randomness (0.0 = equal masses,
                           1.0 = masses from 0 to 2x mean, 0.5 = default)
        """
        const = CosmologicalConstants()

        self.n_particles = n_particles
        self.box_size_m = box_size_m if box_size_m is not None else const.R_hubble
        self.total_mass_kg = total_mass_kg if total_mass_kg is not None else const.M_observable_kg
        self.a_start = a_start
        self.use_dark_energy = use_dark_energy
        self.damping_factor = damping_factor_override
        self.mass_randomize = np.clip(mass_randomize, 0.0, 1.0)

        self.particles = []
        self.time = 0.0

        # Initialize particles
        self._initialize_particles()
        
    def _initialize_particles(self) -> None:
        """Create initial particle distribution with Hubble flow."""
        lcdm = LambdaCDMParameters()

        # Use model-appropriate Hubble parameter for initial velocity
        # ΛCDM: H includes dark energy (Ω_Λ) → higher expansion rate
        # Matter-only: H without dark energy → lower expansion rate
        # This ensures each model's N-body matches its own Friedmann solution
        if self.use_dark_energy:
            H_start = lcdm.H_at_time(self.a_start)
            print(f"[ParticleSystem] Using ΛCDM H(a={self.a_start:.3f}) = {H_start:.3e} s⁻¹")
        else:
            H_start = lcdm.H_matter_only(self.a_start)
            print(f"[ParticleSystem] Using matter-only H(a={self.a_start:.3f}) = {H_start:.3e} s⁻¹")

        if self.damping_factor is not None:
            damping_factor = self.damping_factor
        else:
            damping_factor = 1
        damping_factor = np.clip(damping_factor, 0.0, 1.0)

        print("[ParticleSystem] Damping factor for initial:", damping_factor)

        # Generate particle masses
        mean_mass_kg = self.total_mass_kg / self.n_particles
        if self.mass_randomize > 0 and self.n_particles > 1:
            # Generate random masses with specified randomization level
            # mass_randomize=1.0: uniform in [0, 2*mean], so range is 2*mean
            # mass_randomize=0.5: uniform in [0.5*mean, 1.5*mean], range is mean
            # mass_randomize=0.0: all masses equal to mean
            half_range = self.mass_randomize * mean_mass_kg
            raw_masses = np.random.uniform(
                mean_mass_kg - half_range,
                mean_mass_kg + half_range,
                self.n_particles
            )
            # Ensure no negative masses (shouldn't happen unless randomize > 1, but be safe)
            raw_masses = np.maximum(raw_masses, 1e-10 * mean_mass_kg)
            # Normalize to preserve total mass exactly
            particle_masses_kg = raw_masses * (self.total_mass_kg / np.sum(raw_masses))
            print(f"[ParticleSystem] Mass randomize={self.mass_randomize:.2f}: "
                  f"min={np.min(particle_masses_kg):.2e}, max={np.max(particle_masses_kg):.2e}, "
                  f"mean={np.mean(particle_masses_kg):.2e} kg")
        else:
            particle_masses_kg = np.full(self.n_particles, mean_mass_kg)

        # Scale box_size so that the RMS radius matches the target
        # For a uniform sphere of radius R, RMS radius = R * sqrt(3/5) ≈ 0.775*R
        # We want RMS = box_size/2, so R_sphere = box_size/2 / 0.775
        # This means we need to use a sphere of radius: box_size/2 / sqrt(3/5)
        sphere_radius_m = (self.box_size_m / 2) / np.sqrt(3/5)

        # First, generate all positions using rejection sampling
        # This keeps position RNG calls separate from velocity RNG calls
        positions = []
        for i in range(self.n_particles):
            # Random position uniformly in sphere of radius sphere_radius_m
            # Using rejection sampling for clarity
            while True:
                pos = np.random.uniform(-sphere_radius_m, sphere_radius_m, 3)
                if np.linalg.norm(pos) <= sphere_radius_m:
                    break
            positions.append(pos)

        # CRITICAL: Center positions FIRST before calculating velocities
        # Random particle distribution creates non-zero COM position
        # We must center BEFORE velocity calculation so v_hubble = H*r uses centered positions
        positions_arr = np.array(positions)
        com_position = np.mean(positions_arr, axis=0)

        print(f"[ParticleSystem] Centering COM position: [{com_position[0]:.3e}, {com_position[1]:.3e}, {com_position[2]:.3e}] m")

        # Center the positions array
        centered_positions = positions_arr - com_position

        # CRITICAL: Normalize to exact target RMS radius
        # Random particle rejection sampling creates slight RMS variation even with same seed
        # This causes initialization artifacts in model comparisons (matter-only appearing
        # to "exceed LCDM" initially when it's just starting 0.5% larger by chance)
        # Normalization ensures exact comparison: any deviation is real physics, not randomness
        current_rms = np.sqrt(np.mean(np.sum(centered_positions**2, axis=1)))
        target_rms = self.box_size_m / 2  # RMS should be half box size

        # Handle edge case: if RMS is already very small (e.g., n=1 particle at origin),
        # skip normalization to avoid division by zero
        if current_rms > 1e-10 * target_rms:  # Only normalize if RMS is non-negligible
            scale_factor = target_rms / current_rms
            centered_positions *= scale_factor

            print(f"[ParticleSystem] Normalized RMS radius: {current_rms:.6e} -> {target_rms:.6e} m (scale={scale_factor:.6f})")

            # Verify normalization succeeded
            final_rms = np.sqrt(np.mean(np.sum(centered_positions**2, axis=1)))
            assert abs(final_rms - target_rms) / target_rms < 1e-10, \
                f"RMS normalization failed: {final_rms:.6e} vs {target_rms:.6e}"
        else:
            print(f"[ParticleSystem] Skipping RMS normalization (current RMS={current_rms:.3e} is negligible)")

        # Now generate velocities using CENTERED and NORMALIZED positions
        # This ensures velocity initialization is independent of rejection sampling randomness
        for i in range(self.n_particles):
            pos = centered_positions[i]  # Use centered and normalized position!

            # Initial velocity: Damped Hubble flow + small peculiar velocity
            # Damping compensates for lack of ongoing Hubble drag during integration
            v_hubble = damping_factor * H_start * pos
            v_peculiar = np.random.normal(0, 1e5, 3)  # ~100 km/s peculiar velocity
            vel = v_hubble + v_peculiar

            particle = Particle(pos, vel, particle_masses_kg[i], particle_id=i)
            self.particles.append(particle)

        # CRITICAL: Remove center-of-mass velocity to prevent bulk motion
        # With Hubble flow v = H*r, random particle positions create non-zero COM velocity
        # This causes the entire system to drift, appearing as unphysical expansion
        velocities = np.array([p.vel for p in self.particles])
        com_velocity = np.mean(velocities, axis=0)

        print(f"[ParticleSystem] Removing COM velocity: [{com_velocity[0]:.3e}, {com_velocity[1]:.3e}, {com_velocity[2]:.3e}] m/s")

        # Apply COM velocity correction to each particle
        for particle in self.particles:
            particle.vel -= com_velocity
    
    def get_positions(self) -> np.ndarray:
        """Get all particle positions as (N, 3) array."""
        return np.array([p.pos for p in self.particles])

    def get_velocities(self) -> np.ndarray:
        """Get all particle velocities as (N, 3) array."""
        return np.array([p.vel for p in self.particles])

    def get_masses(self) -> np.ndarray:
        """Get all particle masses as (N,) array."""
        return np.array([p.mass_kg for p in self.particles])

    def get_accelerations(self) -> np.ndarray:
        """Get all particle accelerations as (N, 3) array."""
        return np.array([p.acc for p in self.particles])

    def set_accelerations(self, accelerations: np.ndarray) -> None:
        """Set accelerations for all particles."""
        for i, particle in enumerate(self.particles):
            particle.acc = accelerations[i]

    def update_positions(self, dt_s: float) -> None:
        """Update positions using current velocities."""
        for particle in self.particles:
            particle.pos += particle.vel * dt_s

    def update_velocities(self, dt_s: float) -> None:
        """Update velocities using current accelerations."""
        for particle in self.particles:
            particle.vel += particle.acc * dt_s

    def apply_periodic_boundaries(self) -> None:
        """Apply periodic boundary conditions."""
        for particle in self.particles:
            # Wrap positions back into box
            particle.pos = np.where(particle.pos > self.box_size_m/2, 
                                   particle.pos - self.box_size_m, 
                                   particle.pos)
            particle.pos = np.where(particle.pos < -self.box_size_m/2, 
                                   particle.pos + self.box_size_m, 
                                   particle.pos)
    
    def kinetic_energy(self) -> float:
        """Calculate total kinetic energy in Joules."""
        KE = 0.0
        for particle in self.particles:
            v2 = np.sum(particle.vel**2)
            KE += 0.5 * particle.mass_kg * v2
        return KE

    @staticmethod
    def calculate_system_size(positions: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """
        Calculate characteristic size of system.

        Returns (rms_radius_m, max_radius_m, com) where com shows universe center drift.
        """

        # Center of mass
        com = np.mean(positions, axis=0)

        # Distances from center
        r_m = np.linalg.norm(positions - com, axis=1)

        # RMS distance (mean behavior)
        rms_radius_m = np.sqrt(np.mean(r_m**2))

        # Maximum distance (catches runaway particles)
        max_radius_m = np.max(r_m)

        return rms_radius_m, max_radius_m, com
    
    
    def __len__(self):
        return self.n_particles
    
    def __repr__(self):
        return f"ParticleSystem(n={self.n_particles}, t={self.time:.2e}s)"


class HMEAGrid:
    """Represents the external HMEA nodes as boundary conditions"""
    
    def __init__(self, node_params: Optional[ExternalNodeParameters] = None, n_nodes: int = 8):
        """Initialize HMEA grid (typically 26 nodes in 3x3x3-1 cubic lattice)."""
        self.params = node_params if node_params is not None else ExternalNodeParameters()
        self.n_nodes = n_nodes
        self.nodes = []
        
        # Create grid topology
        self._create_grid()
        
    def _create_grid(self) -> None:
        """
        Create 3x3x3 grid of HMEA nodes (26 total, excluding center).

        Grid is perfectly symmetric to ensure tidal forces cancel at origin.
        Any drift indicates either numerical issues or particle asymmetry.
        """
        S = self.params.S

        # 3x3x3 grid positions: -1, 0, +1 in each direction
        # Skip (0,0,0) - that's our universe
        node_id = 0
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                for k in [-1, 0, 1]:
                    # Skip center - that's us!
                    if i == 0 and j == 0 and k == 0:
                        continue

                    # Position with spacing S (perfectly symmetric)
                    pos = np.array([i, j, k], dtype=float) * S

                    node = {
                        'id': node_id,
                        'position': pos,
                        'mass': self.params.M_ext_kg,
                    }
                    self.nodes.append(node)
                    node_id += 1
    
    def get_positions(self) -> np.ndarray:
        """Get all node positions as (N, 3) array."""
        return np.array([node['position'] for node in self.nodes])

    def get_masses(self) -> np.ndarray:
        """Get all node masses as (N,) array."""
        return np.array([node['mass'] for node in self.nodes])

    def calculate_tidal_acceleration_batch(self, positions: np.ndarray, use_numba: bool = True) -> np.ndarray:
        """
        Calculate tidal acceleration for multiple positions.

        Args:
            positions: (N, 3) particle positions in meters
            use_numba: If True, use Numba JIT for speedup

        Returns:
            Accelerations array with shape (N, 3) in m/s².
        """
        const = CosmologicalConstants()

        if use_numba:
            # Use Numba JIT-compiled version (much faster)
            from .tidal_forces_numba import calculate_tidal_forces_numba

            node_positions = self.get_positions()
            node_masses = self.get_masses()

            return calculate_tidal_forces_numba(
                positions,
                node_positions,
                node_masses,
                const.G
            )
        else:
            # Original NumPy vectorized version (fallback)
            N = len(positions)
            accelerations = np.zeros((N, 3))

            for node in self.nodes:
                node_pos = node['position']
                M_ext_kg = node['mass']

                # Vector from position to node (attractive force toward node)
                r_vec_m = node_pos - positions  # Broadcasting
                r_m = np.linalg.norm(r_vec_m, axis=1, keepdims=True)

                # Avoid singularities
                r_m = np.maximum(r_m, 1e10)

                # Tidal acceleration for all particles (attractive toward node)
                a_tidal = const.G * M_ext_kg * r_vec_m / r_m**3

                accelerations += a_tidal

            return accelerations
    
    def __repr__(self):
        return (f"HMEAGrid(n_nodes={self.n_nodes}, "
                f"M_ext_kg={self.params.M_ext_kg:.2e} kg, "
                f"S={self.params.S_Gpc:.1f} Gpc)")
