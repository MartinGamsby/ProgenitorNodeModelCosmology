"""
Particle and Grid Structures
Represents galaxies/clusters as particles and HMEA nodes as boundary conditions
"""

import numpy as np
from .constants import CosmologicalConstants, ExternalNodeParameters, LambdaCDMParameters


class Particle:
    """Represents a single galaxy cluster or tracer particle"""
    
    def __init__(self, position, velocity, mass, particle_id=0):
        """
        Initialize a particle
        
        Parameters:
        -----------
        position : array-like, shape (3,)
            Position vector [x, y, z] in meters
        velocity : array-like, shape (3,)
            Velocity vector [vx, vy, vz] in m/s
        mass : float
            Mass in kg
        particle_id : int
            Unique identifier
        """
        self.pos = np.array(position, dtype=np.float64)
        self.vel = np.array(velocity, dtype=np.float64)
        self.mass = float(mass)
        self.id = particle_id
        self.acc = np.zeros(3, dtype=np.float64)  # Acceleration
        
    def __repr__(self):
        return f"Particle(id={self.id}, mass={self.mass:.2e} kg, pos={self.pos})"


class ParticleSystem:
    """Collection of particles representing the observable universe"""
    
    def __init__(self, n_particles=1000, box_size=None, total_mass=None, a_start=1.0, use_dark_energy=True, damping_factor_override=0.91):
        """
        Initialize a system of particles

        Parameters:
        -----------
        n_particles : int
            Number of particles (galaxy clusters)
        box_size : float
            Size of simulation box [meters]
        total_mass : float
            Total mass to distribute among particles [kg]
        a_start : float
            Scale factor at simulation start time (a=1 at present day)
        use_dark_energy : bool
            Whether dark energy will be used in the simulation (affects initial velocities)
        """
        const = CosmologicalConstants()

        self.n_particles = n_particles
        self.box_size = box_size if box_size is not None else const.R_hubble
        self.total_mass = total_mass if total_mass is not None else const.M_observable
        self.a_start = a_start
        self.use_dark_energy = use_dark_energy
        self.damping_factor = damping_factor_override

        self.particles = []
        self.time = 0.0

        # Initialize particles
        self._initialize_particles()
        
    def _initialize_particles(self):
        """Create initial particle distribution with Hubble flow"""
        lcdm = LambdaCDMParameters()

        H_start = lcdm.H_at_time(self.a_start)

        if self.damping_factor is not None:
            damping_factor = self.damping_factor
        else:
            # where Omega_m(a) = Omega_m / a^3 / [Omega_m / a^3 + Omega_Lambda]
            Omega_m_eff = lcdm.Omega_m / self.a_start**3
            Omega_Lambda_eff = lcdm.Omega_Lambda
            total_omega = Omega_m_eff + Omega_Lambda_eff

            if total_omega > 0:
                q = 0.5 * Omega_m_eff / total_omega - 1.0
            else:
                q = 0.5  # Default to matter-dominated if something goes wrong

            # Damping factor based on deceleration parameter
            # q > 0 (decelerating) → more damping needed
            # q < 0 (accelerating) → less damping needed
            # Range: ~0.15 (strong deceleration) to ~0.65 (acceleration)
            damping_factor = 0.4 - 0.25 * q

            # Clamp to reasonable range
            damping_factor = np.clip(damping_factor, 0.1, 0.7)

        print("[ParticleSystem] Damping factor for initial:", damping_factor)

        particle_mass = self.total_mass / self.n_particles

        # Scale box_size so that the RMS radius matches the target
        # For a uniform sphere of radius R, RMS radius = R * sqrt(3/5) ≈ 0.775*R
        # We want RMS = box_size/2, so R_sphere = box_size/2 / 0.775
        # This means we need to use a sphere of radius: box_size/2 / sqrt(3/5)
        sphere_radius = (self.box_size / 2) / np.sqrt(3/5)

        # First, generate all positions using rejection sampling
        # This keeps position RNG calls separate from velocity RNG calls
        positions = []
        for i in range(self.n_particles):
            # Random position uniformly in sphere of radius sphere_radius
            # Using rejection sampling for clarity
            while True:
                pos = np.random.uniform(-sphere_radius, sphere_radius, 3)
                if np.linalg.norm(pos) <= sphere_radius:
                    break
            positions.append(pos)

        # Now generate velocities with consistent RNG state
        # This ensures velocity initialization is independent of rejection sampling randomness
        for i in range(self.n_particles):
            pos = positions[i]

            # Initial velocity: Damped Hubble flow + small peculiar velocity
            # Damping compensates for lack of ongoing Hubble drag during integration
            v_hubble = damping_factor * H_start * pos
            v_peculiar = np.random.normal(0, 1e5, 3)  # ~100 km/s peculiar velocity
            vel = v_hubble + v_peculiar

            particle = Particle(pos, vel, particle_mass, particle_id=i)
            self.particles.append(particle)

        # CRITICAL: Remove center-of-mass velocity to prevent bulk motion
        # With Hubble flow v = H*r, random particle positions create non-zero COM velocity
        # This causes the entire system to drift, appearing as unphysical expansion
        velocities = np.array([p.vel for p in self.particles])
        com_velocity = np.mean(velocities, axis=0)

        print(f"[ParticleSystem] Removing COM velocity: [{com_velocity[0]:.3e}, {com_velocity[1]:.3e}, {com_velocity[2]:.3e}] m/s")

        for particle in self.particles:
            particle.vel -= com_velocity
    
    def get_positions(self):
        """Get all particle positions as (N, 3) array"""
        return np.array([p.pos for p in self.particles])
    
    def get_velocities(self):
        """Get all particle velocities as (N, 3) array"""
        return np.array([p.vel for p in self.particles])
    
    def get_masses(self):
        """Get all particle masses as (N,) array"""
        return np.array([p.mass for p in self.particles])
    
    def get_accelerations(self):
        """Get all particle accelerations as (N, 3) array"""
        return np.array([p.acc for p in self.particles])
    
    def set_accelerations(self, accelerations):
        """Set accelerations for all particles"""
        for i, particle in enumerate(self.particles):
            particle.acc = accelerations[i]
    
    def update_positions(self, dt):
        """Update positions using current velocities"""
        for particle in self.particles:
            particle.pos += particle.vel * dt
    
    def update_velocities(self, dt):
        """Update velocities using current accelerations"""
        for particle in self.particles:
            particle.vel += particle.acc * dt
    
    def apply_periodic_boundaries(self):
        """Apply periodic boundary conditions"""
        for particle in self.particles:
            # Wrap positions back into box
            particle.pos = np.where(particle.pos > self.box_size/2, 
                                   particle.pos - self.box_size, 
                                   particle.pos)
            particle.pos = np.where(particle.pos < -self.box_size/2, 
                                   particle.pos + self.box_size, 
                                   particle.pos)
    
    def kinetic_energy(self):
        """Calculate total kinetic energy"""
        KE = 0.0
        for particle in self.particles:
            v2 = np.sum(particle.vel**2)
            KE += 0.5 * particle.mass * v2
        return KE

    @staticmethod
    def calculate_system_size(positions):
        """
        Calculate characteristic size of system

        Returns:
        --------
        tuple: (rms_radius, max_radius)
            rms_radius: RMS distance from center of mass (typical particle distance)
            max_radius: Maximum particle distance from COM (detects runaway particles)
        """

        # Center of mass
        com = np.mean(positions, axis=0)

        # Distances from center
        r = np.linalg.norm(positions - com, axis=1)

        # RMS distance (mean behavior)
        rms_radius = np.sqrt(np.mean(r**2))

        # Maximum distance (catches runaway particles)
        max_radius = np.max(r)

        return rms_radius, max_radius
    
    
    def __len__(self):
        return self.n_particles
    
    def __repr__(self):
        return f"ParticleSystem(n={self.n_particles}, t={self.time:.2e}s)"


class HMEAGrid:
    """Represents the external HMEA nodes as boundary conditions"""
    
    def __init__(self, node_params=None, n_nodes=8):
        """
        Initialize HMEA grid
        
        Parameters:
        -----------
        node_params : ExternalNodeParameters
            Parameters for the HMEA nodes
        n_nodes : int
            Number of nodes (typically 6-8 for 3D grid)
        """
        self.params = node_params if node_params is not None else ExternalNodeParameters()
        self.n_nodes = n_nodes
        self.nodes = []
        
        # Create grid topology
        self._create_grid()
        
    def _create_grid(self):
        """
        Create a 3x3x3 grid of HMEA nodes surrounding our universe
        
        Our observable universe is at the center (0,0,0) - no node there.
        26 nodes surround us in a cubic lattice with spacing S.
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
                    
                    # Position with spacing S
                    pos = np.array([i, j, k], dtype=float) * S
                    
                    # Add small irregularity (virialized structure)
                    irregularity = np.random.normal(0, 0.05, 3)  # 5% perturbation
                    pos += irregularity * S
                    
                    node = {
                        'id': node_id,
                        'position': pos,
                        'mass': self.params.M_ext,
                    }
                    self.nodes.append(node)
                    node_id += 1
    
    def get_positions(self):
        """Get all node positions as (N, 3) array"""
        return np.array([node['position'] for node in self.nodes])
    
    def get_masses(self):
        """Get all node masses as (N,) array"""
        return np.array([node['mass'] for node in self.nodes])
    
    def calculate_tidal_acceleration_batch(self, positions):
        """
        Calculate tidal acceleration for multiple positions (vectorized)
        
        Parameters:
        -----------
        positions : array, shape (N, 3)
            Array of positions [meters]
            
        Returns:
        --------
        accelerations : array, shape (N, 3)
            Tidal acceleration vectors [m/s^2]
        """
        const = CosmologicalConstants()
        N = len(positions)
        accelerations = np.zeros((N, 3))
        
        for node in self.nodes:
            node_pos = node['position']
            M_ext = node['mass']
            
            # Vector from position to node (attractive force toward node)
            r_vec = node_pos - positions  # Broadcasting
            r = np.linalg.norm(r_vec, axis=1, keepdims=True)

            # Avoid singularities
            r = np.maximum(r, 1e10)

            # Tidal acceleration for all particles (attractive toward node)
            a_tidal = const.G * M_ext * r_vec / r**3
            
            accelerations += a_tidal
        
        return accelerations
    
    def __repr__(self):
        return (f"HMEAGrid(n_nodes={self.n_nodes}, "
                f"M_ext={self.params.M_ext:.2e} kg, "
                f"S={self.params.S_Gpc:.1f} Gpc)")
