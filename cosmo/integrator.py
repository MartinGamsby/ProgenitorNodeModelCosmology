"""
N-Body Integrator
Handles force calculation and time evolution of the particle system
"""

import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, *args, **kwargs):
        return iterable

from .constants import CosmologicalConstants, LambdaCDMParameters
from .particles import ParticleSystem, HMEAGrid


class Integrator:
    """Base class for N-body integration"""
    
    def __init__(self, particle_system, hmea_grid=None, softening_per_Mobs=1e24, use_external_nodes=True, use_dark_energy=False):
        """
        Initialize integrator

        Parameters:
        -----------
        particle_system : ParticleSystem
            The particle system to evolve
        hmea_grid : HMEAGrid, optional
            External HMEA nodes (if None, runs pure ΛCDM)
        softening : float
            Base gravitational softening length [meters]
            Actual softening scales as: ε = softening × (m/M_observable)^(1/3)
            This makes softening proportional to particle mass^(1/3), so heavier
            particles (fewer particles) have larger softening for stability
        use_external_nodes : bool
            Whether to include external node tidal forces
        use_dark_energy : bool
            Whether to include ΛCDM dark energy acceleration (for ΛCDM model)
        """
        self.particles = particle_system
        self.hmea_grid = hmea_grid
        self.softening_per_Mobs = softening_per_Mobs
        self.use_external_nodes = use_external_nodes
        self.use_dark_energy = use_dark_energy
        self.const = CosmologicalConstants()

        # Calculate adaptive softening based on particle mass
        # ε ∝ m^(1/3) makes softening scale with typical inter-particle distance
        # For reference mass (M_observable with 100 particles): m_ref = 1e52 kg
        m_ref = self.const.M_observable
        particle_mass = self.particles.particles[0].mass  # All particles have same mass
        mass_ratio = particle_mass / m_ref

        # Scale softening
        # More massive particles (fewer particles) → larger softening
        self.softening = self.softening_per_Mobs * mass_ratio

        print(f"[Integrator] Base softening: {self.softening_per_Mobs/self.const.Mpc_to_m:.2f} Mpc")
        print(f"[Integrator] Particle mass: {particle_mass:.2e} kg")
        print(f"[Integrator] Adaptive softening: {self.softening/self.const.Mpc_to_m:.2f} Mpc (mass ratio: {mass_ratio:.2f})")
        
        # ΛCDM parameters for dark energy
        self.lcdm = LambdaCDMParameters()
        
        # History tracking
        self.time_history = []
        self.energy_history = []
        
    def calculate_internal_forces(self):
        """
        Calculate gravitational forces between particles (internal gravity)
        Uses direct N-body summation with softening
        
        Returns:
        --------
        accelerations : array, shape (N, 3)
            Acceleration for each particle [m/s^2]
        """
        N = len(self.particles)
        positions = self.particles.get_positions()
        masses = self.particles.get_masses()
        
        accelerations = np.zeros((N, 3))
        
        # Direct pairwise summation (O(N^2) - slow but accurate)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                
                # Vector from i to j
                r_vec = positions[j] - positions[i]
                r = np.linalg.norm(r_vec)
                
                # Softened distance to prevent singularities
                r_soft = np.sqrt(r**2 + self.softening**2)

                # Newton's law: a = GM/r^2 (with softening)
                # CRITICAL: Use r_soft for BOTH magnitude AND direction
                a_mag = self.const.G * masses[j] / r_soft**2
                a_vec = a_mag * (r_vec / r_soft)

                accelerations[i] += a_vec
        
        return accelerations
    
    def calculate_external_forces(self):
        """
        Calculate tidal forces from external HMEA nodes
        
        Returns:
        --------
        accelerations : array, shape (N, 3)
            Tidal acceleration for each particle [m/s^2]
        """
        if not self.use_external_nodes or self.hmea_grid is None:
            return np.zeros((len(self.particles), 3))
        
        positions = self.particles.get_positions()
        return self.hmea_grid.calculate_tidal_acceleration_batch(positions)
    
    def calculate_dark_energy_forces(self):
        """
        Calculate dark energy acceleration (ΛCDM model)

        Dark energy acceleration: a_Λ = H₀² Ω_Λ R

        Returns:
        --------
        accelerations : array, shape (N, 3)
            Dark energy acceleration for each particle [m/s^2]
        """
        if not self.use_dark_energy:
            return np.zeros((len(self.particles), 3))

        # Get positions (R vectors from origin)
        positions = self.particles.get_positions()

        # Dark energy acceleration: a_Λ = H₀² Ω_Λ R (pointing outward)
        # This is the repulsive "push" from vacuum energy
        a_Lambda = self.lcdm.H0**2 * self.lcdm.Omega_Lambda * positions

        return a_Lambda
    
    def calculate_total_forces(self):
        """
        Calculate total forces (internal + external/dark energy + Hubble drag)

        Returns:
        --------
        accelerations : array, shape (N, 3)
            Total acceleration for each particle [m/s^2]
        """
        a_internal = self.calculate_internal_forces()
        a_external = self.calculate_external_forces()
        a_dark_energy = self.calculate_dark_energy_forces()

        return a_internal + a_external + a_dark_energy
    
    def total_energy(self):
        """
        Calculate total energy (kinetic + potential)
        
        Returns:
        --------
        E_total : float
            Total energy [J]
        """
        # Kinetic energy
        KE = self.particles.kinetic_energy()
        
        # Potential energy (internal)
        PE = self.potential_energy()
        
        return KE + PE
    
    def potential_energy(self):
        """
        Calculate gravitational potential energy
        
        Returns:
        --------
        PE : float
            Potential energy [J]
        """
        N = len(self.particles)
        positions = self.particles.get_positions()
        masses = self.particles.get_masses()
        
        PE = 0.0
        for i in range(N):
            for j in range(i+1, N):  # Avoid double counting
                r_vec = positions[j] - positions[i]
                r = np.linalg.norm(r_vec)
                r_soft = np.sqrt(r**2 + self.softening**2)
                
                PE -= self.const.G * masses[i] * masses[j] / r_soft
        
        return PE


class LeapfrogIntegrator(Integrator):
    """
    Leapfrog (kick-drift-kick) integrator
    Second-order symplectic integrator, conserves energy well
    """
    
    def step(self, dt):
        """
        Take one leapfrog timestep

        For LCDM mode, Hubble drag is NOT included in the force calculation
        but instead applied as an exponential damping after the position update.
        This prevents numerical instability with large timesteps.

        Parameters:
        -----------
        dt : float
            Timestep [seconds]
        """
        a_total = self.calculate_total_forces()

        # Kick (half step)
        self.particles.set_accelerations(a_total)
        self.particles.update_velocities(dt / 2)

        # Drift (full step)
        self.particles.update_positions(dt)

        # Kick (half step)
        a_total = self.calculate_total_forces()

        self.particles.set_accelerations(a_total)
        self.particles.update_velocities(dt / 2)

        # NOTE: Hubble drag is NOT applied in proper-coordinate simulations!
        # In proper coordinates with explicit dark energy, the expansion is handled
        # by the dark energy acceleration term (a_Λ = H²Ω_Λ r).
        # Hubble drag (a_drag = -2Hv) is only appropriate for comoving coordinates
        # where the background expansion is implicit.
        # Applying it here would over-damp the system since velocities include
        # both Hubble flow AND peculiar velocities.

        # Update time
        self.particles.time += dt
    
    def evolve(self, t_end, n_steps, save_interval=10):
        """
        Evolve system from current time to t_end
        
        Parameters:
        -----------
        t_end : float
            Final time [seconds]
        n_steps : int
            Number of timesteps
        save_interval : int
            Save snapshot every N steps
            
        Returns:
        --------
        snapshots : list of dict
            Saved snapshots containing time, positions, velocities
        """
        dt = (t_end - self.particles.time) / n_steps
        
        snapshots = []
        
        print(f"Running leapfrog integration...")
        print(f"  dt = {dt:.2e} s ({dt/(365.25*24*3600*1e6):.2f} Myr)")
        print(f"  Total steps = {n_steps}")
        print(f"  Save interval = {save_interval}")
        
        # Initial snapshot
        snapshots.append(self._save_snapshot())
        
        for step in tqdm(range(n_steps)):
            self.step(dt)
            
            # Save snapshot
            if (step + 1) % save_interval == 0:
                snapshots.append(self._save_snapshot())
            
            # Track energy
            if (step + 1) % (n_steps // 10) == 0:
                self.time_history.append(self.particles.time)
                self.energy_history.append(self.total_energy())
        
        print(f"Integration complete. Time = {self.particles.time/(365.25*24*3600*1e9):.2f} Gyr")
        
        return snapshots
    
    def _save_snapshot(self):
        """Save current state"""
        return {
            'time': self.particles.time,
            'positions': self.particles.get_positions().copy(),
            'velocities': self.particles.get_velocities().copy(),
            'accelerations': self.particles.get_accelerations().copy(),
        }

