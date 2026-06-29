"""
N-Body Integrator
Handles force calculation and time evolution of the particle system
"""

from typing import Optional, List, Dict
import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, *args, **kwargs):
        return iterable

from .constants import CosmologicalConstants, LambdaCDMParameters
from .particles import ParticleSystem, HMEAGrid

is_cuda_available = None

class Integrator:
    """Base class for N-body integration"""
    
    def __init__(self, particle_system: ParticleSystem, hmea_grid: Optional[HMEAGrid] = None,
                 softening_per_Mobs_m: float = 1e24, use_external_nodes: bool = True, use_dark_energy: bool = False,
                 force_method: str = 'auto', barnes_hut_theta: float = 0.5, use_hubble_drag: bool = False,
                 node_substep_threshold: float = 0.0, node_substeps: int = 1):
        """
        Initialize integrator.

        Softening scales as: ε = softening_per_Mobs_m × (m/M_observable_kg)^(1/3)
        This makes softening proportional to particle mass^(1/3), so heavier
        particles (fewer particles) have larger softening for stability.

        Args:
            force_method: 'auto' (uses numba_direct for N>=100), 'direct' for NumPy O(N²),
                          'numba_direct' for Numba JIT O(N²), or 'barnes_hut' for real O(N log N) octree
            barnes_hut_theta: Opening angle for Barnes-Hut (0.3-0.7 typical)
            use_hubble_drag: Apply Hubble drag a_drag = -2H(a)v for matter-only sims
            node_substep_threshold: Adaptive KDK sub-stepping trigger (Section 4),
                          in units of the node softening length (or node spacing S
                          if softening is 0). 0.0 (default) DISABLES sub-stepping
                          (one plain leapfrog step -> byte-identical). > 0.0: a
                          global step where any particle is within
                          node_substep_threshold * S_ref of a node is integrated as
                          node_substeps smaller KDK substeps, refining dt during the
                          close pass.
            node_substeps: KDK substeps per triggered close pass (>= 1). 1 (default)
                          is a no-op even with a positive threshold.
        """
        self.particles = particle_system
        self.hmea_grid = hmea_grid
        self.softening_per_Mobs_m = softening_per_Mobs_m
        self.use_external_nodes = use_external_nodes
        self.use_dark_energy = use_dark_energy
        self.use_hubble_drag = use_hubble_drag
        self.force_method = force_method
        self.barnes_hut_theta = barnes_hut_theta
        # Adaptive KDK sub-stepping config (Section 4). Default OFF (threshold 0
        # or substeps <= 1) -> the integrator takes one plain leapfrog step, so a
        # default sim is byte-identical. The reference scale S_ref is the node
        # softening length when set, else the characteristic node spacing.
        self.node_substep_threshold = float(node_substep_threshold)
        self.node_substeps = int(max(1, node_substeps))
        self._node_substep_ref_m = self._compute_node_substep_ref_m()

        # Determine actual method to use
        N = len(particle_system.particles)
        if force_method == 'auto':
            if N >= 10000:
                global is_cuda_available
                if is_cuda_available == None:
                    from numba import cuda
                    is_cuda_available = cuda.is_available()
                
                    if is_cuda_available:
                        print("GPU is supported and ready!")
                        print(cuda.detect())
                    else:
                        print("CUDA GPU not found or drivers are missing.")
                
                if is_cuda_available:
                    self._active_force_method = 'barnes_hut_gpu'
                else:
                    self._active_force_method = 'barnes_hut'
            elif N >= 1000:
                self._active_force_method = 'barnes_hut'
            elif N >= 100:
                self._active_force_method = 'numba_direct'
            else:
                self._active_force_method = 'direct'
        else:
            self._active_force_method = force_method
        self.const = CosmologicalConstants()

        # Calculate adaptive softening based on mean particle mass
        # ε ∝ m^(1/3) makes softening scale with typical inter-particle distance
        # For reference mass (M_observable_kg with 100 particles): m_ref = 1e52 kg
        m_ref = self.const.M_observable_kg
        mean_particle_mass_kg = np.mean(self.particles.get_masses())
        mass_ratio = mean_particle_mass_kg / m_ref

        # Scale softening: ε ∝ m^(1/3)
        # More massive particles (fewer particles) → larger softening
        self.softening_m = self.softening_per_Mobs_m * (mass_ratio ** (1.0/3.0))

        # Additional safety factor for small N systems
        # Small N → higher probability of close encounters → need larger softening
        # Use a strong scaling: boost_factor = (100/N)^0.5 for N < 100
        # This gives: N=50 → 1.41x, N=25 → 2x, N=10 → 3.16x
        if len(self.particles) < 100:
            n_particles = len(self.particles)
            boost_factor = np.sqrt(100.0 / n_particles)
            boost_factor = max(1.0, min(5.0, boost_factor))  # Clamp to [1.0, 5.0]
            self.softening_m *= boost_factor
            print(f"[Integrator] Small-N boost applied: {boost_factor:.2f}x (N={n_particles})")

        print(f"[Integrator] Base softening: {self.softening_per_Mobs_m/self.const.Mpc_to_m:.2f} Mpc")
        print(f"[Integrator] Mean particle mass_kg: {mean_particle_mass_kg:.2e} kg")
        print(f"[Integrator] Final softening: {self.softening_m/self.const.Mpc_to_m:.2f} Mpc (mass ratio: {mass_ratio:.2f})")
        
        # ΛCDM parameters for dark energy
        self.lcdm = LambdaCDMParameters()
        
        # History tracking
        self.time_history = []
        self.energy_history = []
        
    def calculate_internal_forces(self) -> np.ndarray:
        """
        Calculate gravitational forces between particles using vectorized N-body summation.

        Returns accelerations array with shape (N, 3) in m/s².
        """
        N = len(self.particles)
        positions = self.particles.get_positions()  # Shape: (N, 3)
        masses_kg = self.particles.get_masses()        # Shape: (N,)

        # Vectorized computation using NumPy broadcasting
        # positions[:, np.newaxis, :] has shape (N, 1, 3)
        # positions[np.newaxis, :, :] has shape (1, N, 3)
        # Broadcasting gives shape (N, N, 3) for all pairwise vectors
        r_vec_m = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]  # Shape: (N, N, 3)

        # Distance between all pairs: sqrt(dx^2 + dy^2 + dz^2)
        r_m = np.sqrt(np.sum(r_vec_m**2, axis=2))  # Shape: (N, N)

        # Softened distance to prevent singularities
        r_soft_m = np.sqrt(r_m**2 + self.softening_m**2)  # Shape: (N, N)

        # Avoid division by zero on diagonal (self-interaction)
        # Set diagonal to large value to make acceleration zero
        np.fill_diagonal(r_soft_m, np.inf)

        # Newton's law: a = GM/r_soft^2
        # masses_kg[np.newaxis, :] broadcasts to shape (1, N)
        # Result has shape (N, N) - acceleration magnitude from each j on each i
        a_mag = self.const.G * masses_kg[np.newaxis, :] / r_soft_m**2  # Shape: (N, N)

        # Acceleration vectors: a_vec = a_mag * (r_vec_m / r_soft_m)
        # Need to expand dimensions for broadcasting
        # a_mag[:, :, np.newaxis] has shape (N, N, 1)
        # r_soft_m[:, :, np.newaxis] has shape (N, N, 1)
        # Result has shape (N, N, 3)
        a_vec = a_mag[:, :, np.newaxis] * (r_vec_m / r_soft_m[:, :, np.newaxis])  # Shape: (N, N, 3)

        # Sum over all j (axis=1) to get total acceleration on each i
        accelerations = np.sum(a_vec, axis=1)  # Shape: (N, 3)

        return accelerations

    def calculate_internal_forces_numba_direct(self) -> np.ndarray:
        """
        Calculate gravitational forces using Numba JIT-compiled direct O(N^2) summation.
        14-17x faster than NumPy vectorized method. Exact results (no approximation).

        Returns accelerations array with shape (N, 3) in m/s².
        """
        from cosmo.numba_direct import NumbaDirectSolver

        positions = self.particles.get_positions()
        masses_kg = self.particles.get_masses()

        solver = NumbaDirectSolver(
            softening_m=self.softening_m,
            G=self.const.G
        )
        solver.build_tree(positions, masses_kg)
        return solver.calculate_all_accelerations()

    def calculate_internal_forces_barnes_hut(self) -> np.ndarray:
        """
        Calculate gravitational forces using real Barnes-Hut octree.
        O(N log N) with accuracy controlled by opening angle theta.

        Returns accelerations array with shape (N, 3) in m/s².
        """
        from cosmo.barnes_hut_numba import NumbaBarnesHutTree

        positions = self.particles.get_positions()
        masses_kg = self.particles.get_masses()

        tree = NumbaBarnesHutTree(
            theta=self.barnes_hut_theta,
            softening_m=self.softening_m,
            G=self.const.G
        )
        tree.build_tree(positions, masses_kg)
        return tree.calculate_all_accelerations()

    def calculate_internal_forces_barnes_hut_gpu(self) -> np.ndarray:
        from cosmo.barnes_hut_numba import NumbaBarnesHutGPU

        positions = self.particles.get_positions()
        masses_kg = self.particles.get_masses()

        tree = NumbaBarnesHutGPU(
            theta=self.barnes_hut_theta,
            softening_m=self.softening_m,
            G=self.const.G
        )
        result = tree.calculate(positions, masses_kg)
        if result:
            return result
        # Fallback to CPU for Barnes Hut
        return self.calculate_internal_forces_barnes_hut()

    def calculate_external_forces(self) -> np.ndarray:
        """
        Calculate tidal forces from external HMEA nodes.

        Returns accelerations array with shape (N, 3) in m/s².
        """
        if not self.use_external_nodes or self.hmea_grid is None:
            return np.zeros((len(self.particles), 3))

        positions = self.particles.get_positions()
        return self.hmea_grid.calculate_tidal_acceleration_batch(positions)

    # ------------------------------------------------------------------
    # Adaptive KDK sub-stepping helpers (Section 4)
    # ------------------------------------------------------------------
    def _compute_node_substep_ref_m(self) -> float:
        """Reference length S_ref for the close-pass trigger, in meters.

        Uses the node softening length when it is set (> 0), else the
        characteristic node spacing (median nearest-neighbour distance, or the
        minimum node radius for tiny grids). Returns 0.0 when there are no nodes
        (sub-stepping then never triggers).
        """
        if self.hmea_grid is None:
            return 0.0
        params = getattr(self.hmea_grid, 'params', None)
        soft_m = float(getattr(params, 'node_softening_m', 0.0)) if params else 0.0
        if soft_m > 0.0:
            return soft_m
        # Fall back to a node-spacing scale so the trigger is meaningful even at
        # zero softening (the bounded/legacy laws still slingshot near a node).
        try:
            npos = self.hmea_grid.get_positions()
        except Exception:
            return 0.0
        if npos is None or len(npos) < 2:
            if npos is not None and len(npos) == 1:
                return float(np.linalg.norm(npos[0]))
            return 0.0
        # Median nearest-neighbour spacing (robust, O(M^2) but M is small).
        d2 = np.sum(
            (npos[:, None, :] - npos[None, :, :]) ** 2, axis=2)
        np.fill_diagonal(d2, np.inf)
        nn = np.sqrt(np.min(d2, axis=1))
        return float(np.median(nn))

    def _substep_active(self) -> bool:
        """True when adaptive sub-stepping is configured AND has nodes to watch."""
        return (
            self.use_external_nodes
            and self.hmea_grid is not None
            and self.node_substep_threshold > 0.0
            and self.node_substeps > 1
            and self._node_substep_ref_m > 0.0
        )

    def _in_close_pass(self) -> bool:
        """True when any particle is within threshold * S_ref of any node.

        Used by the leapfrog step to decide whether to subdivide the step into
        KDK substeps. O(N*M) on the current positions; only evaluated when
        sub-stepping is active.
        """
        if not self._substep_active():
            return False
        positions = self.particles.get_positions()           # (N, 3)
        npos = self.hmea_grid.get_positions()                 # (M, 3)
        if positions.size == 0 or npos is None or len(npos) == 0:
            return False
        trigger_m = self.node_substep_threshold * self._node_substep_ref_m
        trigger2 = trigger_m * trigger_m
        # Min squared distance from each particle to the nearest node.
        d2 = np.sum((positions[:, None, :] - npos[None, :, :]) ** 2, axis=2)
        return bool(np.any(np.min(d2, axis=1) < trigger2))
    
    def calculate_dark_energy_forces(self) -> np.ndarray:
        """
        Calculate dark energy acceleration: a_Λ = H₀² Ω_Λ R

        Returns accelerations array with shape (N, 3) in m/s².
        """
        if not self.use_dark_energy:
            return np.zeros((len(self.particles), 3))

        # Get positions (R vectors from origin)
        positions = self.particles.get_positions()

        # Dark energy acceleration: a_Λ = H₀² Ω_Λ R (pointing outward)
        # This is the repulsive "push" from vacuum energy
        a_Lambda = self.lcdm.H0_si**2 * self.lcdm.Omega_Lambda * positions

        return a_Lambda
    
    def calculate_total_forces(self) -> np.ndarray:
        """
        Calculate total forces (internal + external + dark energy).

        Returns accelerations array with shape (N, 3) in m/s².
        """
        # Select internal force method
        if self._active_force_method == 'numba_direct':
            a_internal_mps2 = self.calculate_internal_forces_numba_direct()
        elif self._active_force_method == 'barnes_hut':
            a_internal_mps2 = self.calculate_internal_forces_barnes_hut()
        elif self._active_force_method == 'barnes_hut_gpu':
            a_internal_mps2 = self.calculate_internal_forces_barnes_hut()# TO TEST: calculate_internal_forces_barnes_hut_gpu()
        else:
            a_internal_mps2 = self.calculate_internal_forces()

        a_external_mps2 = self.calculate_external_forces()
        a_dark_energy_mps2 = self.calculate_dark_energy_forces()

        return a_internal_mps2 + a_external_mps2 + a_dark_energy_mps2

    def apply_hubble_drag(self, dt_s: float) -> None:
        """
        Apply cosmological drag to match Friedmann expansion.

        The N-body gravity alone gives ~83% of the Friedmann deceleration.
        This adds the missing ~17% as a drag term proportional to H*v.

        Derived from: a_friedmann = -0.5*H^2*R, a_gravity = -GM/R^2
        Missing: a_missing = a_friedmann - a_gravity ≈ -0.17 * H^2 * R = -0.17 * H * v
        """
        if not self.use_hubble_drag:
            return

        # Get current scale factor from RMS radius ratio
        positions = self.particles.get_positions()
        rms_current = np.sqrt(np.mean(np.sum(positions**2, axis=1)))
        rms_initial = self.particles.box_size_m / 2
        a_current = rms_current / rms_initial * self.particles.a_start

        # Clamp scale factor to avoid numerical issues
        a_current = max(a_current, 0.01)

        # Calculate H(a) for matter-only
        H_current = self.lcdm.H_matter_only(a_current)

        # Apply drag coefficient ~0.20 (empirically tuned to match Friedmann)
        # Accounts for difference between N-body gravity and Friedmann deceleration
        drag_coeff = 0.20
        drag_factor = np.exp(-drag_coeff * H_current * dt_s)
        velocities = self.particles.get_velocities()
        self.particles.set_velocities(velocities * drag_factor)

    def total_energy(self) -> float:
        """Calculate total energy (kinetic + potential) in Joules."""
        # Kinetic energy
        KE = self.particles.kinetic_energy()
        
        # Potential energy (internal)
        PE = self.potential_energy()
        
        return KE + PE
    
    def potential_energy(self) -> float:
        """Gravitational potential energy in Joules (CHUNKED: O(N) memory).

        Computed in row-blocks so it never allocates the full (N, N, 3) pairwise array
        (that OOMed at N~20k: 8.9 GiB). Still O(N^2) in TIME, so it is a diagnostic for
        modest N only — the integration loop SKIPS it at large N (see integrate()).
        Numerically identical (upper-triangle sum) to the old dense version.
        """
        N = len(self.particles)
        positions = self.particles.get_positions()  # (N, 3)
        masses_kg = self.particles.get_masses()      # (N,)
        soft2 = self.softening_m ** 2
        G = self.const.G
        j_idx = np.arange(N)
        # Keep each block's working set ~1e7 elements (block*N*3 floats).
        block = max(1, int(1e7 // max(1, N)))
        PE = 0.0
        for i0 in range(0, N, block):
            i1 = min(i0 + block, N)
            diff = positions[i0:i1, np.newaxis, :] - positions[np.newaxis, :, :]  # (b, N, 3)
            r2 = np.einsum('ijk,ijk->ij', diff, diff)                              # (b, N)
            r_soft = np.sqrt(r2 + soft2)
            mp = masses_kg[i0:i1, np.newaxis] * masses_kg[np.newaxis, :]           # (b, N)
            pe = -G * mp / r_soft
            # Upper triangle only (count each pair once): keep columns j > global row i.
            mask = j_idx[np.newaxis, :] > np.arange(i0, i1)[:, np.newaxis]
            PE += float(np.sum(pe[mask]))
        return PE


class LeapfrogIntegrator(Integrator):
    """
    Leapfrog (kick-drift-kick) integrator
    Second-order symplectic integrator, conserves energy well
    """
    
    def _kdk(self, dt_s: float) -> None:
        """One bare kick-drift-kick leapfrog update (no drag, no time bump)."""
        a_total = self.calculate_total_forces()

        # Kick (half step)
        self.particles.set_accelerations(a_total)
        self.particles.update_velocities(dt_s / 2)

        # Drift (full step)
        self.particles.update_positions(dt_s)

        # Kick (half step)
        a_total = self.calculate_total_forces()

        self.particles.set_accelerations(a_total)
        self.particles.update_velocities(dt_s / 2)

    def step(self, dt_s: float) -> None:
        """
        Take one leapfrog timestep with optional Hubble drag.

        For matter-only simulations with use_hubble_drag=True, applies
        cosmological expansion via Hubble drag term: v_new = v * exp(-2Hdt).

        Adaptive KDK sub-stepping (Section 4): when sub-stepping is enabled
        (node_substep_threshold > 0 AND node_substeps > 1) and a particle is
        within threshold * S_ref of a node at the start of this step, the step is
        integrated as node_substeps smaller KDK substeps to refine dt during the
        close pass. When sub-stepping is OFF (default) OR no close pass is
        detected, exactly ONE KDK step is taken — byte-identical to the legacy
        integrator. Hubble drag and the time advance are applied ONCE per global
        step in both cases.
        """
        if self._substep_active() and self._in_close_pass():
            m = self.node_substeps
            dt_sub = dt_s / m
            for _ in range(m):
                self._kdk(dt_sub)
        else:
            self._kdk(dt_s)

        # Apply Hubble drag if enabled (for matter-only to match Friedmann)
        self.apply_hubble_drag(dt_s)

        # Update time
        self.particles.time += dt_s
    
    def evolve(self, t_end_s: float, n_steps: int, save_interval: int = 10) -> List[Dict]:
        """Evolve system from current time to t_end_s, saving snapshots every N steps."""
        dt_s = (t_end_s - self.particles.time) / n_steps

        snapshots = []

        print(f"Running leapfrog integration...")
        print(f"  dt = {dt_s:.2e} s ({dt_s/(365.25*24*3600*1e6):.2f} Myr)")
        print(f"  Total steps = {n_steps}")
        print(f"  Save interval = {save_interval}")

        # Initial snapshot (after pre-kick, velocities now at t=-dt/2)
        snapshots.append(self._save_snapshot())

        n_particles = self.particles.n_particles
        for step in tqdm(range(n_steps), mininterval=.5 if n_particles > 1000 else (0.25 if n_particles > 300 else 0.1),
                         desc="Integrating", unit="step"):
            self.step(dt_s)

            # Save snapshot
            if (step + 1) % save_interval == 0:
                snapshots.append(self._save_snapshot())

            # Track energy (KE+PE). total_energy()'s PE term is O(N^2) in TIME, so SKIP it
            # for large N (the hero/high-resolution runs) where it would dominate runtime;
            # the energy_history is a diagnostic, not used by a(t)/chi2. Threshold 10000
            # keeps every existing run (<=4000p sweeps) byte-identical.
            if n_particles <= 10000 and (step + 1) % max(1, (n_steps // 10)) == 0:
                self.time_history.append(self.particles.time)
                self.energy_history.append(self.total_energy())

        print(f"Integration complete. Time = {self.particles.time/(365.25*24*3600*1e9):.2f} Gyr")

        return snapshots
    
    def _save_snapshot(self) -> Dict:
        """Save current state."""
        return {
            'time_s': self.particles.time,
            'positions': self.particles.get_positions().copy(),
            'velocities': self.particles.get_velocities().copy(),
            'accelerations': self.particles.get_accelerations().copy(),
        }

