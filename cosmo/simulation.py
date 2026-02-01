"""
Main Simulation Runner
Compares ΛCDM cosmology with External-Node Model
"""

from typing import Optional, List, Dict
import numpy as np
import pickle
import os

from .constants import CosmologicalConstants, SimulationParameters
from .particles import ParticleSystem, HMEAGrid
from .integrator import LeapfrogIntegrator
from .analysis import solve_friedmann_at_times


class CosmologicalSimulation:
    """Main class for running cosmological simulations"""
    
    def __init__(self, sim_params: SimulationParameters, box_size_Gpc: float, a_start: float,
                 use_external_nodes: bool = True, use_dark_energy: Optional[bool] = None,
                 force_method: str = 'auto', barnes_hut_theta: float = 0.5):
        """
        Initialize simulation.

        If use_dark_energy is None, defaults to (not use_external_nodes).

        Args:
            force_method: 'auto' (barnes_hut for N>=1000, numba_direct for N>=100, direct otherwise), 'direct', 'numba_direct', or 'barnes_hut'
        """
        self.const = CosmologicalConstants()
        self.use_external_nodes = use_external_nodes
        self.t_start_Gyr = sim_params.t_start_Gyr
        self.a_start = a_start
        self.box_size_Gpc = box_size_Gpc  # Store initial box size for consistent size calculation
        self.seed = sim_params.seed
        np.random.seed(self.seed)

        # Default: use dark energy only if not using external nodes
        if use_dark_energy is None:
            use_dark_energy = (not use_external_nodes)
        self.use_dark_energy = use_dark_energy

        # Convert box size to meters
        box_size_m = box_size_Gpc * self.const.Gpc_to_m

        # Calculate total mass from center_node_mass
        total_mass_kg = sim_params.center_node_mass_kg

        # Initialize particle system
        print(f"Initializing {sim_params.n_particles} particles in {box_size_Gpc} Gpc box...")
        if sim_params.center_node_mass != 1.0:
            print(f"Total mass: {sim_params.center_node_mass} × M_observable")

        self.particles = ParticleSystem(n_particles=sim_params.n_particles,
                                       box_size_m=box_size_m,
                                       total_mass_kg=total_mass_kg,
                                       a_start=self.a_start,
                                       use_dark_energy=self.use_dark_energy,
                                       mass_randomize=sim_params.mass_randomize)

        # Initialize HMEA grid if using External-Node Model
        self.hmea_grid = None
        if use_external_nodes:
            self.hmea_grid = HMEAGrid(node_params=sim_params.external_params, n_nodes=8)
            print(f"External-Node Model: {self.hmea_grid}")
        else:
            print("Running standard matter-only (no dark energy)")

        # Calculate softening based on center_node_mass (scales with mass for stability)
        # 1Gpc softening per Mobs
        softening_m = sim_params.center_node_mass * 1.0 * self.const.Gpc_to_m
        # Hubble drag disabled - using velocity calibration instead
        use_hubble_drag = False

        self.integrator = LeapfrogIntegrator(
            self.particles,
            self.hmea_grid,
            softening_per_Mobs_m=softening_m,
            use_external_nodes=use_external_nodes,
            use_dark_energy=self.use_dark_energy,
            force_method=force_method,
            barnes_hut_theta=barnes_hut_theta,
            use_hubble_drag=use_hubble_drag
        )
        
        # Simulation results
        self.snapshots = []
        self.expansion_history = []

    def _calibrate_velocity_for_lcdm_match(self, t_duration_Gyr: float, n_steps: int, damping: float = None) -> None:
        """
        Calibrate initial velocity so matter-only NEVER exceeds LCDM.

        Key insight: N-body gravity provides ~80% of Friedmann deceleration.
        This means N-body will overshoot analytic matter-only by ~20% over time.

        Strategy: Calculate where LCDM will be at the END, then calibrate initial
        velocity so that even with reduced N-body deceleration, we end up AT or
        BELOW LCDM at the final timestep.

        The velocity scaling is derived from:
        1. Final LCDM size at t_end
        2. Estimated N-body expansion with reduced deceleration
        3. Required initial velocity to land at LCDM final size

        This naturally accounts for N-body's deceleration deficit without Hubble drag.
        """
        dt_Gyr = t_duration_Gyr / n_steps
        dt_s = dt_Gyr * 1e9 * 365.25 * 24 * 3600

        # Get LCDM and matter-only Friedmann at start and end
        t_end_Gyr = self.t_start_Gyr + t_duration_Gyr
        lcdm_solution = solve_friedmann_at_times(np.array([self.t_start_Gyr, t_end_Gyr]))
        matter_solution = solve_friedmann_at_times(
            np.array([self.t_start_Gyr, t_end_Gyr]),
            Omega_Lambda=0.0
        )

        a_lcdm_start = lcdm_solution['a'][0]
        a_lcdm_end = lcdm_solution['a'][1]
        a_matter_start = matter_solution['a'][0]
        a_matter_end = matter_solution['a'][1]

        # Expansion ratios
        lcdm_expansion = a_lcdm_end / a_lcdm_start
        matter_expansion = a_matter_end / a_matter_start

        # N-body deceleration factor: empirically ~65-70% of Friedmann for long runs
        # This means N-body overshoots matter-only Friedmann by ~1/0.65 = 1.54x
        # The deficit compounds over time, so longer runs need larger correction
        
        if damping is not None:
            nbody_decel_factor = damping
        else:
            # **0.2 to have higher factor faster, but still cap at 1
            nbody_decel_factor = (self.t_start_Gyr/13.8)**0.135
        nbody_decel_factor = np.clip(nbody_decel_factor, 0.01, 1.0)  # min 0.01 to avoid divide by zero
        print("[Velocity Calibration] Damping factor for initial:", nbody_decel_factor)
        #print(self.t_start_Gyr, self.t_start_Gyr/13.8, nbody_decel_factor)
        #exit(1)

        # Predicted N-body expansion if starting with full velocity
        # N-body decelerates less, so expands more than matter-only Friedmann
        # estimated_nbody_expansion ≈ matter_expansion / nbody_decel_factor
        # But this is rough - let's use a simpler approach

        # We want: final N-body size ≤ final LCDM size
        # Target: N-body ends at ~95% of LCDM (margin for safety)
        target_final_relative = 1.0

        # Required velocity damping to achieve target
        # If full velocity gives expansion E, damped velocity gives ~E * damping
        # We want: matter_expansion * overshoot_factor * damping = lcdm_expansion * target_final_relative
        #
        # Overshoot factor ≈ 1/nbody_decel_factor ≈ 1.25
        # So: damping = (lcdm_expansion * target_final_relative) / (matter_expansion * overshoot_factor)

        overshoot_factor = 1.0 / nbody_decel_factor
        velocity_scale = (lcdm_expansion * target_final_relative) / (matter_expansion * overshoot_factor)

        # Clamp to reasonable range
        velocity_scale = np.clip(velocity_scale, 0.5, 1.5)

        print(f"[Velocity Calibration] LCDM expansion: {lcdm_expansion:.4f}x")
        print(f"[Velocity Calibration] Matter-only Friedmann expansion: {matter_expansion:.4f}x")
        print(f"[Velocity Calibration] Estimated N-body overshoot factor: {overshoot_factor:.3f}")
        print(f"[Velocity Calibration] Target final relative to LCDM: {target_final_relative:.2f}")
        print(f"[Velocity Calibration] Velocity scale factor: {velocity_scale:.6f}")

        # Apply calibrated velocity
        velocities = self.particles.get_velocities()
        self.particles.set_velocities(velocities * velocity_scale)

        print(f"[Velocity Calibration] Applied velocity scaling to all particles")

    def _validate_timestep(self, t_duration_Gyr: float, n_steps: int) -> None:
        """
        Validate timestep for leapfrog numerical stability.

        Empirical testing shows dt < 0.05 Gyr required for stability:
        - 150 steps over 20 Gyr (dt=0.133 Gyr): UNSTABLE (1600% energy drift)
        - 500 steps over 20 Gyr (dt=0.040 Gyr): STABLE

        Recommended: dt < 0.04 Gyr for safety margin.

        Raises SystemExit if timestep too large.
        """
        dt_Gyr = t_duration_Gyr / n_steps

        # Critical threshold: dt must be < 0.05 Gyr
        dt_critical = 0.05  # Gyr
        dt_recommended = 0.04  # Gyr (with safety margin)

        # Calculate minimum required steps
        n_steps_minimum = int(np.ceil(t_duration_Gyr / dt_critical))
        n_steps_recommended = int(np.ceil(t_duration_Gyr / dt_recommended))

        if dt_Gyr > dt_critical:
            print("\n" + "="*70)
            print("ERROR: INSUFFICIENT TIMESTEPS FOR NUMERICAL STABILITY")
            print("="*70)
            print(f"Simulation duration: {t_duration_Gyr:.1f} Gyr")
            print(f"Requested steps:     {n_steps}")
            print(f"Timestep (dt):       {dt_Gyr:.4f} Gyr")
            print()
            print("The leapfrog integrator becomes unstable with timesteps > 0.05 Gyr.")
            print("This causes spurious energy injection, making matter-only simulations")
            print("expand faster than LCDM (physically impossible).")
            print()
            print(f"MINIMUM steps required:    {n_steps_minimum} (dt < {dt_critical:.3f} Gyr)")
            print(f"RECOMMENDED steps:         {n_steps_recommended} (dt < {dt_recommended:.3f} Gyr)")
            print()
            print("Example: For a 20 Gyr simulation, use --n-steps 500 or more")
            print("="*70)
            import sys
            sys.exit(1)

        # Warning if close to threshold
        elif dt_Gyr > dt_recommended:
            print("\n" + "!"*70)
            print("WARNING: Timestep is close to stability threshold")
            print("!"*70)
            print(f"Current timestep:  {dt_Gyr:.4f} Gyr")
            print(f"Recommended:       < {dt_recommended:.3f} Gyr")
            print(f"For better stability, consider using {n_steps_recommended} steps or more")
            print("!"*70 + "\n")

    def run(self, t_end_Gyr: float = 13.8, n_steps: int = 1000, save_interval: int = 10, damping=None) -> List[Dict]:
        """Run the simulation and return snapshots."""
        # Set random seed for reproducibility
        np.random.seed(self.seed)
        # Validate timestep before running
        self._validate_timestep(t_end_Gyr, n_steps)

        # Convert to seconds
        t_end = t_end_Gyr * 1e9 * 365.25 * 24 * 3600

        print("\n" + "="*60)
        print("RUNNING COSMOLOGICAL SIMULATION")
        print("="*60)
        print(f"Model: {'External-Node' if self.use_external_nodes else 'Matter-only'}")
        print(f"Duration: {t_end_Gyr} Gyr")
        print(f"Timesteps: {n_steps}")
        print("="*60 + "\n")

        # Velocity calibration for matter-only: find initial velocity so step 2 matches LCDM
        if not self.use_dark_energy:# and not self.use_external_nodes:
            self._calibrate_velocity_for_lcdm_match(t_end_Gyr, n_steps, damping)

        # Run integration
        self.snapshots = self.integrator.evolve(t_end, n_steps, save_interval)
        
        # Calculate expansion history
        self._calculate_expansion_history()
        
        print("\nSimulation complete!")
        return self.snapshots
    
    def _calculate_expansion_history(self) -> None:
        """Calculate the scale factor a(t) from snapshots."""
        self.expansion_history = []

        rms_initial, max_initial, _ = self.calculate_system_size(self.snapshots[0])

        for snapshot in self.snapshots:
            t = snapshot['time_s']
            rms_current, max_current, com = self.calculate_system_size(snapshot)

            # Scale factor a(t) = R(t) / R(t=0)
            # Use RMS for scale factor (typical expansion)
            a = rms_current / rms_initial

            # Physical size: consistent with ΛCDM (a * box_size_initial)
            # This ensures all models start from the same physical size
            size_Gpc = a * self.box_size_Gpc

            # diameter_m = 2 × rms_radius_m
            self.expansion_history.append({
                'time': t,
                'time_Gyr': t / (1e9 * 365.25 * 24 * 3600),
                'scale_factor': a,
                'diameter_m': rms_current*2,
                'size_a': size_Gpc* self.const.Gpc_to_m,
                'max_particle_distance': max_current,
                'com': com,
            })
    
    @staticmethod
    def calculate_system_size(snapshot):
        positions = snapshot['positions']
        return ParticleSystem.calculate_system_size(positions)
    
    def save(self, filename: str) -> None:
        """Save simulation results."""
        data = {
            'snapshots': self.snapshots,
            'expansion_history': self.expansion_history,
            'use_external_nodes': self.use_external_nodes,
            'n_particles': len(self.particles),
            'time_history': self.integrator.time_history,
            'energy_history': self.integrator.energy_history,
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"\nSaved simulation to {filename}")
    
    @staticmethod
    def load(filename: str) -> Dict:
        """Load simulation results."""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data
