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

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, *args, **kwargs):
        return iterable

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

    def _calibrate_velocity_for_lcdm_match(self, t_duration_Gyr: float, n_steps: int, damping: float = None,
                                           percent_sim: float = 0.3) -> None:
        """
        Calibrate initial velocity so matter-only tracks LCDM, never overshooting.

        Goal: Find velocity scale that keeps N-body at or below LCDM throughout
        the calibration period. Uses the maximum scale needed at any step.

        Strategy:
        1. Run N-body test for ~2 Gyr / 20% of simulation (for percent_sim=0.2)
        2. At each step, compare N-body size to LCDM size
        3. Track the velocity scale that would be needed to match LCDM at that step
        4. Use the maximum scale found (most conservative, prevents overshoot)
        """
        # TODO: Make sure we don't do that for both Matter-only and External nodes ... in run_simulation.py!)
        dt_Gyr = t_duration_Gyr / n_steps
        dt_s = dt_Gyr * 1e9 * 365.25 * 24 * 3600

        # Apply user damping override if provided (skip N-body test)
        if damping is not None:
            print(f"[Velocity Calibration] Using user-provided damping: {damping}")
            velocities = self.particles.get_velocities()
            self.particles.set_velocities(velocities * damping)
            print(f"[Velocity Calibration] Applied velocity scaling: {damping:.6f}")
            return

        # Save initial state for restoration
        initial_positions = self.particles.get_positions()
        updated_velocities = self.particles.get_velocities()
        initial_time = self.particles.time

        # Measure initial RMS radius
        rms_initial = np.sqrt(np.mean(np.sum(initial_positions**2, axis=1)))

        # Temporarily disable external forces for calibration
        # We want to measure pure N-body expansion rate
        saved_use_external_nodes = self.integrator.use_external_nodes
        self.integrator.use_external_nodes = False


        # Run test for ~2 Gyr or 20% of simulation, whichever is smaller ( for percent_sim == 0.2 )
        steps_per_Gyr = n_steps / t_duration_Gyr
        calibration_steps = min(int(10.0 * percent_sim * steps_per_Gyr), int(n_steps * percent_sim))
        calibration_steps = max(10, calibration_steps)  # At least 10 steps
        calibration_duration_Gyr = calibration_steps * dt_Gyr
        
        velocity_scale = 1.0
        for tries in tqdm(range(20), mininterval=.1, desc="Preparing", unit="step"):
            
            self.particles.set_positions(initial_positions)
            initial_positions = initial_positions.copy()
            self.particles.set_velocities(updated_velocities)
            updated_velocities = updated_velocities.copy()
            self.particles.time = initial_time

            # Pre-compute LCDM expansion at each step
            t_points = self.t_start_Gyr + np.arange(1, calibration_steps + 1) * dt_Gyr
            lcdm_at_steps = solve_friedmann_at_times(
                np.concatenate([[self.t_start_Gyr], t_points])
            )
            lcdm_a_start = lcdm_at_steps['a'][0]

            # Track max velocity scale needed
            max_velocity_scale = 0.0
            min_velocity_scale = 1.9

            velocity_scale_at_step = 1.0
            last_step_direction = 1
            use_min_velocity = False

            for step in range(calibration_steps):
                self.integrator.step(dt_s)

                # Measure N-body expansion at this step
                current_positions = self.particles.get_positions()
                rms_current = np.sqrt(np.mean(np.sum(current_positions**2, axis=1)))
                nbody_expansion = rms_current / rms_initial

                # Get LCDM expansion at this step
                lcdm_expansion = lcdm_at_steps['a'][step + 1] / lcdm_a_start


                step_direction = (lcdm_expansion / nbody_expansion) - velocity_scale_at_step

                # Calculate velocity scale needed to match LCDM at this step
                # If N-body > LCDM, we need scale < 1 (slow down)
                # If N-body < LCDM, we need scale > 1 (speed up)
                velocity_scale_at_step = lcdm_expansion / nbody_expansion

                if step_direction < 0 and last_step_direction > 0:
                    use_min_velocity = True

                # TODO: Change logic: It can't go up/down??

                if velocity_scale_at_step > max_velocity_scale:
                    max_velocity_scale = velocity_scale_at_step
                if velocity_scale_at_step < min_velocity_scale:
                    min_velocity_scale = velocity_scale_at_step

                last_step_direction = step_direction
                if use_min_velocity:
                    break

            # Use the maximum velocity scale found (most conservative)
            last_velocity_scale = velocity_scale
            if use_min_velocity:
                if min_velocity_scale < 1.0:
                    velocity_scale = min_velocity_scale
                else:
                    velocity_scale = 1.0+(1.0-max_velocity_scale)
            else:
                velocity_scale = max_velocity_scale

            # Clamp to reasonable range
            velocity_scale = np.clip(velocity_scale, 0.1, 1.9)

            if (last_velocity_scale > 1.0 and velocity_scale < 1.0) or (last_velocity_scale < 1.0 and velocity_scale > 1.0):
                break
            updated_velocities *= velocity_scale



        print(f"[Velocity Calibration] Calibration period: {calibration_duration_Gyr:.2f} Gyr ({calibration_steps} steps)")
        print(f"[Velocity Calibration] Max scale needed: {velocity_scale:.6f}")
        print(f"[Velocity Calibration] Velocity scale factor: {velocity_scale:.6f}")

            
        # Restore initial state and external node setting
        self.particles.set_positions(initial_positions)
        self.particles.time = initial_time
        self.integrator.use_external_nodes = saved_use_external_nodes

        # Apply calibrated velocity
        self.particles.set_velocities(updated_velocities)

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

        # Velocity calibration: calibrate initial velocity to match LCDM expansion
        # For matter-only: uses N-body test to measure deceleration deficit
        # For External-Node: uses N-body test including HMEA tidal forces
        if not self.use_dark_energy:
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
