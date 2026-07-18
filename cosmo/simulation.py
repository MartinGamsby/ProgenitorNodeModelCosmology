"""
Main Simulation Runner
Compares ΛCDM cosmology with External-Node Model
"""

from typing import Optional, List, Dict
import numpy as np
import pickle

from .constants import CosmologicalConstants, SimulationParameters
from .particles import ParticleSystem, HMEAGrid
from .integrator import LeapfrogIntegrator
from .analysis import solve_friedmann_at_times
from .visualization import generate_output_filename
from .cache import Cache, CacheType

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, *args, **kwargs):
        return iterable

velocity_cache = None

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
        global velocity_cache
        if not velocity_cache:
            velocity_cache = Cache("velocity")

        self.const = CosmologicalConstants()
        self.sim_params = sim_params
        self.use_external_nodes = use_external_nodes
        self.t_start_Gyr = sim_params.t_start_Gyr
        self.a_start = a_start
        # Start-size lever (Section 6): scale the incoming LCDM-implied box BEFORE
        # building particles. 1.0 (default) -> box unchanged, byte-identical. The
        # scaled box is what self.box_size_Gpc stores, so size_Gpc = a(t)*box and
        # the EdS-critical cloud MASS (derived from box volume in ParticleSystem)
        # both follow the scaled size consistently. The nodes keep their UNSCALED
        # spacing S, so a bigger/smaller cloud spans a different fraction of S ->
        # different differential tidal shear -> a(t) SHAPE moves (M_ext>0). a(t) is
        # an RMS RATIO (scale-free), so a pure size change at M_ext=0 leaves a(t)
        # identical (density stays EdS-critical) -> M=0 == EdS preserved at any size.
        start_size_scale = float(getattr(sim_params, 'start_size_scale', 1.0))
        box_size_Gpc = box_size_Gpc * start_size_scale
        self.start_size_scale = start_size_scale
        self.box_size_Gpc = box_size_Gpc  # Store (scaled) initial box size for consistent size calculation
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

        # EdS-consistent initial conditions: only meaningful when dark energy is
        # OFF (matter-only / external-node). When enabled, ParticleSystem ignores
        # total_mass_kg above and instead carries the EdS critical mass, with a
        # matching H_EdS Hubble flow, so M_ext=0 reproduces analytic EdS a(t).
        self.eds_consistent = bool(getattr(sim_params, 'eds_consistent', True)) and (not self.use_dark_energy)

        # Initialize particle system
        print(f"Initializing {sim_params.n_particles} particles in {box_size_Gpc} Gpc box...")
        if sim_params.center_node_mass != 1.0:
            print(f"centerM={sim_params.center_node_mass} (outer-mass multiplier): "
                  f"outer particles will be added outside R_obs; R_sim = R_obs × {sim_params.center_node_mass**(1/3):.4f}")

        self.particles = ParticleSystem(n_particles=sim_params.n_particles,
                                       box_size_m=box_size_m,
                                       total_mass_kg=total_mass_kg,
                                       a_start=self.a_start,
                                       use_dark_energy=self.use_dark_energy,
                                       mass_randomize=sim_params.mass_randomize,
                                       init_distribution=sim_params.init_distribution,
                                       init_kwargs=sim_params.init_kwargs,
                                       eds_consistent=self.eds_consistent,
                                       t_start_Gyr=self.t_start_Gyr,
                                       center_node_mass=sim_params.center_node_mass,
                                       outer_density_ceiling=sim_params.outer_density_ceiling,
                                       outer_particle_cap=getattr(sim_params, 'outer_particle_cap', 0.0))

        # Initialize HMEA grid if using External-Node Model
        self.hmea_grid = None
        if use_external_nodes:
            self.hmea_grid = HMEAGrid(node_params=sim_params.external_params, n_nodes=8)
            print(f"External-Node Model: {self.hmea_grid}")
        else:
            print("Running standard matter-only (no dark energy)")

        # Physical pre-t_start HMEA boost: the cloud should ARRIVE at t_start with a
        # radial velocity slightly ABOVE pure EdS Hubble flow because the HMEA tidal
        # field has been pulling on it from the Big Bang to t_start. Applied here
        # (after particles + grid exist, before any integration), only for the
        # External-Node + EdS-consistent case. Scales with M_ext so it vanishes as
        # M_ext -> 0 (M=0 == EdS preserved exactly).
        self.pre_start_tidal_boost = (
            bool(getattr(sim_params, 'pre_start_tidal_boost', True))
            and use_external_nodes
            and self.eds_consistent
            and self.hmea_grid is not None
            and self.t_start_Gyr is not None
            and self.t_start_Gyr > 0
        )
        if self.pre_start_tidal_boost:
            self._apply_pre_start_tidal_boost()

        # Softening frozen at the centerM=1 baseline (1 Gpc per M_obs). It must NOT
        # scale with center_node_mass: centerM now means an OUTER-MASS multiplier, and
        # the old centerM->softening coupling made the centerM->chi2 shift a resolution
        # ARTIFACT rather than real added gravity. Freezing keeps centerM=1 byte-
        # identical (the old value WAS 1.0*1.0*Gpc) and makes centerM>1 change a(t)
        # ONLY via added outer-mass gravity. (WS4 correctness gate.)
        softening_m = 1.0 * self.const.Gpc_to_m
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
            use_hubble_drag=use_hubble_drag,
            # Adaptive KDK sub-stepping (Section 4). Default OFF (threshold 0.0)
            # -> one plain leapfrog step (byte-identical).
            node_substep_threshold=float(getattr(sim_params, 'node_substep_threshold', 0.0)),
            node_substeps=int(getattr(sim_params, 'node_substeps', 1)),
        )
        
        # Simulation results
        self.snapshots = []
        self.expansion_history = []

    def _apply_pre_start_tidal_boost(self) -> None:
        """Add the pre-t_start HMEA tidal velocity boost to the initial conditions.

        Physical motivation
        -------------------
        The EdS-consistent ICs set v_i = H_EdS(t_start) * r_i, i.e. the cloud
        arrives at t_start moving at PURE matter-only Hubble flow. But for M_ext>0
        the HMEA nodes have been pulling on the cloud since the Big Bang, so the
        cloud should actually arrive at t_start moving slightly FASTER (a net
        outward boost). This term restores that pre-history.

        Derivation (linear / early-time regime, S >> cloud size)
        --------------------------------------------------------
        Per particle at displacement r from the cloud centre, the HMEA tidal
        acceleration is the SAME node sum the integrator uses:
            g_tid(r, t) = sum_nodes G m_node (r - r_node)/|r - r_node|^3 .
        At early times the cloud is small, so g_tid is ~linear in r and the node
        distances are ~constant; the positions track the EdS background,
        r(t) = r_start * a(t)/a_start. Hence the RADIAL tidal accel scales as
            g_r(t) ~= g_r(t_start) * a(t)/a_start .
        The extra radial velocity imparted from t_i to t_start (proper coords,
        the same frame as v = H_EdS*r) is
            dv_r = integral_{t_i}^{t_start} g_r(t) dt
                 = g_r(t_start)/a_start * integral_{t_i}^{t_start} a_EdS(t) dt .
        With EdS a(t) = a_start (t/t_start)^(2/3):
            integral_{t_i}^{t_start} a(t) dt
              = a_start (3/5) t_start (1 - (t_i/t_start)^(5/3)) .
        Taking t_i -> 0 (full pre-history from the Big Bang) gives the clean,
        parameter-free factor (3/5) t_start:
            dv_r(particle) = g_r(t_start) * (3/5) * t_start_seconds .

        Properties
        ----------
        * VANISHES as M_ext -> 0 (g_tid is linear in the node masses), so the
          M_ext=0 == Einstein-de Sitter invariant is preserved EXACTLY.
        * Monotone-ish increasing in M_ext and decreasing in S (stronger / closer
          nodes pull harder), as required.
        * Uses the real node sum (incl. per-node anisotropy), not an analytic
          Omega_Lambda — it is NOT a fit-to-LCDM knob.

        Only the radial (expansion) component is added; the boost is applied along
        each particle's radial unit vector. The net COM velocity is removed
        afterwards so no bulk drift is introduced.
        """
        positions = self.particles.get_positions()        # (N,3) m, centred
        velocities = self.particles.get_velocities()       # (N,3) m/s

        # Tidal acceleration at t_start positions (same path as the integrator).
        g_tid = self.hmea_grid.calculate_tidal_acceleration_batch(positions)  # (N,3)

        # Radial unit vectors (guard the origin particle).
        r = np.linalg.norm(positions, axis=1, keepdims=True)
        r_safe = np.where(r > 0.0, r, 1.0)
        r_hat = positions / r_safe

        # Radial component of the tidal accel (positive = outward / expansion).
        g_r = np.sum(g_tid * r_hat, axis=1)                # (N,)

        # Integrated pre-history factor (3/5) t_start, in seconds.
        t_start_s = self.t_start_Gyr * self.const.Gyr_to_s
        dv_r = g_r * (3.0 / 5.0) * t_start_s               # (N,) m/s along r_hat

        boosted = velocities + dv_r[:, np.newaxis] * r_hat

        # Remove any net COM velocity the boost introduced (keep the cloud at rest).
        com_v = np.mean(boosted, axis=0)
        boosted = boosted - com_v
        self.particles.set_velocities(boosted)

        rms_dv = float(np.sqrt(np.mean(dv_r ** 2)))
        rms_v = float(np.sqrt(np.mean(np.sum(velocities ** 2, axis=1))))
        mean_g_r = float(np.mean(g_r))
        print(
            f"[Pre-start tidal boost] Applied: RMS dv_r = {rms_dv/1e3:.1f} km/s "
            f"({100.0*rms_dv/max(rms_v,1e-30):.3f}% of Hubble flow), "
            f"mean radial accel = {mean_g_r:.3e} m/s^2 "
            f"(t_i->0 Big-Bang pre-history, factor (3/5)*t_start)."
        )

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

        # Velocity-calibration cache key.
        # NOTE: this whole method is the LEGACY non-EdS calibration path. It is only
        # reached from run() when NOT (eds_consistent and damping is None) — i.e. for
        # eds_consistent=False runs without an explicit damping override. Under the
        # default eds_consistent=True it is NEVER called, so the headline/pinned runs
        # never touch this cache. generate_output_filename omits start_size_scale,
        # node_softening_gpc and node_geometry, but those knobs DO change the
        # calibrated velocity scale (start size changes density/expansion; softening
        # and geometry change the tidal field measured during the test). We thread
        # them into the key (only when non-default, to keep existing keys stable) so a
        # keyed value always matches the run that produced it — closing the
        # "keyed-but-not-run" foot-gun on this legacy path.
        calib_name = generate_output_filename('', self.sim_params, '', '', include_timestamp=False,
                                               include_S=False, include_M=False, include_D=False)
        _calib_extra = []
        _start_size_scale = float(getattr(self.sim_params, 'start_size_scale', 1.0))
        if _start_size_scale != 1.0:
            _calib_extra.append(f"{_start_size_scale}ss")
        _node_softening_gpc = float(getattr(self.sim_params, 'node_softening_gpc', 0.0))
        if _node_softening_gpc != 0.0:
            _calib_extra.append(f"{_node_softening_gpc}soft")
        _node_geometry = getattr(self.sim_params, 'node_geometry', 'cube26')
        if _node_geometry != 'cube26':
            _calib_extra.append(f"{_node_geometry}geom")
        if _calib_extra:
            calib_name = calib_name + "_" + "_".join(_calib_extra)
        cached_velocity = velocity_cache.get_cached_value(calib_name, CacheType.VELOCITY)
        if cached_velocity:
            self.particles.set_velocities(self.particles.get_velocities()*cached_velocity)
            print(f"Using cached calibration of {cached_velocity} for {calib_name}")
            return

        print("Calibrating", end="", flush=True)
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
        
        total_velocity_scale = 1.0
        velocity_scale = 1.0
        for tries in range(20):
            
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

                if not (step % 10):
                    print(".", end="", flush=True)
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
            total_velocity_scale *= velocity_scale

        print(f"\n[Velocity Calibration] Calibration period: {calibration_duration_Gyr:.2f} Gyr ({calibration_steps} steps)")
        print(f"[Velocity Calibration] Velocity scale factor: {total_velocity_scale:.6f}")

            
        # Restore initial state and external node setting
        self.particles.set_positions(initial_positions)
        self.particles.time = initial_time
        self.integrator.use_external_nodes = saved_use_external_nodes

        # Apply calibrated velocity
        self.particles.set_velocities(updated_velocities)
        velocity_cache.add_cached_value(calib_name, CacheType.VELOCITY, total_velocity_scale, save_interval_s=0)

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
        #
        # SKIP calibration when EdS-consistent ICs are active: the Hubble flow and
        # cloud density are already mutually consistent, so the expansion is set by
        # real physics (M_ext=0 -> EdS, M_ext>0 -> tidal acceleration). Calibrating
        # to LCDM here would re-introduce the very fudge this mode removes. An
        # explicit user-provided damping override is still honoured.
        if not self.use_dark_energy:
            if self.eds_consistent and damping is None:
                print("[Velocity Calibration] Skipped (EdS-consistent ICs: "
                      "physical Hubble flow + critical density, no calibration).")
            else:
                self._calibrate_velocity_for_lcdm_match(t_end_Gyr, n_steps, damping)

        # Run integration
        self.snapshots = self.integrator.evolve(t_end, n_steps, save_interval)
        
        # Calculate expansion history
        self._calculate_expansion_history()
        
        print("\nSimulation complete!")
        return self.snapshots
    
    def _calculate_expansion_history(self) -> None:
        """Calculate the scale factor a(t) from snapshots.

        WS4 / Section 2 — OBSERVABLE INNER REGION ONLY
        -----------------------------------------------
        When centerM > 1 the simulation contains both inner (observable) and
        outer (shell) particles.  a(t) — and hence H(z), mu(z), and the growth
        anchor — MUST be computed from the inner observable sub-region only.
        The outer particles still exert gravity (integrator is UNCHANGED) but
        must never enter the size/expansion MEASUREMENT.

        Implementation: the observable mask (True for inner particles, always
        all-True when centerM=1) is applied to every snapshot's position array
        before passing it to calculate_system_size.  When centerM=1 the mask is
        all-True so the result is numerically byte-identical to the pre-WS4 code.

        Guard: if the ParticleSystem was constructed without an observable_mask
        attribute (very old test constructions), we default to all-True.
        """
        self.expansion_history = []

        # --- Observable mask (Section 2 gate) ---
        # getattr guard: older ParticleSystem constructions (pre-WS4) may not
        # have observable_mask; default to all-True for backward compat.
        raw_mask = getattr(self.particles, 'observable_mask', None)
        if raw_mask is None:
            n_total = len(self.particles.particles)
            mask = np.ones(n_total, dtype=bool)
        else:
            mask = np.asarray(raw_mask, dtype=bool)

        # Initial baseline on the INNER observable subset only.
        inner_pos_initial = self.snapshots[0]['positions'][mask]
        rms_initial, _, _ = ParticleSystem.calculate_system_size(inner_pos_initial)

        for snapshot in self.snapshots:
            t = snapshot['time_s']

            # Inner-subset positions for this snapshot.
            inner_pos = snapshot['positions'][mask]
            rms_current, max_current, com = ParticleSystem.calculate_system_size(inner_pos)

            # Scale factor a(t) = R_inner(t) / R_inner(t=0)
            # Use RMS over the OBSERVABLE sub-region only.
            a = rms_current / rms_initial

            # Physical size: consistent with ΛCDM (a * box_size_initial)
            # This ensures all models start from the same physical size
            size_Gpc = a * self.box_size_Gpc

            # max_particle_distance: also restricted to observable subset so
            # downstream callers see the inner cloud's furthest particle.
            # diameter_m = 2 × rms_radius_m (inner only)
            self.expansion_history.append({
                'time': t,
                'time_Gyr': t / (1e9 * 365.25 * 24 * 3600),
                'scale_factor': a,
                'diameter_m': rms_current * 2,
                'size_a': size_Gpc * self.const.Gpc_to_m,
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
