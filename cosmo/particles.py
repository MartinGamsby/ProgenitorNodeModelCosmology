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
                 use_dark_energy: bool = True,
                 mass_randomize: float = 0.5,
                 init_distribution: str = "uniform_sphere",
                 init_kwargs: Optional[dict] = None,
                 eds_consistent: bool = False,
                 t_start_Gyr: Optional[float] = None,
                 center_node_mass: float = 1.0,
                 outer_density_ceiling: float = 1.0):
        """
        Initialize particle system with damped Hubble flow initial conditions.

        Args:
            n_particles: Number of particles
            box_size_m: Box size in meters (default: Hubble radius)
            total_mass_kg: Total mass in kg (default: M_observable)
            a_start: Initial scale factor
            use_dark_energy: Whether dark energy is enabled
            mass_randomize: Mass distribution randomness (0.0 = equal masses,
                           1.0 = masses from 0 to 2x mean, 0.5 = default)
            init_distribution: Position sampler: "uniform_sphere" (default,
                               backward-compatible) or "grf" (Gaussian random field
                               + Zel'dovich displacement shaped by BBKS LCDM P(k)).
            init_kwargs: Optional dict forwarded to the sampler (e.g. Ng for grf).
            eds_consistent: If True, use SELF-CONSISTENT Einstein-de Sitter initial
                            conditions (Ω_m=1): the Hubble flow v_i = H_EdS(t_start)*r_i
                            with H_EdS = 2/(3 t_start), AND the cloud total mass is
                            OVERRIDDEN to the EdS critical mass ρ_crit(t_start)*V so
                            self-gravity supplies the exact EdS deceleration. With
                            no external tidal forces this makes the from-sim a(t)
                            reproduce the analytic EdS solution by construction
                            (the M=0 == EdS invariant). Requires t_start_Gyr.
                            Default False keeps the legacy behaviour.
            t_start_Gyr: Absolute start time in Gyr, REQUIRED when eds_consistent
                         is True (sets H_EdS and ρ_crit). Ignored otherwise.
            center_node_mass: Outer-mass multiplier (>= 1.0, default 1.0). When
                         > 1.0 outer particles are added OUTSIDE the inner
                         observable sphere at the same density and per-particle
                         mass. N grows LINEARLY; R_sim = R_obs*centerM**(1/3).
                         centerM=1.0 -> no outer particles, byte-identical.
                         Only supported for init_distribution="uniform_sphere";
                         centerM>1 with "grf" raises NotImplementedError.
            outer_density_ceiling: Multiplier on inner density for outer particles
                         (default 1.0 = same density). Clipped to
                         MAX_OUTER_DENSITY_CEILING (2.0) upstream.
        """
        const = CosmologicalConstants()
        self.const = const

        self.n_particles = n_particles
        self.box_size_m = box_size_m if box_size_m is not None else const.R_hubble
        self.a_start = a_start
        self.use_dark_energy = use_dark_energy
        self.mass_randomize = np.clip(mass_randomize, 0.0, 1.0)
        self.init_distribution = init_distribution
        self.init_kwargs = init_kwargs if init_kwargs is not None else {}
        self.eds_consistent = eds_consistent
        self.t_start_Gyr = t_start_Gyr
        # centerM: outer-mass multiplier (>= 1.0). 1.0 = no outer particles.
        self.center_node_mass = max(1.0, float(center_node_mass))
        # outer_density_ceiling: density multiplier for outer shell (default 1.0).
        self.outer_density_ceiling = float(outer_density_ceiling)

        # EdS-consistent mode: the cloud must carry the EdS critical (background)
        # density so internal self-gravity matches the Friedmann deceleration.
        # We OVERRIDE total_mass_kg with ρ_crit(t_start) * V_sphere; the velocity
        # field below uses the matching H_EdS = 2/(3 t_start). Velocity & density
        # are mutually consistent => M_ext=0 reproduces EdS a(t) exactly.
        # H_EdS = 2/(3 t_start) is only defined for a positive start time; t_start<=0
        # (e.g. Big-Bang t=0 used by some size-semantics tests) has no finite Hubble
        # rate, so fall back to the legacy (non-EdS) initialisation there.
        if eds_consistent and t_start_Gyr is not None and t_start_Gyr <= 0:
            eds_consistent = False
            self.eds_consistent = False

        if eds_consistent:
            if t_start_Gyr is None:
                raise ValueError("eds_consistent=True requires t_start_Gyr.")
            H_eds = LambdaCDMParameters.H_eds_at_time(t_start_Gyr)
            rho_crit = LambdaCDMParameters.eds_critical_density(H_eds)
            # Uniform sphere: target RMS radius = box_size/2, so physical sphere
            # radius R_sphere = RMS / sqrt(3/5) (RMS = R*sqrt(3/5) for a uniform
            # ball). The carried mass is the critical density times that volume.
            rms_radius_m = self.box_size_m / 2.0
            r_sphere_m = rms_radius_m / np.sqrt(3.0 / 5.0)
            volume_m3 = (4.0 / 3.0) * np.pi * r_sphere_m ** 3
            self.total_mass_kg = rho_crit * volume_m3
        else:
            self.total_mass_kg = total_mass_kg if total_mass_kg is not None else const.M_observable_kg

        self.particles = []
        self.time = 0.0

        # Initialize particles
        self._initialize_particles()
        
    # ------------------------------------------------------------------
    # Private position samplers
    # ------------------------------------------------------------------

    def _init_uniform_sphere(self) -> np.ndarray:
        """Uniform sphere rejection sampler (legacy behaviour).

        Reproduces the ORIGINAL positions loop from the pre-refactor code so
        existing tests that pin output to a fixed np.random seed remain valid.

        Returns:
            positions : (N, 3) float64, raw (not centred / normalised).
        """
        sphere_radius_m = (self.box_size_m / 2) / np.sqrt(3 / 5)
        positions = []
        for i in range(self.n_particles):
            while True:
                pos = np.random.uniform(-sphere_radius_m, sphere_radius_m, 3)
                if np.linalg.norm(pos) <= sphere_radius_m:
                    break
            positions.append(pos)
        return np.array(positions)

    def _init_outer_shell(self, r_inner_m: float, r_outer_m: float, n_outer: int) -> np.ndarray:
        """Uniform rejection sampler for the outer shell r_inner < |r| <= r_outer.

        Draws continue from the CURRENT np.random state (immediately after inner
        draws), so the inner draws are never disturbed. The shell is rejection-sampled
        in the enclosing cube of half-side r_outer.

        Args:
            r_inner_m: Inner (observable) sphere radius in metres.
            r_outer_m: Outer (sim) sphere radius in metres.
            n_outer:   Number of outer particles to generate.

        Returns:
            positions: (n_outer, 3) float64, raw (NOT centred or normalised).
        """
        positions = []
        while len(positions) < n_outer:
            pos = np.random.uniform(-r_outer_m, r_outer_m, 3)
            d = np.linalg.norm(pos)
            if r_inner_m < d <= r_outer_m:
                positions.append(pos)
        return np.array(positions)

    def _init_grf(self) -> np.ndarray:
        """Gaussian random field + Zel'dovich displacement sampler.

        Uses the current np.random seed (set by simulation.py) as the integer
        seed forwarded to sample_grf so the result is reproducible for a given
        SimulationParameters.seed.

        Returns:
            positions : (N, 3) float64, raw (not centred / normalised).
        """
        from .initial_distributions import sample_grf
        # Use the seed established by np.random.seed() in simulation.py.
        # np.random.randint gives a fresh deterministic integer from that stream.
        grf_seed = int(np.random.randint(0, 2**31))
        kwargs = dict(self.init_kwargs)  # copy so we don't mutate the original
        return sample_grf(
            n_particles=self.n_particles,
            box_size_m=self.box_size_m,
            seed=grf_seed,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Main initializer
    # ------------------------------------------------------------------

    def _initialize_particles(self) -> None:
        """Create initial particle distribution with Hubble flow.

        When center_node_mass > 1.0 (centerM > 1), outer particles are added
        OUTSIDE the inner observable sphere at the same per-particle mass and
        outer_density_ceiling * inner_density number density. The inner
        observable region is UNCHANGED (same positions, masses) — only outer
        particles are appended.

        self.observable_mask (bool, N_total) is set here:
          True  = inner observable particle (indices 0 .. N_inner-1)
          False = outer shell particle      (indices N_inner .. N_total-1)
        centerM=1 -> all-True mask of length n_particles (byte-identical).
        """
        lcdm = LambdaCDMParameters()

        # Use model-appropriate Hubble parameter for initial velocity
        # ΛCDM: H includes dark energy (Ω_Λ) → higher expansion rate
        # Matter-only: H without dark energy → lower expansion rate
        # This ensures each model's N-body matches its own Friedmann solution
        if self.eds_consistent:
            # Pure EdS Hubble flow, consistent with the critical density carried by
            # the cloud (set in __init__). H_EdS = 2/(3 t_start) makes a(t) follow
            # the analytic (t/t_start)^(2/3) EdS solution with no calibration.
            H_start = lcdm.H_eds_at_time(self.t_start_Gyr)
            print(f"[ParticleSystem] EdS-consistent H(t_start={self.t_start_Gyr:.3f} Gyr) = {H_start:.3e} /s")
        elif self.use_dark_energy:
            H_start = lcdm.H_at_time(self.a_start)
            print(f"[ParticleSystem] Using LCDM H(a={self.a_start:.3f}) = {H_start:.3e} /s")
        else:
            H_start = lcdm.H_matter_only(self.a_start)
            print(f"[ParticleSystem] Using matter-only H(a={self.a_start:.3f}) = {H_start:.3e} /s")

        # N_inner is the requested observable particle count (always = n_particles).
        n_inner = self.n_particles
        center_m = self.center_node_mass  # >= 1.0

        # Non-EdS + centerM>1 is not supported; EdS path is the one used for WS4.
        if center_m > 1.0 and not self.eds_consistent:
            raise NotImplementedError(
                "centerM > 1.0 (outer-particle generation) is only supported "
                "with eds_consistent=True. For non-EdS / LCDM runs, centerM must "
                "remain 1.0. (WS4 outer-shell design targets the EdS path.)"
            )

        # centerM > 1.0 requires uniform_sphere sampler (GRF outer shell: TODO).
        if center_m > 1.0 and self.init_distribution != "uniform_sphere":
            raise NotImplementedError(
                f"Outer-particle generation (centerM={center_m} > 1.0) is not yet "
                "implemented for init_distribution='grf'. Use init_distribution="
                "'uniform_sphere' for WS4, or set centerM=1.0 for GRF runs. "
                "(Follow-up: add shell sampling to the GRF path.)"
            )

        # ------------------------------------------------------------------
        # INNER-PARTICLE MASS (EdS path: total_mass_kg is the EdS critical mass
        # for the INNER sphere; per-particle mass = total_mass_kg / n_inner).
        # The SAME per-particle mass is reused for outer particles so that outer
        # number density == outer_density_ceiling * inner density (at ceiling=1
        # this is exactly the same density, giving total mass = centerM * inner).
        # ------------------------------------------------------------------
        mean_mass_kg = self.total_mass_kg / n_inner

        if self.mass_randomize > 0 and n_inner > 1:
            # Generate random masses with specified randomization level.
            # mass_randomize=1.0: uniform in [0, 2*mean], range is 2*mean.
            # mass_randomize=0.5: uniform in [0.5*mean, 1.5*mean], range is mean.
            # mass_randomize=0.0: all masses equal to mean.
            half_range = self.mass_randomize * mean_mass_kg
            raw_masses = np.random.uniform(
                mean_mass_kg - half_range,
                mean_mass_kg + half_range,
                n_inner,
            )
            # Ensure no negative masses (shouldn't happen unless randomize > 1, be safe).
            raw_masses = np.maximum(raw_masses, 1e-10 * mean_mass_kg)
            # Normalize to preserve total mass exactly.
            inner_masses_kg = raw_masses * (self.total_mass_kg / np.sum(raw_masses))
            print(f"[ParticleSystem] Mass randomize={self.mass_randomize:.2f}: "
                  f"min={np.min(inner_masses_kg):.2e}, max={np.max(inner_masses_kg):.2e}, "
                  f"mean={np.mean(inner_masses_kg):.2e} kg")
        else:
            inner_masses_kg = np.full(n_inner, mean_mass_kg)

        # ------------------------------------------------------------------
        # POSITION SAMPLING: inner particles FIRST (unchanged), then outer shell.
        # The sampler must return raw (N, 3) positions; post-processing
        # (centre + RMS-norm) follows below, referenced to the INNER subset.
        # ------------------------------------------------------------------
        print(f"[ParticleSystem] init_distribution={self.init_distribution!r}, centerM={center_m:.4f}")
        if self.init_distribution == "uniform_sphere":
            inner_positions_raw = self._init_uniform_sphere()
        elif self.init_distribution == "grf":
            inner_positions_raw = self._init_grf()
        else:
            raise ValueError(
                f"Unknown init_distribution {self.init_distribution!r}. "
                "Valid choices: 'uniform_sphere', 'grf'."
            )

        # CRITICAL: Center positions FIRST before calculating velocities.
        # Random particle distribution creates non-zero COM position.
        # We must center BEFORE velocity calculation so v_hubble = H*r uses centred positions.
        com_position = np.mean(inner_positions_raw, axis=0)
        print(f"[ParticleSystem] Centering COM position: "
              f"[{com_position[0]:.3e}, {com_position[1]:.3e}, {com_position[2]:.3e}] m")
        inner_centered = inner_positions_raw - com_position

        # CRITICAL: Normalize to exact target RMS radius using the INNER subset.
        # We must compute the scale factor from the inner subset RMS so that:
        #   (a) centerM=1 is byte-identical to the old code (inner==all -> same calc).
        #   (b) centerM>1: inner region keeps RMS = box/2 (its density is what's fixed);
        #       outer particles get the SAME scale applied, preserving relative geometry.
        current_inner_rms = np.sqrt(np.mean(np.sum(inner_centered**2, axis=1)))
        target_rms = self.box_size_m / 2  # RMS should be half box size

        # Handle edge case: if inner RMS is negligible (e.g. n=1 particle at origin).
        if current_inner_rms > 1e-10 * target_rms:
            scale_factor = target_rms / current_inner_rms
            inner_centered *= scale_factor
            print(f"[ParticleSystem] Normalized inner RMS: {current_inner_rms:.6e} -> {target_rms:.6e} m "
                  f"(scale={scale_factor:.6f})")
            final_inner_rms = np.sqrt(np.mean(np.sum(inner_centered**2, axis=1)))
            assert abs(final_inner_rms - target_rms) / target_rms < 1e-10, \
                f"Inner RMS normalization failed: {final_inner_rms:.6e} vs {target_rms:.6e}"
        else:
            scale_factor = 1.0
            print(f"[ParticleSystem] Skipping RMS normalization "
                  f"(inner RMS={current_inner_rms:.3e} is negligible)")

        # ------------------------------------------------------------------
        # OUTER PARTICLE GENERATION (centerM > 1 only).
        # R_obs = r_sphere used during inner sampling (before scale; but the
        # scaled inner sphere radius = r_sphere * scale_factor, which is what
        # we need for the shell boundary). We use the scaled inner sphere radius.
        # ------------------------------------------------------------------
        if center_m > 1.0:
            # Inner sphere radius (scaled, physical).
            # _init_uniform_sphere() uses sphere_radius = (box_size/2)/sqrt(3/5).
            # After scale_factor: R_obs_scaled = sphere_radius * scale_factor.
            # However, since we normalised the RMS to box/2, and for a uniform sphere
            # RMS = R * sqrt(3/5), we have R_obs_physical = (box/2) / sqrt(3/5).
            r_obs_m = (self.box_size_m / 2.0) / np.sqrt(3.0 / 5.0)
            r_sim_m = r_obs_m * (center_m ** (1.0 / 3.0))

            # N_outer = round((centerM - 1) * N_inner * ceiling). At ceiling=1
            # this gives N_total = round(centerM * N_inner) (LINEAR, not cubic).
            n_outer = int(np.round((center_m - 1.0) * n_inner * self.outer_density_ceiling))
            print(f"[ParticleSystem] centerM={center_m:.4f}: N_inner={n_inner}, "
                  f"N_outer={n_outer}, N_total={n_inner + n_outer}; "
                  f"R_obs={r_obs_m/self.const.Gpc_to_m:.3f} Gpc, "
                  f"R_sim={r_sim_m/self.const.Gpc_to_m:.3f} Gpc "
                  f"(x{center_m**(1/3):.4f})")

            if n_outer > 0:
                # Draw outer positions using the SAME np.random stream (after inner draws).
                outer_positions_raw = self._init_outer_shell(r_obs_m, r_sim_m, n_outer)
                # Apply the SAME scale factor so inner and outer are in the same frame.
                outer_scaled = outer_positions_raw * scale_factor

                # All outer particles carry the SAME per-particle mean mass as inner.
                outer_masses_kg = np.full(n_outer, mean_mass_kg)

                # Concatenate [inner; outer]. Observable mask: True for inner indices.
                all_positions = np.concatenate([inner_centered, outer_scaled], axis=0)
                all_masses = np.concatenate([inner_masses_kg, outer_masses_kg])
                observable_mask = np.concatenate([
                    np.ones(n_inner, dtype=bool),
                    np.zeros(n_outer, dtype=bool),
                ])
            else:
                # n_outer rounded to 0 (centerM very close to 1.0 with small N_inner).
                all_positions = inner_centered
                all_masses = inner_masses_kg
                observable_mask = np.ones(n_inner, dtype=bool)
        else:
            # centerM == 1.0: no outer particles; all-True mask; byte-identical.
            all_positions = inner_centered
            all_masses = inner_masses_kg
            observable_mask = np.ones(n_inner, dtype=bool)

        # Store the observable mask on the instance (Section 2 consumes this).
        self.observable_mask = observable_mask
        n_total = len(all_positions)

        # Now generate velocities using CENTRED and NORMALISED positions.
        # This ensures velocity initialization is independent of sampling randomness.
        for i in range(n_total):
            pos = all_positions[i]

            # Initial velocity: Hubble flow + small peculiar velocity.
            v_hubble = H_start * pos
            v_peculiar = np.random.normal(0, 1e5, 3)  # ~100 km/s peculiar velocity
            vel = v_hubble + v_peculiar

            particle = Particle(pos, vel, all_masses[i], particle_id=i)
            self.particles.append(particle)

        # CRITICAL: Remove centre-of-mass velocity to prevent bulk motion.
        # With Hubble flow v = H*r, random particle positions create non-zero COM velocity.
        velocities = np.array([p.vel for p in self.particles])
        com_velocity = np.mean(velocities, axis=0)
        print(f"[ParticleSystem] Removing COM velocity: "
              f"[{com_velocity[0]:.3e}, {com_velocity[1]:.3e}, {com_velocity[2]:.3e}] m/s")

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

    def get_observable_mask(self) -> np.ndarray:
        """Return the observable mask (bool, N_total).

        True  = inner observable particle (indices 0 .. N_inner-1).
        False = outer shell particle (only present when centerM > 1.0).
        centerM=1.0 -> all-True mask of length n_particles.

        Section 2 (a(t) computation) uses this mask to restrict the RMS
        radius calculation to the inner observable sub-region only.
        """
        return self.observable_mask

    def set_accelerations(self, accelerations: np.ndarray) -> None:
        """Set accelerations for all particles."""
        for i, particle in enumerate(self.particles):
            particle.acc = accelerations[i]

    def set_velocities(self, velocities: np.ndarray) -> None:
        """Set velocities for all particles."""
        for i, particle in enumerate(self.particles):
            particle.vel = velocities[i]

    def set_positions(self, positions: np.ndarray) -> None:
        """Set positions for all particles."""
        for i, particle in enumerate(self.particles):
            particle.pos = positions[i]

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
        # Total particle count (inner + outer). n_particles is the inner (observable) count.
        return len(self.particles)

    def __repr__(self):
        n_total = len(self.particles)
        n_inner = self.n_particles
        if n_total == n_inner:
            return f"ParticleSystem(n={n_inner}, t={self.time:.2e}s)"
        return (f"ParticleSystem(n_total={n_total}, n_inner={n_inner}, "
                f"centerM={self.center_node_mass:.2f}, t={self.time:.2e}s)")


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
        Create HMEA node grid using the geometry factory.

        The base node positions come from build_node_positions(geometry, S, **kwargs)
        in cosmo/node_geometry.py.  Default geometry "cube26" produces a 3×3×3-1
        cubic lattice (26 nodes) byte-identical to the previous hard-coded loop.

        Per-node masses are drawn from ExternalNodeParameters.node_masses(n).
        When node_mass_amplitude == 0.0 (default), all masses equal M_ext_kg
        (byte-identical to the legacy uniform behavior).
        When node_mass_amplitude > 0.0, masses are log-normally distributed
        with mean M_ext_kg exactly (mean-preserving normalization), preserving
        Omega_Lambda_eff / growth-anchor / isotropic background.

        Per-node RADIAL position perturbation is drawn from
        ExternalNodeParameters.node_scale_factors(n). When node_s_amplitude == 0.0
        (default), all factors are 1.0 => the geometry's symmetric node positions
        (byte-identical to the legacy positions for cube26). When node_s_amplitude > 0.0,
        each node's DISTANCE from the origin is scaled by a mean-preserving
        log-normal factor (mean scale == S preserved) while its DIRECTION (ray)
        is held fixed, breaking the symmetry radially.

        Mass bookkeeping: total external mass = n_nodes * M_ext_kg.  The
        Omega_Lambda_eff formula uses M_ext_kg (per-node mean), so to keep
        Omega_Lambda_eff comparable across geometries the caller should rescale
        M_ext_kg via effective_M_ext_kg() from cosmo/node_geometry.py.
        """
        from .node_geometry import build_node_positions

        S = self.params.S
        geometry = getattr(self.params, 'node_geometry', 'cube26')
        geometry_kwargs = getattr(self.params, 'geometry_kwargs', {})

        # Build base positions via the factory (cube26 is byte-identical to old loop)
        base_positions_arr = build_node_positions(geometry, S, **geometry_kwargs)
        n = len(base_positions_arr)

        scale_factors = self.params.node_scale_factors(n)  # mean == 1.0; ones() when amp=0
        masses = self.params.node_masses(n)

        for node_id in range(n):
            pos = base_positions_arr[node_id] * scale_factors[node_id]
            node = {
                'id': node_id,
                'position': pos,
                'mass': masses[node_id],
            }
            self.nodes.append(node)

        # Update n_nodes to match actual geometry (callers may rely on len(grid.nodes))
        self.n_nodes = n
    
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
