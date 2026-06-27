"""
Cosmological Constants and Parameters
Handles both standard ΛCDM and External-Node Model parameters
"""

import numpy as np

class CosmologicalConstants:
    """Fundamental physical constants in SI units"""
    
    # Physical constants
    G = 6.67430e-11  # Gravitational constant [m^3 kg^-1 s^-2]
    c = 299792458.0  # Speed of light [m/s]
    
    # Unit conversions
    Mpc_to_m = 3.0857e22  # Megaparsec to meters
    Gpc_to_m = 3.0857e25  # Gigaparsec to meters
    km_to_m = 1000.0
    Gyr_to_s = 3.1536e16  # Gigayear to seconds
    
    # Solar mass
    M_sun = 1.989e30  # kg
    
    # Observable universe properties
    M_observable_kg = 1e53  # kg (approximate total mass)
    R_hubble = 4.4e26  # meters (~14 Gpc, current Hubble radius)
    

class LambdaCDMParameters:
    """Standard ΛCDM cosmological parameters"""
    
    def __init__(self):
        # Hubble constant
        self.H0_km_s_Mpc = 70.0  # km/s/Mpc
        self.H0_si = self.H0_km_s_Mpc * 1000 / CosmologicalConstants.Mpc_to_m  # s^-1
        
        # Density parameters (present day)
        self.Omega_Lambda = 0.7  # Dark energy
        self.Omega_m = 0.3  # Matter (dark + baryonic)
        self.Omega_r = 9e-5  # Radiation (negligible today)
        
        # Age of universe
        self.t_universe = 13.8e9  # years
        self.t_universe_s = self.t_universe * 365.25 * 24 * 3600  # seconds
        
    def H_at_time(self, a: float) -> float:
        """Calculate ΛCDM Hubble parameter: H(a) = H₀ √(Ω_m a⁻³ + Ω_Λ)"""
        import numpy as np
        return self.H0_si * np.sqrt(self.Omega_m / a**3 + self.Omega_Lambda)

    def H_matter_only(self, a: float) -> float:
        """Calculate matter-only Hubble parameter: H(a) = H₀ √(Ω_m a⁻³) [no dark energy]"""
        import numpy as np
        return self.H0_si * np.sqrt(self.Omega_m / a**3)

    @staticmethod
    def H_eds_at_time(t_start_Gyr: float) -> float:
        """Einstein-de Sitter Hubble parameter at age t, in s^-1.

        For a flat matter-dominated (Ω_m = 1) universe a(t) ∝ t^(2/3), so
        H(t) = (da/dt)/a = 2 / (3 t). This is the ONLY initial Hubble rate that
        is mutually consistent with the EdS critical density below; together they
        make a finite uniform sphere reproduce the EdS scale factor EXACTLY
        (standard Newtonian cosmology). Using the absolute age t (not the
        Ω_m=0.3 H_matter_only) is what ties the sim's a(t) to the analytic
        einstein_de_sitter null used in the Pantheon comparison.
        """
        return (2.0 / 3.0) / (t_start_Gyr * CosmologicalConstants.Gyr_to_s)

    @staticmethod
    def eds_critical_density(H_si: float) -> float:
        """EdS critical (background) density ρ_crit = 3 H² / (8 π G), Ω_m = 1.

        At Ω_m = 1 the matter density equals the critical density, so a comoving
        patch carrying this density supplies exactly the self-gravity needed to
        decelerate the Hubble flow onto the EdS solution.
        """
        import numpy as np
        return 3.0 * H_si ** 2 / (8.0 * np.pi * CosmologicalConstants.G)

    def __str__(self):
        return (f"ΛCDM Parameters:\n"
                f"  H0 = {self.H0_km_s_Mpc} km/s/Mpc\n"
                f"  Ω_Λ = {self.Omega_Lambda}\n"
                f"  Ω_m = {self.Omega_m}\n"
                f"  Age = {self.t_universe} Gyr")


class ExternalNodeParameters:
    """External-Node Model parameters from the paper"""

    def __init__(self, M_ext_kg: float = None, S: float = None,
                 node_mass_seed: int = 0, node_mass_amplitude: float = 0.0,
                 node_s_amplitude: float = 0.0,
                 node_geometry: str = "cube26",
                 geometry_kwargs: dict = None,
                 vir_n_nodes: int = 26, vir_extent: float = 1.0,
                 vir_mass_rule: str = "radial", vir_mass_spread: float = 0.0,
                 vir_segregation: float = 1.0, vir_s_metric: str = "median",
                 vir_relax_steps: int = 1):
        """Initialize External-Node parameters (M_ext_kg in kg, S in meters).

        Args:
            M_ext_kg: External node mass in kg (mean mass per node).
            S: Node separation in meters.
            node_mass_seed: RNG seed for per-node mass distribution (default 0).
                Also seeds the per-node POSITION perturbation (node_s_amplitude)
                so a single seed selects a coherent orientation for both knobs.
            node_mass_amplitude: Log-normal width of per-node mass distribution.
                0.0 (default) => all 26 nodes have identical mass M_ext_kg (backward compatible).
            node_s_amplitude: Log-normal width of the per-node RADIAL position
                perturbation. 0.0 (default) => all 26 nodes sit on the perfect
                symmetric lattice (backward compatible, byte-identical). >0 scales
                each node's distance from the origin by a mean-preserving factor
                S_i = S * exp(node_s_amplitude * g_i) / <exp(...)>, keeping each
                node on its original ray (direction unchanged) and preserving the
                MEAN radial scale exactly, so it isolates symmetry-breaking from a
                net S change.
            node_geometry: Geometry identifier for the HMEA node layout
                (default "cube26" = current 3×3×3-1 lattice, backward-compatible).
                "virialized" selects the COUPLED (positions, masses) mass-segregated
                generator (see cosmo/node_geometry.build_virialized_grid); the vir_*
                params below are consumed ONLY then (ignored otherwise), and for
                "virialized" node_mass_amplitude is IGNORED (vir_mass_spread owns the
                node-mass distribution).
            geometry_kwargs: Optional dict forwarded to the geometry factory.
                Default None (uses each geometry's built-in defaults). NOT used by
                "virialized" (it reads the vir_* fields instead).
            vir_n_nodes: Virialized node count (default 26, parity with cube26).
            vir_extent: Virialized radius multiplier; raw outer radius ~ vir_extent*S
                before the NN-spacing rescale (default 1.0).
            vir_mass_rule: "radial" (deterministic mass ~ f(r), default) or "massfunc"
                (log-normal mass-function draw + spatial segregation).
            vir_mass_spread: Amplitude of the node-mass distribution about the mean.
                0.0 (default) -> uniform masses (THE falsifiable knob).
            vir_segregation: Mass<->radius coupling strength (default 1.0).
                0.0 -> mass/radius decoupled (no segregation).
            vir_s_metric: NN-spacing definition the generator targets:
                "median" (default) or "mean".
            vir_relax_steps: Virialized BALANCE LEVEL (default 1). 0 -> realistic
                Fibonacci layout (NOT force-balanced); >= 1 -> force-balanced
                cubic-lattice ball (inner nodes feel ~zero net force). See
                cosmo.node_geometry.build_virialized_grid.
            Note: the virialized RNG reuses node_mass_seed (one-seed coherence,
                like node_s_amplitude); there is no separate vir_seed field.
        """
        # Default values - S is tuned to give Ω_Λ_eff ≈ 0.7 with M_ext_kg = 5e55
        self.M_ext_kg = M_ext_kg if M_ext_kg is not None else 5e55  # kg
        self.S = S if S is not None else 31.6 * CosmologicalConstants.Gpc_to_m  # meters
        self.node_mass_seed = node_mass_seed
        self.node_mass_amplitude = node_mass_amplitude
        self.node_s_amplitude = node_s_amplitude
        self.node_geometry = node_geometry
        self.geometry_kwargs = geometry_kwargs if geometry_kwargs is not None else {}
        # Virialized-grid params (consumed only when node_geometry == "virialized").
        self.vir_n_nodes = vir_n_nodes
        self.vir_extent = vir_extent
        self.vir_mass_rule = vir_mass_rule
        self.vir_mass_spread = vir_mass_spread
        self.vir_segregation = vir_segregation
        self.vir_s_metric = vir_s_metric
        self.vir_relax_steps = vir_relax_steps

        # Calculate derived parameters
        self._calculate_derived()

    def build_virialized(self) -> tuple:
        """Return COUPLED (positions, masses) for the virialized geometry.

        Thin adapter that forwards this object's vir_* fields (and M_ext_kg / S /
        node_mass_seed) to cosmo.node_geometry.build_virialized_grid. Only valid
        when node_geometry == "virialized"; HMEAGrid._create_grid calls this on the
        coupled branch.
        """
        from .node_geometry import build_virialized_grid
        return build_virialized_grid(
            self.S,
            n_nodes=self.vir_n_nodes,
            M_ext_kg=self.M_ext_kg,
            vir_extent=self.vir_extent,
            vir_mass_rule=self.vir_mass_rule,
            vir_mass_spread=self.vir_mass_spread,
            vir_segregation=self.vir_segregation,
            vir_s_metric=self.vir_s_metric,
            vir_relax_steps=self.vir_relax_steps,
            seed=self.node_mass_seed,
        )

    def _calculate_derived(self) -> None:
        """Calculate derived quantities."""
        const = CosmologicalConstants()

        # Effective dark energy from tidal acceleration
        # From paper: H0^2 * Omega_Lambda ≈ G*M_ext_kg/S^3
        self.Omega_Lambda_eff = (const.G * self.M_ext_kg) / (self.S**3 * (70*1000/const.Mpc_to_m)**2)

        # Schwarzschild radius of HMEA (for reference)
        self.R_schwarzschild = 2 * const.G * self.M_ext_kg / const.c**2

        # Grid spacing in Gpc
        self.S_Gpc = self.S / const.Gpc_to_m

        # Mass ratio to observable universe
        self.M_ratio = self.M_ext_kg / const.M_observable_kg
        
    def node_masses(self, n_nodes: int = 26) -> np.ndarray:
        """Return per-node mass array of length n_nodes.

        INVARIANTS:
        - (a) Deterministic & reproducible: identical output for the same
          (node_mass_seed, node_mass_amplitude) regardless of global RNG state.
          Uses np.random.default_rng(seed) — independent of particle/simulation RNG.
        - (b) All masses strictly positive: guaranteed by exp() > 0.
        - (c) MEAN-PRESERVING: mean(m_i) == M_ext_kg EXACTLY (within float precision).
          The / w.mean() step enforces this, keeping total external mass /
          Omega_Lambda_eff / growth-anchor / never-exceed-LCDM background fixed.
          The seed selects shear/dipole ORIENTATION only, not the isotropic background.

        When node_mass_amplitude == 0.0 (default): returns np.full(n_nodes, M_ext_kg),
        which is byte-identical to the legacy uniform behavior.
        """
        if self.node_mass_amplitude == 0.0:
            return np.full(n_nodes, self.M_ext_kg)
        rng = np.random.default_rng(self.node_mass_seed)
        g = rng.standard_normal(n_nodes)
        w = np.exp(self.node_mass_amplitude * g)
        return self.M_ext_kg * w / w.mean()

    def node_scale_factors(self, n_nodes: int = 26) -> np.ndarray:
        """Return per-node RADIAL scale factors (length n_nodes), mean == 1.0.

        Multiplies each lattice node's position by its factor, scaling the node's
        DISTANCE from the origin while keeping it on its original ray (direction
        unchanged). This breaks the lattice symmetry RADIALLY without changing the
        net scale S or any node direction.

        INVARIANTS (mirror node_masses):
        - (a) Deterministic & reproducible: identical output for the same
          (node_mass_seed, node_s_amplitude). Uses np.random.default_rng(seed),
          independent of the particle/simulation RNG. A SEPARATE rng draw from
          node_masses() (different call), so the two knobs do not entangle.
        - (b) All factors strictly positive: guaranteed by exp() > 0, so no node
          can cross the origin or flip sides.
        - (c) MEAN-PRESERVING: mean(factor_i) == 1.0 EXACTLY (within float
          precision). The / w.mean() step enforces this, keeping the MEAN radial
          scale (and hence the symmetric-lattice average geometry) fixed; the
          seed selects the shear/dipole ORIENTATION only.

        When node_s_amplitude == 0.0 (default): returns np.ones(n_nodes), so node
        positions are byte-identical to the symmetric lattice (backward compatible).
        """
        if self.node_s_amplitude == 0.0:
            return np.ones(n_nodes)
        rng = np.random.default_rng(self.node_mass_seed)
        g = rng.standard_normal(n_nodes)
        w = np.exp(self.node_s_amplitude * g)
        return w / w.mean()

    def set_grid_spacing(self, S_Gpc: float) -> None:
        """Set grid spacing in Gigaparsecs."""
        self.S = S_Gpc * CosmologicalConstants.Gpc_to_m
        self._calculate_derived()
        
    def set_node_mass(self, M_ext_kg: float) -> None:
        """Set HMEA mass in kg."""
        self.M_ext_kg = M_ext_kg
        self._calculate_derived()
        
    def calculate_required_spacing(self, Omega_Lambda_target: float = 0.7, H0: float = 70) -> float:
        """Calculate required grid spacing: S ≈ (G*M_ext_kg / (H0² * Ω_Λ))^(1/3)"""
        const = CosmologicalConstants()
        H0_si = H0 * 1000 / const.Mpc_to_m  # Convert to s^-1

        S = (const.G * self.M_ext_kg / (H0_si**2 * Omega_Lambda_target))**(1/3)
        return S
    
    def __str__(self):
        return (f"External-Node Parameters:\n"
                f"  M_ext_kg = {self.M_ext_kg:.2e} kg ({self.M_ratio:.0f} × M_obs)\n"
                f"  S = {self.S_Gpc:.1f} Gpc\n"
                f"  Ω_Λ_eff = {self.Omega_Lambda_eff:.3f}\n"
                f"  R_Schwarzschild = {self.R_schwarzschild/CosmologicalConstants.Gpc_to_m:.2e} Gpc")


class SimulationParameters:
    """Parameters for running cosmological simulations"""

    # Maximum allowed outer_density_ceiling (>2 risks over-dense tidal environment).
    MAX_OUTER_DENSITY_CEILING: float = 2.0

    def __init__(self, M_value: float = 800, S_value: float = 24.0, n_particles: int = 300, seed: int = 42,
                 t_start_Gyr: float = 10.8, t_duration_Gyr: float = 6.0, n_steps: int = 150,
                 damping_factor: float = None, center_node_mass: float = 1.0,
                 outer_density_ceiling: float = 1.0,
                 mass_randomize: float = 0.5,
                 node_mass_seed: int = 0, node_mass_amplitude: float = 0.0,
                 node_s_amplitude: float = 0.0,
                 init_distribution: str = "uniform_sphere",
                 init_kwargs: dict = None,
                 eds_consistent: bool = True,
                 pre_start_tidal_boost: bool = True,
                 node_geometry: str = "cube26",
                 geometry_kwargs: dict = None,
                 vir_n_nodes: int = 26, vir_extent: float = 1.0,
                 vir_mass_rule: str = "radial", vir_mass_spread: float = 0.0,
                 vir_segregation: float = 1.0, vir_s_metric: str = "median",
                 vir_relax_steps: int = 1):
        """
        Initialize simulation parameters.

        Args:
            M_value: External node mass as multiple of M_observable
            S_value: Node separation in Gpc
            n_particles: Number of particles
            seed: Random seed
            t_start_Gyr: Start time in Gyr
            t_duration_Gyr: Duration in Gyr
            n_steps: Number of timesteps
            damping_factor: Initial velocity damping (None=auto)
            center_node_mass: Outer-mass multiplier: total simulated mass /
                              inner observable mass (>= 1.0). Default 1.0 =
                              observable sphere only (backward-compatible,
                              byte-identical). >1.0 adds extra Big-Bang matter
                              OUTSIDE the observable sphere at the same
                              EdS-critical density and same per-particle mass;
                              N grows LINEARLY (centerM=2 -> 2x particles),
                              R_sim = R_obs*centerM**(1/3). The inner R_obs
                              stays the OBSERVABLE region used for a(t)/mu(z).
                              Under eds_consistent the inner mass is the EdS
                              critical mass regardless of centerM; centerM ONLY
                              adds OUTER particles. NOTE: softening no longer
                              scales with centerM (frozen at the centerM=1
                              baseline, 1.0 Gpc).
            mass_randomize: Particle mass randomization (0.0=equal masses,
                           1.0=masses from 0 to 2x mean). Default 0.5.
            node_mass_seed: RNG seed for per-node mass distribution (default 0).
                            Independent of particle/simulation RNG.
            node_mass_amplitude: Log-normal width of per-node mass distribution.
                                 0.0 (default) => all 26 nodes have identical mass
                                 M_ext_kg (backward compatible, byte-identical).
            node_s_amplitude: Log-normal width of the per-node RADIAL position
                              perturbation (analogous to node_mass_amplitude, but
                              for node POSITIONS). 0.0 (default) => perfect
                              symmetric lattice (backward compatible,
                              byte-identical). >0 scales each node's distance from
                              the origin by a mean-preserving factor (mean scale
                              == S preserved), breaking lattice symmetry radially.
                              Reuses node_mass_seed for determinism.
            init_distribution: Particle position sampler.
                               "uniform_sphere" (default) — current behaviour,
                               backward-compatible with all existing tests.
                               "grf" — Gaussian random field + Zel'dovich displacement
                               shaped by approximate LCDM P(k) (BBKS transfer function).
            init_kwargs: Optional dict of keyword arguments forwarded to the sampler.
                         Supported for "grf": Ng (int, default 64), n_s, Omega_m, h.
                         Ignored for "uniform_sphere".
            eds_consistent: When True (default) and dark energy is OFF, the cloud
                            uses self-consistent Einstein-de Sitter initial
                            conditions (Hubble flow v=H_EdS*r with H_EdS=2/(3 t_start)
                            AND cloud mass = EdS critical mass). This makes the
                            matter-only (M_ext=0) sim reproduce the analytic EdS
                            expansion by construction, replacing the old
                            velocity-calibration fudge. Ignored for LCDM runs
                            (use_dark_energy=True). Set False to restore the legacy
                            calibrated-velocity behaviour.
            pre_start_tidal_boost: When True (default) AND external nodes are
                            active AND eds_consistent, the cloud's initial radial
                            velocities are boosted at t_start by the velocity the
                            HMEA tidal field would have imparted over the
                            pre-t_start history (Big Bang -> t_start). The cloud
                            should ARRIVE at t_start moving slightly FASTER than
                            pure EdS Hubble flow because the nodes have been pulling
                            on it for billions of years. Derived from the SAME node
                            sum the sim uses (see CosmologicalSimulation
                            ._apply_pre_start_tidal_boost); it scales with M_ext so
                            it VANISHES as M_ext -> 0, preserving M=0 == EdS exactly.
                            This is NOT a fit-to-LCDM knob. Set False to start from
                            pure EdS Hubble flow with no pre-history boost.
            outer_density_ceiling: Multiplier on the inner EdS-critical density
                              for outer particles (default 1.0 = exactly critical,
                              same density as inner). Values > 1.0 raise the outer
                              number density above critical. Capped at
                              MAX_OUTER_DENSITY_CEILING = 2.0 (enforce/clip);
                              higher values re-introduce over-dense tidal
                              environments and are not physically motivated.
                              Has no effect when centerM == 1.0.
            node_geometry:  Geometry identifier for the HMEA node layout (default
                            "cube26" = current 3×3×3-1 lattice, backward-compatible).
                            Other choices: "cube_dense", "fcc", "bcc" (all
                            volume-filling / virialized). See cosmo/node_geometry.py.
            geometry_kwargs: Optional dict of keyword arguments forwarded to the
                            geometry factory (e.g. n_per_side=7 for "cube_dense").
                            Default None (uses each geometry's own defaults).
            vir_n_nodes / vir_extent / vir_mass_rule / vir_mass_spread /
            vir_segregation / vir_s_metric / vir_relax_steps: parameters of the
                            "virialized" COUPLED (positions, masses) mass-segregated
                            grid, consumed ONLY when node_geometry == "virialized" (see
                            ExternalNodeParameters / node_geometry.build_virialized_grid).
                            Defaults: 26, 1.0, "radial", 0.0, 1.0, "median", 1.
                            vir_relax_steps is a BALANCE LEVEL: 0 -> realistic (not
                            force-balanced) Fibonacci layout; >= 1 (default) ->
                            force-balanced cubic-lattice ball (inner nodes ~ zero net
                            force). The virialized RNG reuses node_mass_seed.
        """
        self.M_value = M_value
        self.S_value = S_value
        self.n_particles = n_particles
        self.seed = seed
        self.t_start_Gyr = t_start_Gyr
        self.t_duration_Gyr = t_duration_Gyr
        self.n_steps = n_steps
        self.damping_factor = damping_factor
        # centerM >= 1.0: values below 1.0 are a misconfiguration; clip to 1.0.
        self.center_node_mass = max(1.0, float(center_node_mass))
        # outer_density_ceiling: clip to [0, MAX_OUTER_DENSITY_CEILING].
        _max_ceil = SimulationParameters.MAX_OUTER_DENSITY_CEILING
        if outer_density_ceiling > _max_ceil:
            import warnings
            warnings.warn(
                f"outer_density_ceiling={outer_density_ceiling} exceeds maximum "
                f"{_max_ceil}; clipping to {_max_ceil}. Higher values risk an "
                "over-dense outer tidal environment and are not physically motivated.",
                UserWarning,
                stacklevel=2,
            )
        self.outer_density_ceiling = float(np.clip(outer_density_ceiling, 0.0, _max_ceil))
        self.mass_randomize = mass_randomize
        self.node_mass_seed = node_mass_seed
        self.node_mass_amplitude = node_mass_amplitude
        self.node_s_amplitude = node_s_amplitude
        self.init_distribution = init_distribution
        self.init_kwargs = init_kwargs if init_kwargs is not None else {}
        self.eds_consistent = eds_consistent
        self.pre_start_tidal_boost = pre_start_tidal_boost
        self.node_geometry = node_geometry
        self.geometry_kwargs = geometry_kwargs if geometry_kwargs is not None else {}
        # Virialized-grid params (consumed only when node_geometry == "virialized").
        self.vir_n_nodes = vir_n_nodes
        self.vir_extent = vir_extent
        self.vir_mass_rule = vir_mass_rule
        self.vir_mass_spread = vir_mass_spread
        self.vir_segregation = vir_segregation
        self.vir_s_metric = vir_s_metric
        self.vir_relax_steps = vir_relax_steps

        # Calculate derived quantities
        self._calculate_derived()

    def _calculate_derived(self) -> None:
        """Calculate derived quantities."""
        const = CosmologicalConstants()

        # Convert to physical units
        self.M_ext_kg = self.M_value * const.M_observable_kg
        self.S = self.S_value * const.Gpc_to_m

        # Calculate end time
        self.t_end_Gyr = self.t_start_Gyr + self.t_duration_Gyr

        # center_node_mass_kg: legacy property kept for backward compat.
        # Under eds_consistent this value is NOT used to drive inner cloud mass
        # (ParticleSystem overrides to EdS critical). It is still read on the
        # legacy non-EdS path (CosmologicalSimulation.__init__ total_mass_kg).
        self.center_node_mass_kg = self.center_node_mass * const.M_observable_kg

        # Create external node parameters for this configuration
        self.external_params = ExternalNodeParameters(
            M_ext_kg=self.M_ext_kg,
            S=self.S,
            node_mass_seed=self.node_mass_seed,
            node_mass_amplitude=self.node_mass_amplitude,
            node_s_amplitude=self.node_s_amplitude,
            node_geometry=self.node_geometry,
            geometry_kwargs=self.geometry_kwargs,
            vir_n_nodes=self.vir_n_nodes,
            vir_extent=self.vir_extent,
            vir_mass_rule=self.vir_mass_rule,
            vir_mass_spread=self.vir_mass_spread,
            vir_segregation=self.vir_segregation,
            vir_s_metric=self.vir_s_metric,
            vir_relax_steps=self.vir_relax_steps,
        )

    def __str__(self):
        return (f"Simulation Parameters:\n"
                f"  M = {self.M_value} × M_obs\n"
                f"  S = {self.S_value} Gpc\n"
                f"  centerM = {self.center_node_mass} (outer-mass multiplier; 1.0=obs only)\n"
                f"  outer_density_ceiling = {self.outer_density_ceiling}\n"
                f"  Particles = {self.n_particles}\n"
                f"  Seed = {self.seed}\n"
                f"  Time = {self.t_start_Gyr} → {self.t_end_Gyr} Gyr ({self.t_duration_Gyr} Gyr)\n"
                f"  Steps = {self.n_steps}\n"
                f"  Ω_Λ_eff = {self.external_params.Omega_Lambda_eff:.3f}")
