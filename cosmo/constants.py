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
        """Calculate Hubble parameter: H(a) = H₀ √(Ω_m a⁻³ + Ω_Λ)"""
        import numpy as np
        return self.H0_si * np.sqrt(self.Omega_m / a**3 + self.Omega_Lambda)

    def __str__(self):
        return (f"ΛCDM Parameters:\n"
                f"  H0 = {self.H0_km_s_Mpc} km/s/Mpc\n"
                f"  Ω_Λ = {self.Omega_Lambda}\n"
                f"  Ω_m = {self.Omega_m}\n"
                f"  Age = {self.t_universe} Gyr")


class ExternalNodeParameters:
    """External-Node Model parameters from the paper"""
    
    def __init__(self, M_ext_kg: float = None, S: float = None):
        """Initialize External-Node parameters (M_ext_kg in kg, S in meters)."""
        # Default values - S is tuned to give Ω_Λ_eff ≈ 0.7 with M_ext_kg = 5e55
        self.M_ext_kg = M_ext_kg if M_ext_kg is not None else 5e55  # kg
        self.S = S if S is not None else 31.6 * CosmologicalConstants.Gpc_to_m  # meters
        
        # Calculate derived parameters
        self._calculate_derived()
        
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

    def __init__(self, M_value: float = 800, S_value: float = 24.0, n_particles: int = 300, seed: int = 42,
                 t_start_Gyr: float = 10.8, t_duration_Gyr: float = 6.0, n_steps: int = 150,
                 damping_factor: float = None, center_node_mass: float = 1.0,
                 mass_randomize: float = 0.5):
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
            center_node_mass: Central node mass as multiple of M_observable.
                              Default 1.0 = 1 x M_observable_kg.
                              Affects total_mass_kg and softening scaling.
            mass_randomize: Particle mass randomization (0.0=equal masses,
                           1.0=masses from 0 to 2x mean). Default 0.5.
        """
        self.M_value = M_value
        self.S_value = S_value
        self.n_particles = n_particles
        self.seed = seed
        self.t_start_Gyr = t_start_Gyr
        self.t_duration_Gyr = t_duration_Gyr
        self.n_steps = n_steps
        self.damping_factor = damping_factor
        self.center_node_mass = center_node_mass
        self.mass_randomize = mass_randomize

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

        # Calculate center node mass in kg
        self.center_node_mass_kg = self.center_node_mass * const.M_observable_kg

        # Create external node parameters for this configuration
        self.external_params = ExternalNodeParameters(M_ext_kg=self.M_ext_kg, S=self.S)

    def __str__(self):
        return (f"Simulation Parameters:\n"
                f"  M = {self.M_value} × M_obs\n"
                f"  S = {self.S_value} Gpc\n"
                f"  Center Node Mass = {self.center_node_mass} × M_obs\n"
                f"  Particles = {self.n_particles}\n"
                f"  Seed = {self.seed}\n"
                f"  Time = {self.t_start_Gyr} → {self.t_end_Gyr} Gyr ({self.t_duration_Gyr} Gyr)\n"
                f"  Steps = {self.n_steps}\n"
                f"  Ω_Λ_eff = {self.external_params.Omega_Lambda_eff:.3f}")
