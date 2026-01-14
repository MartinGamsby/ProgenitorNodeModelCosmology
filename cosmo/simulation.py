"""
Main Simulation Runner
Compares ΛCDM cosmology with External-Node Model
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle
import os

from .constants import (CosmologicalConstants, LambdaCDMParameters, 
                             ExternalNodeParameters, SimulationParameters)
from .particles import ParticleSystem, HMEAGrid
from .integrator import LeapfrogIntegrator


class CosmologicalSimulation:
    """Main class for running cosmological simulations"""
    
    def __init__(self, n_particles=1000, box_size_Gpc=20.0,
                 use_external_nodes=True, external_node_params=None,
                 t_start_Gyr=10.8, a_start=None):
        """
        Initialize simulation

        Parameters:
        -----------
        n_particles : int
            Number of tracer particles (galaxy clusters)
        box_size_Gpc : float
            Size of simulation box [Gpc]
        use_external_nodes : bool
            True = External-Node Model, False = ΛCDM
        external_node_params : ExternalNodeParameters, optional
            Parameters for HMEA nodes
        t_start_Gyr : float
            Simulation start time since Big Bang (in Gyr)
        a_start : float, optional
            Scale factor at start time (a=1 at present day)
        """
        self.const = CosmologicalConstants()
        self.use_external_nodes = use_external_nodes
        self.t_start_Gyr = t_start_Gyr
        self.a_start = a_start if a_start is not None else 1.0

        # Convert box size to meters
        box_size = box_size_Gpc * self.const.Gpc_to_m

        # Initialize particle system
        print(f"Initializing {n_particles} particles in {box_size_Gpc} Gpc box...")
        self.particles = ParticleSystem(n_particles=n_particles,
                                       box_size=box_size,
                                       total_mass=self.const.M_observable,
                                       a_start=self.a_start)
        
        # Initialize HMEA grid if using External-Node Model
        self.hmea_grid = None
        if use_external_nodes:
            self.hmea_grid = HMEAGrid(node_params=external_node_params, n_nodes=8)
            print(f"External-Node Model: {self.hmea_grid}")
        else:
            print("Running standard matter-only (no dark energy)")
        
        # Create integrator
        softening = 1.0 * self.const.Mpc_to_m  # 1 Mpc softening
        self.integrator = LeapfrogIntegrator(
            self.particles, 
            self.hmea_grid, 
            softening=softening,
            use_external_nodes=use_external_nodes,
            use_dark_energy=(not use_external_nodes)  # ΛCDM uses dark energy, External-Node doesn't
        )
        
        # Simulation results
        self.snapshots = []
        self.expansion_history = []
        
    def run(self, t_end_Gyr=13.8, n_steps=1000, save_interval=10):
        """
        Run the simulation
        
        Parameters:
        -----------
        t_end_Gyr : float
            End time in Gigayears
        n_steps : int
            Number of timesteps
        save_interval : int
            Save snapshot every N steps
            
        Returns:
        --------
        snapshots : list
            Simulation snapshots
        """
        # Convert to seconds
        t_end = t_end_Gyr * 1e9 * 365.25 * 24 * 3600
        
        print("\n" + "="*60)
        print("RUNNING COSMOLOGICAL SIMULATION")
        print("="*60)
        print(f"Model: {'External-Node' if self.use_external_nodes else 'Matter-only'}")
        print(f"Duration: {t_end_Gyr} Gyr")
        print(f"Timesteps: {n_steps}")
        print("="*60 + "\n")
        
        # Run integration
        self.snapshots = self.integrator.evolve(t_end, n_steps, save_interval)
        
        # Calculate expansion history
        self._calculate_expansion_history()
        
        print("\nSimulation complete!")
        return self.snapshots
    
    def _calculate_expansion_history(self):
        """Calculate the scale factor a(t) from snapshots"""
        self.expansion_history = []
        
        initial_size = self._calculate_system_size(self.snapshots[0])
        
        for snapshot in self.snapshots:
            t = snapshot['time']
            current_size = self._calculate_system_size(snapshot)
            
            # Scale factor a(t) = R(t) / R(t=0)
            a = current_size / initial_size
            
            self.expansion_history.append({
                'time': t,
                'time_Gyr': t / (1e9 * 365.25 * 24 * 3600),
                'scale_factor': a,
                'size': current_size,
            })
    
    def _calculate_system_size(self, snapshot):
        """Calculate characteristic size of system (RMS radius)"""
        positions = snapshot['positions']
        
        # Center of mass
        com = np.mean(positions, axis=0)
        
        # RMS distance from center
        r = np.linalg.norm(positions - com, axis=1)
        rms_radius = np.sqrt(np.mean(r**2))
        
        return rms_radius
    
    def save(self, filename):
        """Save simulation results"""
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
    def load(filename):
        """Load simulation results"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data


def compare_models(n_particles=500, box_size_Gpc=20.0, t_end_Gyr=13.8, n_steps=500, output_dir="./results"):
    """
    Run both ΛCDM and External-Node Model and compare results
    
    Parameters:
    -----------
    n_particles : int
        Number of particles
    box_size_Gpc : float
        Simulation box size [Gpc]
    t_end_Gyr : float
        Simulation duration [Gyr]
    n_steps : int
        Number of timesteps
    """
    print("\n" + "="*70)
    print("COMPARING ΛCDM vs EXTERNAL-NODE MODEL")
    print("="*70)
    
    # Run ΛCDM simulation
    print("\n### Running ΛCDM Simulation ###\n")
    sim_lcdm = CosmologicalSimulation(n_particles=n_particles,
                                      box_size_Gpc=box_size_Gpc,
                                      use_external_nodes=False)
    sim_lcdm.run(t_end_Gyr=t_end_Gyr, n_steps=n_steps, save_interval=10)
    
    # Run External-Node simulation
    print("\n### Running External-Node Model Simulation ###\n")
    sim_external = CosmologicalSimulation(n_particles=n_particles,
                                          box_size_Gpc=box_size_Gpc,
                                          use_external_nodes=True)
    sim_external.run(t_end_Gyr=t_end_Gyr, n_steps=n_steps, save_interval=10)
    
    # Save results
    sim_lcdm.save(os.path.join(output_dir, 'simulation_lcdm.pkl'))
    sim_external.save(os.path.join(output_dir, 'simulation_external.pkl'))
    
    return sim_lcdm, sim_external


def plot_expansion_comparison(sim_lcdm, sim_external, save_path='expansion_comparison.png'):
    """
    Plot expansion history comparison
    
    Parameters:
    -----------
    sim_lcdm : CosmologicalSimulation
        ΛCDM simulation
    sim_external : CosmologicalSimulation
        External-Node simulation
    save_path : str
        Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('ΛCDM vs External-Node Model Comparison', fontsize=16, fontweight='bold')
    
    # Extract data
    lcdm_hist = sim_lcdm.expansion_history
    ext_hist = sim_external.expansion_history
    
    t_lcdm = np.array([h['time_Gyr'] for h in lcdm_hist])
    a_lcdm = np.array([h['scale_factor'] for h in lcdm_hist])
    
    t_ext = np.array([h['time_Gyr'] for h in ext_hist])
    a_ext = np.array([h['scale_factor'] for h in ext_hist])
    
    # Plot 1: Scale factor evolution
    ax1 = axes[0, 0]
    ax1.plot(t_lcdm, a_lcdm, 'b-', linewidth=2, label='ΛCDM', alpha=0.7)
    ax1.plot(t_ext, a_ext, 'r--', linewidth=2, label='External-Node', alpha=0.7)
    ax1.set_xlabel('Time [Gyr]', fontsize=12)
    ax1.set_ylabel('Scale Factor a(t)', fontsize=12)
    ax1.set_title('Cosmic Expansion History', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Hubble parameter H(t) = (da/dt)/a
    ax2 = axes[0, 1]
    
    # SMOOTH the data first to reduce noise from finite particles
    from scipy.ndimage import gaussian_filter1d
    
    # Apply Gaussian smoothing (sigma=2 for gentle smoothing)
    a_lcdm_smooth = gaussian_filter1d(a_lcdm, sigma=2)
    a_ext_smooth = gaussian_filter1d(a_ext, sigma=2)
    
    # Numerical derivative on SMOOTHED data
    H_lcdm = np.gradient(a_lcdm_smooth, t_lcdm * 1e9 * 365.25 * 24 * 3600) / a_lcdm_smooth
    H_ext = np.gradient(a_ext_smooth, t_ext * 1e9 * 365.25 * 24 * 3600) / a_ext_smooth
    
    # Convert to km/s/Mpc
    const = CosmologicalConstants()
    H_lcdm_hubble = H_lcdm * const.Mpc_to_m / 1000
    H_ext_hubble = H_ext * const.Mpc_to_m / 1000
    
    ax2.plot(t_lcdm, H_lcdm_hubble, 'b-', linewidth=2, label='ΛCDM', alpha=0.7)
    ax2.plot(t_ext, H_ext_hubble, 'r--', linewidth=2, label='External-Node', alpha=0.7)
    ax2.axhline(70, color='gray', linestyle=':', label='H₀ = 70 km/s/Mpc')
    ax2.set_xlabel('Time [Gyr]', fontsize=12)
    ax2.set_ylabel('Hubble Parameter [km/s/Mpc]', fontsize=12)
    ax2.set_title('Expansion Rate Evolution', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    # Plot 3: Acceleration (d²a/dt²)
    ax3 = axes[1, 0]
    
    # Second derivative on SMOOTHED data
    acc_lcdm = np.gradient(np.gradient(a_lcdm_smooth, t_lcdm), t_lcdm)
    acc_ext = np.gradient(np.gradient(a_ext_smooth, t_ext), t_ext)
    
    ax3.plot(t_lcdm, acc_lcdm, 'b-', linewidth=2, label='ΛCDM', alpha=0.7)
    ax3.plot(t_ext, acc_ext, 'r--', linewidth=2, label='External-Node', alpha=0.7)
    ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_xlabel('Time [Gyr]', fontsize=12)
    ax3.set_ylabel('Acceleration d²a/dt² [Gyr⁻²]', fontsize=12)
    ax3.set_title('Cosmic Acceleration', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Energy conservation
    ax4 = axes[1, 1]
    
    if len(sim_lcdm.integrator.energy_history) > 0 and len(sim_external.integrator.energy_history) > 0:
        E_lcdm = np.array(sim_lcdm.integrator.energy_history)
        E_ext = np.array(sim_external.integrator.energy_history)
        
        # Fractional energy change
        dE_lcdm = (E_lcdm - E_lcdm[0]) / abs(E_lcdm[0])
        dE_ext = (E_ext - E_ext[0]) / abs(E_ext[0])
        
        t_energy_lcdm = np.array(sim_lcdm.integrator.time_history) / (1e9 * 365.25 * 24 * 3600)
        t_energy_ext = np.array(sim_external.integrator.time_history) / (1e9 * 365.25 * 24 * 3600)
        
        ax4.plot(t_energy_lcdm, dE_lcdm, 'b-', linewidth=2, label='ΛCDM', alpha=0.7)
        ax4.plot(t_energy_ext, dE_ext, 'r--', linewidth=2, label='External-Node', alpha=0.7)
    
    ax4.set_xlabel('Time [Gyr]', fontsize=12)
    ax4.set_ylabel('Fractional Energy Change ΔE/E₀', fontsize=12)
    ax4.set_title('Energy Conservation', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison plot to {save_path}")
    
    return fig


def quick_test():
    """Quick test with small parameters"""
    print("\n" + "="*70)
    print("QUICK TEST RUN (reduced parameters for speed)")
    print("="*70)
    
    sim_lcdm, sim_external = compare_models(
        n_particles=100,      # Reduced for speed
        box_size_Gpc=15.0,
        t_end_Gyr=10.0,       # Shorter simulation
        n_steps=200           # Fewer steps
    )
    
    # Plot comparison
    plot_expansion_comparison(sim_lcdm, sim_external)
    
    print("\n" + "="*70)
    print("Quick test complete!")
    print("="*70)


if __name__ == "__main__":
    quick_test()
