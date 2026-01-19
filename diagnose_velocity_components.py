"""
Track radial vs tangential velocity components to understand energy injection
"""

import numpy as np
from scipy.integrate import odeint
from cosmo.constants import CosmologicalConstants, LambdaCDMParameters
from cosmo.particles import ParticleSystem
from cosmo.integrator import LeapfrogIntegrator

# Setup
const = CosmologicalConstants()
lcdm = LambdaCDMParameters()

t_start_gyr = 3.8
t_duration_gyr = 20.0
n_particles = 50
n_steps = 150
damping = 0.9

t_start = t_start_gyr * const.Gyr_to_s
t_duration = t_duration_gyr * const.Gyr_to_s

# Get scale factor
def friedmann_equation(a, t, H0, Omega_m, Omega_Lambda):
    if a <= 0:
        return 1e-10
    H = H0 * np.sqrt(Omega_m / a**3 + Omega_Lambda)
    return H * a

a0 = 0.001
t_max = 25.0 * const.Gyr_to_s
t_span = np.linspace(0, t_max, 300)
a_full = odeint(friedmann_equation, a0, t_span,
                args=(lcdm.H0, lcdm.Omega_m, lcdm.Omega_Lambda)).flatten()

idx_start = np.argmin(np.abs(t_span - t_start))
a_start = a_full[idx_start]
idx_today = np.argmin(np.abs(t_span - 13.8 * const.Gyr_to_s))
box_size = 14.5 * const.Gpc_to_m * (a_start / a_full[idx_today])

# Create system
np.random.seed(42)
particles = ParticleSystem(
    n_particles=n_particles,
    box_size=box_size,
    a_start=a_start,
    use_dark_energy=False,
    damping_factor_override=damping
)
particles.time = t_start

integrator = LeapfrogIntegrator(
    particles, hmea_grid=None, softening_per_Mobs=1e24,
    use_external_nodes=False, use_dark_energy=False
)

dt = t_duration / n_steps

print(f"{'Step':<8} {'t[Gyr]':<10} {'v_rad[km/s]':<15} {'v_tan[km/s]':<15} {'v_tot[km/s]':<15} {'a_rad[m/s^2]':<15}")
print("-" * 85)

for step in range(n_steps):
    integrator.step(dt)

    if step % 15 == 0 or step == n_steps - 1 or step == 90:  # Include step 91 specifically
        pos = particles.get_positions()
        vel = particles.get_velocities()
        acc = particles.get_accelerations()

        # Compute radial unit vectors
        r_mag = np.linalg.norm(pos, axis=1, keepdims=True)
        r_unit = pos / r_mag

        # Radial components
        v_radial = np.sum(vel * r_unit, axis=1)
        a_radial = np.sum(acc * r_unit, axis=1)

        # Tangential components
        v_radial_vec = v_radial[:, np.newaxis] * r_unit
        v_tangential_vec = vel - v_radial_vec
        v_tangential = np.linalg.norm(v_tangential_vec, axis=1)

        # RMS values
        v_rad_rms = np.sqrt(np.mean(v_radial**2)) / 1000  # km/s
        v_tan_rms = np.sqrt(np.mean(v_tangential**2)) / 1000  # km/s
        v_tot_rms = np.sqrt(np.mean(np.sum(vel**2, axis=1))) / 1000  # km/s
        a_rad_mean = np.mean(a_radial)

        t_gyr = particles.time / const.Gyr_to_s

        print(f"{step+1:<8} {t_gyr:<10.2f} {v_rad_rms:<15.1f} {v_tan_rms:<15.1f} {v_tot_rms:<15.1f} {a_rad_mean:<+15.3e}")

print("\n=== ANALYSIS ===")
print("If v_radial is increasing but a_radial is negative (inward),")
print("then energy is being added incorrectly by the integrator.")
