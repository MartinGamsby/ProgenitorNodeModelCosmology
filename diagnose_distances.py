"""
Check minimum inter-particle distances during simulation
"""

import numpy as np
from scipy.integrate import odeint
from cosmo.constants import CosmologicalConstants, LambdaCDMParameters
from cosmo.particles import ParticleSystem
from cosmo.integrator import LeapfrogIntegrator

# Setup
const = CosmologicalConstants()
lcdm = LambdaCDMParameters()

# Simulation parameters
t_start_gyr = 3.8
t_duration_gyr = 20.0
n_particles = 50
n_steps = 150
damping = 0.9

# Convert to SI
t_start = t_start_gyr * const.Gyr_to_s
t_duration = t_duration_gyr * const.Gyr_to_s

# Get scale factor at start
def friedmann_equation(a, t, H0, Omega_m, Omega_Lambda):
    if a <= 0:
        return 1e-10
    H = H0 * np.sqrt(Omega_m / a**3 + Omega_Lambda)
    return H * a

a0 = 0.001
t_max = 25.0 * const.Gyr_to_s
t_span = np.linspace(0, t_max, 300)

a_full = odeint(friedmann_equation, a0, t_span,
                args=(lcdm.H0, lcdm.Omega_m, lcdm.Omega_Lambda))
a_full = a_full.flatten()

idx_start = np.argmin(np.abs(t_span - t_start))
a_start = a_full[idx_start]
idx_today = np.argmin(np.abs(t_span - 13.8 * const.Gyr_to_s))

box_size = 14.5 * const.Gpc_to_m * (a_start / a_full[idx_today])

# Create particle system
np.random.seed(42)
particles = ParticleSystem(
    n_particles=n_particles,
    box_size=box_size,
    a_start=a_start,
    use_dark_energy=False,
    damping_factor_override=damping
)
particles.time = t_start

# Create integrator
integrator = LeapfrogIntegrator(
    particles,
    hmea_grid=None,
    softening=1e21,  # 1 Mpc = 3.086e22 m
    use_external_nodes=False,
    use_dark_energy=False
)

dt = t_duration / n_steps
softening = 1e21

print(f"Softening length: {softening / const.Mpc_to_m:.2f} Mpc")
print(f"\n{'Step':<8} {'t[Gyr]':<10} {'r_min[Mpc]':<15} {'r_min/soft':<12} {'Status'}")
print("-" * 60)

# Check distances at intervals
for step in range(n_steps):
    integrator.step(dt)

    if step % 15 == 0 or step == n_steps - 1:
        pos = particles.get_positions()

        # Calculate all pairwise distances
        min_dist = np.inf
        for i in range(len(pos)):
            for j in range(i+1, len(pos)):
                r = np.linalg.norm(pos[i] - pos[j])
                if r < min_dist:
                    min_dist = r

        t_gyr = particles.time / const.Gyr_to_s
        min_dist_mpc = min_dist / const.Mpc_to_m
        ratio = min_dist / softening

        status = "OK"
        if ratio < 1.0:
            status = "CLOSE!"
        if ratio < 0.1:
            status = "COLLISION!"

        print(f"{step+1:<8} {t_gyr:<10.2f} {min_dist_mpc:<15.2f} {ratio:<12.3f} {status}")
