"""
Diagnostic script to investigate matter-only instability
Tracks accelerations, velocities, positions, and energy throughout simulation
"""

import numpy as np
import pickle
from pathlib import Path
from scipy.integrate import odeint
from cosmo.constants import CosmologicalConstants, LambdaCDMParameters
from cosmo.particles import ParticleSystem
from cosmo.integrator import LeapfrogIntegrator

# Setup
const = CosmologicalConstants()
lcdm = LambdaCDMParameters()

# Simulation parameters (matching user's problem case)
t_start_gyr = 3.8
t_duration_gyr = 20.0
n_particles = 50
n_steps = 150
damping = 0.9

# Convert to SI
t_start = t_start_gyr * const.Gyr_to_s
t_duration = t_duration_gyr * const.Gyr_to_s
t_end = t_start + t_duration

# Get scale factor at start using Friedmann equation
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

# Find scale factor at start time
idx_start = np.argmin(np.abs(t_span - t_start))
a_start = a_full[idx_start]
idx_today = np.argmin(np.abs(t_span - 13.8 * const.Gyr_to_s))

print(f"Scale factor at t_start: {a_start:.4f}")

# Initial box size
box_size_today = 14.5 * const.Gpc_to_m
box_size = box_size_today * (a_start / a_full[idx_today])
print(f"Initial box size: {box_size / const.Gpc_to_m:.2f} Gpc")

# Create particle system (matter-only mode)
np.random.seed(42)
particles = ParticleSystem(
    n_particles=n_particles,
    box_size=box_size,
    a_start=a_start,
    use_dark_energy=False,  # Matter-only
    damping_factor_override=damping
)
particles.time = t_start  # Set starting time

# Create integrator (matter-only: no external nodes, no dark energy)
integrator = LeapfrogIntegrator(
    particles,
    hmea_grid=None,
    softening=1e21,
    use_external_nodes=False,
    use_dark_energy=False
)

# Diagnostic output
dt = t_duration / n_steps
print(f"\n=== DIAGNOSTIC SETUP ===")
print(f"Timestep: {dt:.3e} s = {dt / const.Gyr_to_s:.4f} Gyr")
print(f"Total steps: {n_steps}")
print(f"Damping factor: {damping}")

# Initial state
pos0 = particles.get_positions()
vel0 = particles.get_velocities()
r0_rms = np.sqrt(np.mean(np.sum(pos0**2, axis=1)))
v0_rms = np.sqrt(np.mean(np.sum(vel0**2, axis=1)))

print(f"\n=== INITIAL STATE ===")
print(f"RMS radius: {r0_rms / const.Gpc_to_m:.3f} Gpc")
print(f"RMS velocity: {v0_rms:.3e} m/s")
print(f"v/r ratio: {v0_rms / r0_rms:.3e} s^-1 (H0 = {lcdm.H0:.3e} s^-1)")

# Calculate initial accelerations
a_internal = integrator.calculate_internal_forces()
a_internal_rms = np.sqrt(np.mean(np.sum(a_internal**2, axis=1)))
print(f"Internal gravity RMS: {a_internal_rms:.3e} m/s^2")

# Energy
E0 = integrator.total_energy()
KE0 = particles.kinetic_energy()
PE0 = integrator.potential_energy()
print(f"\n=== INITIAL ENERGY ===")
print(f"Total: {E0:.3e} J")
print(f"Kinetic: {KE0:.3e} J")
print(f"Potential: {PE0:.3e} J")
print(f"KE/|PE|: {KE0 / abs(PE0):.3f}")

# Track diagnostics at intervals
diagnostics = []
n_diagnostics = 10  # Sample 10 times during simulation
diagnostic_interval = max(1, n_steps // n_diagnostics)

print(f"\n=== RUNNING SIMULATION ===")
print(f"Will sample every {diagnostic_interval} steps...")

for step in range(n_steps):
    # Take step
    integrator.step(dt)

    # Sample diagnostics
    if step % diagnostic_interval == 0 or step == n_steps - 1:
        pos = particles.get_positions()
        vel = particles.get_velocities()
        acc = particles.get_accelerations()

        r_rms = np.sqrt(np.mean(np.sum(pos**2, axis=1)))
        v_rms = np.sqrt(np.mean(np.sum(vel**2, axis=1)))
        a_rms = np.sqrt(np.mean(np.sum(acc**2, axis=1)))

        E = integrator.total_energy()
        KE = particles.kinetic_energy()
        PE = integrator.potential_energy()

        t_gyr = particles.time / const.Gyr_to_s

        # Compute radial vs tangential components
        r_unit = pos / np.linalg.norm(pos, axis=1, keepdims=True)
        v_radial = np.sum(vel * r_unit, axis=1)
        v_radial_rms = np.sqrt(np.mean(v_radial**2))

        a_radial = np.sum(acc * r_unit, axis=1)
        a_radial_rms = np.sqrt(np.mean(a_radial**2))
        a_radial_mean = np.mean(a_radial)  # Should be negative (inward) for matter-only

        diagnostics.append({
            'step': step + 1,
            't_gyr': t_gyr,
            'r_rms': r_rms / const.Gpc_to_m,
            'v_rms': v_rms,
            'a_rms': a_rms,
            'v_radial_rms': v_radial_rms,
            'a_radial_rms': a_radial_rms,
            'a_radial_mean': a_radial_mean,
            'E_total': E,
            'KE': KE,
            'PE': PE,
            'dE': (E - E0) / abs(E0) * 100  # Energy drift percentage
        })

        print(f"\nStep {step+1}/{n_steps} (t={t_gyr:.2f} Gyr):")
        print(f"  r_rms: {r_rms / const.Gpc_to_m:.3f} Gpc")
        print(f"  v_rms: {v_rms:.3e} m/s ({v_rms / v0_rms:.3f}x initial)")
        print(f"  a_radial_mean: {a_radial_mean:.3e} m/s^2 {'(OUTWARD!)' if a_radial_mean > 0 else '(inward)'}")
        print(f"  Energy drift: {(E - E0) / abs(E0) * 100:.2f}%")

# Final analysis
print(f"\n=== FINAL STATE ===")
pos_final = particles.get_positions()
vel_final = particles.get_velocities()
r_final_rms = np.sqrt(np.mean(np.sum(pos_final**2, axis=1)))
v_final_rms = np.sqrt(np.mean(np.sum(vel_final**2, axis=1)))

print(f"RMS radius: {r_final_rms / const.Gpc_to_m:.3f} Gpc ({r_final_rms / r0_rms:.3f}x expansion)")
print(f"RMS velocity: {v_final_rms:.3e} m/s ({v_final_rms / v0_rms:.3f}x initial)")

E_final = integrator.total_energy()
print(f"\nEnergy conservation: {(E_final - E0) / abs(E0) * 100:.2f}% drift")

# Check what ballistic would give
r_ballistic = r0_rms + v0_rms * t_duration
print(f"\nBallistic estimate: {r_ballistic / const.Gpc_to_m:.3f} Gpc")
print(f"Actual: {r_final_rms / const.Gpc_to_m:.3f} Gpc")
print(f"Ratio: {r_final_rms / r_ballistic:.3f}x ballistic")

# Save diagnostics
output_dir = Path("./results/diagnostics")
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "matter_only_diagnostics.pkl", 'wb') as f:
    pickle.dump(diagnostics, f)

print(f"\nDiagnostics saved to {output_dir / 'matter_only_diagnostics.pkl'}")

# Summary table
print("\n=== DIAGNOSTIC SUMMARY ===")
print("Step    t[Gyr]  r[Gpc]  v/v0   a_radial[m/s^2]  dE[%]")
print("-" * 60)
for d in diagnostics:
    print(f"{d['step']:4d}  {d['t_gyr']:6.2f}  {d['r_rms']:6.2f}  "
          f"{d['v_rms']/v0_rms:6.3f}  {d['a_radial_mean']:+.3e}  {d['dE']:+6.2f}")
