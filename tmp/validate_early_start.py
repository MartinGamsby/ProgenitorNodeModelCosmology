"""
Validation sweep: check matter-only never-exceeds-LCDM and timestep stability
across t_start values from 5.8 down to 2.9 Gyr.

Run from repo root:  python tmp/validate_early_start.py
"""
import sys
import math
import numpy as np

sys.path.insert(0, '.')

from cosmo.analysis import solve_friedmann_at_times, calculate_initial_conditions, detect_runaway_particles
from cosmo.constants import CosmologicalConstants, LambdaCDMParameters
from cosmo.particles import ParticleSystem
from cosmo.integrator import LeapfrogIntegrator

GYR_S = 1e9 * 365.25 * 24 * 3600

def rms_radius(positions):
    return np.sqrt(np.mean(np.sum(positions ** 2, axis=1)))

def auto_damping(t_start_Gyr):
    return np.clip((t_start_Gyr / 13.8) ** 0.135, 0.0, 1.0)

def run_matter_only_check(t_start_Gyr, n_particles=30, seed=42):
    """
    Run matter-only from t_start to 13.8 Gyr with n_steps = ceil(duration/0.04).
    Returns dict with:
      - max_excess: max(size_matter/size_lcdm - 1) over all snapshots
      - max_runaway: max(max_r/rms_r) over all snapshots
      - dt_Gyr: timestep used
      - n_steps: steps used
      - ok_invariant: True if matter_only never exceeded LCDM (within 0.01%)
      - ok_runaway: True if no runaway particles
    """
    t_end = 13.8
    t_duration = t_end - t_start_Gyr
    n_steps = math.ceil(t_duration / 0.04)
    dt_Gyr = t_duration / n_steps
    dt_s = dt_Gyr * GYR_S

    ic = calculate_initial_conditions(t_start_Gyr)
    box_size_m = ic['box_size_Gpc'] * CosmologicalConstants.Gpc_to_m
    a_start = ic['a_start']

    total_mass_kg = CosmologicalConstants.M_observable_kg

    np.random.seed(seed)
    p_matter = ParticleSystem(
        n_particles=n_particles,
        box_size_m=box_size_m,
        total_mass_kg=total_mass_kg,
        a_start=a_start,
        use_dark_energy=False,
    )
    np.random.seed(seed)
    p_lcdm = ParticleSystem(
        n_particles=n_particles,
        box_size_m=box_size_m,
        total_mass_kg=total_mass_kg,
        a_start=a_start,
        use_dark_energy=True,
    )

    # Apply auto-damping to matter-only velocities
    damping = auto_damping(t_start_Gyr)
    p_matter.set_velocities(p_matter.get_velocities() * damping)

    integ_matter = LeapfrogIntegrator(p_matter, use_dark_energy=False, use_external_nodes=False)
    integ_lcdm   = LeapfrogIntegrator(p_lcdm,   use_dark_energy=True,  use_external_nodes=False)

    rms0_matter = rms_radius(p_matter.get_positions())
    rms0_lcdm   = rms_radius(p_lcdm.get_positions())

    max_excess  = 0.0
    max_runaway = 0.0

    for step in range(n_steps):
        integ_matter.step(dt_s)
        integ_lcdm.step(dt_s)

        rms_m = rms_radius(p_matter.get_positions())
        rms_l = rms_radius(p_lcdm.get_positions())

        rel = rms_m / rms_l - 1.0
        if rel > max_excess:
            max_excess = rel

        # runaway check
        positions_m = p_matter.get_positions()
        com = np.mean(positions_m, axis=0)
        dists = np.linalg.norm(positions_m - com, axis=1)
        max_r = np.max(dists)
        cur_rms = np.sqrt(np.mean(dists**2))
        ratio = max_r / cur_rms if cur_rms > 0 else 0.0
        if ratio > max_runaway:
            max_runaway = ratio

    ok_invariant = (max_excess <= 0.0001)   # 0.01% tolerance
    ok_runaway   = (max_runaway <= 2.5)

    return {
        't_start_Gyr': t_start_Gyr,
        'dt_Gyr': dt_Gyr,
        'n_steps': n_steps,
        'max_excess': max_excess,
        'max_runaway': max_runaway,
        'ok_invariant': ok_invariant,
        'ok_runaway': ok_runaway,
        'damping': damping,
        'a_start': a_start,
    }


if __name__ == '__main__':
    T_STARTS = [5.8, 4.8, 3.8, 3.3, 2.9]

    print(f"\n{'t_start':>8}  {'a_start':>7}  {'dt_Gyr':>7}  {'n_steps':>7}  {'damping':>7}  "
          f"{'max_excess%':>11}  {'max_runaway':>11}  {'invariant':>9}  {'runaway_ok':>10}")
    print("-" * 105)

    results = []
    for t in T_STARTS:
        print(f"  Running t_start={t:.1f} Gyr ...", end='', flush=True)
        r = run_matter_only_check(t)
        results.append(r)
        inv_sym  = "PASS" if r['ok_invariant'] else "FAIL"
        run_sym  = "PASS" if r['ok_runaway']   else "FAIL"
        print(f"\r{r['t_start_Gyr']:>8.1f}  {r['a_start']:>7.4f}  {r['dt_Gyr']:>7.4f}  "
              f"{r['n_steps']:>7d}  {r['damping']:>7.4f}  "
              f"{r['max_excess']*100:>11.4f}  {r['max_runaway']:>11.4f}  "
              f"{inv_sym:>9}  {run_sym:>10}")

    # Determine safe floor
    safe = [r for r in results if r['ok_invariant'] and r['ok_runaway']]
    if safe:
        floor = min(r['t_start_Gyr'] for r in safe)
        a_floor = next(r['a_start'] for r in results if r['t_start_Gyr'] == floor)
        z_max = 1.0 / a_floor - 1.0
        print(f"\nSafe floor: t_start >= {floor:.1f} Gyr  (a~{a_floor:.3f}, z_max~{z_max:.2f})")
    else:
        print("\nNo safe t_start found in sweep!")
