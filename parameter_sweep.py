#!/usr/bin/env python3
"""
Parameter Sweep - Find Best Fit to ΛCDM
Test multiple External-Node configurations

Uses cosmo.parameter_sweep module for search algorithms.
This script handles simulation setup, callback wiring, and output formatting.
"""

import numpy as np
import pickle
import os
from cosmo.constants import CosmologicalConstants, SimulationParameters
from cosmo.simulation import CosmologicalSimulation
from cosmo.analysis import (
    calculate_initial_conditions,
    solve_friedmann_at_times,
    calculate_hubble_parameters
)
from cosmo.factories import run_and_extract_results
from cosmo.parameter_sweep import (
    SearchMethod, SweepConfig, MatchWeights, SimResult, LCDMBaseline,
    build_m_list, build_s_list, build_center_mass_list, run_sweep
)

const = CosmologicalConstants()

# Configuration
SEARCH_METHOD = SearchMethod.LINEAR_SEARCH
QUICK_SEARCH = True#False
MANY_SEARCH = False
SEARCH_CENTER_MASS = True

config = SweepConfig(
    quick_search=QUICK_SEARCH,
    many_search=MANY_SEARCH,
    search_center_mass=SEARCH_CENTER_MASS,
    t_start_Gyr=3.8,
    t_duration_Gyr=10.0,
    damping_factor=0.98,
    s_min_gpc=15,
    s_max_gpc=100 if MANY_SEARCH else 60,
    save_interval=10
)

weights = MatchWeights()

print("="*70)
print("PARAMETER SWEEP: Finding Best Match to ΛCDM")
print("="*70)

# Calculate initial conditions once (reused for all configs)
initial_conditions = calculate_initial_conditions(config.t_start_Gyr)
BOX_SIZE = initial_conditions['box_size_Gpc']
A_START = initial_conditions['a_start']

# Build parameter lists for info display
m_list = build_m_list(config.many_search)
s_list = build_s_list(config.s_min_gpc, config.s_max_gpc)
center_masses = build_center_mass_list(config.search_center_mass, config.many_search)
nbConfigs_bruteforce = len(m_list) * len(s_list) * len(center_masses)

print(f"Using {SEARCH_METHOD.name} on S for each M value...")
print(f"M values to test: {len(m_list)}")
if config.search_center_mass:
    print(f"Center M values to test: {len(center_masses)}")
print(f"S range: [{config.s_min_gpc}, {config.s_max_gpc}]")
print(f"(Brute force would test {nbConfigs_bruteforce} configurations)")

# First, solve ΛCDM baseline (analytic solution, not N-body simulation)
print("\n1. Computing ΛCDM baseline...")

# Compute time array matching N-body snapshot times
# N-body saves initial snapshot + every SAVE_INTERVAL steps
snapshot_steps = np.arange(0, config.n_steps + 1, config.save_interval)
t_relative_Gyr = (snapshot_steps / config.n_steps) * config.t_duration_Gyr
t_absolute_Gyr = config.t_start_Gyr + t_relative_Gyr

# Solve ΛCDM at exact N-body snapshot times
lcdm_solution = solve_friedmann_at_times(t_absolute_Gyr, Omega_Lambda=None)
H_lcdm_hubble = lcdm_solution['H_hubble']

# Extract expansion history
t_lcdm = lcdm_solution['t_Gyr'] - config.t_start_Gyr  # Offset to start at 0
a_lcdm_array = lcdm_solution['a']
size_lcdm_full = BOX_SIZE * (a_lcdm_array / A_START)  # Full resolution curve
a_lcdm = a_lcdm_array[-1]
size_lcdm_final = size_lcdm_full[-1]

# Scale box_size so that the RMS radius matches the target
# For a uniform sphere of radius R, RMS radius = R * sqrt(3/5) ≈ 0.775*R
radius_lcdm_max = size_lcdm_final / 2 / np.sqrt(3/5)
print(f"   ΛCDM final a(t) = {a_lcdm:.4f}, size = {size_lcdm_final:.2f} Gpc")

# Create baseline object for parameter sweep module
baseline = LCDMBaseline(
    t_Gyr=t_lcdm,
    size_Gpc=size_lcdm_full,
    H_hubble=H_lcdm_hubble,
    size_final_Gpc=size_lcdm_final,
    radius_max_Gpc=radius_lcdm_max,
    a_final=a_lcdm
)

# Track simulation count for efficiency reporting
sim_count = 0

def sim_callback(M_factor: int, S_gpc: int, centerM: int, seed: int) -> SimResult:
    """
    Run a single External-Node simulation and return raw results.

    This callback is passed to the parameter sweep module.
    """
    global sim_count
    sim_count += 1
    desc = f"M={M_factor}, S={S_gpc}, centerM={centerM}"
    print(f"\n2. Testing {desc} (sim #{sim_count})")

    # Create simulation parameters
    sim_params = SimulationParameters(
        M_value=M_factor,
        S_value=S_gpc,
        n_particles=config.particle_count,
        seed=seed,
        t_start_Gyr=config.t_start_Gyr,
        t_duration_Gyr=config.t_duration_Gyr,
        n_steps=config.n_steps,
        damping_factor=config.damping_factor,
        center_node_mass=centerM
    )

    # Run simulation
    sim_ext = CosmologicalSimulation(sim_params, BOX_SIZE, A_START,
                                      use_external_nodes=True, use_dark_energy=False)
    ext_results = run_and_extract_results(sim_ext, config.t_duration_Gyr, config.n_steps,
                                           save_interval=config.save_interval)

    a_ext = ext_results['a'][-1]
    size_ext_final = ext_results['diameter_Gpc'][-1]
    size_ext_curve = ext_results['diameter_Gpc']
    radius_max_final = ext_results['max_radius_Gpc'][-1]
    t_ext = ext_results['t_Gyr']

    hubble_ext = calculate_hubble_parameters(t_ext, ext_results['a'], smooth_sigma=0.0)

    print(f"   External-Node final a(t) = {a_ext:.4f}, size = {size_ext_final:.2f} Gpc")

    return SimResult(
        size_curve_Gpc=size_ext_curve,
        hubble_curve=hubble_ext,
        size_final_Gpc=size_ext_final,
        radius_max_Gpc=radius_max_final,
        a_final=a_ext,
        t_Gyr=t_ext,
        params=sim_params.external_params
    )


# Run the sweep
results = run_sweep(config, SEARCH_METHOD, sim_callback, baseline, weights, seed=42)

print("\n" + "="*70)

print("RESULTS BY MASS")
print("="*70)

# Sort by best match
results.reverse()  # Original order was descending M

print(f"\n{'Config':<20} {'M×M_obs':<10} {'centerM':<10} {'S[Gpc]':<10} {'Match%':<10} {'Diff%':<10} {'Curve%':<10} {'End%':<10} {'Radius%':<10} {'Hubble%':<10}")
print("-" * 70)
for r in results:
    print(f"{r['desc']:<20} {r['M_factor']:<10} {r['centerM']:<10} {r['S_gpc']:<10.1f} "
          f"{r['match_avg_pct']:<10.2f} {r['diff_pct']:<10.2f} {r['match_curve_pct']:<10.2f} {r['match_end_pct']:<10.2f} {r['match_max_pct']:<10.2f} {r['match_hubble_curve_pct']:<10.2f}")
print("\n" + "="*70)

print("RESULTS SUMMARY")
print("="*70)

# Sort by best match
results.sort(key=lambda x: x['diff_pct'])

print(f"\n{'Config':<20} {'M×M_obs':<10} {'centerM':<10} {'S[Gpc]':<10} {'Match%':<10} {'Diff%':<10} {'Curve%':<10} {'End%':<10} {'Radius%':<10} {'Hubble%':<10}")
print("-" * 70)
for r in results:
    print(f"{r['desc']:<20} {r['M_factor']:<10} {r['centerM']:<10} {r['S_gpc']:<10.1f} "
          f"{r['match_avg_pct']:<10.2f} {r['diff_pct']:<10.2f} {r['match_curve_pct']:<10.2f} {r['match_end_pct']:<10.2f} {r['match_max_pct']:<10.2f} {r['match_hubble_curve_pct']:<10.2f}")

best = results[0]
print(f"\n★ BEST MATCH: {best['desc']}")
print(f"   M = {best['M_factor']} × M_obs")
print(f"   S = {best['S_gpc']:.1f} Gpc")
print(f"   centerM = {best['centerM']} M_obs ")
print(f"   Match: {best['match_avg_pct']:.1f}%")

print(f"\n{'='*70}")
print(f"EFFICIENCY SUMMARY")
print(f"{'='*70}")
print(f"Total simulations run: {sim_count}")
print(f"Brute force would require: {nbConfigs_bruteforce}")
print(f"Speedup: {nbConfigs_bruteforce/sim_count:.1f}×")

# Save best configuration
os.makedirs('./results', exist_ok=True)
with open('./results/best_config.pkl', 'wb') as f:
    pickle.dump(best, f)

print(f"\n✓ Saved best configuration to results/best_config.pkl")
