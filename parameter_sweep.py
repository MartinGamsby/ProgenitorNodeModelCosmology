#!/usr/bin/env python3
"""
Parameter Sweep - Find Best Fit to ΛCDM
Test multiple External-Node configurations

Uses cosmo.parameter_sweep module for search algorithms.
This script handles simulation setup, callback wiring, and output formatting.
"""

import numpy as np
import math
import csv
import os
from typing import List
from cosmo.constants import CosmologicalConstants, SimulationParameters
from cosmo.factories import (
    setup_simulation_context,
    run_external_node_simulation,
    results_to_sim_result
)
from cosmo.parameter_sweep import (
    SearchMethod, SweepConfig, MatchWeights, SimResult, LCDMBaseline,
    build_m_list, build_s_list, build_center_mass_list, run_sweep,
    CSV_COLUMNS
)

const = CosmologicalConstants()

# Configuration
SEARCH_METHOD = SearchMethod.LINEAR_SEARCH
QUICK_SEARCH = False
MULTIPLY_PARTICLES = False
SEARCH_CENTER_MASS = True
MANY_SEARCH = 3 if QUICK_SEARCH else (10 if SEARCH_CENTER_MASS else 20)#21#3 and 10 are probably fine. You can go to 12,20,21!,31!!,...61!!!,...101!!!!

config = SweepConfig(
    quick_search=QUICK_SEARCH,
    many_search=MANY_SEARCH,
    search_center_mass=SEARCH_CENTER_MASS,
    t_start_Gyr=5.8,
    t_duration_Gyr=8.0,
    damping_factor=None,
    s_min_gpc=15,
    s_max_gpc=60+MANY_SEARCH,
    save_interval=10
)

weights = MatchWeights()#TODO:Remove

print("="*70)
print("PARAMETER SWEEP: Finding Best Match to ΛCDM")
print("="*70)

# Setup initial conditions and LCDM baseline (shared with run_simulation.py)
print("\n1. Computing initial conditions and ΛCDM baseline...")
BOX_SIZE, A_START, lcdm_result = setup_simulation_context(
    config.t_start_Gyr, config.t_duration_Gyr, config.n_steps, config.save_interval
)

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

# Create baseline object for parameter sweep module
size_lcdm_final = lcdm_result['diameter_Gpc'][-1]
radius_lcdm_max = size_lcdm_final / 2 / np.sqrt(3/5)
a_lcdm = lcdm_result['a'][-1]
print(f"   ΛCDM final a(t) = {a_lcdm:.4f}, size = {size_lcdm_final:.2f} Gpc")

baseline = LCDMBaseline(
    t_Gyr=lcdm_result['t'],
    size_Gpc=lcdm_result['diameter_Gpc'],
    H_hubble=lcdm_result['H_hubble'],
    size_final_Gpc=size_lcdm_final,
    radius_max_Gpc=radius_lcdm_max,
    a_final=a_lcdm
)

# Track simulation count for efficiency reporting
sim_count = 1

def sim(M_factor: int, S_gpc: int, centerM: int, seed: int) -> SimResult:
    """
    Run a single External-Node simulation and return raw results.

    Uses shared factory functions for consistency with run_simulation.py.
    """
    global sim_count
    sim_count += 1
    desc = f"M={M_factor}, S={S_gpc}, centerM={centerM}"
    print(f"\n2. Testing {desc} (sim #{sim_count})")

    # Create simulation parameters (mass_randomize=0.0 matches CLI default)
    sim_params = SimulationParameters(
        M_value=M_factor,
        S_value=S_gpc,
        n_particles=config.particle_count*(int(math.log(centerM*centerM, 2)+1) if MULTIPLY_PARTICLES else 1),
        seed=seed,
        t_start_Gyr=config.t_start_Gyr,
        t_duration_Gyr=config.t_duration_Gyr,
        n_steps=config.n_steps,
        damping_factor=config.damping_factor,
        center_node_mass=centerM,
        mass_randomize=0.0  # Matches CLI default for deterministic results
    )

    # Run simulation and convert to SimResult (both use shared factory functions)
    ext_results = run_external_node_simulation(sim_params, BOX_SIZE, A_START, config.save_interval)
    print(f"   External-Node final a(t) = {ext_results['a'][-1]:.4f}, size = {ext_results['diameter_Gpc'][-1]:.2f} Gpc")

    return results_to_sim_result(ext_results, sim_params)

def sim_callback(M_factor: int, S_gpc: int, centerM: int, seeds: List[int] = [42]) -> List[SimResult]:
    results = []
    for seed in seeds:
        results.append(sim(M_factor, S_gpc, centerM, seed))
    return results
   

# Run the sweep
results = run_sweep(config, SEARCH_METHOD, sim_callback, baseline, weights, seeds=[123] if QUICK_SEARCH else [42,123])

# Build best per S
best_per_s = {}
for r in results:
    s = r['S_gpc']
    if s not in best_per_s or r['match_avg_pct'] > best_per_s[s]['match_avg_pct']:
        best_per_s[s] = r

# Sort by S ascending
best_per_s_list = [best_per_s[s] for s in sorted(best_per_s.keys())]

# Print best per S results
print("\n" + "="*70)
print("BEST PER S")
print("="*70)

print(f"\n{'Config':<20} {'M×M_obs':<10} {'centerM':<10} {'S[Gpc]':<10} {'Match%':<10} {'Diff%':<10} {'Curve%':<10} {'Half%':<10} {'End%':<10} {'Hubble%':<10}")
print("-" * 100)
for r in best_per_s_list:
    print(f"{r['desc']:<20} {r['M_factor']:<10} {r['centerM']:<10} {r['S_gpc']:<10.1f} "
          f"{r['match_avg_pct']:<10.2f} {r['diff_pct']:<10.2f} {r['match_curve_pct']:<10.2f} {r['match_half_curve_pct']:<10.2f} {r['match_end_pct']:<10.2f} {r['match_hubble_curve_pct']:<10.2f}")

# Find overall best
best = max(results, key=lambda x: x['match_avg_pct'])
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


# Save all results to CSV
os.makedirs('./results', exist_ok=True)
csv_path = './results/sweep_results.csv'

with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(results)

print(f"\n✓ Saved {len(results)} results to {csv_path}")

# Save best per S to CSV
csv_path_best_s = './results/sweep_best_per_S.csv'
with open(csv_path_best_s, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(best_per_s_list)

print(f"\n✓ Saved {len(best_per_s_list)} best-per-S results to {csv_path_best_s}")
