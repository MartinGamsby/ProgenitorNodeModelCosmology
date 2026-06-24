#!/usr/bin/env python3
"""
Small from-data Pantheon+ sweep for Stage 3 validation.
Coarse M x S grid, modest particles/steps. Results reported in summary.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import math
from cosmo.constants import CosmologicalConstants, SimulationParameters
from cosmo.factories import setup_simulation_context, run_external_node_simulation, results_to_sim_result
from cosmo.parameter_sweep import (
    SearchMethod, SweepConfig, MatchWeights, SimResult, LCDMBaseline,
    build_m_list, build_s_list, run_sweep, SKIP_CACHE, compute_pantheon_metrics,
)
from cosmo.pantheon import load_pantheon
import cosmo.parameter_sweep as ps_module
ps_module.SKIP_CACHE = True  # Disable cache for this small test sweep

const = CosmologicalConstants()

# Load Pantheon+ real data
pantheon_data = load_pantheon()
print(f"Loaded Pantheon+ data: {pantheon_data['n']} SNe")

# Coarse grid: small number of particles & steps for speed
T_START = 5.8
T_DURATION = 8.0  # -> t_end = 13.8 Gyr (today)

config = SweepConfig(
    quick_search=True,   # 200 particles
    many_search=3,
    search_center_mass=False,
    t_start_Gyr=T_START,
    t_duration_Gyr=T_DURATION,
    s_min_gpc=15,
    s_max_gpc=60,
    save_interval=10,
    objective="pantheon",
)
# Override n_steps manually via the property
# quick_search gives n_steps=250, which is fine

print(f"Config: {config.particle_count} particles, {config.n_steps} steps, "
      f"t=[{T_START}, {T_START+T_DURATION}] Gyr")

# Setup initial conditions and LCDM baseline (needed for simulation setup)
BOX_SIZE, A_START, lcdm_result = setup_simulation_context(
    T_START, T_DURATION, config.n_steps, config.save_interval
)

# Coarse M x S grid: hand-pick a few representative values for speed
M_LIST = [50, 200, 500, 2000, 5000, 20000]  # ascending
S_LIST = [15, 20, 25, 30, 35, 40, 50, 60]

print(f"\nSearching {len(M_LIST)} M values x {len(S_LIST)} S values = {len(M_LIST)*len(S_LIST)} configs")
print(f"M values: {M_LIST}")
print(f"S values: {S_LIST}")

sim_count = 0

def sim_callback(M_factor, S_gpc, centerM, seeds):
    global sim_count
    results = []
    for seed in seeds:
        sim_count += 1
        sim_params = SimulationParameters(
            M_value=M_factor,
            S_value=S_gpc,
            n_particles=config.particle_count,
            seed=seed,
            t_start_Gyr=T_START,
            t_duration_Gyr=T_DURATION,
            n_steps=config.n_steps,
            damping_factor=None,
            center_node_mass=1,
            mass_randomize=0.0,
        )
        ext_results = run_external_node_simulation(sim_params, BOX_SIZE, A_START, config.save_interval)
        results.append(results_to_sim_result(ext_results, sim_params))
    return results

# Run brute-force over the coarse grid
all_results = []
for M in M_LIST:
    for S in S_LIST:
        print(f"  M={M:6d}, S={S:3d} Gpc ... ", end="", flush=True)
        sim_results = sim_callback(M, S, 1, [42])
        for sim_result in sim_results:
            metrics = compute_pantheon_metrics(sim_result, pantheon_data, T_START)
            result = {
                'M_factor': M,
                'S_gpc': S,
                'centerM': 1,
                **metrics,
            }
            all_results.append(result)
        r = all_results[-1]
        print(f"chi2_dof={r['chi2_dof']:.3f}, R2={r['R2']:.4f}, "
              f"n_sne={r['n_sne_used']}, match={r['match_avg_pct']:.2f}%")

# Sort by match_avg_pct (best first)
all_results.sort(key=lambda x: x['match_avg_pct'], reverse=True)

print("\n" + "="*70)
print("BEST CONFIGS (from-data Pantheon+ chi^2)")
print("="*70)
print(f"{'M':>8} {'S':>5} {'chi2_dof':>10} {'R2':>8} {'n_sne':>6} {'match%':>8}")
print("-"*55)
for r in all_results[:10]:
    print(f"{r['M_factor']:>8} {r['S_gpc']:>5} {r['chi2_dof']:>10.3f} {r['R2']:>8.4f} "
          f"{r['n_sne_used']:>6} {r['match_avg_pct']:>8.2f}%")

best = all_results[0]
print(f"\n*** BEST FIT: M={best['M_factor']}, S={best['S_gpc']} Gpc")
print(f"    chi2_dof = {best['chi2_dof']:.4f}")
print(f"    R2       = {best['R2']:.4f}")
print(f"    n_sne    = {best['n_sne_used']}")
print(f"    match%   = {best['match_avg_pct']:.2f}%")

# Compare with LCDM and matter-only references using analytic model
print("\n--- Reference models (analytic, full Pantheon+) ---")
from cosmo.hubble_diagram import evaluate_model
from cosmo.constants import SimulationParameters as SP
z, mu_obs, sigma = pantheon_data['z'], pantheon_data['mu'], pantheon_data['sigma']
for model in ('lcdm', 'matter_only'):
    ev = evaluate_model(z, mu_obs, sigma, model=model)
    print(f"  {model:<15}: chi2_dof={ev['chi2_dof']:.4f}, R2={ev['R2']:.4f}")

print(f"\nTotal simulations run: {sim_count}")
