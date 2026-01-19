#!/usr/bin/env python3
"""
Parameter Sweep - Find Best Fit to ΛCDM
Test multiple External-Node configurations
"""

import numpy as np
import pickle
import os
from cosmo.constants import CosmologicalConstants, SimulationParameters
from cosmo.simulation import CosmologicalSimulation
from cosmo.analysis import (
    calculate_initial_conditions,
    compare_expansion_histories
)
from cosmo.factories import (
    create_external_node_simulation,
    create_lcdm_simulation,
    run_and_extract_results
)

const = CosmologicalConstants()

PARTICLE_COUNT = 26  # Small for speed
T_START_GYR = 3.8
T_DURATION_GYR = 10.0
N_STEPS = 200

print("="*70)
print("PARAMETER SWEEP: Finding Best Match to ΛCDM")
print("="*70)

# Calculate initial conditions once (reused for all configs)
initial_conditions = calculate_initial_conditions(T_START_GYR)
BOX_SIZE = initial_conditions['box_size_Gpc']
A_START = initial_conditions['a_start']

# Test different configurations
configs = []
for M in range(500, 1250+1, 250):
    for S in range(25, 40+1, 5):
        desc = f"M={M}×M_obs, S={S}Gpc"
        configs.append((M, S, desc))

print(f"Running {len(configs)} configurations...")


# First, run ΛCDM baseline
print("\n1. Running ΛCDM baseline...")
sim_lcdm = create_lcdm_simulation(PARTICLE_COUNT, BOX_SIZE, T_START_GYR, A_START)
lcdm_results = run_and_extract_results(sim_lcdm, T_DURATION_GYR, N_STEPS)
a_lcdm = lcdm_results['a'][-1]
size_lcdm = lcdm_results['size_Gpc'][-1]
print(f"   ΛCDM final a(t) = {a_lcdm:.4f}, size = {size_lcdm:.2f} Gpc")

results = []

for M_factor, S_gpc, desc in configs:
    print(f"\n2. Testing {desc}: M={M_factor}×M_obs, S={S_gpc:.1f}Gp ({len(results)+1}/{len(configs)})")

    # Create simulation parameters
    sim_params = SimulationParameters(
        M_value=M_factor,
        S_value=S_gpc,
        n_particles=PARTICLE_COUNT,
        seed=42,
        t_start_Gyr=T_START_GYR,
        t_duration_Gyr=T_DURATION_GYR,
        n_steps=N_STEPS
    )

    # Run simulation
    sim_ext = create_external_node_simulation(sim_params, BOX_SIZE, A_START)
    ext_results = run_and_extract_results(sim_ext, T_DURATION_GYR, N_STEPS)
    a_ext = ext_results['a'][-1]
    size_ext = ext_results['size_Gpc'][-1]

    # Calculate match
    match_pct = compare_expansion_histories(size_ext, size_lcdm)
    diff_pct = 100 - match_pct

    print(f"   External-Node final a(t) = {a_ext:.4f}, size = {size_ext:.2f} Gpc")
    print(f"   Match: {match_pct:.2f}% ({diff_pct:.1f}% diff)")

    results.append({
        'M_factor': M_factor,
        'S_gpc': S_gpc,
        'desc': desc,
        'a_ext': a_ext,
        'size_ext': size_ext,
        'match_pct': match_pct,
        'diff_pct': diff_pct,
        'params': sim_params.external_params
    })

print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

# Sort by best match
results.sort(key=lambda x: x['diff_pct'])

print(f"\n{'Config':<20} {'M×M_obs':<10} {'S[Gpc]':<10} {'Match%':<10} {'Diff%':<10}")
print("-" * 70)

for r in results:
    print(f"{r['desc']:<20} {r['M_factor']:<10} {r['S_gpc']:<10.1f} "
          f"{r['match_pct']:<10.2f} {r['diff_pct']:<10.2f}")

best = results[0]
print(f"\n★ BEST MATCH: {best['desc']}")
print(f"   M = {best['M_factor']} × M_obs")
print(f"   S = {best['S_gpc']:.1f} Gpc")
print(f"   Match: {best['match_pct']:.1f}%")

# Save best configuration
os.makedirs('./results', exist_ok=True)
with open('./results/best_config.pkl', 'wb') as f:
    pickle.dump(best, f)

print(f"\n✓ Saved best configuration to results/best_config.pkl")
