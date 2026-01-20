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
from cosmo.factories import run_and_extract_results

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
SMin_gpc = 10   # Min box size to test
SMax_gpc = 1000  # Max box size to test

Mlist = [i for i in range(5, 2500, 250)]
Slist = [i for i in range(SMin_gpc, SMax_gpc+1, 1)]
nbConfigs_bruteforce = len(Mlist)*len(Slist)


print(f"Using ternary search on S for each M value...")
print(f"M values to test: {len(Mlist)}")
print(f"S range: [{SMin_gpc}, {SMax_gpc}]")
print(f"(Brute force would test {nbConfigs_bruteforce} configurations)")


# First, run ΛCDM baseline
print("\n1. Running ΛCDM baseline...")
# Create sim_params for LCDM
lcdm_params = SimulationParameters(n_particles=PARTICLE_COUNT, seed=42,
                                    t_start_Gyr=T_START_GYR, t_duration_Gyr=T_DURATION_GYR, n_steps=N_STEPS)
sim_lcdm = CosmologicalSimulation(lcdm_params, BOX_SIZE, A_START,
                                   use_external_nodes=False, use_dark_energy=True)
lcdm_results = run_and_extract_results(sim_lcdm, T_DURATION_GYR, N_STEPS)
a_lcdm = lcdm_results['a'][-1]
size_lcdm = lcdm_results['size_Gpc'][-1]
print(f"   ΛCDM final a(t) = {a_lcdm:.4f}, size = {size_lcdm:.2f} Gpc")

results = []
sim_count = 0  # Track total simulations


def sim(M_factor, S_gpc, desc):
    global sim_count
    sim_count += 1
    print(f"\n2. Testing {desc}: M={M_factor}×M_obs, S={S_gpc:.1f}Gpc (sim #{sim_count})")

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
    sim_ext = CosmologicalSimulation(sim_params, BOX_SIZE, A_START,
                                      use_external_nodes=True, use_dark_energy=False)
    ext_results = run_and_extract_results(sim_ext, T_DURATION_GYR, N_STEPS)
    a_ext = ext_results['a'][-1]
    size_ext = ext_results['size_Gpc'][-1]

    # Calculate match
    match_pct = compare_expansion_histories(size_ext, size_lcdm)
    diff_pct = 100 - match_pct

    print(f"   External-Node final a(t) = {a_ext:.4f}, size = {size_ext:.2f} Gpc")
    print(f"   Match: {match_pct:.2f}% ({diff_pct:.1f}% diff)")

    return {
        'M_factor': M_factor,
        'S_gpc': S_gpc,
        'desc': desc,
        'a_ext': a_ext,
        'size_ext': size_ext,
        'match_pct': match_pct,
        'diff_pct': diff_pct,
        'params': sim_params.external_params
    }

def ternary_search_S(M_factor, S_min=SMin_gpc, S_max=SMax_gpc):
    """
    Ternary search for optimal S given fixed M.
    Assumes unimodal (bell curve) match quality.

    Returns: (best_S, best_match_pct, best_result_dict)
    """
    # Track all evaluated results to return the best
    evaluated = {}

    def evaluate_S(S_val):
        """Evaluate and cache simulation result for given S"""
        S_val = round(S_val)  # Round to integer
        if S_val not in evaluated:
            result = sim(M_factor, S_val, f"M={M_factor}, S={S_val}")
            evaluated[S_val] = result
        return evaluated[S_val]['match_pct']

    begin = S_min
    end = S_max

    while end - begin > 3:
        # Ternary search: divide range into thirds
        low = (begin * 2 + end) // 3
        high = (begin + end * 2) // 3

        if evaluate_S(low) > evaluate_S(high):
            # Maximum is in [begin, high)
            end = high - 1
        else:
            # Maximum is in (low, end]
            begin = low + 1

    # Exhaustively check remaining small range
    for S_val in range(begin, end + 1):
        evaluate_S(S_val)

    # Find best result from all evaluations
    best_S = max(evaluated.keys(), key=lambda s: evaluated[s]['match_pct'])
    best_result = evaluated[best_S]
    best_match = best_result['match_pct']

    return best_S, best_match, best_result


# Ternary search for each M
for M in Mlist:
    print(f"\n{'='*70}")
    print(f"Searching optimal S for M={M}×M_obs")
    print(f"{'='*70}")
    S_best, match_pct, result = ternary_search_S(M)
    results.append(result)
    print(f"\n   → Best S for M={M}: S={S_best:.1f} Gpc, match={match_pct:.2f}%")

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
