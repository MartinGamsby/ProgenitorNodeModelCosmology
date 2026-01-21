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
    compare_expansion_histories,
    solve_friedmann_equation
)
from cosmo.factories import run_and_extract_results

const = CosmologicalConstants()

QUICK_SEARCH = False

T_START_GYR = 3.8
T_DURATION_GYR = 10.0
DAMPING_FACTOR = 0.92

# Quick quick, to test the search
if QUICK_SEARCH:
    PARTICLE_COUNT = 20#140
    N_STEPS = 500
else:
    PARTICLE_COUNT = 140
    N_STEPS = 550


print("="*70)
print("PARAMETER SWEEP: Finding Best Match to ΛCDM")
print("="*70)

# Calculate initial conditions once (reused for all configs)
initial_conditions = calculate_initial_conditions(T_START_GYR)
BOX_SIZE = initial_conditions['box_size_Gpc']
A_START = initial_conditions['a_start']

# Test different configurations
configs = []

if QUICK_SEARCH:    
    SMin_gpc = 30   # Min box size to test
    SMax_gpc = 50   # Max box size to test
    #Mlist = [i for i in range(1000, 0, -100)]
    Mlist = [800]
else:
    SMin_gpc = 1   # Min box size to test
    SMax_gpc = 75   # Max box size to test
    #Mlist = [i for i in range(3000, 100, -100)]
    # Fibonacci sequence
    Mlist = [1, 2, 3, 5, 8 ,13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765, 10946]    
    Mlist.reverse()
Slist = [i for i in range(SMin_gpc, SMax_gpc+1, 1)]
nbConfigs_bruteforce = len(Mlist)*len(Slist)


print(f"Using ternary search on S for each M value...")
print(f"M values to test: {len(Mlist)}")
print(f"S range: [{SMin_gpc}, {SMax_gpc}]")
print(f"(Brute force would test {nbConfigs_bruteforce} configurations)")


# First, solve ΛCDM baseline (analytic solution, not N-body simulation)
print("\n1. Computing ΛCDM baseline...")
lcdm_solution = solve_friedmann_equation(
    T_START_GYR,
    T_START_GYR + T_DURATION_GYR,
    Omega_Lambda=None  # Use default LCDM value (0.7)
)
# Extract expansion history
t_lcdm = lcdm_solution['t_Gyr'] - T_START_GYR  # Offset to start at 0
a_lcdm_array = lcdm_solution['a']
size_lcdm_full = BOX_SIZE * (a_lcdm_array / A_START)  # Full resolution curve
a_lcdm = a_lcdm_array[-1]
size_lcdm_final = size_lcdm_full[-1]
print(f"   ΛCDM final a(t) = {a_lcdm:.4f}, size = {size_lcdm_final:.2f} Gpc")

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
        n_steps=N_STEPS,
        damping_factor=DAMPING_FACTOR
    )

    # Run simulation
    sim_ext = CosmologicalSimulation(sim_params, BOX_SIZE, A_START,
                                      use_external_nodes=True, use_dark_energy=False)
    ext_results = run_and_extract_results(sim_ext, T_DURATION_GYR, N_STEPS)
    a_ext = ext_results['a'][-1]
    size_ext_final = ext_results['size_Gpc'][-1]
    size_ext_curve = ext_results['size_Gpc']  # Full expansion history
    t_ext = ext_results['t_Gyr']  # Time points from simulation

    # Interpolate LCDM curve to match External-Node time points
    size_lcdm_curve = np.interp(t_ext, t_lcdm, size_lcdm_full)

    # Calculate match using full curve comparison
    match_curve_pct = compare_expansion_histories(size_ext_curve, size_lcdm_curve)
    match_end_pct = compare_expansion_histories(size_ext_final, size_lcdm_final)
    match_avg_pct = (match_curve_pct*2+match_end_pct)/3
    diff_pct = 100 - match_avg_pct

    print(f"   External-Node final a(t) = {a_ext:.4f}, size = {size_ext_final:.2f} Gpc")
    print(f"   Match: {match_avg_pct:.2f}% (avg diff across all timesteps: {diff_pct:.2f}%)")

    return {
        'M_factor': M_factor,
        'S_gpc': S_gpc,
        'desc': desc,
        'a_ext': a_ext,
        'size_ext': size_ext_final,
        'match_curve_pct': match_curve_pct,
        'match_avg_pct': match_avg_pct,
        'match_end_pct': match_end_pct,
        'diff_pct': diff_pct,
        'params': sim_params.external_params
    }

def ternary_search_S(M_factor, S_min=SMin_gpc, S_max=SMax_gpc, S_hint=None, hint_window=10):
    """
    Ternary search for optimal S given fixed M.
    Assumes unimodal (bell curve) match quality.

    Args:
        M_factor: Mass factor to test
        S_min: Minimum S value
        S_max: Maximum S value
        S_hint: Previous best S (warm start)
        hint_window: Search within ±hint_window of S_hint first

    Returns: (best_S, best_match_pct, best_result_dict)
    """
    # Track all evaluated results to return the best
    evaluated = {}

    def evaluate_S(S_val):
        """Evaluate and cache simulation result for given S"""
        S_val = round(S_val)  # Round to integer
        if S_val not in evaluated:
            result = sim(M_factor, S_val, f"M={M_factor}, S={S_val}")
            #results.append(result)
            evaluated[S_val] = result
        return evaluated[S_val]['match_avg_pct']

    # Warm start: if we have a hint, search locally first
    if S_hint is not None:
        begin = max(S_min, S_hint - hint_window)
        end = min(S_max, S_hint + hint_window)
        print(f"   Warm start: searching [{begin}, {end}] around S={S_hint}")
    else:
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
    best_S = max(evaluated.keys(), key=lambda s: evaluated[s]['match_avg_pct'])
    best_result = evaluated[best_S]
    best_match = best_result['match_avg_pct']

    return best_S, best_match, best_result


# Ternary search for each M
prev_best_S = None


#for M in Mlist:
#    for S in Slist:
#        desc = f"M={M}×M_obs, S={S}Gpc"
#        results.append(sim(M, S, desc))
for M in Mlist:
    print(f"\n{'='*70}")
    print(f"Searching optimal S for M={M}×M_obs")
    print(f"{'='*70}")
    S_best, match_avg_pct, result = ternary_search_S(M, S_hint=prev_best_S, 
                                                 S_max=prev_best_S if prev_best_S else SMax_gpc,
                                                 hint_window=prev_best_S//4 if prev_best_S else SMax_gpc//4)  # Going from high mass to low mass, it needs to be lower
    results.append(result)
    print(f"\n   → Best S for M={M}: S={S_best:.1f} Gpc, match={match_avg_pct:.2f}%")
    prev_best_S = S_best  # Use as hint for next M

print("\n" + "="*70)

print("RESULTS BY MASS")
print("="*70)

# Sort by best match
results.reverse() # Original order was descending M

print(f"\n{'Config':<20} {'M×M_obs':<10} {'S[Gpc]':<10} {'Match%':<10} {'Diff%':<10} {'Curve%':<10}  {'End%':<10} ")
print("-" * 70)
for r in results:
    print(f"{r['desc']:<20} {r['M_factor']:<10} {r['S_gpc']:<10.1f} "
          f"{r['match_avg_pct']:<10.2f} {r['diff_pct']:<10.2f} {r['match_curve_pct']:<10.2f} {r['match_end_pct']:<10.2f}")
print("\n" + "="*70)

print("RESULTS SUMMARY")
print("="*70)

# Sort by best match
results.sort(key=lambda x: x['diff_pct'])

print(f"\n{'Config':<20} {'M×M_obs':<10} {'S[Gpc]':<10} {'Match%':<10} {'Diff%':<10} {'Curve%':<10}  {'End%':<10} ")
print("-" * 70)
for r in results:
    print(f"{r['desc']:<20} {r['M_factor']:<10} {r['S_gpc']:<10.1f} "
          f"{r['match_avg_pct']:<10.2f} {r['diff_pct']:<10.2f} {r['match_curve_pct']:<10.2f} {r['match_end_pct']:<10.2f}")
best = results[0]
print(f"\n★ BEST MATCH: {best['desc']}")
print(f"   M = {best['M_factor']} × M_obs")
print(f"   S = {best['S_gpc']:.1f} Gpc")
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
