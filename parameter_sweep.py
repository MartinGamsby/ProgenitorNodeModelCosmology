#!/usr/bin/env python3
"""
Parameter Sweep - Find Best Fit to ΛCDM
Test multiple External-Node configurations
"""

import numpy as np
import pickle
import os
from enum import Enum
from cosmo.constants import CosmologicalConstants, SimulationParameters
from cosmo.simulation import CosmologicalSimulation
from cosmo.analysis import (
    calculate_initial_conditions,
    compare_expansion_histories,
    compare_expansion_history,
    solve_friedmann_at_times,
    calculate_hubble_parameters
)
from cosmo.factories import run_and_extract_results

const = CosmologicalConstants()

# Search method enum
class SearchMethod(Enum):
    BRUTE_FORCE = 1
    TERNARY_SEARCH = 2
    LINEAR_SEARCH = 3

# Configuration defaults
SEARCH_METHOD = SearchMethod.LINEAR_SEARCH
QUICK_SEARCH = False
MANY_SEARCH = True
T_START_GYR = 3.8
T_DURATION_GYR = 10.0
DAMPING_FACTOR = 0.98
PARTICLE_COUNT = 50 if QUICK_SEARCH else 200#300
N_STEPS = 250 if QUICK_SEARCH else 300
SAVE_INTERVAL = 10  # Must match value used in sim() function


print("="*70)
print("PARAMETER SWEEP: Finding Best Match to ΛCDM")
print("="*70)

# Calculate initial conditions once (reused for all configs)
initial_conditions = calculate_initial_conditions(T_START_GYR)
BOX_SIZE = initial_conditions['box_size_Gpc']
A_START = initial_conditions['a_start']

# Test different configurations
configs = []

Mlist = []
M = 10
while M < 500:
    Mlist.append(M)
    M += 1 if MANY_SEARCH else 10
while M < 1000:
    Mlist.append(M)
    M += 5 if MANY_SEARCH else 50
while M < 2000:
    Mlist.append(M)
    M += 10 if MANY_SEARCH else 100
while M < 5000:
    Mlist.append(M)
    M += 100 if MANY_SEARCH else 500
while M < (100000 if MANY_SEARCH else 25000)+1:
    Mlist.append(M)
    M += 1000 if MANY_SEARCH else 10000

Mlist.reverse()

SMin_gpc = 10    # Min box size to test
SMax_gpc = 250   # Max box size to test

Slist = [i for i in range(SMin_gpc, SMax_gpc+1, 1)]
nbConfigs_bruteforce = len(Mlist)*len(Slist)


print(f"Using ternary search on S for each M value...")
print(f"M values to test: {len(Mlist)}")
print(f"S range: [{SMin_gpc}, {SMax_gpc}]")
print(f"(Brute force would test {nbConfigs_bruteforce} configurations)")


# First, solve ΛCDM baseline (analytic solution, not N-body simulation)
print("\n1. Computing ΛCDM baseline...")

# Compute time array matching N-body snapshot times
# N-body saves initial snapshot + every SAVE_INTERVAL steps
snapshot_steps = np.arange(0, N_STEPS + 1, SAVE_INTERVAL)
t_relative_Gyr = (snapshot_steps / N_STEPS) * T_DURATION_GYR
t_absolute_Gyr = T_START_GYR + t_relative_Gyr

# Solve ΛCDM at exact N-body snapshot times
lcdm_solution = solve_friedmann_at_times(t_absolute_Gyr, Omega_Lambda=None)
H_lcdm_hubble = lcdm_solution['H_hubble']

# Extract expansion history
t_lcdm = lcdm_solution['t_Gyr'] - T_START_GYR  # Offset to start at 0
a_lcdm_array = lcdm_solution['a']
size_lcdm_full = BOX_SIZE * (a_lcdm_array / A_START)  # Full resolution curve
a_lcdm = a_lcdm_array[-1]
size_lcdm_final = size_lcdm_full[-1]

# Scale box_size so that the RMS radius matches the target
# For a uniform sphere of radius R, RMS radius = R * sqrt(3/5) ≈ 0.775*R
# We want RMS = size_lcdm_Gpc, so R_sphere = size_lcdm_Gpc / 0.775
# This means we need to use a sphere of radius: size_lcdm_Gpc/2 / sqrt(3/5)
radius_lcdm_max = size_lcdm_final / 2 / np.sqrt(3/5)
print(f"   ΛCDM final a(t) = {a_lcdm:.4f}, size = {size_lcdm_final:.2f} Gpc")

results = []
sim_count = 0  # Track total simulations

def sim(M_factor, S_gpc, desc, seed):
    global sim_count
    sim_count += 1
    print(f"\n2. Testing {desc}: M={M_factor}×M_obs, S={S_gpc:.1f}Gpc (sim #{sim_count})")

    # Create simulation parameters
    sim_params = SimulationParameters(
        M_value=M_factor,
        S_value=S_gpc,
        n_particles=PARTICLE_COUNT,
        seed=seed,
        t_start_Gyr=T_START_GYR,
        t_duration_Gyr=T_DURATION_GYR,
        n_steps=N_STEPS,
        damping_factor=DAMPING_FACTOR
    )

    # Run simulation
    sim_ext = CosmologicalSimulation(sim_params, BOX_SIZE, A_START,
                                      use_external_nodes=True, use_dark_energy=False)
    ext_results = run_and_extract_results(sim_ext, T_DURATION_GYR, N_STEPS, save_interval=SAVE_INTERVAL)
    a_ext = ext_results['a'][-1]
    size_ext_final = ext_results['diameter_Gpc'][-1]
    size_ext_curve = ext_results['diameter_Gpc']  # Full expansion history
    radius_max_final = ext_results['max_radius_Gpc'][-1]
    t_ext = ext_results['t_Gyr']  # Time points from simulation

    hubble_ext = calculate_hubble_parameters(t_ext, ext_results['a'], smooth_sigma=0.0)

    # LCDM curve should now match N-body time points exactly (no interpolation needed)
    size_lcdm_curve = size_lcdm_full


    half_point = len(size_lcdm_curve)//2

    # Calculate match using full curve comparison
    match_curve_pct = compare_expansion_histories(size_ext_curve[half_point:], size_lcdm_curve[half_point:])
    match_end_pct = compare_expansion_history(size_ext_final, size_lcdm_final)
    match_max_pct = compare_expansion_history(radius_max_final, radius_lcdm_max)
    match_hubble_curve_pct = compare_expansion_histories(hubble_ext[half_point:], H_lcdm_hubble[half_point:])

    #match_avg_pct = (match_hubble_curve_pct*1 + match_curve_pct*3 + match_end_pct*5 + match_max_pct*1)/10
    #match_avg_pct = match_curve_pct#TODOOO
    #match_avg_pct = match_end_pct#TODOOOO
    match_avg_pct = (match_hubble_curve_pct*0.1 + match_curve_pct*0.6 + match_end_pct*0.2 + match_max_pct*0.1)
    diff_pct = 100 - match_avg_pct

    print(f"   External-Node final a(t) = {a_ext:.4f}, size = {size_ext_final:.2f} Gpc")
    print(f"   Match: {match_avg_pct:.2f}% (curve {(match_curve_pct):.2f}%, end {(match_end_pct):.2f}%, radius {(match_max_pct):.2f}%, Hubble {(match_hubble_curve_pct):.2f}%)")

    return {
        'M_factor': M_factor,
        'S_gpc': S_gpc,
        'desc': desc,
        'a_ext': a_ext,
        'size_ext': size_ext_final,
        'match_curve_pct': match_curve_pct,
        'match_avg_pct': match_avg_pct,
        'match_end_pct': match_end_pct,
        'match_max_pct': match_max_pct,
        'match_hubble_curve_pct': match_hubble_curve_pct,
        'diff_pct': diff_pct,
        'params': sim_params.external_params
    }

def sim_check(M_factor, S_gpc, desc):
    return sim(M_factor, S_gpc, desc, seed=42)
    # Take the worst of 2 seeds to avoid lucky runs
    result1 = sim(M_factor, S_gpc, desc, seed=42)
    result2 = sim(M_factor, S_gpc, desc, seed=123)
    if result1['match_avg_pct'] < result2['match_avg_pct']:
        return result1
    return result2    

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
            result = sim_check(M_factor, S_val, f"M={M_factor}, S={S_val}")
            results.append(result)
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

if SEARCH_METHOD == SearchMethod.BRUTE_FORCE:
    for M in Mlist:
        for S in Slist:
            desc = f"M={M}×M_obs, S={S}Gpc"
            results.append(sim_check(M, S, desc))
elif SEARCH_METHOD == SearchMethod.TERNARY_SEARCH:
    for M in Mlist:
        print(f"\n{'='*70}")
        print(f"Searching optimal S for M={M}×M_obs")
        print(f"{'='*70}")
        S_best, match_avg_pct, result = ternary_search_S(M, S_hint=prev_best_S, 
                                                    S_max=prev_best_S if prev_best_S else SMax_gpc,
                                                    hint_window=prev_best_S//4 if prev_best_S else SMax_gpc//4)  # Going from high mass to low mass, it needs to be lower
        #results.append(result)
        print(f"\n   → Best S for M={M}: S={S_best:.1f} Gpc, match={match_avg_pct:.2f}%")
        prev_best_S = S_best  # Use as hint for next M
        if S_best == SMin_gpc or S_best == SMax_gpc:
            print("   ⚠️  Warning: Best S is at search boundary. Consider expanding S range for better results.")
            if S_best == SMin_gpc:
                break  # No point in going to lower M if S is already at minimum
elif SEARCH_METHOD == SearchMethod.LINEAR_SEARCH:
    for M in Mlist:
        print(f"\n{'='*70}")
        print(f"Linear search for M={M}×M_obs")
        print(f"{'='*70}")

        S_min = SMin_gpc
        S_max = prev_best_S if prev_best_S else SMax_gpc # prev_best_S+1?

        current_evaluated = []
        S_list = range(S_max, S_min - 1, -1) 
        i = 0
        while i < len(S_list):
            S = S_list[i]
            desc = f"M={M}, S={S}"
            result = sim_check(M, S, desc)
            results.append(result)
            print(f"S={S}, M={M}, Match={result['match_avg_pct']}")

            if result['match_avg_pct'] <= 0:
                print("\tMatch below 0%, stopping search for this M.")
                break

            if current_evaluated:
                if current_evaluated[-1][1]['match_avg_pct'] > result['match_avg_pct']*1.00025:
                    print("\tMatch decreasing > 0.025%, stopping search for this M.")
                    break
                # if almost equal 0, skip one S
                diff = result['match_avg_pct'] - current_evaluated[-1][1]['match_avg_pct']
                print(f"\tMATCH CHANGE: {diff:.4f}%")
                if S > 40:  # Only skip if we have room to skip
                    if abs(diff) < 0.002:
                        print("\tMatch change < 0.002%, skipping S/10 S.")
                        i += int(S/10)
                    elif diff > 0 and diff < 0.01:
                        print("\tMatch change < 0.01%, skipping 2 S.")
                        i += 2
                    elif diff > 0 and diff < 0.02:
                        print("\tMatch change < 0.02%, skipping 1 S.")
                        i += 1
            current_evaluated.append((S, result))
            i += 1
            
        # Find best result from current evaluations
        if prev_best_S == SMin_gpc:
            break  # No point in going to lower M if S is already at minimum
        prev_best_S, best_result = max(current_evaluated, key=lambda x: x[1]['match_avg_pct'])
        
print("\n" + "="*70)

print("RESULTS BY MASS")
print("="*70)

# Sort by best match
results.reverse() # Original order was descending M

print(f"\n{'Config':<20} {'M×M_obs':<10} {'S[Gpc]':<10} {'Match%':<10} {'Diff%':<10} {'Curve%':<10} {'End%':<10} {'Radius%':<10} {'Hubble%':<10}")
print("-" * 70)
for r in results:
    print(f"{r['desc']:<20} {r['M_factor']:<10} {r['S_gpc']:<10.1f} "
          f"{r['match_avg_pct']:<10.2f} {r['diff_pct']:<10.2f} {r['match_curve_pct']:<10.2f} {r['match_end_pct']:<10.2f} {r['match_max_pct']:<10.2f} {r['match_hubble_curve_pct']:<10.2f}")
print("\n" + "="*70)

print("RESULTS SUMMARY")
print("="*70)

# Sort by best match
results.sort(key=lambda x: x['diff_pct'])

print(f"\n{'Config':<20} {'M×M_obs':<10} {'S[Gpc]':<10} {'Match%':<10} {'Diff%':<10} {'Curve%':<10} {'End%':<10} {'Radius%':<10} {'Hubble%':<10}")
print("-" * 70)
for r in results:
    print(f"{r['desc']:<20} {r['M_factor']:<10} {r['S_gpc']:<10.1f} "
          f"{r['match_avg_pct']:<10.2f} {r['diff_pct']:<10.2f} {r['match_curve_pct']:<10.2f} {r['match_end_pct']:<10.2f} {r['match_max_pct']:<10.2f} {r['match_hubble_curve_pct']:<10.2f}")
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
