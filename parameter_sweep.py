#!/usr/bin/env python3
"""
Parameter Sweep - Find Best Fit to ΛCDM
Test multiple External-Node configurations
"""

import numpy as np
import pickle
from cosmo.constants import CosmologicalConstants, ExternalNodeParameters
from cosmo.simulation import CosmologicalSimulation

const = CosmologicalConstants()

PARTICLE_COUNT = 50  # Small for speed
BOX_SIZE = 11.59

print("="*70)
print("PARAMETER SWEEP: Finding Best Match to ΛCDM")
print("="*70)

# First, run ΛCDM baseline
print("\n1. Running ΛCDM baseline...")
sim_lcdm = CosmologicalSimulation(
    n_particles=PARTICLE_COUNT,
    box_size_Gpc=BOX_SIZE,
    use_external_nodes=False
)
sim_lcdm.run(t_end_Gyr=10.0, n_steps=100, save_interval=10)
a_lcdm = sim_lcdm.expansion_history[-1]['scale_factor']
print(f"   ΛCDM final a(t) = {a_lcdm:.4f}")

# Test different configurations
configs = [
    (700, 22, "Lighter/Closer"),
    (750, 23, "Slightly lighter"),
    (800, 24, "Current"),
    (850, 25, "Slightly heavier"),
    (900, 26, "Heavier/Farther"),
]

results = []

for M_factor, S_gpc, desc in configs:
    print(f"\n2. Testing {desc}: M={M_factor}×M_obs, S={S_gpc:.1f}Gpc")
    
    M = M_factor * const.M_observable
    S = S_gpc * const.Gpc_to_m
    params = ExternalNodeParameters(M_ext=M, S=S)
    
    # Run simulation
    sim_ext = CosmologicalSimulation(
        n_particles=PARTICLE_COUNT,
        box_size_Gpc=BOX_SIZE,
        use_external_nodes=True,
        external_node_params=params
    )
    
    sim_ext.run(t_end_Gyr=10.0, n_steps=100, save_interval=10)
    a_ext = sim_ext.expansion_history[-1]['scale_factor']
    
    ratio = a_ext / a_lcdm
    diff_percent = abs(ratio - 1.0) * 100
    
    print(f"   External-Node final a(t) = {a_ext:.4f}")
    print(f"   Ratio (Ext/ΛCDM) = {ratio:.4f} ({diff_percent:.1f}% diff)")
    
    results.append({
        'M_factor': M_factor,
        'S_gpc': S_gpc,
        'desc': desc,
        'a_ext': a_ext,
        'ratio': ratio,
        'diff_percent': diff_percent,
        'params': params
    })

print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

# Sort by closest match
results.sort(key=lambda x: x['diff_percent'])

print(f"\n{'Config':<20} {'M×M_obs':<10} {'S[Gpc]':<10} {'Ratio':<10} {'Diff%':<10}")
print("-" * 70)

for r in results:
    print(f"{r['desc']:<20} {r['M_factor']:<10} {r['S_gpc']:<10.1f} "
          f"{r['ratio']:<10.4f} {r['diff_percent']:<10.2f}")

best = results[0]
print(f"\n BEST MATCH: {best['desc']}")
print(f"   M = {best['M_factor']} × M_obs")
print(f"   S = {best['S_gpc']:.1f} Gpc")
print(f"   Match: {100-best['diff_percent']:.1f}%")

# Save best configuration
with open('./results/best_config.pkl', 'wb') as f:
    pickle.dump(best, f)

print(f"\n Saved best configuration to best_config.pkl")
