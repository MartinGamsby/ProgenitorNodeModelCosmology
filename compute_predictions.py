#!/usr/bin/env python3
"""
Compute quantitative predictions for paper Section 7.

1. Phantom energy (w < -1): Run extended simulation, measure w(t) deviation
2. Dipole anisotropy: Estimate H_0 asymmetry from HMEA grid irregularity

These predictions distinguish the External-Node Model from LambdaCDM.
"""

import numpy as np
import sys
import os

from cosmo.constants import CosmologicalConstants, LambdaCDMParameters, SimulationParameters
from cosmo.simulation import CosmologicalSimulation
from cosmo.analysis import (
    calculate_initial_conditions,
    solve_friedmann_at_times,
    extract_expansion_history,
    calculate_hubble_parameters
)
from cosmo.factories import run_and_extract_results

const = CosmologicalConstants()
lcdm = LambdaCDMParameters()


def compute_phantom_w():
    """
    Compute effective equation of state w(t) for External-Node Model.

    LambdaCDM has w = -1 exactly (cosmological constant).
    External-Node tidal force scales as ~(S-R)^-2 at large R,
    so w should drift below -1 as R->S ("phantom energy").

    w_eff = -1 - (2/3) * d(ln H) / d(ln a)
    For LambdaCDM: w_eff ~ -0.7 at z=0 (since matter contribution makes w_eff > -1)
    The key test is whether External-Node w_eff drops BELOW LambdaCDM w_eff.
    """
    print("="*70)
    print("PREDICTION 1: Phantom Energy Behavior (w < -1)")
    print("="*70)

    # Best-fit parameters
    M_VALUE = 855
    S_VALUE = 25
    T_START = 3.8

    # Run extended simulation: 3.8 -> 23.8 Gyr (10 Gyr beyond today)
    T_DURATION = 20.0  # 20 Gyr total
    N_STEPS = 600
    N_PARTICLES = 200
    SAVE_INTERVAL = 10

    print(f"\nRunning extended External-Node simulation: t={T_START}->{T_START+T_DURATION} Gyr")
    print(f"Parameters: M={M_VALUE}*M_obs, S={S_VALUE} Gpc, N={N_PARTICLES}")

    sim_params = SimulationParameters(
        M_value=M_VALUE, S_value=S_VALUE,
        n_particles=N_PARTICLES, seed=42,
        t_start_Gyr=T_START, t_duration_Gyr=T_DURATION,
        n_steps=N_STEPS, damping_factor=0.98
    )

    initial_conditions = calculate_initial_conditions(T_START)

    # Run External-Node simulation
    sim_ext = CosmologicalSimulation(
        sim_params, initial_conditions['box_size_Gpc'], initial_conditions['a_start'],
        use_external_nodes=True, use_dark_energy=False
    )
    ext_results = run_and_extract_results(sim_ext, T_DURATION, N_STEPS, save_interval=SAVE_INTERVAL)

    t_ext = ext_results['t_Gyr']
    a_ext = ext_results['a']

    # Compute LambdaCDM baseline at same times
    t_absolute = T_START + t_ext
    lcdm_solution = solve_friedmann_at_times(t_absolute)
    a_lcdm = lcdm_solution['a']

    # Compute H(t) for both models
    H_ext = calculate_hubble_parameters(t_ext, a_ext, smooth_sigma=0.0)
    H_lcdm = lcdm_solution['H_hubble']

    # Compute effective equation of state: w_eff = -1 - (2/3) * d(ln H)/d(ln a)
    # Use central differences, skip edge points
    def compute_w_eff(t_Gyr, a, H):
        """Compute w_eff from H(a) using numerical derivatives."""
        ln_H = np.log(np.abs(H))
        ln_a = np.log(a)

        # Central differences for d(ln H)/d(ln a)
        d_ln_H = np.gradient(ln_H, ln_a)
        w_eff = -1.0 - (2.0/3.0) * d_ln_H
        return w_eff

    w_ext = compute_w_eff(t_ext, a_ext, H_ext)
    w_lcdm = compute_w_eff(t_ext, a_lcdm, H_lcdm)

    # Report at key epochs
    # Skip first and last 10% for edge effects
    n = len(t_ext)
    start_idx = n // 10
    end_idx = n - n // 10

    print(f"\n{'Epoch':<25} {'t_abs [Gyr]':<15} {'w_ext':<12} {'w_LambdaCDM':<12} {'Deltaw':<12}")
    print("-" * 76)

    epochs = {
        'Today (13.8 Gyr)': 13.8,
        'Near future (15 Gyr)': 15.0,
        'Medium future (18 Gyr)': 18.0,
        'Far future (20 Gyr)': 20.0,
        'Very far (23 Gyr)': 23.0,
    }

    for name, t_target in epochs.items():
        t_rel = t_target - T_START
        idx = np.argmin(np.abs(t_ext - t_rel))
        if start_idx <= idx <= end_idx:
            delta_w = w_ext[idx] - w_lcdm[idx]
            print(f"{name:<25} {t_target:<15.1f} {w_ext[idx]:<12.4f} {w_lcdm[idx]:<12.4f} {delta_w:<12.4f}")

    # Find where w_ext first drops significantly below w_lcdm
    w_diff = w_ext[start_idx:end_idx] - w_lcdm[start_idx:end_idx]
    t_abs_window = T_START + t_ext[start_idx:end_idx]

    # Threshold: Deltaw < -0.05 (5% phantom deviation)
    phantom_mask = w_diff < -0.05
    if np.any(phantom_mask):
        first_phantom_idx = np.argmax(phantom_mask)
        t_phantom = t_abs_window[first_phantom_idx]
        w_at_phantom = w_ext[start_idx + first_phantom_idx]
        print(f"\n* Phantom threshold (Deltaw < -0.05) reached at t = {t_phantom:.1f} Gyr")
        print(f"  w_ext = {w_at_phantom:.4f} at that time")
    else:
        print(f"\n* No significant phantom behavior in simulation window")
        print(f"  Max Deltaw = {np.min(w_diff):.4f}")

    # Universe size at end vs S
    diameter_ext_final = ext_results['diameter_Gpc'][-1]
    R_final = diameter_ext_final / 2  # RMS radius
    R_to_S = R_final / S_VALUE
    print(f"\n  Final universe RMS radius: {R_final:.1f} Gpc")
    print(f"  Grid spacing S: {S_VALUE} Gpc")
    print(f"  R/S ratio: {R_to_S:.3f} (phantom effects significant when R/S -> 1)")

    return {
        't_ext': t_ext, 'w_ext': w_ext, 'w_lcdm': w_lcdm,
        'R_final': R_final, 'R_to_S': R_to_S
    }


def compute_dipole_anisotropy():
    """
    Estimate dipole anisotropy in H_0 from HMEA grid irregularity.

    In a virialized meta-structure, nodes have BOTH:
    1. Position irregularity: ~5% deviation from perfect lattice
    2. Mass variation: nodes have different masses (virialized, not identical)

    Both contribute to asymmetric tidal field -> dipole in H_0.

    Tidal acceleration: a_tidal = G*M_node / (S - R)^2
    Position offset changes (S - R), mass variation changes M directly.
    Combined effect adds in quadrature for uncorrelated variations.
    """
    print("\n" + "="*70)
    print("PREDICTION 2: Dipole Anisotropy in H_0")
    print("="*70)

    # Best-fit parameters
    M_VALUE = 855
    S_VALUE = 25  # Gpc
    S_m = S_VALUE * const.Gpc_to_m
    M_ext_kg = M_VALUE * const.M_observable_kg

    # Grid irregularity parameters for virialized structure
    position_irregularity = 0.05  # 5% position deviation
    mass_irregularity = 0.20      # 20% mass variation (virialized structure has diverse node masses)

    delta_S = position_irregularity * S_VALUE  # Gpc
    delta_M_frac = mass_irregularity  # fractional mass variation

    # Current universe size
    R_universe = 14.5 / 2  # Gpc (RMS radius ~7.25 Gpc)

    # Omega_Lambda_eff for this configuration
    H0_si = lcdm.H0_si
    Omega_Lambda_eff = (const.G * M_ext_kg) / (S_m**3 * H0_si**2)

    print(f"\nParameters:")
    print(f"  M_mean = {M_VALUE}*M_obs, S_mean = {S_VALUE} Gpc")
    print(f"  Omega_Lambda_eff = {Omega_Lambda_eff:.4f}")
    print(f"  Position irregularity: {position_irregularity*100:.0f}% -> deltaS = {delta_S:.2f} Gpc")
    print(f"  Mass variation: {mass_irregularity*100:.0f}% -> deltaM/M = {delta_M_frac:.2f}")
    print(f"  Universe RMS radius: {R_universe:.2f} Gpc")

    S_minus_R = S_VALUE - R_universe  # Distance from edge to nearest node

    # === POSITION CONTRIBUTION ===
    # Tidal acceleration: a = GM/(S-R)^2
    # With node displaced by delta: da/a ~ 2*delta/(S-R)
    # For single nearest node:
    delta_a_position_single = 2 * delta_S / S_minus_R

    # For 6 face nodes with random displacements, net dipole ~ delta/sqrt(6)
    delta_a_position_grid = 2 * (delta_S / np.sqrt(6)) / S_minus_R

    # === MASS CONTRIBUTION ===
    # Tidal acceleration: a = GM/(S-R)^2
    # With mass varied by deltaM: da/a = deltaM/M directly
    # For single nearest node:
    delta_a_mass_single = delta_M_frac

    # For 6 face nodes with random mass variations, net dipole ~ deltaM/sqrt(6)
    delta_a_mass_grid = delta_M_frac / np.sqrt(6)

    # === COMBINED EFFECT ===
    # Position and mass variations are uncorrelated, so add in quadrature
    delta_a_combined_single = np.sqrt(delta_a_position_single**2 + delta_a_mass_single**2)
    delta_a_combined_grid = np.sqrt(delta_a_position_grid**2 + delta_a_mass_grid**2)

    # Convert tidal acceleration asymmetry to H_0 asymmetry
    # H^2 ~ (8piG/3)rho + Lambda_eff/3
    # DeltaH/H ~ (1/2) * (Omega_Lambda_eff / (Omega_m + Omega_Lambda_eff)) * Deltaa/a
    omega_factor = Omega_Lambda_eff / (lcdm.Omega_m + Omega_Lambda_eff)

    delta_H_position_single = 0.5 * omega_factor * delta_a_position_single
    delta_H_position_grid = 0.5 * omega_factor * delta_a_position_grid
    delta_H_mass_single = 0.5 * omega_factor * delta_a_mass_single
    delta_H_mass_grid = 0.5 * omega_factor * delta_a_mass_grid
    delta_H_combined_single = 0.5 * omega_factor * delta_a_combined_single
    delta_H_combined_grid = 0.5 * omega_factor * delta_a_combined_grid

    # Hubble tension is ~10% (67 vs 73 km/s/Mpc)
    hubble_tension_pct = ((73 - 67) / 70) * 100  # ~8.6%

    print(f"\n--- Position Irregularity Only (5%) ---")
    print(f"  Single node: DeltaH_0/H_0 = {delta_H_position_single*100:.2f}%")
    print(f"  Full grid:   DeltaH_0/H_0 = {delta_H_position_grid*100:.2f}%")

    print(f"\n--- Mass Variation Only (20%) ---")
    print(f"  Single node: DeltaH_0/H_0 = {delta_H_mass_single*100:.2f}%")
    print(f"  Full grid:   DeltaH_0/H_0 = {delta_H_mass_grid*100:.2f}%")

    print(f"\n--- Combined (Position + Mass, in quadrature) ---")
    print(f"  Single nearest node (worst case):")
    print(f"    DeltaH_0/H_0 = {delta_H_combined_single*100:.2f}%")
    print(f"    DeltaH_0 = {delta_H_combined_single * 70:.2f} km/s/Mpc")
    print(f"  Full grid (statistical average):")
    print(f"    DeltaH_0/H_0 = {delta_H_combined_grid*100:.2f}%")
    print(f"    DeltaH_0 = {delta_H_combined_grid * 70:.2f} km/s/Mpc")

    print(f"\n--- Comparison to Hubble Tension ---")
    print(f"  Observed tension: ~{hubble_tension_pct:.1f}% ({73-67} km/s/Mpc)")
    print(f"  Predicted dipole (combined, single node): {delta_H_combined_single*100:.2f}%")
    print(f"  Predicted dipole (combined, full grid): {delta_H_combined_grid*100:.2f}%")

    # Assess significance
    if delta_H_combined_grid * 100 >= hubble_tension_pct * 0.5:
        print(f"\n* Predicted dipole ({delta_H_combined_grid*100:.1f}%) is significant fraction of Hubble Tension ({hubble_tension_pct:.1f}%)")
        print(f"  External-Node Model predicts measurable H_0 anisotropy")
    else:
        print(f"\n* Predicted dipole ({delta_H_combined_grid*100:.1f}%) is smaller than Hubble Tension ({hubble_tension_pct:.1f}%)")

    # What mass variation would be needed to explain Hubble Tension (with 5% position)?
    # Combined: sqrt(pos^2 + mass^2) = target
    # mass^2 = target^2 - pos^2
    target_delta_a = (hubble_tension_pct / 100) / (0.5 * omega_factor) * np.sqrt(6)
    if target_delta_a**2 > delta_a_position_grid**2:
        required_mass_var = np.sqrt(target_delta_a**2 - delta_a_position_grid**2)
        print(f"\n  To explain full Hubble Tension with 5% position irregularity:")
        print(f"    Need ~{required_mass_var*100:.0f}% mass variation")
    else:
        print(f"\n  Position irregularity alone could explain Hubble Tension")

    return {
        'delta_H_position_single_pct': delta_H_position_single * 100,
        'delta_H_position_grid_pct': delta_H_position_grid * 100,
        'delta_H_mass_single_pct': delta_H_mass_single * 100,
        'delta_H_mass_grid_pct': delta_H_mass_grid * 100,
        'delta_H_combined_single_pct': delta_H_combined_single * 100,
        'delta_H_combined_grid_pct': delta_H_combined_grid * 100,
    }


if __name__ == "__main__":
    print("\n" + "="*70)
    print("COMPUTING QUANTITATIVE PREDICTIONS FOR EXTERNAL-NODE MODEL")
    print("="*70)

    phantom_results = compute_phantom_w()
    dipole_results = compute_dipole_anisotropy()

    print("\n" + "="*70)
    print("SUMMARY OF PREDICTIONS")
    print("="*70)

    print(f"\n1. Phantom Energy:")
    print(f"   R/S ratio at t=23.8 Gyr: {phantom_results['R_to_S']:.3f}")
    print(f"   (Phantom effects grow as R/S -> 1)")

    print(f"\n2. Dipole Anisotropy (5% position + 20% mass variation):")
    print(f"   Position only (grid): {dipole_results['delta_H_position_grid_pct']:.2f}%")
    print(f"   Mass only (grid):     {dipole_results['delta_H_mass_grid_pct']:.2f}%")
    print(f"   Combined (grid):      {dipole_results['delta_H_combined_grid_pct']:.2f}%")
    print(f"   Combined (single):    {dipole_results['delta_H_combined_single_pct']:.2f}% (worst-case)")
    print(f"   Hubble Tension:       ~8.6%")

    print(f"\nDone.")
