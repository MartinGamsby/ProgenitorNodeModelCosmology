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

    The HMEA grid has 5% position irregularity. An observer at the center
    of a perfect grid sees isotropic expansion. With irregularity, one hemisphere
    has slightly closer/farther nodes -> different tidal acceleration -> different H_0.

    We estimate this analytically from the tidal force formula:
    a_tidal = G*M_ext*r / |r_node - r|^3

    For small offset delta from center:
    DeltaH_0/H_0 ~ (3delta/S) * (Omega_Lambda_eff / (Omega_m + Omega_Lambda_eff))
    """
    print("\n" + "="*70)
    print("PREDICTION 2: Dipole Anisotropy in H_0")
    print("="*70)

    # Best-fit parameters
    M_VALUE = 855
    S_VALUE = 25  # Gpc
    S_m = S_VALUE * const.Gpc_to_m
    M_ext_kg = M_VALUE * const.M_observable_kg

    # HMEA grid irregularity: 5% of S
    irregularity_fraction = 0.05
    delta_S = irregularity_fraction * S_VALUE  # Gpc

    # Current universe size
    R_universe = 14.5 / 2  # Gpc (RMS radius ~7.25 Gpc)

    # Omega_Lambda_eff for this configuration
    H0_si = lcdm.H0_si
    Omega_Lambda_eff = (const.G * M_ext_kg) / (S_m**3 * H0_si**2)

    print(f"\nParameters:")
    print(f"  M = {M_VALUE}*M_obs, S = {S_VALUE} Gpc")
    print(f"  Omega_Lambda_eff = {Omega_Lambda_eff:.4f}")
    print(f"  Grid irregularity: {irregularity_fraction*100:.0f}% -> deltaS = {delta_S:.2f} Gpc")
    print(f"  Universe RMS radius: {R_universe:.2f} Gpc")

    # Tidal acceleration from nearest node at distance (S - R):
    # a_tidal = G*M_ext / (S - R)^2  (leading order for particle at edge)
    #
    # With node displaced by delta toward observer:
    #   Closer hemisphere: d = S - delta  ->  a+ = G*M / (S-delta-R)^2
    #   Farther hemisphere: d = S + delta  ->  a- = G*M / (S+delta-R)^2
    #
    # Fractional difference: Deltaa/a ~ 4delta / (S - R)  (leading order)

    S_minus_R = S_VALUE - R_universe  # Distance from edge to nearest node

    # Conservative: single nearest node displacement
    delta_a_over_a_single = 2 * delta_S / S_minus_R

    # Full grid: 26 nodes partially cancel. Net dipole from irregular positions.
    # RMS displacement of COM of nearest 6 face nodes (each shifted by +/-delta randomly):
    # Net dipole ~ delta/sqrt(6) for 6 nearest nodes
    delta_effective = delta_S / np.sqrt(6)  # Statistical reduction from multiple nodes
    delta_a_over_a_grid = 2 * delta_effective / S_minus_R

    # Convert tidal acceleration asymmetry to H_0 asymmetry
    # H^2 ~ (8piG/3)rho + Lambda/3. The tidal contribution acts like variable Lambda.
    # DeltaH/H ~ (1/2) * (Omega_Lambda_eff / (Omega_m + Omega_Lambda_eff)) * Deltaa_tidal/a_tidal
    omega_factor = Omega_Lambda_eff / (lcdm.Omega_m + Omega_Lambda_eff)

    delta_H_single = 0.5 * omega_factor * delta_a_over_a_single
    delta_H_grid = 0.5 * omega_factor * delta_a_over_a_grid

    # Hubble tension is ~10% (67 vs 73 km/s/Mpc)
    hubble_tension_pct = ((73 - 67) / 70) * 100  # ~8.6%

    print(f"\n--- Single Nearest Node (worst case) ---")
    print(f"  Node displacement: delta = {delta_S:.2f} Gpc")
    print(f"  S - R = {S_minus_R:.2f} Gpc")
    print(f"  Deltaa/a_tidal = {delta_a_over_a_single*100:.2f}%")
    print(f"  DeltaH_0/H_0 = {delta_H_single*100:.3f}%")
    print(f"  DeltaH_0 = {delta_H_single * 70:.2f} km/s/Mpc")

    print(f"\n--- Full Grid (statistical average) ---")
    print(f"  Effective dipole offset: delta_eff = {delta_effective:.3f} Gpc")
    print(f"  DeltaH_0/H_0 = {delta_H_grid*100:.3f}%")
    print(f"  DeltaH_0 = {delta_H_grid * 70:.2f} km/s/Mpc")

    print(f"\n--- Comparison to Hubble Tension ---")
    print(f"  Observed tension: ~{hubble_tension_pct:.1f}% ({73-67} km/s/Mpc)")
    print(f"  Predicted dipole (single node): {delta_H_single*100:.3f}%")
    print(f"  Predicted dipole (full grid): {delta_H_grid*100:.3f}%")

    if delta_H_grid * 100 < 1.0:
        print(f"\n* Predicted dipole ({delta_H_grid*100:.2f}%) is much smaller than Hubble Tension ({hubble_tension_pct:.1f}%)")
        print(f"  External-Node Model does NOT explain the Hubble Tension via simple dipole")
        print(f"  Would need >20% grid irregularity to produce ~1% dipole")
    else:
        print(f"\n* Predicted dipole is comparable to or larger than Hubble Tension")

    # What irregularity would be needed to explain Hubble Tension?
    required_delta = (hubble_tension_pct / 100) / (omega_factor / (S_minus_R)) * np.sqrt(6)
    required_irregularity = required_delta / S_VALUE * 100
    print(f"\n  To explain full Hubble Tension: need ~{required_irregularity:.0f}% grid irregularity")

    return {
        'delta_H_single_pct': delta_H_single * 100,
        'delta_H_grid_pct': delta_H_grid * 100,
        'required_irregularity_pct': required_irregularity
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

    print(f"\n2. Dipole Anisotropy:")
    print(f"   DeltaH_0/H_0 ~ {dipole_results['delta_H_grid_pct']:.2f}% (full grid, 5% irregularity)")
    print(f"   DeltaH_0/H_0 ~ {dipole_results['delta_H_single_pct']:.2f}% (worst-case single node)")
    print(f"   Hubble Tension requires ~{dipole_results['required_irregularity_pct']:.0f}% grid irregularity")

    print(f"\nDone.")
