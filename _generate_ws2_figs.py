"""
WS2 figure generation script — run once to produce the validation PNGs.

Figures produced:
  F5  results/figures/ws2/eds_overlay_M0.png
      (M=0 sim vs EdS; visual proof of PF1)
  F6  results/figures/ws2/mu_z_panel_M855_S37.8.png
      (Nominal config mu(z) + Delta-mu panel vs Pantheon+)
  F7  results/figures/ws2/shear_vs_nma.png
  F8  results/figures/ws2/dipole_vs_nma.png
      (Anisotropy: uniform vs node_mass_amplitude > 0)

Usage:  python _generate_ws2_figs.py
"""

import math
import os
import sys

# ── headless backend & UTF-8 stdout ──────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")

from cosmo.encoding import configure_utf8_stdout
configure_utf8_stdout()

import numpy as np

from cosmo.constants import SimulationParameters
from cosmo.factories import run_external_node_simulation, setup_simulation_context
from cosmo.anisotropy import anisotropy_summary
from cosmo.plots import (
    figure_path,
    plot_eds_overlay,
    plot_mu_z_panel,
    plot_shear_vs_lever,
    plot_dipole_vs_lever,
)
from cosmo.sim_distance import sim_to_distance_modulus

_TODAY_GYR = 13.8

# ─────────────────────────────────────────────────────────────────────────────
# F5 — M=0 == EdS overlay
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("F5: M=0 EdS overlay")
print("=" * 60)

T_START_F5 = 2.9
T_DUR_F5   = _TODAY_GYR - T_START_F5
N_STEPS_F5 = math.ceil(T_DUR_F5 / 0.04)  # dt ~ 0.040 Gyr

sp_m0 = SimulationParameters(
    M_value=0.0,          # zero external mass => EdS BY CONSTRUCTION
    S_value=37.8,
    n_particles=80,
    seed=42,
    t_start_Gyr=T_START_F5,
    t_duration_Gyr=T_DUR_F5,
    n_steps=N_STEPS_F5,
)

box_size_Gpc, a_start, _ = setup_simulation_context(T_START_F5, T_DUR_F5, N_STEPS_F5)
ext_m0 = run_external_node_simulation(sp_m0, box_size_Gpc, a_start)

a_sim_m0  = ext_m0["a"]
t_Gyr_m0  = ext_m0["t_Gyr"]

out_f5 = plot_eds_overlay(
    a_sim_m0, t_Gyr_m0, t_start_Gyr=T_START_F5,
    workstream="ws2", name="eds_overlay_M0",
    title="M=0 N-body sim vs analytic EdS & ΛCDM  (PF1 visual proof)",
)
print(f"Saved: {out_f5}")


# ─────────────────────────────────────────────────────────────────────────────
# F6 — Per-config mu(z) + Delta-mu panel (nominal M=855, S=37.8)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("F6: mu(z) panel for M=855, S=37.8")
print("=" * 60)

# Try to load Pantheon data; fall back gracefully if absent.
_pantheon_path = os.path.join(
    os.path.dirname(__file__),
    "data", "pantheon_plus", "Pantheon+SH0ES.dat",
)

T_START_F6 = 2.9
T_DUR_F6   = _TODAY_GYR - T_START_F6
N_STEPS_F6 = math.ceil(T_DUR_F6 / 0.04)

sp_nom = SimulationParameters(
    M_value=855.0,
    S_value=37.8,
    n_particles=80,
    seed=42,
    t_start_Gyr=T_START_F6,
    t_duration_Gyr=T_DUR_F6,
    n_steps=N_STEPS_F6,
)

box_size_Gpc, a_start, _ = setup_simulation_context(T_START_F6, T_DUR_F6, N_STEPS_F6)
ext_nom = run_external_node_simulation(sp_nom, box_size_Gpc, a_start)
a_sim_nom = ext_nom["a"]
t_Gyr_nom = ext_nom["t_Gyr"]

if os.path.isfile(_pantheon_path):
    import cosmo.pantheon as _pantheon_mod
    import cosmo.hubble_diagram as _hd

    data = _pantheon_mod.load_pantheon(path=_pantheon_path, z_min=0.01)
    sd_nom = sim_to_distance_modulus(
        data["z"], a_sim_nom, t_Gyr_nom, t_start_Gyr=T_START_F6
    )
    z_in  = sd_nom["z"]
    mu_in_obs = data["mu"][sd_nom["in_range"]]
    sigma_in  = data["sigma"][sd_nom["in_range"]]
    results_nom = {
        "external_node_nbody": _hd.evaluate_precomputed(
            z_in, mu_in_obs, sigma_in, sd_nom["mu"],
            model_name="external_node_nbody",
        ),
        "lcdm": _hd.evaluate_model(z_in, mu_in_obs, sigma_in, model="lcdm"),
        "einstein_de_sitter": _hd.evaluate_model(
            z_in, mu_in_obs, sigma_in, model="einstein_de_sitter"
        ),
    }
    out_f6 = plot_mu_z_panel(
        sd_nom, results_nom, sp_nom,
        workstream="ws2", name="mu_z_panel_M855_S37.8",
        data=data, in_range_mask=sd_nom["in_range"],
    )
else:
    # No Pantheon data — synthetic reference mu(z) as stand-in
    print("  Pantheon file not found; using synthetic mu(z) reference.")
    from cosmo.distances import model_distance_modulus
    z_ref = np.linspace(0.02, 1.0, 100)
    sd_nom = sim_to_distance_modulus(z_ref, a_sim_nom, t_Gyr_nom, t_start_Gyr=T_START_F6)
    mu_obs = model_distance_modulus(sd_nom["z"], "lcdm") + 0.02
    sigma  = np.full_like(mu_obs, 0.15)
    import cosmo.hubble_diagram as _hd
    results_nom = {
        "external_node_nbody": _hd.evaluate_precomputed(
            sd_nom["z"], mu_obs, sigma, sd_nom["mu"],
            model_name="external_node_nbody",
        ),
        "lcdm": _hd.evaluate_model(sd_nom["z"], mu_obs, sigma, model="lcdm"),
        "einstein_de_sitter": _hd.evaluate_model(
            sd_nom["z"], mu_obs, sigma, model="einstein_de_sitter"
        ),
    }
    out_f6 = plot_mu_z_panel(
        sd_nom, results_nom, sp_nom,
        workstream="ws2", name="mu_z_panel_M855_S37.8",
    )
print(f"Saved: {out_f6}")


# ─────────────────────────────────────────────────────────────────────────────
# F7 + F8 — Anisotropy vs node_mass_amplitude
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("F7+F8: Shear & dipole vs node_mass_amplitude")
print("=" * 60)

T_START_ANISO = 5.8
T_DUR_ANISO   = _TODAY_GYR - T_START_ANISO
N_STEPS_ANISO = math.ceil(T_DUR_ANISO / 0.04)  # dt < 0.05 Gyr required

amplitudes   = [0.0, 0.25, 0.5, 0.75, 1.0]
shear_vals   = []
dipole_vals  = []

for amp in amplitudes:
    sp_a = SimulationParameters(
        M_value=855.0,
        S_value=37.8,
        n_particles=80,
        seed=42,
        t_start_Gyr=T_START_ANISO,
        t_duration_Gyr=T_DUR_ANISO,
        n_steps=N_STEPS_ANISO,
        node_mass_amplitude=amp,
        node_mass_seed=7,
    )
    box_size_Gpc, a_start, _ = setup_simulation_context(
        T_START_ANISO, T_DUR_ANISO, N_STEPS_ANISO
    )
    result = run_external_node_simulation(sp_a, box_size_Gpc, a_start)
    sim_obj = result["sim"]

    # Final snapshot positions and velocities from the particle system
    pos_final = sim_obj.particles.get_positions()   # (N, 3) metres
    vel_final = sim_obj.particles.get_velocities()  # (N, 3) m/s

    summary = anisotropy_summary(pos_final, vel_final)
    shear   = summary["shape"]["shear_index"]
    dipole  = abs(summary["hubble_dipole"]["best_dipole"])
    shear_vals.append(shear)
    dipole_vals.append(dipole)
    print(f"  amp={amp:.2f}  shear={shear:.4f}  dipole={dipole:.4f}")

out_f7 = plot_shear_vs_lever(
    amplitudes, shear_vals, "node_mass_amplitude",
    workstream="ws2", name="shear_vs_nma",
    title="Shear index vs node_mass_amplitude  (PF2 visual proof)",
)
out_f8 = plot_dipole_vs_lever(
    amplitudes, dipole_vals, "node_mass_amplitude",
    workstream="ws2", name="dipole_vs_nma",
    title="Hubble dipole ΔH/H vs node_mass_amplitude  (PF2 visual proof)",
)
print(f"Saved: {out_f7}")
print(f"Saved: {out_f8}")

# ─────────────────────────────────────────────────────────────────────────────
# Heatmaps from existing sweep CSV (F1 + F9) — no new sim needed
# ─────────────────────────────────────────────────────────────────────────────
_sweep_csv = os.path.join(os.path.dirname(__file__),
                          "results", "sweep_results_pantheon.csv")
if os.path.isfile(_sweep_csv):
    print("\n" + "=" * 60)
    print("F1+F9: heatmaps from sweep_results_pantheon.csv")
    print("=" * 60)
    from cosmo.plots import plots_from_csv
    out_csv_figs = plots_from_csv(_sweep_csv, workstream="ws2")
    for p in out_csv_figs:
        print(f"Saved: {p}")
else:
    print("\nNo sweep CSV found; skipping F1+F9 heatmaps.")

print("\nAll WS2 figures done.")
