#!/usr/bin/env python3
"""
Anisotropy Report — Deliverable B

Runs two External-Node simulations side-by-side:
  (A) UNIFORM masses      (node_mass_amplitude = 0)
  (B) ANISOTROPIC masses  (node_mass_amplitude > 0, user-specified seed)

Computes the four anisotropy diagnostics on the FINAL snapshot for both runs
and prints a comparison table:

  metric           | uniform  | anisotropic | delta
  ---------------------------------------------------------
  shear_index      | ...      | ...         | ...
  max_min_ratio    | ...      | ...         | ...
  hubble_dipole    | ...      | ...         | ...
  expansion_spread | ...      | ...         | ...

Dipole DIRECTION (best_axis) vs node-mass principal axis is also reported —
they should be roughly aligned when amplitude is non-trivial.

Usage
-----
    python anisotropy_report.py
    python anisotropy_report.py --node-mass-seed 7 --node-mass-amplitude 0.5
    python anisotropy_report.py --particles 80 --n-steps 100

"""

import argparse
import sys

# ---------------------------------------------------------------------------
# UTF-8 stdout/stderr reconfiguration (Windows cp1252 guard).
# Must happen BEFORE any print() that might emit Greek characters.
# ---------------------------------------------------------------------------
from cosmo.encoding import configure_utf8_stdout
configure_utf8_stdout()

import numpy as np

from cosmo.cli import add_common_arguments
from cosmo.constants import SimulationParameters
from cosmo.factories import run_external_node_simulation
from cosmo.analysis import calculate_initial_conditions
from cosmo.anisotropy import anisotropy_summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Anisotropy diagnostic: uniform vs anisotropic HMEA node masses."
    )
    add_common_arguments(parser)
    # add_common_arguments already registers --node-mass-seed, --node-mass-amplitude,
    # --t-start, --t-duration, --particles, and --n-steps. Re-adding them would raise
    # argparse.ArgumentError (conflicting option string). Override only the defaults
    # this report needs (notably a NON-ZERO node_mass_amplitude so the anisotropic run
    # actually breaks the lattice symmetry).
    parser.set_defaults(
        node_mass_seed=42,
        node_mass_amplitude=0.5,
        t_start=5.8,
        t_duration=8.0,
        particles=80,
        n_steps=150,
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def _run_sim(
    *,
    t_start_Gyr: float,
    t_duration_Gyr: float,
    n_particles: int,
    n_steps: int,
    seed: int,
    M_value: float,
    S_value: float,
    node_mass_seed: int,
    node_mass_amplitude: float,
) -> dict:
    """Run one External-Node simulation; return factory results dict."""
    ic = calculate_initial_conditions(t_start_Gyr)
    box_size_Gpc = ic["box_size_Gpc"]
    a_start = ic["a_start"]

    sim_params = SimulationParameters(
        M_value=M_value,
        S_value=S_value,
        n_particles=n_particles,
        seed=seed,
        t_start_Gyr=t_start_Gyr,
        t_duration_Gyr=t_duration_Gyr,
        n_steps=n_steps,
        node_mass_seed=node_mass_seed,
        node_mass_amplitude=node_mass_amplitude,
    )

    return run_external_node_simulation(sim_params, box_size_Gpc, a_start)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    M_value = getattr(args, "M", 800)
    S_value = getattr(args, "S", 25.0)
    seed = getattr(args, "seed", 42)

    common = dict(
        t_start_Gyr=args.t_start,
        t_duration_Gyr=args.t_duration,
        n_particles=args.particles,
        n_steps=args.n_steps,
        seed=seed,
        M_value=M_value,
        S_value=S_value,
    )

    print("=" * 60)
    print("ANISOTROPY REPORT — Deliverable B")
    print("=" * 60)
    print(f"  t_start      = {args.t_start} Gyr")
    print(f"  t_duration   = {args.t_duration} Gyr")
    print(f"  n_particles  = {args.particles}")
    print(f"  n_steps      = {args.n_steps}")
    print(f"  M_value      = {M_value}")
    print(f"  S_value      = {S_value}")
    print(f"  aniso seed   = {args.node_mass_seed}")
    print(f"  aniso amp    = {args.node_mass_amplitude}")
    print()

    print("[1/2] Running UNIFORM mass simulation ...")
    res_u = _run_sim(**common, node_mass_seed=0, node_mass_amplitude=0.0)

    print("[2/2] Running ANISOTROPIC mass simulation ...")
    res_a = _run_sim(
        **common,
        node_mass_seed=args.node_mass_seed,
        node_mass_amplitude=args.node_mass_amplitude,
    )

    # Extract initial and final snapshots
    sim_u = res_u["sim"]
    sim_a = res_a["sim"]

    snap0_u = sim_u.snapshots[0]
    snapN_u = sim_u.snapshots[-1]
    snap0_a = sim_a.snapshots[0]
    snapN_a = sim_a.snapshots[-1]

    # Compute diagnostics on final snapshot (and expansion anisotropy)
    diag_u = anisotropy_summary(
        snapN_u["positions"],
        snapN_u["velocities"],
        positions_initial=snap0_u["positions"],
        positions_final=snapN_u["positions"],
    )
    diag_a = anisotropy_summary(
        snapN_a["positions"],
        snapN_a["velocities"],
        positions_initial=snap0_a["positions"],
        positions_final=snapN_a["positions"],
    )

    # ---- Table ----
    def fmt(v: float) -> str:
        return f"{v:.4f}"

    metrics_u = {
        "shear_index":       diag_u["shape"]["shear_index"],
        "max_min_ratio":     diag_u["axis_rms"]["max_min_ratio"],
        "hubble_dipole":     abs(diag_u["hubble_dipole"]["dipole"]),
        "best_dipole":       diag_u["hubble_dipole"]["best_dipole"],
        "expansion_spread":  diag_u["expansion"]["spread"] if diag_u["expansion"] else float("nan"),
    }
    metrics_a = {
        "shear_index":       diag_a["shape"]["shear_index"],
        "max_min_ratio":     diag_a["axis_rms"]["max_min_ratio"],
        "hubble_dipole":     abs(diag_a["hubble_dipole"]["dipole"]),
        "best_dipole":       diag_a["hubble_dipole"]["best_dipole"],
        "expansion_spread":  diag_a["expansion"]["spread"] if diag_a["expansion"] else float("nan"),
    }

    header = f"{'metric':<22}| {'uniform':>10} | {'anisotropic':>12} | {'delta':>10}"
    sep    = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for key in metrics_u:
        vu = metrics_u[key]
        va = metrics_a[key]
        delta = va - vu
        print(f"{key:<22}| {fmt(vu):>10} | {fmt(va):>12} | {fmt(delta):>10}")
    print(sep)

    # ---- Dipole directions ----
    print()
    ba_u = diag_u["hubble_dipole"]["best_axis"]
    ba_a = diag_a["hubble_dipole"]["best_axis"]
    pa_u = diag_u["shape"]["principal_axis"]
    pa_a = diag_a["shape"]["principal_axis"]
    print(f"Uniform    best dipole axis      : [{ba_u[0]:+.3f}, {ba_u[1]:+.3f}, {ba_u[2]:+.3f}]")
    print(f"Anisotropic best dipole axis     : [{ba_a[0]:+.3f}, {ba_a[1]:+.3f}, {ba_a[2]:+.3f}]")
    print(f"Uniform    shape principal axis  : [{pa_u[0]:+.3f}, {pa_u[1]:+.3f}, {pa_u[2]:+.3f}]")
    print(f"Anisotropic shape principal axis : [{pa_a[0]:+.3f}, {pa_a[1]:+.3f}, {pa_a[2]:+.3f}]")
    print()

    if args.particles < 2000:
        print(
            "NOTE: dipole estimates are noisy for N < 2000 particles. "
            "Re-run with --particles 2000 or more for publication-quality numbers."
        )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
