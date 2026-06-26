#!/usr/bin/env python3
"""
WS4 Figure Generator — centerM outer-mass hypothesis vs Pantheon+
=================================================================

Produces TWO figures under results/figures/ws4/:

  Fig A (mu_z_panel_<tag>.png)
      Best-config inner-region mu(z) overlaid on the REAL Pantheon+ data,
      with LCDM and EdS reference curves, each annotated with chi2/dof.
      "Best" = minimum chi2/dof vs Pantheon among anchor_ok configs.

  Fig B (inner_at_vs_centerM.png)
      Inner-region a(t) growth and chi2/dof vs centerM for a fixed
      (M, S) in the small-M corner, showing where the inner region is
      undisturbed (centerM=1) and where adding outer mass helps or hurts.

Usage
-----
    python _generate_ws4_figs.py                      # run sweep then plot
    python _generate_ws4_figs.py --from-csv results/ws1_sweep_ws4_centerm.csv
    python _generate_ws4_figs.py --config sweeps/ws4_centerm.json

The sweep is configured by sweeps/ws4_centerm.json by default.
If a CSV already exists it is reused (pass --force-sweep to override).

Grid actually run (reduced from the full JSON to fit in reasonable time):
    M in {1, 2, 5, 20} (hypothesis corner + low-M context)
    centerM in {1.0, 1.5, 2.0, 3.0}
    S: co-fit per M in [18, 40] Gpc
    outer_density_ceiling: 1.0
    particle_count: 400, n_steps: 273, t_start: 2.9 Gyr
"""

from cosmo.encoding import configure_utf8_stdout
configure_utf8_stdout()

import argparse
import csv as _csv
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from cosmo.constants import CosmologicalConstants, SimulationParameters
from cosmo.factories import (
    run_external_node_simulation,
    setup_simulation_context,
    results_to_sim_result,
)
from cosmo.parameter_sweep import (
    SweepConfig, MatchWeights, LCDMBaseline,
    expected_growth_factor, GROWTH_ANCHOR_TOL,
    compute_pantheon_metrics,
)
from cosmo.pantheon import load_pantheon
from cosmo.sim_distance import sim_to_distance_modulus
from cosmo.distances import model_distance_modulus
import cosmo.hubble_diagram as hd_engine
from cosmo.plots import figure_path, plot_mu_z_panel, _footer, _DPI, _BBOX

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TODAY_GYR: float = 13.8
_CHI2_LCDM_REF: float = 0.436
_CHI2_EDS_REF: float = 0.843
_WS = "ws4"

# Reduced sweep grid (covers hypothesis corner + baseline, fits in time)
_DEFAULT_M_VALUES = [1, 2, 5, 20]
_DEFAULT_CENTERM_VALUES = [1.0, 1.5, 2.0, 3.0]
_DEFAULT_S_MIN = 18
_DEFAULT_S_MAX = 40
_DEFAULT_PARTICLES = 400
_DEFAULT_N_STEPS = 273
_DEFAULT_T_START = 2.9

# ---------------------------------------------------------------------------
# Pure helpers (unit-testable, no I/O)
# ---------------------------------------------------------------------------

def select_best_row(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the row with minimum chi2_dof among anchor_ok rows.

    Falls back to all finite-chi2 rows if none are anchor_ok.
    Returns None if rows is empty or all chi2_dof are non-finite.
    """
    def _chi2(r):
        try:
            v = float(r["chi2_dof"])
            return v if math.isfinite(v) else None
        except (KeyError, ValueError, TypeError):
            return None

    ok_rows = [r for r in rows if str(r.get("anchor_ok", "False")).lower() in ("true", "1")]
    pool = ok_rows if ok_rows else rows
    scored = [(v, r) for r in pool if (v := _chi2(r)) is not None]
    if not scored:
        return None
    return min(scored, key=lambda vr: vr[0])[1]


def format_chi2_annotation(chi2_dof: float, label: str) -> str:
    """Return a concise chi2/dof annotation string for figure legends."""
    if math.isfinite(chi2_dof):
        return f"{label}  χ²/dof={chi2_dof:.3f}"
    return f"{label}  χ²/dof=n/a"


def load_sweep_csv(csv_path: str) -> List[Dict[str, Any]]:
    """Load a sweep CSV and return a list of row dicts with float coercion."""
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = _csv.DictReader(fh)
        rows = list(reader)
    return rows


# ---------------------------------------------------------------------------
# Sweep runner (reduced grid)
# ---------------------------------------------------------------------------

def run_ws4_sweep(
    cfg: Dict[str, Any],
    csv_path: str,
) -> List[Dict[str, Any]]:
    """Run the WS4 centerM sweep and write results to csv_path.

    Uses the reduced grid defined at module top. Returns all result rows.
    """
    from sweep import (
        _FixedSweepConfig, expand_grid, _cofit_S_for_cell,
        SWEEP_CSV_COLS, _compute_reference_chi2,
    )
    from cosmo.parameter_sweep import MatchWeights, LCDMBaseline

    t_start = cfg["t_start_Gyr"]
    t_dur = _TODAY_GYR - t_start

    print("[ws4] Computing initial conditions ...")
    box_size_Gpc, a_start, lcdm_result = setup_simulation_context(
        t_start, t_dur, cfg["n_steps"], save_interval=10
    )
    pantheon_data = load_pantheon()
    print(f"[ws4] Loaded {pantheon_data['n']} SNe Ia")

    baseline = LCDMBaseline(
        t_Gyr=lcdm_result["t"],
        size_Gpc=lcdm_result["diameter_Gpc"],
        H_hubble=lcdm_result["H_hubble"],
        size_final_Gpc=lcdm_result["diameter_Gpc"][-1],
        radius_max_Gpc=lcdm_result["diameter_Gpc"][-1] / 2 / math.sqrt(3.0 / 5.0),
        a_final=lcdm_result["a"][-1],
    )
    weights = MatchWeights()

    chi2_lcdm, chi2_eds = _compute_reference_chi2(pantheon_data, t_start)
    print(f"[ws4] Reference: LCDM chi2/dof={chi2_lcdm:.4f}, EdS chi2/dof={chi2_eds:.4f}")

    cells = expand_grid(cfg)
    center_masses = (
        [float(x) for x in cfg["centerM"]]
        if isinstance(cfg["centerM"], list)
        else [float(cfg["centerM"])]
    )
    outer_density_ceilings = [float(x) for x in cfg.get("outer_density_ceilings", [1.0])]

    total = len(cells) * len(center_masses) * len(outer_density_ceilings)
    print(f"\n[ws4] Grid: {len(cells)} M-cells × {len(center_masses)} centerM "
          f"× {len(outer_density_ceilings)} ceiling = {total} outer combos (co-fit S)")

    all_rows: List[Dict] = []
    t0 = time.perf_counter()
    combo_num = 0

    for centerM_val in center_masses:
        for ceiling_val in outer_density_ceilings:
            cell_cfg = dict(cfg)
            cell_cfg["centerM"] = centerM_val
            cell_cfg["outer_density_ceiling"] = ceiling_val

            prev_best_S = None
            for cell in sorted(cells, key=lambda c: c["M"], reverse=True):
                combo_num += 1
                t_cell = time.perf_counter()
                row, best_S = _cofit_S_for_cell(
                    cell, cell_cfg, box_size_Gpc, a_start,
                    pantheon_data, baseline, weights,
                    chi2_lcdm, chi2_eds,
                    prev_best_S=prev_best_S,
                )
                elapsed = time.perf_counter() - t_cell
                prev_best_S = best_S
                all_rows.append(row)
                chi_str = (f"{row['chi2_dof']:.4f}"
                           if math.isfinite(row["chi2_dof"]) else "FAIL")
                print(f"  [{combo_num}/{total}] M={cell['M']:3d}  "
                      f"centerM={centerM_val}  ceil={ceiling_val}  "
                      f"=> S={best_S}  chi2/dof={chi_str}  ({elapsed:.1f}s)")

    total_elapsed = time.perf_counter() - t0
    print(f"\n[ws4] Sweep done: {len(all_rows)} rows in {total_elapsed:.1f}s")

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = _csv.DictWriter(fh, fieldnames=SWEEP_CSV_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[ws4] CSV written: {csv_path}")

    return all_rows


# ---------------------------------------------------------------------------
# Figure A — mu(z) overlay on Pantheon+ for the best config
# ---------------------------------------------------------------------------

def generate_fig_a(
    best_row: Dict[str, Any],
    cfg: Dict[str, Any],
    out_tag: str = "ws4_centerm",
) -> str:
    """Generate Fig A: best-config mu(z) panel over Pantheon+ data.

    Returns path to saved PNG.
    """
    t_start = cfg["t_start_Gyr"]
    t_dur = _TODAY_GYR - t_start

    M = int(float(best_row["M_factor"]))
    S = float(best_row["S_gpc"])
    centerM = float(best_row.get("centerM", 1.0))
    chi2_best = float(best_row.get("chi2_dof", float("nan")))
    anchor_ok = str(best_row.get("anchor_ok", "False")).lower() in ("true", "1")

    print(f"\n[Fig A] Best config: M={M}, S={S:.1f}, centerM={centerM}, "
          f"chi2/dof={chi2_best:.4f}, anchor_ok={anchor_ok}")

    box_size_Gpc, a_start, _ = setup_simulation_context(
        t_start, t_dur, cfg["n_steps"], save_interval=10
    )

    sim_params = SimulationParameters(
        M_value=M,
        S_value=S,
        n_particles=cfg["particle_count"],
        seed=42,
        t_start_Gyr=t_start,
        t_duration_Gyr=t_dur,
        n_steps=cfg["n_steps"],
        damping_factor=None,
        center_node_mass=centerM,
        outer_density_ceiling=float(best_row.get("outer_density_ceiling", 1.0)),
        mass_randomize=0.0,
        node_mass_seed=int(float(best_row.get("node_mass_seed", 42))),
        node_mass_amplitude=float(best_row.get("node_mass_amplitude", 0.0)),
        node_s_amplitude=float(best_row.get("node_s_amplitude", 0.0)),
        init_distribution=str(best_row.get("init_distribution", "uniform_sphere")),
        node_geometry=str(best_row.get("node_geometry", "cube26")),
    )

    print(f"[Fig A] Running sim for best config ...")
    ext = run_external_node_simulation(sim_params, box_size_Gpc, a_start, 10)
    a_curve = ext["a"]
    t_Gyr = ext["t_Gyr"]

    pantheon_data = load_pantheon()
    z = pantheon_data["z"]
    mu_obs = pantheon_data["mu"]
    sigma = pantheon_data["sigma"]

    sim_dist = sim_to_distance_modulus(z, a_curve, t_Gyr, t_start_Gyr=t_start)
    in_range = sim_dist["in_range"]
    z_in = z[in_range]
    mu_in = mu_obs[in_range]
    sg_in = sigma[in_range]
    mu_sim_in = sim_dist["mu"]

    results: Dict[str, Any] = {}
    for model_key in ("external_node_nbody", "lcdm", "einstein_de_sitter"):
        if model_key == "external_node_nbody":
            mu_m = mu_sim_in
        else:
            analytic_name = "lcdm" if model_key == "lcdm" else "einstein_de_sitter"
            mu_m = model_distance_modulus(z_in, analytic_name)
        try:
            ev = hd_engine.evaluate_precomputed(z_in, mu_in, sg_in, mu_m,
                                                 model_name=model_key)
            results[model_key] = ev
        except Exception as exc:
            print(f"  [Fig A] WARNING: {model_key} evaluation failed: {exc}")
            results[model_key] = {"chi2_dof": float("nan"), "R2": float("nan"),
                                   "DeltaM": 0.0, "residuals": None}

    sim_chi2 = results["external_node_nbody"].get("chi2_dof", float("nan"))
    print(f"[Fig A] Re-evaluated chi2/dof from sim: {sim_chi2:.4f}  "
          f"(sweep had {chi2_best:.4f})")

    name = f"mu_z_panel_{out_tag}_M{M}_S{S:.0f}_cM{centerM}"
    path = plot_mu_z_panel(
        sim_dist, results, sim_params, _WS, name,
        data=pantheon_data, in_range_mask=in_range,
    )
    print(f"[Fig A] Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Figure B — inner a(t) growth and chi2/dof vs centerM
# ---------------------------------------------------------------------------

def generate_fig_b(
    rows: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    out_tag: str = "ws4_centerm",
) -> str:
    """Generate Fig B: inner a(t) growth and chi2/dof vs centerM.

    Picks the (M, S) corner with best chi2/dof at centerM=1 (the "small-M"
    baseline) and plots the centerM ladder for that fixed (M, S).

    Returns path to saved PNG.
    """
    t_start = cfg["t_start_Gyr"]
    t_dur = _TODAY_GYR - t_start

    # Find the best (M, S) corner at centerM=1 as the fixed reference axis
    baseline_rows = [
        r for r in rows
        if abs(float(r.get("centerM", 1.0)) - 1.0) < 0.05
        and math.isfinite(float(r.get("chi2_dof", float("nan"))))
    ]
    if not baseline_rows:
        # Fallback: use first valid row's M/S
        baseline_rows = [r for r in rows if math.isfinite(float(r.get("chi2_dof", float("nan"))))]

    if not baseline_rows:
        raise ValueError("No valid rows for Fig B; cannot select (M, S) corner.")

    ref_row = min(baseline_rows, key=lambda r: float(r.get("chi2_dof", float("nan"))))
    ref_M = int(float(ref_row["M_factor"]))
    ref_S = float(ref_row["S_gpc"])
    print(f"\n[Fig B] Fixed (M={ref_M}, S={ref_S:.1f}) — best chi2/dof at centerM=1")

    # Gather unique centerM values from the sweep rows for this (M, S)
    # Use a ladder from config, interpolating sweep results if available.
    centerm_vals_sorted = sorted(set(
        float(r["centerM"]) for r in rows
        if int(float(r.get("M_factor", 0))) == ref_M
        and abs(float(r.get("S_gpc", 0)) - ref_S) < 1.0
        and math.isfinite(float(r.get("chi2_dof", float("nan"))))
    ))

    if not centerm_vals_sorted:
        # Fallback to config list
        centerm_vals_sorted = sorted(
            float(x) for x in (
                cfg["centerM"] if isinstance(cfg["centerM"], list) else [cfg["centerM"]]
            )
        )

    print(f"[Fig B] centerM ladder: {centerm_vals_sorted}")

    # Collect chi2/dof from sweep for each centerM (no re-sim needed)
    chi2_from_sweep: Dict[float, float] = {}
    growth_inner: Dict[float, float] = {}  # a[-1]/a[0] for inner region

    for cm in centerm_vals_sorted:
        matching = [
            r for r in rows
            if int(float(r.get("M_factor", 0))) == ref_M
            and abs(float(r.get("S_gpc", 0)) - ref_S) < 1.0
            and abs(float(r.get("centerM", 0)) - cm) < 0.05
        ]
        if matching:
            best_match = min(matching, key=lambda r: float(r.get("chi2_dof", float("nan"))))
            chi2_from_sweep[cm] = float(best_match.get("chi2_dof", float("nan")))
            growth_inner[cm] = float(best_match.get("growth_factor", float("nan")))
        else:
            chi2_from_sweep[cm] = float("nan")
            growth_inner[cm] = float("nan")

    # Re-run sims for a(t) curves at each centerM (needed for the growth plot)
    print("[Fig B] Running sims for inner a(t) vs centerM ladder ...")
    box_size_Gpc, a_start, _ = setup_simulation_context(
        t_start, t_dur, cfg["n_steps"], save_interval=10
    )

    at_curves: Dict[float, np.ndarray] = {}
    t_Gyr_ref: Optional[np.ndarray] = None

    for cm in centerm_vals_sorted:
        sim_params = SimulationParameters(
            M_value=ref_M,
            S_value=ref_S,
            n_particles=cfg["particle_count"],
            seed=42,
            t_start_Gyr=t_start,
            t_duration_Gyr=t_dur,
            n_steps=cfg["n_steps"],
            damping_factor=None,
            center_node_mass=cm,
            outer_density_ceiling=float(cfg.get("outer_density_ceiling",
                                                 cfg.get("outer_density_ceilings", [1.0])[0])),
            mass_randomize=0.0,
            node_mass_seed=0,
            node_mass_amplitude=0.0,
            node_s_amplitude=0.0,
            init_distribution="uniform_sphere",
            node_geometry="cube26",
        )
        try:
            ext = run_external_node_simulation(sim_params, box_size_Gpc, a_start, 10)
            at_curves[cm] = ext["a"]
            if t_Gyr_ref is None:
                t_Gyr_ref = ext["t_Gyr"]
            print(f"  centerM={cm}: growth={ext['a'][-1]/ext['a'][0]:.4f}  "
                  f"chi2/dof(sweep)={chi2_from_sweep.get(cm, float('nan')):.4f}")
        except Exception as exc:
            print(f"  centerM={cm}: sim FAILED: {exc}")
            at_curves[cm] = None

    # EdS reference a(t)
    t_abs = t_start + (t_Gyr_ref if t_Gyr_ref is not None
                       else np.linspace(0, t_dur, cfg["n_steps"] + 1))
    a_eds_ref = (t_abs / t_start) ** (2.0 / 3.0)
    a_eds_ref /= a_eds_ref[0]

    t_plot = (t_Gyr_ref if t_Gyr_ref is not None
              else np.linspace(0, t_dur, cfg["n_steps"] + 1))

    # Build the figure
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10, 8),
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.15},
    )

    # Colour map across centerM values
    cmap = plt.cm.plasma
    n_cm = len(centerm_vals_sorted)
    colors = [cmap(i / max(n_cm - 1, 1)) for i in range(n_cm)]

    for idx, cm in enumerate(centerm_vals_sorted):
        a_cur = at_curves.get(cm)
        if a_cur is None:
            continue
        # Normalize so a[0] = 1
        a_norm = a_cur / a_cur[0]
        t_abs_all = t_start + t_plot
        lw = 2.5 if cm == 1.0 else 1.8
        ls = "-" if cm == 1.0 else "--"
        label = f"centerM={cm}  growth={a_norm[-1]:.3f}"
        ax_top.plot(t_abs_all, a_norm, color=colors[idx], lw=lw, ls=ls, label=label)

    # EdS reference
    t_abs_eds = t_start + t_plot
    ax_top.plot(t_abs_eds, a_eds_ref, color="green", lw=1.5, ls="-.", label="EdS (analytic)")

    ax_top.set_xlabel("Age of Universe [Gyr]", fontsize=11)
    ax_top.set_ylabel("Inner a(t)  [normalized, a(t_start)=1]", fontsize=11)
    ax_top.set_title(
        f"WS4: Inner-region a(t) vs centerM  (M={ref_M}, S={ref_S:.1f} Gpc)\n"
        f"centerM=1 is the undisturbed baseline; outer matter added at >1",
        fontsize=11,
    )
    ax_top.legend(fontsize=8, loc="upper left")
    ax_top.grid(True, alpha=0.3)

    # Bottom panel: chi2/dof vs centerM with reference lines
    cm_vals_plot = [cm for cm in centerm_vals_sorted
                    if math.isfinite(chi2_from_sweep.get(cm, float("nan")))]
    chi2_vals_plot = [chi2_from_sweep[cm] for cm in cm_vals_plot]

    ax_bot.plot(cm_vals_plot, chi2_vals_plot, "o-", color="#ff7f0e", lw=2.0,
                ms=7, label="Sim χ²/dof vs Pantheon+")
    ax_bot.axhline(_CHI2_LCDM_REF, color="#1f77b4", ls="--", lw=1.8,
                   label=f"ΛCDM ref = {_CHI2_LCDM_REF}")
    ax_bot.axhline(_CHI2_EDS_REF, color="#2ca02c", ls="-.", lw=1.5,
                   label=f"EdS ref = {_CHI2_EDS_REF}")

    ax_bot.set_xlabel("centerM  (outer-mass multiplier)", fontsize=11)
    ax_bot.set_ylabel("χ²/dof vs Pantheon+SH0ES", fontsize=11)
    ax_bot.set_title("χ²/dof vs centerM — inner region undisturbed at centerM=1", fontsize=10)
    ax_bot.legend(fontsize=9, loc="upper right")
    ax_bot.grid(True, alpha=0.3)

    _footer(ax_bot, f"M={ref_M} S={ref_S:.1f}")

    path = figure_path(_WS, f"inner_at_vs_centerM_{out_tag}")
    fig.savefig(path, dpi=_DPI, bbox_inches=_BBOX)
    plt.close(fig)
    print(f"[Fig B] Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Summary printout
# ---------------------------------------------------------------------------

def print_results_summary(rows: List[Dict[str, Any]]) -> None:
    """Print a table of chi2/dof by centerM and the best config."""
    from itertools import groupby

    print("\n" + "=" * 70)
    print("WS4 RESULTS SUMMARY")
    print("=" * 70)
    print(f"  References: LCDM chi2/dof={_CHI2_LCDM_REF}  EdS chi2/dof={_CHI2_EDS_REF}")
    print()

    # Table: centerM vs best chi2/dof (across M and S)
    centerm_vals = sorted(set(float(r.get("centerM", 1.0)) for r in rows))
    print(f"  {'centerM':>8}  {'best chi2/dof':>14}  {'M':>6}  {'S':>6}  {'anchor_ok':>9}")
    print("  " + "-" * 50)
    for cm in centerm_vals:
        cm_rows = [r for r in rows if abs(float(r.get("centerM", 1.0)) - cm) < 0.05
                   and math.isfinite(float(r.get("chi2_dof", float("nan"))))]
        if not cm_rows:
            print(f"  {cm:>8.2f}  {'  (no valid rows)':>14}")
            continue
        best = min(cm_rows, key=lambda r: float(r["chi2_dof"]))
        chi2 = float(best["chi2_dof"])
        M = int(float(best["M_factor"]))
        S = float(best["S_gpc"])
        ok = str(best.get("anchor_ok", "?"))
        print(f"  {cm:>8.2f}  {chi2:>14.4f}  {M:>6}  {S:>6.1f}  {ok:>9}")

    print()
    best_overall = select_best_row(rows)
    if best_overall:
        cm_best = float(best_overall.get("centerM", 1.0))
        chi2_best = float(best_overall["chi2_dof"])
        delta = chi2_best - _CHI2_LCDM_REF
        direction = "CLOSER to LCDM" if delta < 0 else "FURTHER from LCDM"
        print(f"  BEST OVERALL: M={best_overall['M_factor']}, "
              f"S={float(best_overall['S_gpc']):.1f}, "
              f"centerM={cm_best}, chi2/dof={chi2_best:.4f}")
        print(f"  vs LCDM ref {_CHI2_LCDM_REF}: delta={delta:+.4f} ({direction})")

        # Compare centerM=1 baseline
        base_rows = [r for r in rows
                     if abs(float(r.get("centerM", 1.0)) - 1.0) < 0.05
                     and math.isfinite(float(r.get("chi2_dof", float("nan"))))]
        if base_rows:
            base_best = min(base_rows, key=lambda r: float(r["chi2_dof"]))
            base_chi2 = float(base_best["chi2_dof"])
            improvement = base_chi2 - chi2_best
            if cm_best > 1.05 and improvement > 0.005:
                verdict = (f"centerM>{1} DID improve chi2 by {improvement:.4f} "
                           f"vs the centerM=1 floor ({base_chi2:.4f}).")
            elif cm_best > 1.05:
                verdict = (f"centerM>{1} did NOT meaningfully improve chi2 "
                           f"(best={chi2_best:.4f} vs centerM=1 floor={base_chi2:.4f}, "
                           f"delta={improvement:+.4f}). "
                           "Node masses are mean-preserving; outer matter is nearly degenerate.")
            else:
                verdict = f"Best config is at centerM=1 (floor={base_chi2:.4f})."
            print(f"\n  VERDICT: {verdict}")
        else:
            print(f"\n  VERDICT: no centerM=1 baseline rows for comparison.")

    print("=" * 70)


# ---------------------------------------------------------------------------
# Default config for the reduced sweep
# ---------------------------------------------------------------------------

def _make_reduced_cfg(base_cfg: Optional[Dict] = None) -> Dict[str, Any]:
    """Return the reduced sweep config (covers hypothesis corner)."""
    cfg: Dict[str, Any] = {
        "M_values": _DEFAULT_M_VALUES,
        "S_values": "co-fit",
        "s_min_gpc": _DEFAULT_S_MIN,
        "s_max_gpc": _DEFAULT_S_MAX,
        "node_mass_amplitudes": [0.0],
        "node_s_amplitudes": [0.0],
        "node_mass_seeds": [42],
        "init_distributions": ["uniform_sphere"],
        "node_geometries": ["cube26"],
        "geometry_kwargs": {},
        "s_cofit_method": "linear",
        "particle_count": _DEFAULT_PARTICLES,
        "n_steps": _DEFAULT_N_STEPS,
        "t_start_Gyr": _DEFAULT_T_START,
        "centerM": _DEFAULT_CENTERM_VALUES,
        "outer_density_ceilings": [1.0],
        "results_dir": "results",
        "tag": "ws4_centerm",
        "objective": "pantheon",
    }
    if base_cfg is not None:
        # Allow the JSON config to override grid size; always include baseline centerM=1.0
        cfg.update(base_cfg)
        # Ensure centerM is a list and always includes 1.0
        raw_cm = cfg["centerM"]
        cm_list = [float(x) for x in raw_cm] if isinstance(raw_cm, list) else [float(raw_cm)]
        if 1.0 not in cm_list:
            cm_list = [1.0] + cm_list
        cfg["centerM"] = sorted(set(cm_list))
        # Reduce M_values to hypothesis corner for speed (keep M <= 20 from config)
        cfg_M = cfg.get("M_values", _DEFAULT_M_VALUES)
        cfg["M_values"] = [m for m in cfg_M if m <= 20] or _DEFAULT_M_VALUES
    return cfg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="WS4 figure generator — centerM outer-mass vs Pantheon+.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="sweeps/ws4_centerm.json",
                   help="Path to JSON sweep config.")
    p.add_argument("--from-csv", default=None, metavar="PATH",
                   help="Load sweep results from an existing CSV (skip simulation).")
    p.add_argument("--force-sweep", action="store_true",
                   help="Re-run sweep even if CSV already exists.")
    p.add_argument("--out-tag", default="ws4_centerm",
                   help="Tag appended to figure filenames.")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    # Load config
    base_cfg: Optional[Dict] = None
    if os.path.isfile(args.config):
        with open(args.config, encoding="utf-8") as fh:
            base_cfg = json.load(fh)
        print(f"[ws4] Loaded config from {args.config}")
    else:
        print(f"[ws4] Config not found ({args.config}), using reduced defaults.")

    cfg = _make_reduced_cfg(base_cfg)

    tag = cfg.get("tag", "ws4_centerm")
    csv_path = os.path.join(cfg.get("results_dir", "results"), f"ws1_sweep_{tag}.csv")

    # Determine whether to run sweep or load CSV
    if args.from_csv:
        csv_path = args.from_csv
        print(f"[ws4] Loading sweep results from {csv_path}")
        rows = load_sweep_csv(csv_path)
    elif os.path.isfile(csv_path) and not args.force_sweep:
        print(f"[ws4] Found existing CSV: {csv_path}  (use --force-sweep to re-run)")
        rows = load_sweep_csv(csv_path)
    else:
        print(f"[ws4] Running sweep -> {csv_path}")
        rows = run_ws4_sweep(cfg, csv_path)

    if not rows:
        print("[ws4] ERROR: no rows available. Cannot generate figures.", file=sys.stderr)
        return 1

    print_results_summary(rows)

    best_row = select_best_row(rows)
    if best_row is None:
        print("[ws4] ERROR: no valid best row found.", file=sys.stderr)
        return 1

    # Generate figures
    path_a = generate_fig_a(best_row, cfg, out_tag=args.out_tag)
    path_b = generate_fig_b(rows, cfg, out_tag=args.out_tag)

    print(f"\n[ws4] Figure A (mu(z) overlay):           {path_a}")
    print(f"[ws4] Figure B (inner a(t) vs centerM):   {path_b}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
