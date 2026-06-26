#!/usr/bin/env python3
"""
WS1 — Overarching Multi-Parameter Sweep Tool
=============================================

One config-driven, resumable, cached sweep over ALL parameter axes:
  M, S, node_mass_amplitude, node_s_amplitude, node_mass_seed,
  init_distribution, particle count, node_geometry.

S can be supplied as an explicit grid OR discovered automatically via a
per-M linear-search co-fit on the pantheon objective (recommended).

Outputs
-------
  results/ws1_sweep_<tag>.csv          — tidy results (load_best_config-compatible
                                          superset; includes chi2_lcdm, chi2_eds,
                                          growth_factor, anchor_ok, runaway flag)
  results/figures/ws1/<name>.png       — M-S chi2/growth/runaway/mu(z) figures

CLI
---
  python sweep.py                                     # built-in default config
  python sweep.py --config sweeps/coarse.json         # JSON config file
  python sweep.py --plots-only results/ws1_sweep.csv  # regenerate figures only
  python sweep.py --probe-only                        # time sims, then exit
  python sweep.py --tag my_run                        # custom CSV/figure prefix

Config shape (JSON, all keys optional; omit to use defaults)
------------------------------------------------------------
{
  "M_values":            [100, 500, 1000, 5000],
  "S_values":            "co-fit",          // or list of ints
  "s_min_gpc":           20,
  "s_max_gpc":           80,
  "node_mass_amplitudes":[0.0, 0.5],
  "node_s_amplitudes":   [0.0],
  "node_mass_seeds":     [42],
  "init_distributions":  ["uniform_sphere"],
  "particle_count":      400,
  "n_steps":             273,
  "t_start_Gyr":         2.9,
  "centerM":             1,
  "node_geometries":     ["cube26"],
  "geometry_kwargs":     {},
  "s_cofit_method":      "linear",          // "linear" or "ternary"
  "figures_dir":         "results/figures/ws1",
  "results_dir":         "results",
  "tag":                 "ws1"
}

Columns emitted (CSV superset of _BEST_ISO_COLS)
------------------------------------------------
M_factor, S_gpc, centerM, node_mass_amplitude, node_s_amplitude,
node_mass_seed, init_distribution, node_geometry,
chi2_dof, chi2, chi2_lcdm, chi2_eds,
R2, n_sne_used, growth_factor, growth_target, anchor_ok, runaway,
match_avg_pct, diff_pct
"""

from cosmo.encoding import configure_utf8_stdout
configure_utf8_stdout()

import argparse
import csv
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from cosmo.constants import CosmologicalConstants, SimulationParameters
from cosmo.factories import (
    setup_simulation_context,
    run_external_node_simulation,
    results_to_sim_result,
)
from cosmo.parameter_sweep import (
    SearchMethod, SweepConfig, MatchWeights, SimResult, LCDMBaseline,
    build_cache_name, compute_pantheon_metrics, worst_callback,
    linear_search_S, ternary_search_S, expected_growth_factor, GROWTH_ANCHOR_TOL,
)
from cosmo.pantheon import load_pantheon

const = CosmologicalConstants()

# ---------------------------------------------------------------------------
# Column contracts
# ---------------------------------------------------------------------------

# Columns that must appear in the tidy results CSV.
# The best-isotropic subset (amplitude=0) is load_best_config-compatible:
# hubble_diagram_nbody.py --from-best-config reads M_factor, S_gpc, centerM, chi2_dof.
SWEEP_CSV_COLS = [
    "M_factor", "S_gpc", "centerM", "outer_density_ceiling",
    "node_mass_amplitude", "node_s_amplitude",
    "node_mass_seed", "init_distribution", "node_geometry",
    "chi2_dof", "chi2", "chi2_lcdm", "chi2_eds",
    "R2", "n_sne_used",
    "growth_factor", "growth_target", "anchor_ok", "runaway",
    "match_avg_pct", "diff_pct",
]

# Subset compatible with load_best_config (rows where node_mass_amplitude=0)
BEST_ISO_COLS = [
    "M_factor", "S_gpc", "centerM",
    "chi2_dof", "chi2", "R2",
    "n_sne_used", "growth_factor", "anchor_ok",
    "node_mass_amplitude", "node_mass_seed", "init_distribution",
    "match_avg_pct", "diff_pct",
]

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    # Parameter axes
    "M_values": [100, 200, 500, 1000, 2000, 5000, 10000],
    "S_values": "co-fit",       # "co-fit" => per-M linear-search; or list of ints
    "s_min_gpc": 20,
    "s_max_gpc": 80,
    "node_mass_amplitudes": [0.0, 0.5],
    "node_s_amplitudes": [0.0],
    "node_mass_seeds": [42],
    "init_distributions": ["uniform_sphere"],
    "node_geometries": ["cube26"],
    "geometry_kwargs": {},
    # Fixed physics
    "particle_count": 400,
    "n_steps": 273,
    "t_start_Gyr": 2.9,
    "centerM": 1,             # float or list of floats — outer-mass multiplier (WS4)
    "outer_density_ceilings": [1.0],   # list of outer density ceilings to sweep (WS4)
    # S co-fit method (when S_values=="co-fit")
    "s_cofit_method": "linear",   # "linear" or "ternary"
    # Output
    "results_dir": "results",
    "tag": "ws1",
}

# ---------------------------------------------------------------------------
# SweepConfig subclass that hard-codes particle_count and n_steps from config
# ---------------------------------------------------------------------------

class _FixedSweepConfig(SweepConfig):
    """SweepConfig whose particle_count / n_steps are pinned from the run config dict."""

    def __init__(self, particle_count: int, n_steps: int, **kwargs):
        super().__init__(**kwargs)
        self._particle_count = particle_count
        self._n_steps = n_steps

    @property
    def particle_count(self) -> int:
        return self._particle_count

    @property
    def n_steps(self) -> int:
        return self._n_steps


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: Optional[str]) -> Dict[str, Any]:
    """Load a JSON config and merge with defaults. Returns a complete config dict."""
    cfg = dict(DEFAULT_CONFIG)
    if path is not None:
        with open(path, "r", encoding="utf-8") as f:
            user = json.load(f)
        cfg.update(user)
    return cfg


# ---------------------------------------------------------------------------
# Grid expansion
# ---------------------------------------------------------------------------

def expand_grid(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expand the full factorial grid from cfg, collapsing amplitude=0 runs to a
    SINGLE nm_seed=42 run (seed is a no-op when amplitude==0).

    Each element of the returned list is a 'cell' dict with keys:
        M, amplitude, nm_seed, s_amplitude, init, geometry.
    S is NOT in the cell — it is either supplied from cfg['S_values'] (list)
    or found by per-M co-fit at runtime.

    Returns list of cell dicts.
    """
    amp_list   = cfg["node_mass_amplitudes"]
    seed_list  = cfg["node_mass_seeds"]
    samp_list  = cfg["node_s_amplitudes"]
    init_list  = cfg["init_distributions"]
    geom_list  = cfg["node_geometries"]
    M_list     = cfg["M_values"]

    cells = []
    for M in M_list:
        for geom in geom_list:
            for init in init_list:
                for samp in samp_list:
                    for amp in amp_list:
                        if amp == 0.0:
                            cells.append(dict(M=M, amplitude=0.0, nm_seed=42,
                                              s_amplitude=samp, init=init,
                                              geometry=geom))
                        else:
                            for seed in seed_list:
                                cells.append(dict(M=M, amplitude=amp, nm_seed=seed,
                                                  s_amplitude=samp, init=init,
                                                  geometry=geom))
    return cells


# ---------------------------------------------------------------------------
# Sim callback factory
# ---------------------------------------------------------------------------

def _make_sweep_config_for_cell(cell: Dict, cfg: Dict) -> _FixedSweepConfig:
    return _FixedSweepConfig(
        particle_count=cfg["particle_count"],
        n_steps=cfg["n_steps"],
        quick_search=False,
        many_search=3,
        leet_search=False,
        search_center_mass=False,
        t_start_Gyr=cfg["t_start_Gyr"],
        t_duration_Gyr=13.8 - cfg["t_start_Gyr"],
        damping_factor=None,
        s_min_gpc=cfg["s_min_gpc"],
        s_max_gpc=cfg["s_max_gpc"],
        save_interval=10,
        objective="pantheon",
        node_mass_seed=cell["nm_seed"],
        node_mass_amplitude=cell["amplitude"],
        node_s_amplitude=cell["s_amplitude"],
        init_distribution=cell["init"],
        node_geometry=cell["geometry"],
        geometry_kwargs=cfg.get("geometry_kwargs", {}),
        outer_density_ceiling=cfg.get("outer_density_ceiling", 1.0),
    )


def _make_sim_callback(sweep_cfg: _FixedSweepConfig, box_size_Gpc: float, a_start: float):
    """Return a sim_callback(M, S, centerM, seeds) -> [SimResult]."""

    def _sim(M_factor, S_gpc, centerM, seed):
        sim_params = SimulationParameters(
            M_value=M_factor,
            S_value=S_gpc,
            n_particles=sweep_cfg.particle_count,
            seed=seed,
            t_start_Gyr=sweep_cfg.t_start_Gyr,
            t_duration_Gyr=sweep_cfg.t_duration_Gyr,
            n_steps=sweep_cfg.n_steps,
            damping_factor=sweep_cfg.damping_factor,
            center_node_mass=centerM,
            outer_density_ceiling=getattr(sweep_cfg, "outer_density_ceiling", 1.0),
            mass_randomize=0.0,
            node_mass_seed=sweep_cfg.node_mass_seed,
            node_mass_amplitude=sweep_cfg.node_mass_amplitude,
            node_s_amplitude=getattr(sweep_cfg, "node_s_amplitude", 0.0),
            init_distribution=sweep_cfg.init_distribution,
            node_geometry=getattr(sweep_cfg, "node_geometry", "cube26"),
            geometry_kwargs=getattr(sweep_cfg, "geometry_kwargs", {}),
        )
        ext_results = run_external_node_simulation(
            sim_params, box_size_Gpc, a_start, sweep_cfg.save_interval
        )
        return results_to_sim_result(ext_results, sim_params)

    def sim_callback(M_factor, S_gpc, centerM, seeds):
        return [_sim(M_factor, S_gpc, centerM, s) for s in seeds]

    return sim_callback


# ---------------------------------------------------------------------------
# LCDM / EdS reference chi2 helpers
# ---------------------------------------------------------------------------

def _compute_reference_chi2(pantheon_data: dict, t_start_Gyr: float) -> Tuple[float, float]:
    """Return (chi2_dof_lcdm, chi2_dof_eds) from the analytic curves."""
    from cosmo.distances import model_distance_modulus
    from cosmo.hubble_diagram import evaluate_precomputed

    z = pantheon_data["z"]
    mu_obs = pantheon_data["mu"]
    sigma = pantheon_data["sigma"]

    results = {}
    for model in ("lcdm", "einstein_de_sitter"):
        mu_model = model_distance_modulus(z, model)
        try:
            ev = evaluate_precomputed(z, mu_obs, sigma, mu_model)
            results[model] = ev.get("chi2_dof", float("nan"))
        except Exception:
            results[model] = float("nan")

    return results["lcdm"], results["einstein_de_sitter"]


# ---------------------------------------------------------------------------
# Single-cell runner (one M + one fixed S)
# ---------------------------------------------------------------------------

def _run_cell_fixed_S(
    cell: Dict, S: int, cfg: Dict,
    box_size_Gpc: float, a_start: float,
    pantheon_data: Dict,
    baseline, weights,
    chi2_lcdm: float, chi2_eds: float,
) -> Dict:
    """Run a single (cell, S) combination and return a result row."""
    sweep_cfg = _make_sweep_config_for_cell(cell, cfg)
    sim_cb = _make_sim_callback(sweep_cfg, box_size_Gpc, a_start)
    centerM = cfg["centerM"]

    sim_result, metrics = worst_callback(
        sim_cb, sweep_cfg,
        M_factor=cell["M"], S_val=S, centerM=centerM,
        seeds=[42],
        baseline=baseline,
        weights=weights,
        pantheon_data=pantheon_data,
    )

    growth_factor  = metrics.get("growth_factor", float("nan"))
    growth_target  = metrics.get("growth_target") or expected_growth_factor(cfg["t_start_Gyr"])
    anchor_ok = (
        math.isfinite(growth_factor) and
        abs(growth_factor / growth_target - 1.0) <= GROWTH_ANCHOR_TOL
        if growth_target else False
    )
    runaway = not anchor_ok

    return {
        "M_factor":              cell["M"],
        "S_gpc":                 S,
        "centerM":               centerM,
        "outer_density_ceiling": cfg.get("outer_density_ceiling", 1.0),
        "node_mass_amplitude":   cell["amplitude"],
        "node_s_amplitude":      cell["s_amplitude"],
        "node_mass_seed":        cell["nm_seed"],
        "init_distribution":     cell["init"],
        "node_geometry":         cell["geometry"],
        "chi2_dof":              metrics.get("chi2_dof", float("inf")),
        "chi2":                  metrics.get("chi2", float("inf")),
        "chi2_lcdm":             chi2_lcdm,
        "chi2_eds":              chi2_eds,
        "R2":                    metrics.get("R2", float("nan")),
        "n_sne_used":            metrics.get("n_sne_used", 0),
        "growth_factor":         growth_factor,
        "growth_target":         growth_target,
        "anchor_ok":             anchor_ok,
        "runaway":               runaway,
        "match_avg_pct":         metrics.get("match_avg_pct", 0.0),
        "diff_pct":              metrics.get("diff_pct", 100.0),
    }


# ---------------------------------------------------------------------------
# Per-M S co-fit (the key inner loop)
# ---------------------------------------------------------------------------

def _cofit_S_for_cell(
    cell: Dict, cfg: Dict,
    box_size_Gpc: float, a_start: float,
    pantheon_data: Dict,
    baseline, weights,
    chi2_lcdm: float, chi2_eds: float,
    prev_best_S: Optional[int] = None,
) -> Tuple[Dict, int]:
    """
    Run the per-M linear (or ternary) search for the best S for this cell.

    Returns (best_row_dict, best_S).
    """
    sweep_cfg = _make_sweep_config_for_cell(cell, cfg)
    sim_cb = _make_sim_callback(sweep_cfg, box_size_Gpc, a_start)
    centerM = cfg["centerM"]
    method = cfg.get("s_cofit_method", "linear")

    if method == "ternary":
        best_S, _, best_result_dict, _ = ternary_search_S(
            sweep_cfg, cell["M"], centerM, sim_cb,
            baseline, weights,
            cfg["s_min_gpc"], cfg["s_max_gpc"],
            s_hint=prev_best_S,
            hint_window=(prev_best_S // 4) if prev_best_S else (cfg["s_max_gpc"] // 4),
            seeds=[42],
            pantheon_data=pantheon_data,
        )
        metrics_from_dict = {k: best_result_dict[k] for k in best_result_dict
                             if k not in ("M_factor", "S_gpc", "centerM", "desc",
                                          "a_ext", "size_ext", "params")}
        S = best_S
        raw_metrics = metrics_from_dict
    else:
        # linear_search_S — this is the "was good" method from Stage-3
        best_S, best_result_dict, _, _ = linear_search_S(
            sweep_cfg, cell["M"], centerM, sim_cb,
            baseline, weights,
            cfg["s_min_gpc"],
            prev_best_S if prev_best_S else cfg["s_max_gpc"],
            prev_best_S=prev_best_S,
            seeds=[42],
            pantheon_data=pantheon_data,
        )
        if best_S is None:
            # Search found nothing; fall back to s_min
            best_S = cfg["s_min_gpc"]
            best_result_dict = {"match_avg_pct": 0.0, "diff_pct": 100.0,
                                "chi2_dof": float("inf"), "chi2": float("inf"),
                                "R2": float("nan"), "n_sne_used": 0}
        S = best_S
        raw_metrics = best_result_dict

    growth_factor = raw_metrics.get("growth_factor", float("nan"))
    growth_target = raw_metrics.get("growth_target") or expected_growth_factor(cfg["t_start_Gyr"])
    anchor_ok = (
        math.isfinite(growth_factor) and
        abs(growth_factor / growth_target - 1.0) <= GROWTH_ANCHOR_TOL
    ) if (growth_target and math.isfinite(growth_factor)) else False
    runaway = not anchor_ok

    row = {
        "M_factor":              cell["M"],
        "S_gpc":                 S,
        "centerM":               centerM,
        "outer_density_ceiling": cfg.get("outer_density_ceiling", 1.0),
        "node_mass_amplitude":   cell["amplitude"],
        "node_s_amplitude":      cell["s_amplitude"],
        "node_mass_seed":        cell["nm_seed"],
        "init_distribution":     cell["init"],
        "node_geometry":         cell["geometry"],
        "chi2_dof":              raw_metrics.get("chi2_dof", float("inf")),
        "chi2":                  raw_metrics.get("chi2", float("inf")),
        "chi2_lcdm":             chi2_lcdm,
        "chi2_eds":              chi2_eds,
        "R2":                    raw_metrics.get("R2", float("nan")),
        "n_sne_used":            raw_metrics.get("n_sne_used", 0),
        "growth_factor":         growth_factor,
        "growth_target":         growth_target,
        "anchor_ok":             anchor_ok,
        "runaway":               runaway,
        "match_avg_pct":         raw_metrics.get("match_avg_pct", 0.0),
        "diff_pct":              raw_metrics.get("diff_pct", 100.0),
    }
    return row, S


# ---------------------------------------------------------------------------
# Probe timing
# ---------------------------------------------------------------------------

def probe_timing(
    box_size_Gpc: float, a_start: float,
    pantheon_data: Dict, baseline, weights, cfg: Dict,
    n_probe: int = 5,
) -> float:
    """Time n_probe representative sims and return seconds/sim."""
    import cosmo.parameter_sweep as _ps
    saved_skip = _ps.SKIP_CACHE
    _ps.SKIP_CACHE = True

    probe_cells = [
        dict(M=100, amplitude=0.0, nm_seed=42, s_amplitude=0.0,
             init="uniform_sphere", geometry="cube26"),
        dict(M=500, amplitude=0.0, nm_seed=42, s_amplitude=0.0,
             init="uniform_sphere", geometry="cube26"),
        dict(M=1000, amplitude=0.0, nm_seed=42, s_amplitude=0.0,
             init="uniform_sphere", geometry="cube26"),
        dict(M=5000, amplitude=0.0, nm_seed=42, s_amplitude=0.0,
             init="uniform_sphere", geometry="cube26"),
        dict(M=10000, amplitude=0.0, nm_seed=42, s_amplitude=0.0,
             init="uniform_sphere", geometry="cube26"),
    ][:n_probe]

    S_probe = cfg.get("s_min_gpc", 30)
    t0 = time.perf_counter()
    for cell in probe_cells:
        _run_cell_fixed_S(cell, S_probe, cfg, box_size_Gpc, a_start,
                          pantheon_data, baseline, weights, float("nan"), float("nan"))
    elapsed = time.perf_counter() - t0
    _ps.SKIP_CACHE = saved_skip

    n = len(probe_cells)
    sps = elapsed / max(n, 1)
    print(f"[probe] {n} sims in {elapsed:.1f}s => {sps:.2f} s/sim")
    return sps


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def generate_figures(csv_path: str, cfg: Dict, best_row: Optional[Dict],
                     box_size_Gpc: float, a_start: float,
                     pantheon_data: Optional[Dict]) -> List[str]:
    """
    Generate F1-F4 + F9 figures from the sweep CSV and (optionally) a mu(z) panel
    for the best config.

    Returns list of saved PNG paths.
    """
    import pandas as pd
    from cosmo.plots import (
        plot_ms_heatmap, plot_growth_map, plot_runaway_boundary,
        plot_mu_z_panel, plots_from_csv,
    )

    tag = cfg.get("tag", "ws1")
    ws = "ws1"

    print(f"\n[figures] Reading {csv_path} ...")
    df = pd.read_csv(csv_path)
    saved: List[str] = []

    # F1 — chi2/dof vs Pantheon
    if "chi2_dof" in df.columns:
        p = plot_ms_heatmap(df, "chi2_dof", ws, f"ms_chi2_dof_heatmap_{tag}",
                            title="M-S χ²/dof vs Pantheon+",
                            colorbar_label="χ²/dof", vmin=0.4, vmax=1.2)
        saved.append(p); print(f"  F1 => {p}")

    # F2 — chi2 vs LCDM (as reference value stamped per row)
    if "chi2_lcdm" in df.columns:
        p = plot_ms_heatmap(df, "chi2_lcdm", ws, f"ms_chi2_lcdm_{tag}",
                            title="ΛCDM χ²/dof (reference, stamped per row)",
                            colorbar_label="χ²/dof (ΛCDM)")
        saved.append(p); print(f"  F2 => {p}")

    # F3 — chi2 vs EdS null
    if "chi2_eds" in df.columns:
        p = plot_ms_heatmap(df, "chi2_eds", ws, f"ms_chi2_eds_{tag}",
                            title="EdS null χ²/dof (reference, stamped per row)",
                            colorbar_label="χ²/dof (EdS)")
        saved.append(p); print(f"  F3 => {p}")

    # F4 — growth map
    if "growth_factor" in df.columns:
        t_start = cfg.get("t_start_Gyr", 2.9)
        tgt = expected_growth_factor(t_start)
        p = plot_growth_map(df, ws, f"growth_map_{tag}", target_growth=tgt)
        saved.append(p); print(f"  F4 => {p}")

    # F9 — runaway boundary
    if "anchor_ok" in df.columns:
        p = plot_runaway_boundary(df, ws, f"runaway_boundary_{tag}")
        saved.append(p); print(f"  F9 => {p}")

    # F6 — mu(z) panel for the best config (if we have the data)
    if (best_row is not None and pantheon_data is not None
            and math.isfinite(best_row.get("chi2_dof", float("inf")))):
        _generate_mu_z_panel(best_row, cfg, box_size_Gpc, a_start, pantheon_data,
                             ws, tag, saved)

    return saved


def _generate_mu_z_panel(
    best_row: Dict, cfg: Dict,
    box_size_Gpc: float, a_start: float,
    pantheon_data: Dict,
    ws: str, tag: str, saved: List[str],
):
    """Generate the mu(z) panel for the best config."""
    from cosmo.plots import plot_mu_z_panel
    from cosmo.sim_distance import sim_to_distance_modulus
    import cosmo.hubble_diagram as hd_engine
    from cosmo.distances import model_distance_modulus

    M = int(best_row["M_factor"])
    S = int(best_row["S_gpc"])
    centerM = float(best_row.get("centerM", cfg["centerM"]))
    t_start = cfg["t_start_Gyr"]
    t_dur = 13.8 - t_start

    print(f"  [mu_z] Running best config M={M} S={S} for mu(z) panel ...")
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
        mass_randomize=0.0,
        node_mass_seed=int(best_row.get("node_mass_seed", 42)),
        node_mass_amplitude=float(best_row.get("node_mass_amplitude", 0.0)),
        node_s_amplitude=float(best_row.get("node_s_amplitude", 0.0)),
        init_distribution=str(best_row.get("init_distribution", "uniform_sphere")),
    )
    try:
        ext = run_external_node_simulation(sim_params, box_size_Gpc, a_start, 10)
        a_curve = ext["a"]
        t_Gyr   = ext["t_Gyr"]

        z = pantheon_data["z"]
        mu_obs = pantheon_data["mu"]
        sigma  = pantheon_data["sigma"]

        sim_dist = sim_to_distance_modulus(z, a_curve, t_Gyr, t_start)
        in_range = sim_dist["in_range"]
        z_in   = z[in_range]
        mu_in  = mu_obs[in_range]
        sg_in  = sigma[in_range]
        mu_sim = sim_dist["mu"]

        results: Dict[str, Any] = {}
        for model_key in ("external_node_nbody", "lcdm", "einstein_de_sitter"):
            if model_key == "external_node_nbody":
                mu_m = mu_sim
            else:
                analytic_name = "lcdm" if model_key == "lcdm" else "einstein_de_sitter"
                mu_m = model_distance_modulus(z_in, analytic_name)
            try:
                ev = hd_engine.evaluate_precomputed(z_in, mu_in, sg_in, mu_m)
                results[model_key] = ev
            except Exception:
                results[model_key] = {"chi2_dof": float("nan"), "R2": float("nan"),
                                      "DeltaM": 0.0, "residuals": None}

        name = f"mu_z_panel_M{M}_S{S}_{tag}"
        p = plot_mu_z_panel(
            sim_dist, results, sim_params, ws, name,
            data=pantheon_data, in_range_mask=in_range,
        )
        saved.append(p)
        print(f"  F6 => {p}")
    except Exception as exc:
        print(f"  [mu_z] WARNING: could not generate panel: {exc}")


# ---------------------------------------------------------------------------
# Plots-only mode
# ---------------------------------------------------------------------------

def run_plots_only(csv_path: str, cfg: Dict) -> List[str]:
    """Regenerate all figures from an existing CSV (no sims)."""
    from cosmo.plots import plots_from_csv
    print(f"[plots-only] Reading {csv_path}")
    tag = cfg.get("tag", "ws1")

    import pandas as pd
    df = pd.read_csv(csv_path)
    saved = plots_from_csv(csv_path, workstream="ws1")

    # Additional metrics not covered by plots_from_csv
    from cosmo.plots import plot_ms_heatmap
    for col, label in [("chi2_lcdm", "ΛCDM χ²/dof reference"),
                       ("chi2_eds", "EdS χ²/dof reference")]:
        if col in df.columns:
            p = plot_ms_heatmap(df, col, "ws1", f"ms_{col}_heatmap_{tag}",
                                title=f"{label}", colorbar_label=col)
            saved.append(p)
            print(f"  {col} => {p}")

    print(f"[plots-only] {len(saved)} figures written.")
    return saved


# ---------------------------------------------------------------------------
# Main sweep runner
# ---------------------------------------------------------------------------

def run_sweep(cfg: Dict, probe_only: bool = False) -> Tuple[str, str, List[str]]:
    """
    Run the full overarching sweep.

    Returns (csv_path, best_iso_csv_path, figure_paths).
    """
    print("=" * 70)
    print("OVERARCHING SWEEP (WS1)")
    print("=" * 70)
    M_list    = cfg["M_values"]
    S_mode    = cfg["S_values"]
    cofit     = (S_mode == "co-fit")
    t_start   = cfg["t_start_Gyr"]
    t_dur     = 13.8 - t_start

    print(f"  M grid ({len(M_list)}): {M_list}")
    if cofit:
        print(f"  S: co-fit per M  [{cfg['s_min_gpc']}..{cfg['s_max_gpc']}]  "
              f"method={cfg.get('s_cofit_method','linear')}")
    else:
        print(f"  S grid ({len(S_mode)}): {S_mode}")
    print(f"  amplitudes: {cfg['node_mass_amplitudes']}")
    print(f"  s_amplitudes: {cfg['node_s_amplitudes']}")
    print(f"  seeds: {cfg['node_mass_seeds']}")
    print(f"  inits: {cfg['init_distributions']}")
    print(f"  geometries: {cfg['node_geometries']}")
    # centerM may be a scalar or a list (WS4 sweep axis)
    raw_centerM = cfg["centerM"]
    center_masses: List[float] = (
        [float(x) for x in raw_centerM] if isinstance(raw_centerM, list)
        else [float(raw_centerM)]
    )
    # outer_density_ceilings may be a list (WS4 sweep axis); default [1.0]
    outer_density_ceilings: List[float] = [
        float(x) for x in cfg.get("outer_density_ceilings", [1.0])
    ]
    print(f"  particles={cfg['particle_count']}, n_steps={cfg['n_steps']}, "
          f"t_start={t_start}, centerM={center_masses}, "
          f"outer_density_ceilings={outer_density_ceilings}")

    # Setup
    print("\n[setup] Computing initial conditions and loading Pantheon+ data ...")
    n_steps = cfg["n_steps"]
    box_size_Gpc, a_start, lcdm_result = setup_simulation_context(
        t_start, t_dur, n_steps, save_interval=10
    )
    pantheon_data = load_pantheon()
    print(f"[setup] Loaded {pantheon_data['n']} SNe Ia")

    baseline = LCDMBaseline(
        t_Gyr=lcdm_result['t'],
        size_Gpc=lcdm_result['diameter_Gpc'],
        H_hubble=lcdm_result['H_hubble'],
        size_final_Gpc=lcdm_result['diameter_Gpc'][-1],
        radius_max_Gpc=lcdm_result['diameter_Gpc'][-1] / 2 / math.sqrt(3.0 / 5.0),
        a_final=lcdm_result['a'][-1],
    )
    weights = MatchWeights()

    # Compute reference chi2 values (constant for this run)
    print("[setup] Computing LCDM / EdS reference chi2 ...")
    chi2_lcdm, chi2_eds = _compute_reference_chi2(pantheon_data, t_start)
    print(f"[setup] LCDM chi2/dof={chi2_lcdm:.4f},  EdS chi2/dof={chi2_eds:.4f}")

    # Probe timing
    sps = probe_timing(box_size_Gpc, a_start, pantheon_data, baseline, weights, cfg)
    if probe_only:
        print("[probe] --probe-only: exiting after timing.")
        return "", "", []

    # Expand grid (M / amplitude / geometry / init / s_amplitude combos; NOT centerM/ceiling)
    cells = expand_grid(cfg)
    n_cm = len(center_masses)
    n_ceil = len(outer_density_ceilings)
    if cofit:
        # Each cell => one co-fit run per M
        n_outer = len(cells) * n_cm * n_ceil
        est_inner = max(1, (cfg["s_max_gpc"] - cfg["s_min_gpc"]) // 5)  # rough estimate
        total_est = n_outer * est_inner
        print(f"\n[grid] {len(cells)} cells x {n_cm} centerM x {n_ceil} ceilings "
              f"= {n_outer} outer combos (co-fit S per M; ~{est_inner} S evals each "
              f"=> ~{total_est} total sims; estimated {total_est * sps / 60:.1f} min)")
    else:
        S_list = list(S_mode)
        n_combos = len(cells) * len(S_list) * n_cm * n_ceil
        print(f"\n[grid] {len(cells)} cells x {len(S_list)} S x {n_cm} centerM "
              f"x {n_ceil} ceilings = {n_combos} sims; "
              f"estimated {n_combos * sps / 60:.1f} min")

    # Run
    os.makedirs(cfg["results_dir"], exist_ok=True)
    tag = cfg.get("tag", "ws1")
    csv_path = os.path.join(cfg["results_dir"], f"ws1_sweep_{tag}.csv")

    all_rows: List[Dict] = []
    t_sweep_start = time.perf_counter()

    # Outer loops over centerM and outer_density_ceiling axes (WS4).
    # For each combination, clone cfg with the specific scalar values so that
    # _run_cell_fixed_S / _cofit_S_for_cell read them from cfg as before.
    for centerM_val in center_masses:
        for ceiling_val in outer_density_ceilings:
            cell_cfg = dict(cfg)
            cell_cfg["centerM"] = centerM_val
            cell_cfg["outer_density_ceiling"] = ceiling_val

            if cofit:
                # Group cells by (geometry, init, s_amplitude, amplitude, nm_seed) so we can
                # warm-start the S search across M values (as in the original linear_search).
                # Within each group iterate M in DESCENDING order (matching original approach).
                from itertools import groupby

                def _group_key(c):
                    return (c["geometry"], c["init"], c["s_amplitude"],
                            c["amplitude"], c["nm_seed"])

                # Sort so groupby works
                sorted_cells = sorted(cells, key=_group_key)
                cell_num = 0
                total_cells = len(cells)

                for group_key, group_iter in groupby(sorted_cells, key=_group_key):
                    group = list(group_iter)
                    # Sort descending by M for warm-start
                    group.sort(key=lambda c: c["M"], reverse=True)
                    prev_best_S: Optional[int] = None

                    for cell in group:
                        cell_num += 1
                        t0 = time.perf_counter()
                        row, best_S = _cofit_S_for_cell(
                            cell, cell_cfg, box_size_Gpc, a_start,
                            pantheon_data, baseline, weights,
                            chi2_lcdm, chi2_eds,
                            prev_best_S=prev_best_S,
                        )
                        elapsed = time.perf_counter() - t0
                        prev_best_S = best_S
                        all_rows.append(row)

                        chi_str = (f"{row['chi2_dof']:.4f}"
                                   if math.isfinite(row["chi2_dof"]) else "FAIL")
                        print(
                            f"  [{cell_num}/{total_cells}] M={cell['M']:6d}  "
                            f"centerM={centerM_val}  ceil={ceiling_val}  "
                            f"geo={cell['geometry']:<10s} init={cell['init']:<14s} "
                            f"amp={cell['amplitude']:.2f}  "
                            f"=> best_S={best_S}  chi2/dof={chi_str}  ({elapsed:.1f}s)"
                        )
            else:
                S_list = list(S_mode)
                total = len(cells) * len(S_list)
                i = 0
                for cell in cells:
                    for S in S_list:
                        i += 1
                        t0 = time.perf_counter()
                        row = _run_cell_fixed_S(
                            cell, S, cell_cfg, box_size_Gpc, a_start,
                            pantheon_data, baseline, weights, chi2_lcdm, chi2_eds,
                        )
                        elapsed = time.perf_counter() - t0
                        all_rows.append(row)
                        chi_str = (f"{row['chi2_dof']:.4f}"
                                   if math.isfinite(row["chi2_dof"]) else "FAIL")
                        print(
                            f"  [{i}/{total}] M={cell['M']:6d} S={S:3d}  "
                            f"centerM={centerM_val}  ceil={ceiling_val}  "
                            f"amp={cell['amplitude']:.2f}  "
                            f"=> chi2/dof={chi_str}  ({elapsed:.1f}s)"
                        )

    total_elapsed = time.perf_counter() - t_sweep_start
    n = len(all_rows)
    print(f"\n[sweep] Done: {n} rows in {total_elapsed:.1f}s "
          f"({total_elapsed/max(n,1):.1f} s/sim)")

    # Write full CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SWEEP_CSV_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[out] Full results: {csv_path}  ({n} rows)")

    # Write best-isotropic subset (load_best_config-compatible)
    iso_rows = [r for r in all_rows if r.get("node_mass_amplitude", 0.0) == 0.0]
    best_iso_csv = os.path.join(cfg["results_dir"], f"sweep_results_pantheon_{tag}.csv")
    with open(best_iso_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=BEST_ISO_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(iso_rows)
    print(f"[out] Best-iso CSV: {best_iso_csv}  ({len(iso_rows)} rows)")

    # Best config: objective = minimize chi2/dof vs the real Pantheon+ points.
    # chi2_lcdm / chi2_eds are reference benchmarks only, NOT the selection target.
    finite_rows = [r for r in all_rows if math.isfinite(r["chi2_dof"])]
    bound_rows  = [r for r in finite_rows if r.get("anchor_ok", False)]
    best_row    = min(bound_rows, key=lambda r: r["chi2_dof"]) if bound_rows else (
                  min(finite_rows, key=lambda r: r["chi2_dof"]) if finite_rows else None)

    _print_summary(all_rows, chi2_lcdm, chi2_eds, t_start)

    # Figures
    figs = generate_figures(csv_path, cfg, best_row,
                            box_size_Gpc, a_start, pantheon_data)

    return csv_path, best_iso_csv, figs


# ---------------------------------------------------------------------------
# Summary printout
# ---------------------------------------------------------------------------

def _print_summary(all_rows: List[Dict], chi2_lcdm: float, chi2_eds: float,
                   t_start: float):
    finite = [r for r in all_rows if math.isfinite(r["chi2_dof"])]
    bound  = [r for r in finite if r.get("anchor_ok", False)]
    iso    = [r for r in bound if r.get("node_mass_amplitude", 0.0) == 0.0]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    tgt = expected_growth_factor(t_start)
    print(f"  Reference:  LCDM chi2/dof={chi2_lcdm:.4f}   "
          f"EdS chi2/dof={chi2_eds:.4f}")
    print(f"  Growth target (physical): {tgt:.3f}")

    if bound:
        best = min(bound, key=lambda r: r["chi2_dof"])
        print(f"\n  BEST (all knobs, anchor_ok):")
        print(f"    M={best['M_factor']}, S={best['S_gpc']}, "
              f"amp={best['node_mass_amplitude']:.2f}, "
              f"geo={best['node_geometry']}, init={best['init_distribution']}")
        print(f"    chi2/dof={best['chi2_dof']:.4f}  "
              f"(LCDM={chi2_lcdm:.4f}, EdS={chi2_eds:.4f})")
        print(f"    growth={best.get('growth_factor', float('nan')):.3f}  "
              f"anchor_ok={best.get('anchor_ok')}")

    if iso:
        best_iso = min(iso, key=lambda r: r["chi2_dof"])
        worst_iso = max(iso, key=lambda r: r["chi2_dof"])
        print(f"\n  BEST ISOTROPIC (amp=0, anchor_ok):")
        print(f"    M={best_iso['M_factor']}, S={best_iso['S_gpc']}, "
              f"geo={best_iso['node_geometry']}, init={best_iso['init_distribution']}")
        print(f"    chi2/dof range: {worst_iso['chi2_dof']:.4f} .. {best_iso['chi2_dof']:.4f}")
        print(f"    (LCDM={chi2_lcdm:.4f}, EdS={chi2_eds:.4f})")
        print(f"    growth={best_iso.get('growth_factor', float('nan')):.3f}  R2={best_iso.get('R2', float('nan')):.5f}")

    runaway_count = sum(1 for r in all_rows if r.get("runaway", False))
    print(f"\n  Runaway cells (rejected by growth anchor): {runaway_count}/{len(all_rows)}")

    print("\n  To reproduce:")
    print("    python sweep.py")
    print("  To regenerate figures only:")
    print(f"    python sweep.py --plots-only results/ws1_sweep_ws1.csv")
    print("  To run a bigger sweep:")
    print("    python sweep.py --config sweeps/coarse.json")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser():
    p = argparse.ArgumentParser(
        description="WS1 overarching multi-parameter sweep tool.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=None,
                   help="Path to JSON config file.")
    p.add_argument("--probe-only", action="store_true",
                   help="Time a few sims and exit (no sweep).")
    p.add_argument("--plots-only", default=None, metavar="CSV",
                   help="Regenerate figures from an existing CSV (no sims).")
    p.add_argument("--tag", default=None,
                   help="Override the 'tag' key in config (sets CSV/figure prefix).")
    p.add_argument("--results-dir", default=None,
                   help="Override results directory.")
    return p


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.tag:
        cfg["tag"] = args.tag
    if args.results_dir:
        cfg["results_dir"] = args.results_dir

    if args.plots_only:
        run_plots_only(args.plots_only, cfg)
        sys.exit(0)

    csv_path, best_iso_csv, figs = run_sweep(cfg, probe_only=args.probe_only)

    if figs:
        print("\nFigures written:")
        for p in figs:
            print(f"  {p}")
    if best_iso_csv:
        print(f"\nTo visualize the best config:")
        print(f"  python hubble_diagram_nbody.py --from-best-config {best_iso_csv}")
