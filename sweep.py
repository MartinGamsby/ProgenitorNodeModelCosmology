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
  "grf_support":         "sphere",          // GRF cloud geometry: "sphere" (default) or "box" (legacy)
  "particle_count":      400,
  "n_steps":             273,
  "t_start_Gyr":         2.9,
  "centerM":             1,
  "node_geometries":     ["cube26"],
  "geometry_kwargs":     {},
  "s_cofit_method":      "linear",          // "linear" or "ternary"
  "objective":           "pantheon",        // "pantheon" (chi2 vs SNe) or "lcdm" (R^2 vs LCDM)
  "figures_dir":         "results/figures/ws1",
  "results_dir":         "results",
  "tag":                 "ws1"
}

This is the SINGLE sweep driver. The retired root scripts map onto JSON configs:
  parameter_sweep.py  (lcdm exploration) -> sweeps/lcdm_example.json ("objective":"lcdm")
  pantheon_knob_sweep.py (grf knob grid) -> sweeps/knob_grf.json

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
    # vir_mass_spread is an IDENTITY column: when swept (vir_mass_spreads) it
    # distinguishes cells that share (M,S) but differ in the mass-function width,
    # so it must be recorded AND keyed (else the spreads collide on resume).
    "vir_mass_spread",
    "chi2_dof", "chi2", "chi2_lcdm", "chi2_eds",
    "R2", "n_sne_used",
    "growth_factor", "growth_target", "anchor_ok", "runaway",
    "match_avg_pct", "diff_pct",
    # Observer-from-particle columns (populated only when score_observers is on).
    # When on, chi2_dof above IS the best-observer value (the headline); center_chi2_dof
    # is the centre baseline for comparison; frac_below_* = fraction of observers
    # at/below the LCDM / EdS references ("how typical a good vantage is").
    "best_observer_chi2", "center_chi2_dof", "observer_median_chi2",
    "frac_below_lcdm", "frac_below_eds",
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
    # GRF cloud support geometry (consumed only for init_distribution "grf"):
    # "sphere" (default, WS5 §8 fix: confine to the uniform_sphere radius so only
    # clustering differs) or "box" (legacy GRF-perturbed cube). Keys the grf cache
    # ("sphsup" token for sphere; box keeps the bare grfinit key) -> keyed == run.
    "grf_support": "sphere",
    "node_geometries": ["cube26"],
    "geometry_kwargs": {},
    # Fixed physics
    "particle_count": 400,
    "n_steps": 273,
    "t_start_Gyr": 2.9,
    "centerM": 1,             # float or list of floats — outer-mass multiplier (WS4)
    "outer_density_ceilings": [1.0],   # list of outer density ceilings to sweep (WS4)
    # Node Plummer softening length in Gpc (Section 4 slingshot taming knob).
    # 0.0 (default) keeps the legacy hard 1e10 m floor (byte-identical, no cache
    # slug). Set ~1.0 to tame the runaway slingshot for both cube26 and virialized.
    "node_softening_gpc": 0.0,
    # Start-size lever (Section 6): multiplier on the LCDM-implied initial cloud
    # size. 1.0 (default) = current LCDM-implied size (byte-identical a(t), no cache
    # slug). != 1.0 starts the cloud bigger/smaller relative to the FIXED node
    # spacing S, changing the tidal shear and hence the a(t) SHAPE (falsifiable).
    "start_size_scale": 1.0,
    # S co-fit method (when S_values=="co-fit")
    "s_cofit_method": "linear",   # "linear" or "ternary"
    # Scoring objective for the per-cell worst_callback:
    #   "pantheon" (default) -> chi2/dof vs the REAL Pantheon+ SNe (headline mode).
    #   "lcdm"               -> R^2-style match vs the analytic LCDM baseline
    #                           (compute_match_metrics; match_avg_pct / diff_pct).
    # The objective is threaded onto the SweepConfig, so build_cache_name stamps a
    # "<objective>obj" slug (lcdm and pantheon caches never collide) and
    # worst_callback picks the right scorer. Folded in from the retired root
    # parameter_sweep.py so its lcdm exploration mode is not lost.
    "objective": "pantheon",
    # Observer-from-particle scoring (opt-in). When True, each cell ALSO scores a
    # sample of per-particle observers; the BEST observer becomes the headline
    # chi2_dof (the co-fit + best-cell selection optimize on it), center_chi2_dof
    # keeps the centre baseline, and frac_below_lcdm/eds report how typical a good
    # vantage is. Default False (centre-only, byte-identical) so existing
    # sweeps/tests are unaffected; the real sweep configs set it True.
    "score_observers": False,
    "observer_definition": "local_rms",  # or "hubble_flow"
    "observer_sample": 128,              # observers scored per cell (strided sample)
    "observer_k": -1,                    # neighbours per observer (-1 = whole cloud)
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
    SINGLE nm_seed=42 run (seed is a no-op when amplitude==0)...

    ... EXCEPT for the virialized massfunc rule with a non-zero mass spread, where
    node_mass_seed MATERIALLY changes the realized grid even at amplitude==0 (it
    drives the log-normal mass draw + segregation permutation in
    cosmo/node_geometry.py:583-586). For that case we do NOT collapse: one cell is
    emitted per seed so a multi-seed config actually sweeps distinct realizations
    (paired with the matching virseed cache token in build_cache_name -> keyed ==
    run). The collapse is preserved for every genuine no-op case (radial rule,
    spread==0, non-virialized geometry) so those keys/cells are byte-identical.

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
    # Config-wide virialized mass-function knobs (mirror _make_sweep_config_for_cell
    # defaults). Only when massfunc + spread>0 does the seed matter at amp=0.
    vir_mass_rule   = cfg.get("vir_mass_rule", "radial")
    # vir_mass_spread (the node mass-function "distribution size") can be SWEPT via the
    # plural "vir_mass_spreads" list; otherwise the scalar "vir_mass_spread" is used.
    # Only virialized geometry consumes it, so the spread axis is applied to virialized
    # cells ONLY (other geoms get the single scalar -> no redundant cells / no cache
    # collisions). Each distinct spread keys the cache ("<spread>vsp") AND reaches the
    # sim via _make_sweep_config_for_cell (keyed == run).
    default_spread = cfg.get("vir_mass_spread", 0.0)
    swept_spreads = cfg.get("vir_mass_spreads")  # optional list -> sweep this axis

    cells = []
    for M in M_list:
        for geom in geom_list:
            spread_vals = (list(swept_spreads)
                           if (geom == "virialized" and swept_spreads)
                           else [default_spread])
            for spread in spread_vals:
                # The seed matters (drives the log-normal mass draw) only for the
                # virialized massfunc rule with THIS spread > 0.
                seed_matters_at_amp0 = (geom == "virialized"
                                        and vir_mass_rule == "massfunc"
                                        and spread > 0.0)
                for init in init_list:
                    for samp in samp_list:
                        for amp in amp_list:
                            cell_extra = dict(geometry=geom, vir_spread=spread)
                            if amp == 0.0:
                                # The seed is a genuine no-op for non-virialized geoms,
                                # the radial rule, or spread==0 -> collapse to seed 42.
                                # For virialized+massfunc+spread>0 the seed reshapes the
                                # grid -> emit one cell per seed (the B3a fix).
                                if seed_matters_at_amp0:
                                    for seed in seed_list:
                                        cells.append(dict(M=M, amplitude=0.0,
                                                          nm_seed=seed,
                                                          s_amplitude=samp, init=init,
                                                          **cell_extra))
                                else:
                                    cells.append(dict(M=M, amplitude=0.0, nm_seed=42,
                                                      s_amplitude=samp, init=init,
                                                      **cell_extra))
                            else:
                                for seed in seed_list:
                                    cells.append(dict(M=M, amplitude=amp, nm_seed=seed,
                                                      s_amplitude=samp, init=init,
                                                      **cell_extra))
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
        # Observer-from-particle scoring (opt-in via the config). When on, the scorer
        # makes the BEST observer the headline chi2_dof + reports the fraction of
        # observers below the LCDM/EdS refs (set run-wide on cfg by the driver).
        score_observers=cfg.get("score_observers", False),
        observer_definition=cfg.get("observer_definition", "local_rms"),
        observer_sample=cfg.get("observer_sample", 128),
        observer_k=cfg.get("observer_k", -1),
        lcdm_ref=cfg.get("lcdm_ref"),
        eds_ref=cfg.get("eds_ref"),
        # Objective threaded from config (default "pantheon"). When "lcdm" the
        # per-cell worst_callback scores against the analytic LCDM baseline; the
        # cache slug includes "<objective>obj" so the two never collide.
        objective=cfg.get("objective", "pantheon"),
        node_mass_seed=cell["nm_seed"],
        node_mass_amplitude=cell["amplitude"],
        node_s_amplitude=cell["s_amplitude"],
        init_distribution=cell["init"],
        # GRF support geometry (WS5 §8). Consumed only when init_distribution ==
        # "grf"; threaded here so build_cache_name (keys off the SweepConfig) and the
        # actual sim (SimulationParameters init_kwargs, see _build_sim_params) agree
        # -> keyed == run. Default "sphere" mirrors the sample_grf default and adds NO
        # cache token for the legacy "box" support (so a pre-fix box cache stays valid
        # as box); "sphere" appends the "sphsup" discriminator so the new default
        # recomputes instead of reusing the old box cache.
        grf_support=cfg.get("grf_support", "sphere"),
        node_geometry=cell["geometry"],
        geometry_kwargs=cfg.get("geometry_kwargs", {}),
        outer_density_ceiling=cfg.get("outer_density_ceiling", 1.0),
        # Virialized COUPLED-grid params. Consumed only when node_geometry ==
        # "virialized"; threaded here so build_cache_name (which reads them off the
        # SweepConfig) and the actual sim (SimulationParameters, see _make_sim_callback)
        # agree. Defaults mirror SweepConfig / SimulationParameters exactly, so
        # non-virialized configs are byte-identical.
        vir_n_nodes=cfg.get("vir_n_nodes", 26),
        vir_extent=cfg.get("vir_extent", 1.0),
        vir_mass_rule=cfg.get("vir_mass_rule", "radial"),
        # Swept per-cell when "vir_mass_spreads" is given (expand_grid puts it in the
        # cell as vir_spread); else the config-wide scalar. Keys the cache ("<>vsp")
        # and reaches build_virialized_grid -> keyed == run.
        vir_mass_spread=cell.get("vir_spread", cfg.get("vir_mass_spread", 0.0)),
        vir_segregation=cfg.get("vir_segregation", 1.0),
        vir_s_metric=cfg.get("vir_s_metric", "median"),
        vir_relax_steps=cfg.get("vir_relax_steps", 1),
        # Extent->node-count coupling (Section 6 / item 10). Threaded here so
        # build_cache_name (keys off the SweepConfig) and the actual sim
        # (SimulationParameters, see _make_sim_callback) agree -> keyed == run.
        # Default False mirrors SweepConfig/SimulationParameters (byte-identical,
        # no cache slug). When True a bigger vir_extent auto-raises the node count.
        vir_extent_couples_nodes=cfg.get("vir_extent_couples_nodes", False),
        # Node-softening (Section 4 slingshot taming knob). Threaded here so
        # build_cache_name (keys off the SweepConfig) and the actual sim
        # (SimulationParameters, see _make_sim_callback) agree -> keyed == run.
        # Default 0.0 mirrors SweepConfig/SimulationParameters (byte-identical).
        node_softening_gpc=cfg.get("node_softening_gpc", 0.0),
        # Close-range tidal force law + adaptive KDK sub-stepping (Section 4).
        # Threaded here so build_cache_name (keys off the SweepConfig) and the
        # actual sim (SimulationParameters, see _make_sim_callback) agree ->
        # keyed == run. Defaults ("plummer" / 0.0 / 1) mirror SweepConfig /
        # SimulationParameters and add NO cache slug (byte-identical). "bounded"
        # selects the regularized "can't cross the midpoint" close-range law;
        # node_substep_threshold>0 AND node_substeps>1 enable adaptive sub-stepping
        # during close node passes. WITHOUT this threading these config axes would
        # be silently ignored (the SweepConfig defaults would key AND run plummer /
        # no-substep), so the comparison sweep's bounded-law row would be a dead axis.
        node_force_law=cfg.get("node_force_law", "plummer"),
        node_substep_threshold=cfg.get("node_substep_threshold", 0.0),
        node_substeps=cfg.get("node_substeps", 1),
        # Virialized RELAXATION MODE (Section 2 Option A vs Option B). Threaded here
        # so build_cache_name (keys off the SweepConfig) and the actual sim
        # (SimulationParameters -> ExternalNodeParameters.build_virialized) agree ->
        # keyed == run. "lattice" (default, Option A) is byte-identical and adds no
        # slug; "gradient" (Option B) is the TRUE iterative relaxation of a realistic
        # segregated blob (vir_relax_steps becomes the iteration count). vir_relax_rate
        # / vir_hold_outer_frac tune the gradient descent (used only in gradient mode).
        # WITHOUT this threading Option B was UNREACHABLE from a sweep config (the sim
        # path always ran "lattice"), so the A-vs-B comparison would have been invalid.
        vir_relax_mode=cfg.get("vir_relax_mode", "lattice"),
        vir_relax_rate=cfg.get("vir_relax_rate", 0.1),
        vir_hold_outer_frac=cfg.get("vir_hold_outer_frac", 0.3),
        # Start-size lever (Section 6). Threaded here so build_cache_name (keys off
        # the SweepConfig) and the actual sim (SimulationParameters, see
        # _make_sim_callback) agree -> keyed == run. Default 1.0 mirrors
        # SweepConfig/SimulationParameters (byte-identical, no cache slug).
        start_size_scale=cfg.get("start_size_scale", 1.0),
    )


def _build_sim_params(
    sweep_cfg: _FixedSweepConfig, M_factor, S_gpc, centerM, seed,
) -> SimulationParameters:
    """Build the SimulationParameters for one (M, S, centerM, seed) of a cell.

    SINGLE SOURCE OF TRUTH for "the params a sweep cell actually runs with".
    Every consumer that must reproduce a cell's exact simulation — the sim
    callback (`_make_sim_callback`, the real run that feeds the CSV chi2) AND the
    mu(z) figure panel (`_generate_mu_z_panel`) — goes through here. This closes
    the figure<->CSV chi2 conflict: the panel can never again omit a knob
    (geometry, geometry_kwargs, vir_*, node_softening_gpc, start_size_scale) that
    the real run threaded, which would otherwise re-run a different a(t).

    The cell-specific axes (M, S, centerM, seed, amplitudes, init, geometry) ride
    in via `sweep_cfg` built by `_make_sweep_config_for_cell`; the config-wide
    knobs (geometry_kwargs, vir_*, softening, start-size) live on `sweep_cfg` too.
    Defaults mirror SimulationParameters/SweepConfig, so non-virialized,
    default-knob cells stay byte-identical.
    """
    # GRF support (WS5 §8): for grf runs, EXPLICITLY thread the support that keys the
    # cache into the sampler via init_kwargs={"support": ...} so the value the cache
    # encodes is the value the sampler actually uses (keyed == run, no reliance on the
    # sample_grf implicit default). For uniform_sphere, leave init_kwargs as None so
    # the run is byte-identical (the sampler ignores support).
    init_kwargs = None
    if sweep_cfg.init_distribution == "grf":
        init_kwargs = {"support": getattr(sweep_cfg, "grf_support", "sphere")}
    return SimulationParameters(
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
        # GRF support reaches the sampler via init_kwargs (None for uniform_sphere ->
        # byte-identical; {"support": grf_support} for grf -> keyed == run).
        init_kwargs=init_kwargs,
        node_geometry=getattr(sweep_cfg, "node_geometry", "cube26"),
        geometry_kwargs=getattr(sweep_cfg, "geometry_kwargs", {}),
        # Virialized COUPLED-grid params: read from the same SweepConfig that
        # build_cache_name keys off, so the sim sees exactly what the cache key
        # encodes. Defaults mirror SimulationParameters/SweepConfig, so
        # non-virialized runs are unaffected.
        vir_n_nodes=getattr(sweep_cfg, "vir_n_nodes", 26),
        vir_extent=getattr(sweep_cfg, "vir_extent", 1.0),
        vir_mass_rule=getattr(sweep_cfg, "vir_mass_rule", "radial"),
        vir_mass_spread=getattr(sweep_cfg, "vir_mass_spread", 0.0),
        vir_segregation=getattr(sweep_cfg, "vir_segregation", 1.0),
        vir_s_metric=getattr(sweep_cfg, "vir_s_metric", "median"),
        vir_relax_steps=getattr(sweep_cfg, "vir_relax_steps", 1),
        # Extent->node-count coupling (Section 6 / item 10): read from the same
        # SweepConfig that build_cache_name keys off, so the sim runs exactly what
        # the cache key encodes (keyed == run). Default False -> byte-identical.
        vir_extent_couples_nodes=getattr(sweep_cfg, "vir_extent_couples_nodes", False),
        # Node-softening: read from the same SweepConfig that build_cache_name
        # keys off, so the sim runs exactly what the cache key encodes.
        node_softening_gpc=getattr(sweep_cfg, "node_softening_gpc", 0.0),
        # Close-range force law + adaptive sub-stepping (Section 4): read from the
        # same SweepConfig that build_cache_name keys off, so the sim runs exactly
        # what the cache key encodes (keyed == run). Defaults -> byte-identical.
        node_force_law=getattr(sweep_cfg, "node_force_law", "plummer"),
        node_substep_threshold=getattr(sweep_cfg, "node_substep_threshold", 0.0),
        node_substeps=getattr(sweep_cfg, "node_substeps", 1),
        # Virialized relaxation mode (Section 2 Option A/B): read from the same
        # SweepConfig that build_cache_name keys off, so the sim runs exactly what
        # the cache key encodes (keyed == run). Defaults ("lattice"/0.1/0.3) ->
        # byte-identical; "gradient" reaches Option B.
        vir_relax_mode=getattr(sweep_cfg, "vir_relax_mode", "lattice"),
        vir_relax_rate=getattr(sweep_cfg, "vir_relax_rate", 0.1),
        vir_hold_outer_frac=getattr(sweep_cfg, "vir_hold_outer_frac", 0.3),
        # Start-size lever (Section 6): read from the same SweepConfig that
        # build_cache_name keys off, so the sim runs exactly what the cache key
        # encodes (keyed == run). Default 1.0 -> byte-identical, no slug.
        start_size_scale=getattr(sweep_cfg, "start_size_scale", 1.0),
    )


def _make_sim_callback(sweep_cfg: _FixedSweepConfig, box_size_Gpc: float, a_start: float):
    """Return a sim_callback(M, S, centerM, seeds) -> [SimResult]."""

    def _sim(M_factor, S_gpc, centerM, seed):
        sim_params = _build_sim_params(sweep_cfg, M_factor, S_gpc, centerM, seed)
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
        "best_observer_chi2":    metrics.get("best_observer_chi2", float("nan")),
        "center_chi2_dof":       metrics.get("center_chi2_dof", float("nan")),
        "observer_median_chi2":  metrics.get("observer_median_chi2", float("nan")),
        "frac_below_lcdm":       metrics.get("frac_below_lcdm", float("nan")),
        "frac_below_eds":        metrics.get("frac_below_eds", float("nan")),
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
        "best_observer_chi2":    raw_metrics.get("best_observer_chi2", float("nan")),
        "center_chi2_dof":       raw_metrics.get("center_chi2_dof", float("nan")),
        "observer_median_chi2":  raw_metrics.get("observer_median_chi2", float("nan")),
        "frac_below_lcdm":       raw_metrics.get("frac_below_lcdm", float("nan")),
        "frac_below_eds":        raw_metrics.get("frac_below_eds", float("nan")),
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

    # centerM / outer_density_ceiling may be sweep AXES (lists) in cfg; the per-cell
    # runner expects SCALARS (it reads cfg["centerM"] straight into build_cache_name).
    # Probe with a single representative scalar so timing never sees a list.
    raw_cm = cfg["centerM"]
    probe_centerM = float(raw_cm[0]) if isinstance(raw_cm, list) else float(raw_cm)
    raw_ceil = cfg.get("outer_density_ceilings", cfg.get("outer_density_ceiling", 1.0))
    probe_ceil = float(raw_ceil[0]) if isinstance(raw_ceil, list) else float(raw_ceil)
    probe_cfg = dict(cfg)
    probe_cfg["centerM"] = probe_centerM
    probe_cfg["outer_density_ceiling"] = probe_ceil

    S_probe = cfg.get("s_min_gpc", 30)
    t0 = time.perf_counter()
    for cell in probe_cells:
        _run_cell_fixed_S(cell, S_probe, probe_cfg, box_size_Gpc, a_start,
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


def _cell_from_best_row(best_row: Dict) -> Dict:
    """Reconstruct the `expand_grid` cell dict that produced `best_row`.

    The CSV row carries every cell-identifying axis (geometry, amplitudes, seed,
    init); the config-wide knobs (geometry_kwargs, vir_*, softening, start-size)
    are NOT per-cell and stay in `cfg`. Feeding this cell to
    `_make_sweep_config_for_cell` reproduces the EXACT SweepConfig the run used,
    so the mu(z) panel re-runs the same a(t) the CSV chi2 was scored on.
    """
    return dict(
        M=int(best_row["M_factor"]),
        amplitude=float(best_row.get("node_mass_amplitude", 0.0)),
        nm_seed=int(best_row.get("node_mass_seed", 42)),
        s_amplitude=float(best_row.get("node_s_amplitude", 0.0)),
        init=str(best_row.get("init_distribution", "uniform_sphere")),
        geometry=str(best_row.get("node_geometry", "cube26")),
    )


def _generate_mu_z_panel(
    best_row: Dict, cfg: Dict,
    box_size_Gpc: float, a_start: float,
    pantheon_data: Dict,
    ws: str, tag: str, saved: List[str],
):
    """Generate the mu(z) panel for the best config.

    The panel re-runs the SAME full config the sweep cell ran (geometry,
    geometry_kwargs, vir_*, node_softening_gpc, start_size_scale) by going through
    `_make_sweep_config_for_cell` + `_build_sim_params` — the identical machinery
    `_make_sim_callback` (the real run that fed the CSV chi2) uses. This keeps the
    figure annotation and the CSV `chi2_dof` in agreement; a stale hand-rolled
    SimulationParameters here used to drop the vir_*/softening knobs and re-run a
    different a(t), producing the figure<->CSV chi2 conflict.
    """
    from cosmo.plots import plot_mu_z_panel
    from cosmo.sim_distance import sim_to_distance_modulus
    import cosmo.hubble_diagram as hd_engine
    from cosmo.distances import model_distance_modulus

    M = int(best_row["M_factor"])
    S = int(best_row["S_gpc"])
    centerM = float(best_row.get("centerM", cfg["centerM"]))
    t_start = cfg["t_start_Gyr"]

    # Single source of truth: rebuild the cell + SweepConfig the run used, then
    # the SimulationParameters via the SAME _build_sim_params the sim callback
    # uses. seed=42 matches the co-fit seed (linear/ternary search seeds=[42]).
    cell = _cell_from_best_row(best_row)
    sweep_cfg = _make_sweep_config_for_cell(cell, cfg)
    sim_params = _build_sim_params(sweep_cfg, M, S, centerM, seed=42)

    print(f"  [mu_z] Running best config M={M} S={S} geo={cell['geometry']} "
          f"for mu(z) panel ...")
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

        # chi2 reconciliation proof: the figure-recomputed chi2_dof (this fresh
        # run, scored by the SAME evaluate_precomputed the CSV scorer uses) MUST
        # now agree with the authoritative CSV value (best_row["chi2_dof"], from
        # compute_pantheon_metrics). Any residual is per-seed RNG of a fresh run
        # and must be << 0.01 — not the ~0.38 gap the old param-dropping panel had.
        _emit_chi2_reconciliation(best_row, results, ws, tag)
    except Exception as exc:
        print(f"  [mu_z] WARNING: could not generate panel: {exc}")


def _emit_chi2_reconciliation(
    best_row: Dict, results: Dict, ws: str, tag: str,
) -> None:
    """Print + write the figure<->CSV chi2 reconciliation proof note.

    Authoritative chi2/dof = the CSV scorer (`compute_pantheon_metrics`, which is
    `best_row["chi2_dof"]`). The figure now re-runs the cell's full config, so its
    recomputed chi2/dof (the `external_node_nbody` panel result) matches the CSV
    to within fresh-run RNG noise. Writes a gitignored note next to the figures.
    """
    csv_chi2 = float(best_row.get("chi2_dof", float("nan")))
    fig_chi2 = float(results.get("external_node_nbody", {}).get("chi2_dof", float("nan")))
    diff = abs(fig_chi2 - csv_chi2)

    lines = [
        "chi2 reconciliation (figure <-> CSV)",
        f"  tag                 : {tag}",
        f"  cell                : M={int(best_row['M_factor'])} S={int(best_row['S_gpc'])} "
        f"geo={best_row.get('node_geometry', 'cube26')}",
        f"  CSV chi2_dof        : {csv_chi2:.6f}   (authoritative: compute_pantheon_metrics)",
        f"  figure chi2_dof     : {fig_chi2:.6f}   (re-run of the SAME full config)",
        f"  |diff|              : {diff:.6f}   (REQUIRED < 0.01)",
        f"  status              : {'OK' if diff < 0.01 else 'MISMATCH (knob still dropped?)'}",
    ]
    note = "\n".join(lines)
    print("  [mu_z] " + note.replace("\n", "\n  [mu_z] "))

    try:
        from cosmo.plots import figure_path
        # Reuse the figures dir helper to keep the note beside the panel (the
        # whole results/ tree is gitignored).
        png = figure_path(ws, f"chi2_reconciliation_{tag}")
        txt = os.path.splitext(png)[0] + ".txt"
        with open(txt, "w", encoding="utf-8") as fh:
            fh.write(note + "\n")
        print(f"  [mu_z] reconciliation note => {txt}")
    except Exception as exc:
        print(f"  [mu_z] WARNING: could not write reconciliation note: {exc}")


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

# ---------------------------------------------------------------------------
# Resume / per-cell checkpoint
# ---------------------------------------------------------------------------
# A long sweep must survive interruption (Claude restarts, force-kills, power
# loss). The metrics Cache is NOT a reliable checkpoint here: it flushes only
# every ~100 s and silently falls back to read-only on a stale .lock (a
# force-killed run leaves its lock behind because atexit never runs). So the
# sweep checkpoints at its OWN level: every finished cell is APPENDED to the
# results CSV immediately (flushed), and on startup any cell already present in
# that CSV is SKIPPED. This is verifiable (the CSV is the list of done cells)
# and independent of the metrics cache.

_RESUME_NUM_COLS = ("M_factor", "S_gpc", "centerM", "outer_density_ceiling",
                    "node_mass_amplitude", "node_s_amplitude", "node_mass_seed",
                    "vir_mass_spread")
_RESUME_TXT_COLS = ("init_distribution", "node_geometry")
_ROW_FLOAT_COLS = ("centerM", "outer_density_ceiling", "node_mass_amplitude",
                   "node_s_amplitude", "vir_mass_spread", "chi2_dof", "chi2",
                   "chi2_lcdm", "chi2_eds",
                   "R2", "growth_factor", "growth_target", "match_avg_pct", "diff_pct",
                   "best_observer_chi2", "center_chi2_dof", "observer_median_chi2",
                   "frac_below_lcdm", "frac_below_eds")
_ROW_INT_COLS = ("M_factor", "S_gpc", "node_mass_seed", "n_sne_used")
_ROW_BOOL_COLS = ("anchor_ok", "runaway")


def _resume_key(identity: Dict, include_S: bool) -> tuple:
    """Stable identity tuple for a sweep cell, matching disk (str) and live (num).

    In fixed-S mode S is part of the identity; in co-fit mode S is an OUTPUT
    (searched per cell), so it is excluded from the key.
    """
    def num(x):
        try:
            return round(float(x), 6)
        except (TypeError, ValueError):
            return None
    num_cols = [c for c in _RESUME_NUM_COLS if include_S or c != "S_gpc"]
    return (tuple(num(identity.get(c)) for c in num_cols)
            + tuple(str(identity.get(c)) for c in _RESUME_TXT_COLS))


def _parse_csv_row(raw: Dict) -> Dict:
    """Parse a CSV-loaded row (all strings) back to typed values for resume/summary."""
    row = dict(raw)
    for c in _ROW_FLOAT_COLS:
        v = row.get(c, "")
        if v not in ("", None):
            try:
                row[c] = float(v)
            except ValueError:
                row[c] = float("inf") if c == "chi2_dof" else float("nan")
    for c in _ROW_INT_COLS:
        v = row.get(c, "")
        if v not in ("", None):
            try:
                row[c] = int(float(v))
            except ValueError:
                pass
    for c in _ROW_BOOL_COLS:
        if c in row:
            row[c] = str(row[c]).strip().lower() in ("true", "1")
    return row


def _select_best_row(all_rows: List[Dict], objective: str) -> Optional[Dict]:
    """Pick the best row for the configured objective.

    pantheon -> minimize finite chi2/dof (prefer anchor_ok rows).
    lcdm     -> maximize match_avg_pct (compute_match_metrics emits no chi2_dof,
                so chi2_dof is inf for every lcdm row; selecting on it would be
                meaningless). anchor_ok still gates when available.
    """
    if objective == "lcdm":
        scored = [r for r in all_rows
                  if math.isfinite(r.get("match_avg_pct", float("nan")))]
        bound = [r for r in scored if r.get("anchor_ok", False)]
        pool = bound or scored
        return max(pool, key=lambda r: r.get("match_avg_pct", 0.0)) if pool else None
    # pantheon (default)
    finite_rows = [r for r in all_rows if math.isfinite(r["chi2_dof"])]
    bound_rows = [r for r in finite_rows if r.get("anchor_ok", False)]
    if bound_rows:
        return min(bound_rows, key=lambda r: r["chi2_dof"])
    return min(finite_rows, key=lambda r: r["chi2_dof"]) if finite_rows else None


def run_sweep(cfg: Dict, probe_only: bool = False) -> Tuple[str, str, List[str]]:
    """
    Run the full overarching sweep.

    Resumable: finished cells are appended to the results CSV as they complete,
    and a re-run skips cells already present (unless cfg['resume'] is False).

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
    # Expose the references to per-cell SweepConfigs (used as the observer
    # fraction-below thresholds when score_observers is on).
    cfg["lcdm_ref"] = chi2_lcdm
    cfg["eds_ref"] = chi2_eds

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

    # --- Resume: load cells already completed by a prior (interrupted) run ---
    resume = bool(cfg.get("resume", True))
    csv_existed = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    done_keys = set()
    if resume and csv_existed:
        with open(csv_path, newline="", encoding="utf-8") as _rf:
            for _raw in csv.DictReader(_rf):
                _pr = _parse_csv_row(_raw)
                # Old CSVs (written before the vir_mass_spread column existed) default
                # to the config's scalar spread, so their resume key matches new cells
                # of that same spread -> a fixed-spread sweep (e.g. core_v3) still
                # resumes correctly across the schema change.
                _pr.setdefault("vir_mass_spread", cfg.get("vir_mass_spread", 0.0))
                all_rows.append(_pr)
                done_keys.add(_resume_key(_pr, include_S=not cofit))
        print(f"[resume] {len(done_keys)} cell(s) already in {csv_path} -> skipping them")
    elif csv_existed and not resume:
        print(f"[resume] --no-resume: overwriting {csv_path}")
        csv_existed = False  # force a fresh header below

    # Per-cell checkpoint writer: append each finished cell immediately (flushed)
    # so an interruption loses at most the single in-flight cell. When APPENDING to an
    # existing CSV, reuse ITS header columns: a file written before the vir_mass_spread
    # column existed stays in its old schema (no ragged rows / corruption mid-file).
    # A FRESH file gets the full SWEEP_CSV_COLS (with vir_mass_spread).
    _writer_cols = SWEEP_CSV_COLS
    if csv_existed:
        with open(csv_path, newline="", encoding="utf-8") as _hf:
            _hdr = next(csv.reader(_hf), None)
        if _hdr:
            _writer_cols = _hdr
    _ckpt_f = open(csv_path, "a" if csv_existed else "w", newline="", encoding="utf-8")
    _ckpt_w = csv.DictWriter(_ckpt_f, fieldnames=_writer_cols, extrasaction="ignore")
    if not csv_existed:
        _ckpt_w.writeheader()
        _ckpt_f.flush()

    def _checkpoint(row: Dict) -> None:
        _ckpt_w.writerow(row)
        _ckpt_f.flush()

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
                        _ident = dict(M_factor=cell["M"], centerM=centerM_val,
                                      outer_density_ceiling=ceiling_val,
                                      node_mass_amplitude=cell["amplitude"],
                                      node_s_amplitude=cell["s_amplitude"],
                                      node_mass_seed=cell["nm_seed"],
                                      init_distribution=cell["init"],
                                      node_geometry=cell["geometry"],
                                      vir_mass_spread=cell.get(
                                          "vir_spread",
                                          cell_cfg.get("vir_mass_spread", 0.0)))
                        if _resume_key(_ident, include_S=False) in done_keys:
                            print(f"  [{cell_num}/{total_cells}] M={cell['M']:6d}  "
                                  f"centerM={centerM_val}  [skip: already done]")
                            continue
                        t0 = time.perf_counter()
                        row, best_S = _cofit_S_for_cell(
                            cell, cell_cfg, box_size_Gpc, a_start,
                            pantheon_data, baseline, weights,
                            chi2_lcdm, chi2_eds,
                            prev_best_S=prev_best_S,
                        )
                        elapsed = time.perf_counter() - t0
                        prev_best_S = best_S
                        row["vir_mass_spread"] = cell.get(
                            "vir_spread", cell_cfg.get("vir_mass_spread", 0.0))
                        all_rows.append(row)
                        _checkpoint(row)
                        done_keys.add(_resume_key(_ident, include_S=False))

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
                        _ident = dict(M_factor=cell["M"], S_gpc=S, centerM=centerM_val,
                                      outer_density_ceiling=ceiling_val,
                                      node_mass_amplitude=cell["amplitude"],
                                      node_s_amplitude=cell["s_amplitude"],
                                      node_mass_seed=cell["nm_seed"],
                                      init_distribution=cell["init"],
                                      node_geometry=cell["geometry"],
                                      vir_mass_spread=cell.get(
                                          "vir_spread",
                                          cell_cfg.get("vir_mass_spread", 0.0)))
                        if _resume_key(_ident, include_S=True) in done_keys:
                            print(f"  [{i}/{total}] M={cell['M']:6d} S={S:3d}  "
                                  f"centerM={centerM_val}  [skip: already done]")
                            continue
                        t0 = time.perf_counter()
                        row = _run_cell_fixed_S(
                            cell, S, cell_cfg, box_size_Gpc, a_start,
                            pantheon_data, baseline, weights, chi2_lcdm, chi2_eds,
                        )
                        row["vir_mass_spread"] = cell.get(
                            "vir_spread", cell_cfg.get("vir_mass_spread", 0.0))
                        elapsed = time.perf_counter() - t0
                        all_rows.append(row)
                        _checkpoint(row)
                        done_keys.add(_resume_key(_ident, include_S=True))
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

    # The results CSV was written incrementally per-cell (the resume checkpoint),
    # so there is no full rewrite here — just close the handle. csv_path now holds
    # every resumed + newly-computed row.
    _ckpt_f.close()
    print(f"[out] Full results: {csv_path}  ({n} rows)")

    # Write best-isotropic subset (load_best_config-compatible)
    iso_rows = [r for r in all_rows if r.get("node_mass_amplitude", 0.0) == 0.0]
    best_iso_csv = os.path.join(cfg["results_dir"], f"sweep_results_pantheon_{tag}.csv")
    with open(best_iso_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=BEST_ISO_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(iso_rows)
    print(f"[out] Best-iso CSV: {best_iso_csv}  ({len(iso_rows)} rows)")

    # Best config selection depends on the objective:
    #   pantheon -> minimize chi2/dof vs the real Pantheon+ points
    #               (chi2_lcdm / chi2_eds are reference benchmarks only).
    #   lcdm     -> maximize match_avg_pct vs the analytic LCDM baseline
    #               (chi2_dof is not produced by compute_match_metrics).
    objective = cfg.get("objective", "pantheon")
    best_row = _select_best_row(all_rows, objective)

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
    p.add_argument("--no-resume", action="store_true",
                   help="Ignore any existing results CSV and recompute every cell "
                        "(default: resume by skipping cells already in the CSV).")
    return p


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.tag:
        cfg["tag"] = args.tag
    if args.results_dir:
        cfg["results_dir"] = args.results_dir
    cfg["resume"] = not args.no_resume

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
