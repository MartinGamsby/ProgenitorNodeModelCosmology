#!/usr/bin/env python3
"""
Runtime calibration + projection for the core_v3 sweep family (plan-v2 B5/B6, NO BLIND LAUNCH).
=============================================================================================

The user must NOT be surprised by a multi-day run. This script measures the REAL
per-sim wall-time at the PRODUCTION settings (2000 particles / 546 steps) for the
geometry x close-range-treatment combinations the sweep actually runs, then PROJECTS
the total wall-time for every core_v3 arm, the seed arm, and each satellite family,
and prints (and writes) a clear TABLE of projected hours so the user can decide to
launch ALL / core-only / trim.

What it measures (SHORT timing run -- NOT the full sweep)
--------------------------------------------------------
For each probe (geometry x treatment) it runs `--evals` (default 2) real sims with
the cache BYPASSED (SKIP_CACHE) at the production resolution, and reports seconds/sim:

  geometry  in {cube26, virialized-A (lattice, 150 nodes), virialized-B (gradient, 150 nodes)}
  treatment in {none (plummer, NO substep), plummer 1 Gpc (NO substep),
                bounded+substep (1 Gpc cap, threshold=2.0, substeps=8)}

The none/plummer arms have NO adaptive substep, so they are CHEAPER than the
bounded+substep arms; the table shows that explicitly per geometry.

How it PROJECTS (assumptions stated in the output)
--------------------------------------------------
Each co-fit cell runs a ternary S-search that evaluates a handful of DISTINCT S
values (the search caches per S). Measured count over the core range
build_s_list(3,35) -> [3..30]:
  - COLD (first M in a warm-start group, no hint): ~9 distinct-S sims.
  - WARM (subsequent M values, hint window = prev_best_S//4): ~5-7 distinct-S sims.
Per 7-M arm: ~9 + 6*~6 ~= 45 sims => ~6.4 sims/cell. We project with a STATED,
slightly-conservative EVALS_PER_CELL (default 7) and also print the per-arm cell
count so the user can rescale. Projection per arm =
  (#cells) * EVALS_PER_CELL * (per-sim seconds for that geometry x treatment).

Output
------
  results/runtime_projection.csv  (gitignored)  + a printed table.

Run
---
  PYTHONIOENCODING=utf-8 python _calibrate_runtime.py            # full calibration + projection
  PYTHONIOENCODING=utf-8 python _calibrate_runtime.py --evals 1  # faster, 1 sim/probe (noisier)
  PYTHONIOENCODING=utf-8 python _calibrate_runtime.py --project-only  # skip timing, use fallback s/sim
"""

from cosmo.encoding import configure_utf8_stdout
configure_utf8_stdout()

import argparse
import csv
import glob
import json
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import cosmo.parameter_sweep as _ps
from cosmo.factories import (
    setup_simulation_context,
    run_external_node_simulation,
)
from sweep import load_config, expand_grid, _make_sweep_config_for_cell, _build_sim_params

# Production resolution the core sweep runs at (decisions-v2 / core_v3 arms).
PROD_PARTICLES = 2000
PROD_STEPS = 546
T_START_GYR = 2.9
S_PROBE = 10  # a representative interior S for the timing sim (in [3..35])

# Stated projection assumption: distinct-S sims a ternary co-fit cell averages.
# (Measured ~6.4 over [3..30]; round UP to stay honest about the upper bound.)
EVALS_PER_CELL = 7

# Fallback per-sim seconds (R4 COST SIGNAL: cube26 ~99 s, virialized-150 ~239 s with
# bounded+substep) so --project-only still emits a useful table without a timing run.
_FALLBACK_SPS = {
    ("cube26", "none"): 99.0 * 0.6,          # no substep ~ cheaper than bounded
    ("cube26", "plummer1"): 99.0 * 0.6,
    ("cube26", "bounded"): 99.0,
    ("virA", "none"): 239.0 * 0.6,
    ("virA", "plummer1"): 239.0 * 0.6,
    ("virA", "bounded"): 239.0,
    ("virB", "none"): 239.0 * 0.6,
    ("virB", "plummer1"): 239.0 * 0.6,
    ("virB", "bounded"): 239.0,
}

_REPO = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Probe definitions: (geometry-label, treatment-label) -> a config dict at the
# PRODUCTION resolution. Mirrors the core_v3 arm knobs exactly.
# ---------------------------------------------------------------------------

def _base_probe_cfg() -> Dict:
    return {
        "M_values": [100],
        "S_values": "co-fit",
        "s_min_gpc": 3, "s_max_gpc": 35, "s_cofit_method": "ternary",
        "node_mass_amplitudes": [0.0], "node_s_amplitudes": [0.0],
        "node_mass_seeds": [42],
        "particle_count": PROD_PARTICLES, "n_steps": PROD_STEPS,
        "t_start_Gyr": T_START_GYR, "centerM": 1, "objective": "pantheon",
        "init_distributions": ["grf"], "grf_support": "sphere",
    }


def _apply_geometry(cfg: Dict, geom: str) -> Dict:
    cfg = dict(cfg)
    if geom == "cube26":
        cfg["node_geometries"] = ["cube26"]
    elif geom == "virA":
        cfg["node_geometries"] = ["virialized"]
        cfg.update(vir_n_nodes=150, vir_mass_rule="massfunc", vir_mass_spread=0.8,
                   vir_segregation=1.0, vir_s_metric="median", vir_relax_steps=1,
                   vir_relax_mode="lattice")
    elif geom == "virB":
        cfg["node_geometries"] = ["virialized"]
        cfg.update(vir_n_nodes=150, vir_mass_rule="massfunc", vir_mass_spread=0.8,
                   vir_segregation=1.0, vir_s_metric="median", vir_relax_steps=40,
                   vir_relax_mode="gradient", vir_relax_rate=0.1, vir_hold_outer_frac=0.3)
    else:
        raise ValueError(geom)
    return cfg


def _apply_treatment(cfg: Dict, treat: str) -> Dict:
    cfg = dict(cfg)
    if treat == "none":
        cfg.update(node_softening_gpc=0.0, node_force_law="plummer",
                   node_substep_threshold=0.0, node_substeps=1)
    elif treat == "plummer1":
        cfg.update(node_softening_gpc=1.0, node_force_law="plummer",
                   node_substep_threshold=0.0, node_substeps=1)
    elif treat == "bounded":
        cfg.update(node_softening_gpc=1.0, node_force_law="bounded",
                   node_substep_threshold=2.0, node_substeps=8)
    else:
        raise ValueError(treat)
    return cfg


GEOMS = ["cube26", "virA", "virB"]
TREATS = ["none", "plummer1", "bounded"]


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def time_one_probe(geom: str, treat: str, box_size_Gpc: float, a_start: float,
                   evals: int) -> float:
    """Run `evals` real production-resolution sims (cache bypassed) -> seconds/sim."""
    cfg = _apply_treatment(_apply_geometry(_base_probe_cfg(), geom), treat)
    cell = expand_grid(cfg)[0]
    sweep_cfg = _make_sweep_config_for_cell(cell, cfg)

    saved_skip = _ps.SKIP_CACHE
    _ps.SKIP_CACHE = True
    try:
        t0 = time.perf_counter()
        for _ in range(evals):
            sim_params = _build_sim_params(sweep_cfg, cell["M"], S_PROBE, 1, seed=42)
            run_external_node_simulation(sim_params, box_size_Gpc, a_start, 10)
        elapsed = time.perf_counter() - t0
    finally:
        _ps.SKIP_CACHE = saved_skip
    return elapsed / max(evals, 1)


def calibrate(evals: int, project_only: bool) -> Dict[Tuple[str, str], float]:
    """Return {(geom,treat): seconds_per_sim} -- measured, or fallback if project_only."""
    if project_only:
        print("[calibrate] --project-only: using R4 fallback per-sim seconds (no timing run).")
        return dict(_FALLBACK_SPS)

    print(f"[calibrate] Timing {len(GEOMS)*len(TREATS)} probes x {evals} sim(s) each at "
          f"{PROD_PARTICLES}p/{PROD_STEPS} (cache bypassed). This is the only slow part.")
    box_size_Gpc, a_start, _ = setup_simulation_context(
        T_START_GYR, 13.8 - T_START_GYR, PROD_STEPS, save_interval=10
    )
    sps: Dict[Tuple[str, str], float] = {}
    for geom in GEOMS:
        for treat in TREATS:
            t = time_one_probe(geom, treat, box_size_Gpc, a_start, evals)
            sps[(geom, treat)] = t
            print(f"  [{geom:>6s} x {treat:<9s}] {t:7.1f} s/sim")
    return sps


# ---------------------------------------------------------------------------
# Arm -> (geometry-label, treatment-label) classification from a loaded cfg
# ---------------------------------------------------------------------------

def classify(cfg: Dict) -> Tuple[str, str]:
    geoms = cfg.get("node_geometries", ["cube26"])
    if "virialized" in geoms:
        geom = "virB" if cfg.get("vir_relax_mode", "lattice") == "gradient" else "virA"
    else:
        geom = "cube26"
    soft = float(cfg.get("node_softening_gpc", 0.0))
    law = cfg.get("node_force_law", "plummer")
    if law == "bounded":
        treat = "bounded"
    elif soft != 0.0:
        treat = "plummer1"
    else:
        treat = "none"
    return geom, treat


def _scale_sps(sps: Dict[Tuple[str, str], float], geom: str, treat: str,
               particle_count: int, n_nodes: Optional[int]) -> float:
    """Per-sim seconds for an arm, scaling the measured 2000p/150-node probe to the
    arm's actual particle_count and (virialized) node count.

    Cost ~ O(N_part) for particle self-gravity + O(N_part x N_nodes) for the tidal
    node force, which dominates for virialized. We scale LINEARLY in particle_count
    (a defensible first-order estimate) and, for virialized arms with a non-150 node
    count (the extent satellite), linearly in node count for the node-force part.
    cube26 (26 nodes) probes are scaled in particle_count only.
    """
    base = sps[(geom, treat)]
    pf = particle_count / float(PROD_PARTICLES)
    s = base * pf
    if geom in ("virA", "virB") and n_nodes and n_nodes != 150:
        # crude: assume node-force dominates -> scale by node-count ratio too.
        s *= n_nodes / 150.0
    return s


# ---------------------------------------------------------------------------
# Projection over the families
# ---------------------------------------------------------------------------

def _arm_paths(family: str, numbered_only: bool = True) -> List[str]:
    d = os.path.join(_REPO, "sweeps", family)
    paths = sorted(glob.glob(os.path.join(d, "*.json")))
    return [p for p in paths if os.path.basename(p) != "_manifest.json"]


def project(sps: Dict[Tuple[str, str], float]) -> Tuple[List[Dict], Dict[str, float]]:
    """Build per-arm projection rows + group subtotals (hours)."""
    rows: List[Dict] = []

    def add_family(label: str, paths: List[str]):
        for p in paths:
            cfg = load_config(p)
            cells = expand_grid(cfg)
            n_cells = len(cells)
            geom, treat = classify(cfg)
            pc = cfg.get("particle_count", PROD_PARTICLES)
            n_nodes = None
            if geom in ("virA", "virB"):
                n_nodes = cfg.get("vir_n_nodes", 150)
                if cfg.get("vir_extent_couples_nodes", False):
                    from cosmo.node_geometry import extent_coupled_n_nodes
                    n_nodes = extent_coupled_n_nodes(n_nodes, cfg.get("vir_extent", 1.0))
            per_sim = _scale_sps(sps, geom, treat, pc, n_nodes)
            sims = n_cells * EVALS_PER_CELL
            hours = sims * per_sim / 3600.0
            rows.append(dict(
                group=label, arm=os.path.basename(p), tag=cfg.get("tag", ""),
                geom=geom, treat=treat, particle_count=pc,
                n_nodes=(n_nodes if n_nodes else 26),
                cells=n_cells, evals_per_cell=EVALS_PER_CELL, sims=sims,
                per_sim_s=round(per_sim, 1), hours=round(hours, 2),
            ))

    # Core arms 01..12, seed arm 13, then satellites.
    core_all = _arm_paths("core_v3")
    core = [p for p in core_all if "seedsweep" not in os.path.basename(p)]
    seed = [p for p in core_all if "seedsweep" in os.path.basename(p)]
    add_family("core", core)
    add_family("seed", seed)
    add_family("sat_startsize", _arm_paths("satellite_startsize"))
    add_family("sat_convergence", _arm_paths("satellite_convergence"))
    add_family("sat_extent", _arm_paths("satellite_extent"))

    subtotals: Dict[str, float] = {}
    for r in rows:
        subtotals[r["group"]] = subtotals.get(r["group"], 0.0) + r["hours"]
    subtotals["satellites_total"] = (subtotals.get("sat_startsize", 0.0)
                                     + subtotals.get("sat_convergence", 0.0)
                                     + subtotals.get("sat_extent", 0.0))
    subtotals["grand_total"] = sum(r["hours"] for r in rows)
    return rows, subtotals


# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------

def emit(sps: Dict[Tuple[str, str], float], rows: List[Dict],
         subtotals: Dict[str, float], results_dir: str) -> str:
    os.makedirs(results_dir, exist_ok=True)
    out = os.path.join(results_dir, "runtime_projection.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["# per-sim seconds at 2000p/546 (cache bypassed)"])
        w.writerow(["geometry", "treatment", "seconds_per_sim"])
        for geom in GEOMS:
            for treat in TREATS:
                w.writerow([geom, treat, round(sps[(geom, treat)], 1)])
        w.writerow([])
        w.writerow(["# projection assumptions"])
        w.writerow(["evals_per_cell", EVALS_PER_CELL])
        w.writerow(["note", "ternary co-fit ~6.4 distinct-S sims/cell over [3..30]; "
                            "7 is a conservative upper bound"])
        w.writerow([])
        hdr = ["group", "arm", "tag", "geom", "treat", "particle_count", "n_nodes",
               "cells", "evals_per_cell", "sims", "per_sim_s", "hours"]
        w.writerow(hdr)
        for r in rows:
            w.writerow([r[k] for k in hdr])
        w.writerow([])
        w.writerow(["# subtotals (hours)"])
        for k in ("core", "seed", "sat_startsize", "sat_convergence", "sat_extent",
                  "satellites_total", "grand_total"):
            if k in subtotals:
                w.writerow([k, round(subtotals[k], 2)])

    # Pretty table to stdout.
    print("\n" + "=" * 78)
    print("PER-SIM WALL-TIME (measured at 2000p/546, cache bypassed)")
    print("=" * 78)
    print(f"  {'geometry':<8s} {'none':>10s} {'plummer1':>10s} {'bounded+sub':>12s}")
    for geom in GEOMS:
        print(f"  {geom:<8s} "
              f"{sps[(geom,'none')]:>10.1f} {sps[(geom,'plummer1')]:>10.1f} "
              f"{sps[(geom,'bounded')]:>12.1f}")
    print("  (none/plummer have NO adaptive substep -> cheaper than bounded+substep)")

    print("\n" + "=" * 78)
    print(f"PROJECTED WALL-TIME PER ARM   (EVALS_PER_CELL={EVALS_PER_CELL}, "
          f"ternary co-fit; ~6.4 measured)")
    print("=" * 78)
    print(f"  {'arm':<40s} {'geo':>6s} {'treat':>9s} {'N':>5s} "
          f"{'cells':>5s} {'s/sim':>7s} {'hours':>7s}")
    cur = None
    for r in rows:
        if r["group"] != cur:
            cur = r["group"]
            print(f"  -- {cur} --")
        print(f"  {r['arm']:<40s} {r['geom']:>6s} {r['treat']:>9s} "
              f"{r['particle_count']:>5d} {r['cells']:>5d} "
              f"{r['per_sim_s']:>7.1f} {r['hours']:>7.2f}")

    print("\n" + "=" * 78)
    print("SUBTOTALS")
    print("=" * 78)
    def hd(h): return f"{h:6.1f} h  ({h/24.0:4.1f} d)"
    print(f"  CORE (12 arms, 84 cells)            : {hd(subtotals['core'])}")
    print(f"  SEED arm (15 cells)                 : {hd(subtotals['seed'])}")
    print(f"  CORE + SEED                         : {hd(subtotals['core']+subtotals['seed'])}")
    print(f"  satellite: start_size (18 cells)    : {hd(subtotals['sat_startsize'])}")
    print(f"  satellite: convergence (3 cells)    : {hd(subtotals['sat_convergence'])}")
    print(f"  satellite: extent (6 cells)         : {hd(subtotals['sat_extent'])}")
    print(f"  satellites TOTAL                    : {hd(subtotals['satellites_total'])}")
    print(f"  GRAND TOTAL (everything)            : {hd(subtotals['grand_total'])}")
    print("=" * 78)
    print(f"\n[out] projection table -> {out}")
    print("[note] Launch CORE-ONLY (default) or everything via launch_sweep_detached.ps1 "
          "(-IncludeSatellites).")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evals", type=int, default=2,
                    help="real sims per probe for timing (default 2).")
    ap.add_argument("--project-only", action="store_true",
                    help="skip the timing run; use R4 fallback per-sim seconds.")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    sps = calibrate(args.evals, args.project_only)
    rows, subtotals = project(sps)
    emit(sps, rows, subtotals, args.results_dir)


if __name__ == "__main__":
    main()
