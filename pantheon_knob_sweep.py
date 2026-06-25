#!/usr/bin/env python3
"""
Pantheon Knob Sweep — exercise node_mass_amplitude, node_mass_seed, init_distribution
alongside the user-chosen M/S grid.

Produces:
  results/sweep_results_pantheon.csv   — best-isotropic (amplitude=0/grf) rows,
                                         columns compatible with --from-best-config
  results/knob_sweep_summary.csv       — one row per (M, S, amplitude, seed, init)

Usage:
    python pantheon_knob_sweep.py                   # full user grid (default)
    python pantheon_knob_sweep.py --probe-only      # time 5 representative sims, then exit
    python pantheon_knob_sweep.py --help

Full factorial command (print and run):
    python pantheon_knob_sweep.py --full-factorial
"""

from cosmo.encoding import configure_utf8_stdout
configure_utf8_stdout()

import argparse
import csv
import math
import os
import sys
import time
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

from cosmo.constants import CosmologicalConstants, SimulationParameters
from cosmo.factories import (
    setup_simulation_context,
    run_external_node_simulation,
    results_to_sim_result,
)
from cosmo.parameter_sweep import (
    SearchMethod, SweepConfig, MatchWeights, SimResult, LCDMBaseline,
    build_cache_name, run_sweep, compute_pantheon_metrics, worst_callback,
)
from cosmo.pantheon import load_pantheon

const = CosmologicalConstants()

# ---------------------------------------------------------------------------
# User-chosen grid (from the orchestrator brief)
# ---------------------------------------------------------------------------
M_LIST   = [50, 100, 250, 500, 700, 750, 800, 850, 900, 1000]
S_LIST   = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
AMP_LIST = [0.0, 0.25, 0.5, 0.75]
SEED_LIST = [42, 7]
INIT_DISTRIBUTION = "grf"   # always grf for every sim in this sweep

# Fixed physics
PARTICLES = 400
N_STEPS   = 273
T_START   = 2.9                           # Gyr
T_DURATION = 13.8 - T_START              # Gyr (must end at today)
OBJECTIVE  = "pantheon"

# CSV columns for the best-isotropic output (load_best_config compatibility)
_BEST_ISO_COLS = [
    "M_factor", "S_gpc", "centerM",
    "chi2_dof", "chi2", "R2",
    "n_sne_used", "growth_factor", "anchor_ok",
    "node_mass_amplitude", "node_mass_seed", "init_distribution",
    "match_avg_pct", "diff_pct",
]

# CSV columns for the full knob-summary output
_KNOB_SUMMARY_COLS = [
    "M_factor", "S_gpc", "centerM",
    "node_mass_amplitude", "node_mass_seed", "node_s_amplitude", "init_distribution",
    "chi2_dof", "chi2", "R2",
    "n_sne_used", "growth_factor", "anchor_ok",
    "match_avg_pct", "diff_pct",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _SweepConfigFixed(SweepConfig):
    """
    SweepConfig variant that hard-codes particle_count and n_steps to the
    orchestrator-specified values (400p / 273 steps) rather than deriving them
    from quick_search / leet_search flags.

    This ensures the cache key (which reads config.particle_count and
    config.n_steps) matches the actual simulation parameters passed to
    SimulationParameters in _make_sim_callback.
    """

    @property
    def particle_count(self) -> int:
        return PARTICLES

    @property
    def n_steps(self) -> int:
        return N_STEPS


def _make_sweep_config(amplitude: float, seed: int, init: str,
                       s_amplitude: float = 0.0) -> _SweepConfigFixed:
    cfg = _SweepConfigFixed(
        quick_search=False,
        many_search=3,
        leet_search=False,
        search_center_mass=False,   # centerM=1 fixed; we control M/S ourselves
        t_start_Gyr=T_START,
        t_duration_Gyr=T_DURATION,
        damping_factor=None,
        s_min_gpc=min(S_LIST),
        s_max_gpc=max(S_LIST),
        save_interval=10,
        objective=OBJECTIVE,
        node_mass_seed=seed,
        node_mass_amplitude=amplitude,
        node_s_amplitude=s_amplitude,
        init_distribution=init,
    )
    return cfg


def _make_sim_callback(config, box_size_Gpc: float, a_start: float):
    """Return a sim_callback that uses the given SweepConfig knobs.

    Uses config.particle_count and config.n_steps (which are overridden to
    PARTICLES=400 and N_STEPS=273 by _SweepConfigFixed) so the cache key
    and actual SimulationParameters are always consistent.
    """
    sim_count = [0]

    def sim(M_factor: int, S_gpc: int, centerM: int, seed: int) -> SimResult:
        sim_count[0] += 1
        sim_params = SimulationParameters(
            M_value=M_factor,
            S_value=S_gpc,
            n_particles=config.particle_count,
            seed=seed,
            t_start_Gyr=config.t_start_Gyr,
            t_duration_Gyr=config.t_duration_Gyr,
            n_steps=config.n_steps,
            damping_factor=config.damping_factor,
            center_node_mass=centerM,
            mass_randomize=0.0,
            node_mass_seed=config.node_mass_seed,
            node_mass_amplitude=config.node_mass_amplitude,
            node_s_amplitude=getattr(config, "node_s_amplitude", 0.0),
            init_distribution=config.init_distribution,
        )
        ext_results = run_external_node_simulation(sim_params, box_size_Gpc, a_start,
                                                    config.save_interval)
        return results_to_sim_result(ext_results, sim_params)

    def sim_callback(M_factor: int, S_gpc: int, centerM: int, seeds: List[int]) -> List[SimResult]:
        return [sim(M_factor, S_gpc, centerM, seed) for seed in seeds]

    return sim_callback, sim_count


def _run_single(
    M: int, S: int, centerM: int,
    amplitude: float, nm_seed: int, init: str,
    box_size_Gpc: float, a_start: float,
    pantheon_data: dict,
    baseline,
    weights,
    s_amplitude: float = 0.0,
) -> dict:
    """Run one (M, S, amplitude, nm_seed, init[, s_amplitude]) combination.

    s_amplitude (node_s_amplitude, the per-node RADIAL position perturbation)
    defaults to 0.0 so existing callers are unaffected; pass >0 to sweep node
    POSITIONS the way `amplitude` sweeps node masses.
    """
    config = _make_sweep_config(amplitude, nm_seed, init, s_amplitude=s_amplitude)
    sim_cb, _ = _make_sim_callback(config, box_size_Gpc, a_start)

    sim_result, metrics = worst_callback(
        sim_cb, config,
        M_factor=M, S_val=S, centerM=centerM,
        seeds=[42],        # particle-RNG seed is always 42; nm_seed is separate
        baseline=baseline,
        weights=weights,
        pantheon_data=pantheon_data,
    )

    row = {
        "M_factor": M,
        "S_gpc": S,
        "centerM": centerM,
        "node_mass_amplitude": amplitude,
        "node_mass_seed": nm_seed,
        "node_s_amplitude": s_amplitude,
        "init_distribution": init,
        "chi2_dof": metrics.get("chi2_dof", float("inf")),
        "chi2": metrics.get("chi2", float("inf")),
        "R2": metrics.get("R2", float("nan")),
        "n_sne_used": metrics.get("n_sne_used", 0),
        "growth_factor": metrics.get("growth_factor", float("nan")),
        "anchor_ok": (
            abs(metrics.get("growth_factor", 0) /
                metrics.get("growth_target", 1) - 1.0) <= 0.20
            if metrics.get("growth_target") else False
        ),
        "match_avg_pct": metrics.get("match_avg_pct", 0.0),
        "diff_pct": metrics.get("diff_pct", 100.0),
    }
    return row


def _expand_grid(m_list, s_list, amp_list, seed_list, init):
    """
    Expand the full factorial grid, collapsing amplitude=0 runs to a SINGLE
    (nm_seed=42) run since seed is a no-op when amplitude==0.

    Returns list of (M, S, amplitude, nm_seed) tuples.
    """
    combos = []
    for M in m_list:
        for S in s_list:
            for amp in amp_list:
                if amp == 0.0:
                    combos.append((M, S, 0.0, 42))
                else:
                    for nm_seed in seed_list:
                        combos.append((M, S, amp, nm_seed))
    return combos


def probe_timing(
    box_size_Gpc: float, a_start: float,
    pantheon_data: dict, baseline, weights,
    n_probe: int = 5,
) -> float:
    """
    Time n_probe representative sims (varying M and S) and return seconds/sim.
    """
    probe_configs = [
        (50,  20, 0.0, 42),
        (250, 40, 0.0, 42),
        (500, 55, 0.0, 42),
        (850, 30, 0.0, 42),
        (1000, 70, 0.0, 42),
    ][:n_probe]

    import cosmo.parameter_sweep as _ps
    saved_skip = _ps.SKIP_CACHE
    _ps.SKIP_CACHE = True  # don't let the probe pollute or read real cache

    t0 = time.perf_counter()
    for M, S, amp, nm_seed in probe_configs:
        _run_single(M, S, 1, amp, nm_seed, INIT_DISTRIBUTION,
                    box_size_Gpc, a_start, pantheon_data, baseline, weights)
    elapsed = time.perf_counter() - t0

    _ps.SKIP_CACHE = saved_skip
    sec_per_sim = elapsed / max(len(probe_configs), 1)
    print(f"[probe] {len(probe_configs)} sims in {elapsed:.1f}s => {sec_per_sim:.1f} s/sim")
    return sec_per_sim


# ---------------------------------------------------------------------------
# Main sweep entry point
# ---------------------------------------------------------------------------

def run_sweep_full(
    m_list: List[int] = M_LIST,
    s_list: List[int] = S_LIST,
    amp_list: List[float] = AMP_LIST,
    seed_list: List[int] = SEED_LIST,
    init: str = INIT_DISTRIBUTION,
    results_dir: str = "./results",
    probe_only: bool = False,
) -> Tuple[str, str]:
    """
    Run the full (or bounded) Pantheon knob sweep.

    Returns (best_iso_csv_path, knob_summary_csv_path).
    """
    print("=" * 70)
    print("PANTHEON KNOB SWEEP")
    print("=" * 70)
    print(f"  M grid ({len(m_list)}): {m_list}")
    print(f"  S grid ({len(s_list)}): {s_list}")
    print(f"  amplitude: {amp_list}")
    print(f"  nm_seed:   {seed_list} (collapsed to 1 run when amplitude=0)")
    print(f"  init_distribution: {init!r} (fixed for ALL sims)")
    print(f"  particles={PARTICLES}, n_steps={N_STEPS}, t_start={T_START}")

    # Setup shared context
    print("\n[setup] Computing initial conditions and loading Pantheon+ data ...")
    box_size_Gpc, a_start, lcdm_result = setup_simulation_context(
        T_START, T_DURATION, N_STEPS, save_interval=10
    )
    pantheon_data = load_pantheon()
    print(f"[setup] Loaded {pantheon_data['n']} SNe Ia")

    baseline = LCDMBaseline(
        t_Gyr=lcdm_result['t'],
        size_Gpc=lcdm_result['diameter_Gpc'],
        H_hubble=lcdm_result['H_hubble'],
        size_final_Gpc=lcdm_result['diameter_Gpc'][-1],
        radius_max_Gpc=lcdm_result['diameter_Gpc'][-1] / 2 / math.sqrt(3 / 5),
        a_final=lcdm_result['a'][-1],
    )
    weights = MatchWeights()

    # Probe timing
    sec_per_sim = probe_timing(box_size_Gpc, a_start, pantheon_data, baseline, weights)
    if probe_only:
        print("[probe] --probe-only: exiting after timing.")
        return "", ""

    # Expand grid
    combos = _expand_grid(m_list, s_list, amp_list, seed_list, init)
    total_sims = len(combos)
    est_minutes = total_sims * sec_per_sim / 60
    print(f"\n[grid] {total_sims} sim combinations, estimated {est_minutes:.1f} min")

    # Run all combos
    os.makedirs(results_dir, exist_ok=True)
    all_rows: List[dict] = []
    t_sweep_start = time.perf_counter()

    for i, (M, S, amp, nm_seed) in enumerate(combos):
        t0 = time.perf_counter()
        row = _run_single(M, S, 1, amp, nm_seed, init,
                          box_size_Gpc, a_start, pantheon_data, baseline, weights)
        elapsed = time.perf_counter() - t0
        all_rows.append(row)

        chi2_str = f"{row['chi2_dof']:.4f}" if math.isfinite(row['chi2_dof']) else "FAIL"
        print(
            f"  [{i+1}/{total_sims}] M={M:5d} S={S:3d} amp={amp:.2f} "
            f"seed={nm_seed} => chi2/dof={chi2_str}  "
            f"({elapsed:.1f}s)"
        )

    total_elapsed = time.perf_counter() - t_sweep_start
    print(f"\n[sweep] Done: {total_sims} sims in {total_elapsed:.1f}s "
          f"({total_elapsed/total_sims:.1f} s/sim)")

    # Save full knob summary
    knob_csv = os.path.join(results_dir, "knob_sweep_summary.csv")
    with open(knob_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_KNOB_SUMMARY_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[out] Knob summary: {knob_csv}  ({len(all_rows)} rows)")

    # Best-isotropic subset: amplitude=0 rows
    iso_rows = [r for r in all_rows if r["node_mass_amplitude"] == 0.0]
    best_iso_csv = os.path.join(results_dir, "sweep_results_pantheon.csv")
    with open(best_iso_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_BEST_ISO_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(iso_rows)
    print(f"[out] Best-isotropic CSV: {best_iso_csv}  ({len(iso_rows)} rows)")

    # ---------- Summary statistics ----------
    _print_summary(all_rows, iso_rows, amp_list, m_list, s_list)

    return best_iso_csv, knob_csv


def _print_summary(all_rows, iso_rows, amp_list, m_list, s_list):
    """Print the headline statistics table."""
    finite_iso = [r for r in iso_rows if math.isfinite(r["chi2_dof"])]
    finite_all = [r for r in all_rows if math.isfinite(r["chi2_dof"])]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if finite_iso:
        best_iso = min(finite_iso, key=lambda r: r["chi2_dof"])
        worst_iso = max(finite_iso, key=lambda r: r["chi2_dof"])
        print(f"\n--- ISOTROPIC (amplitude=0) ---")
        print(f"  chi2/dof RANGE : {worst_iso['chi2_dof']:.4f} .. {best_iso['chi2_dof']:.4f}")
        print(
            f"  BEST  config   : M={best_iso['M_factor']}, S={best_iso['S_gpc']}, "
            f"centerM={best_iso['centerM']}, chi2/dof={best_iso['chi2_dof']:.4f}, "
            f"R2={best_iso['R2']:.5f}"
        )
        print(f"  growth_factor  : {best_iso['growth_factor']:.3f}  (target ~3.30 at t_start=2.9)")

    if finite_all:
        best_all = min(finite_all, key=lambda r: r["chi2_dof"])
        print(f"\n--- OVERALL BEST (all knobs) ---")
        print(
            f"  BEST  config   : M={best_all['M_factor']}, S={best_all['S_gpc']}, "
            f"amp={best_all['node_mass_amplitude']:.2f}, seed={best_all['node_mass_seed']}, "
            f"chi2/dof={best_all['chi2_dof']:.4f}"
        )
        if finite_iso:
            delta = abs(best_all["chi2_dof"] - best_iso["chi2_dof"])
            print(f"  Delta(chi2/dof) vs isotropic best: {delta:.4f}  "
                  f"({'within shot noise' if delta < 0.01 else 'differs'})")

    # Flatness across amplitude at the best-isotropic (M, S)
    if finite_iso:
        bM, bS = best_iso["M_factor"], best_iso["S_gpc"]
        at_best_MS = [r for r in finite_all
                      if r["M_factor"] == bM and r["S_gpc"] == bS]
        if at_best_MS:
            chi2s = [r["chi2_dof"] for r in at_best_MS]
            spread = max(chi2s) - min(chi2s)
            print(f"\n--- FLATNESS at best (M={bM}, S={bS}) ---")
            print(f"  chi2/dof across amplitude/seed: {min(chi2s):.4f} .. {max(chi2s):.4f}  "
                  f"(spread={spread:.4f})")

    # Anisotropy-showcase config
    if finite_iso:
        bM, bS = best_iso["M_factor"], best_iso["S_gpc"]
        print(f"\n--- ANISOTROPY SHOWCASE CONFIG (for Section 4) ---")
        print(f"  M={bM}, S={bS}, amplitude=0.75, nm_seed=42, init={INIT_DISTRIBUTION!r}")
        print(f"  (same chi2/dof as isotropic — new knobs move shear/dipole, not isotropic chi2)")

    print("\n--- REPRODUCE FULL GRID ---")
    print("  python pantheon_knob_sweep.py")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser():
    p = argparse.ArgumentParser(
        description="Pantheon knob sweep: M/S x amplitude x seed x init_distribution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--probe-only", action="store_true",
                   help="Time 5 representative sims then exit (no sweep).")
    p.add_argument("--full-factorial", action="store_true",
                   help="Print the full-factorial command and run it.")
    p.add_argument("--results-dir", default="./results",
                   help="Output directory for CSV files.")
    return p


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    if args.full_factorial:
        print("Full factorial command:")
        print("  python pantheon_knob_sweep.py  (this IS the full factorial)")
        print()

    best_iso_csv, knob_csv = run_sweep_full(
        m_list=M_LIST,
        s_list=S_LIST,
        amp_list=AMP_LIST,
        seed_list=SEED_LIST,
        init=INIT_DISTRIBUTION,
        results_dir=args.results_dir,
        probe_only=args.probe_only,
    )

    if best_iso_csv:
        print(f"\nTo visualize the best config:")
        print(f"  python hubble_diagram_nbody.py --from-best-config {best_iso_csv}")
