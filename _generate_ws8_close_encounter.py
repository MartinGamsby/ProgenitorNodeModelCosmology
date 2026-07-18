#!/usr/bin/env python3
"""
WS8 — Close-encounter law + adaptive substepping COMPARISON (Section 4, items 3/C)
=================================================================================

The user's complaint: ``node_softening_gpc=1`` Gpc is "crazy / a hack" — a blunt
Plummer floor that lets sub-Gpc bodies barely interact — AND the ~40 Myr/step
timestep is too coarse, so the slingshot is partly a TIME-RESOLUTION artifact.

This script MEASURES, on the SAME runaway config, how each candidate close-pass
treatment moves BOTH the slingshot tail AND the Pantheon+ chi2/dof, so the trade
between "taming the runaway" and "distorting the fit" is explicit. It compares,
for {cube26, virialized}:

    legacy hard floor      node_softening_gpc=0   (the unsoftened baseline)
    Plummer 1 Gpc          node_softening_gpc=1   (the blunt floor the user dislikes)
    Plummer 0.1 Gpc        node_softening_gpc=0.1
    bounded/midpoint law   node_softening_gpc=1, node_force_law="bounded"
                           (capped close-range accel; "can't cross the midpoint")
    adaptive-substep       node_substep_threshold>0, node_substeps>1 (no softening)
    bounded+substep        bounded law + adaptive substep together

Plus a STEP-LADDER sub-table (n_steps in {273,546,1092,2184} on the runaway config,
each with the legacy floor) to re-MEASURE (not restate) whether finer time
resolution ALONE tames the tail.

Outputs (results/figures/ws8/, gitignored):
    close_encounter_compare.csv   the full method x geometry table (+ ladder rows)
    close_encounter_compare.png   tail metric + chi2 vs method (bar panels)
and prints both tables to stdout with an honest verdict.

INVARIANTS
----------
- Reuses the PRODUCT knobs (node_force_law, node_substep_*); no diagnostic
  monkeypatch. chi2/dof comes from the SAME authoritative scorer the sweep uses
  (cosmo.parameter_sweep.compute_pantheon_metrics), so the numbers are comparable
  to the sweep CSV.
- Seeded -> reproducible. PNG/CSV only under results/figures/ws8/.
"""

from cosmo.encoding import configure_utf8_stdout
configure_utf8_stdout()

import argparse
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cosmo.constants import CosmologicalConstants, SimulationParameters
from cosmo.factories import run_external_node_simulation, setup_simulation_context
from cosmo.parameter_sweep import SimResult, compute_pantheon_metrics
from cosmo.pantheon import load_pantheon
from cosmo.plots import figure_path, _DPI, _BBOX
from _generate_ws8_figs import (
    displacement_magnitudes,
    resolve_n_steps,
    slingshot_metrics,
)

_WS = "ws8"
_TODAY_GYR = 13.8
_G = CosmologicalConstants.Gpc_to_m

# Runaway config the comparison is built around (a strong node close-pass).
# t_start=2.9 is the project's safe floor (covers Pantheon z<=~2.3) so chi2/dof is
# meaningful; the slingshot still fires at M=1000/S=10.
_RUN_M = 1000.0
_RUN_S_GPC = 10.0
_RUN_N = 300
_RUN_T_START = 2.9
_RUN_SEED = 42

# Baseline coarse-dt step count: ~40 Myr/step over the ~10.9 Gyr run (the timestep
# the user calls too coarse). resolve_n_steps still guards dt < 0.05 Gyr.
_BASE_N_STEPS = 273

# Adaptive-substep config: refine within 2x the softening (or node-spacing) of a
# node into 8 KDK substeps during a close pass.
_SUBSTEP_THRESHOLD = 2.0
_SUBSTEPS = 8

# Step-ladder for the "more steps alone" re-measurement (item 3).
_STEP_LADDER: Tuple[int, ...] = (273, 546, 1092, 2184)

_VIR_KWARGS = dict(
    vir_n_nodes=26, vir_extent=1.0, vir_mass_rule="massfunc",
    vir_mass_spread=0.8, vir_segregation=1.0, vir_s_metric="median",
)


def _make_params(
    geometry: str, *, n_steps: int, node_softening_gpc: float = 0.0,
    node_force_law: str = "plummer", node_substep_threshold: float = 0.0,
    node_substeps: int = 1, M: float = _RUN_M, S_gpc: float = _RUN_S_GPC,
    n_particles: int = _RUN_N, t_start: float = _RUN_T_START, seed: int = _RUN_SEED,
) -> Tuple[SimulationParameters, float, float, int]:
    t_dur = _TODAY_GYR - t_start
    n_steps = resolve_n_steps(t_dur, n_steps)
    box, a_start, _ = setup_simulation_context(
        t_start, t_dur, n_steps, save_interval=max(1, n_steps // 8))
    vir = _VIR_KWARGS if geometry == "virialized" else {}
    sp = SimulationParameters(
        M_value=M, S_value=S_gpc, n_particles=n_particles, seed=seed,
        t_start_Gyr=t_start, t_duration_Gyr=t_dur, n_steps=n_steps,
        damping_factor=None, center_node_mass=1.0, mass_randomize=0.0,
        node_mass_seed=seed, init_distribution="uniform_sphere",
        node_geometry=geometry,
        node_softening_gpc=node_softening_gpc,
        node_force_law=node_force_law,
        node_substep_threshold=node_substep_threshold,
        node_substeps=node_substeps,
        **vir,
    )
    return sp, box, a_start, n_steps


def _run_one(sp, box, a_start, n_steps, pantheon_data) -> Dict[str, Any]:
    """Run ONE sim; return slingshot metrics + chi2/dof + growth via the product path."""
    save_interval = max(1, n_steps // 8)
    ext = run_external_node_simulation(sp, box, a_start, save_interval=save_interval)
    sim = ext["sim"]
    mask = np.asarray(sim.particles.get_observable_mask(), dtype=bool)
    p0 = sim.snapshots[0]["positions"][mask] / _G
    p1 = sim.snapshots[-1]["positions"][mask] / _G
    disp = displacement_magnitudes(p0, p1)
    m = slingshot_metrics(disp)

    # chi2/dof from the SAME authoritative scorer the sweep uses.
    sim_result = SimResult(
        size_curve_Gpc=np.asarray(ext["diameter_Gpc"]),
        hubble_curve=np.asarray(ext["H_hubble"]),
        t_Gyr=np.asarray(ext["t_Gyr"]),
        params=sp.external_params,
        results=None,
        a_curve=np.asarray(ext["a"]),
    )
    metrics = compute_pantheon_metrics(sim_result, pantheon_data, sp.t_start_Gyr)
    return {
        "max_over_median": m["max_over_median"],
        "p99_over_median": m["p99_over_median"],
        "tail_fraction": m["tail_fraction"],
        "max_disp": m["max"],
        "growth_factor": metrics.get("growth_factor", float("nan")),
        "chi2_dof": metrics.get("chi2_dof", float("nan")),
        "n_steps": int(n_steps),
        "n": int(m["n"]),
    }


# (label, kwargs to _make_params) for the method axis.
_METHODS: List[Tuple[str, Dict[str, Any]]] = [
    ("legacy hard floor", dict(node_softening_gpc=0.0)),
    ("Plummer 1 Gpc", dict(node_softening_gpc=1.0)),
    ("Plummer 0.1 Gpc", dict(node_softening_gpc=0.1)),
    ("bounded 1 Gpc", dict(node_softening_gpc=1.0, node_force_law="bounded")),
    ("adaptive substep", dict(node_softening_gpc=0.0,
                              node_substep_threshold=_SUBSTEP_THRESHOLD,
                              node_substeps=_SUBSTEPS)),
    ("bounded+substep", dict(node_softening_gpc=1.0, node_force_law="bounded",
                             node_substep_threshold=_SUBSTEP_THRESHOLD,
                             node_substeps=_SUBSTEPS)),
]

_GEOMETRIES = ("cube26", "virialized")


def run_compare(pantheon_data, *, n_steps: int = _BASE_N_STEPS,
                seed: int = _RUN_SEED) -> List[Dict[str, Any]]:
    """Method x geometry comparison rows (slingshot + chi2 on the runaway config)."""
    rows: List[Dict[str, Any]] = []
    for geom in _GEOMETRIES:
        for label, kw in _METHODS:
            print(f"[compare] {geom:<11} | {label:<18} ...", flush=True)
            sp, box, a_start, ns = _make_params(geom, n_steps=n_steps, seed=seed, **kw)
            r = _run_one(sp, box, a_start, ns, pantheon_data)
            r.update({"geometry": geom, "method": label})
            rows.append(r)
            print(f"    max/median={r['max_over_median']:.2f}  "
                  f"tail={r['tail_fraction']:.3f}  growth={r['growth_factor']:.2f}  "
                  f"chi2/dof={r['chi2_dof']:.3f}")
    return rows


def run_step_ladder(pantheon_data, *, geometry: str = "cube26",
                    ladder: Tuple[int, ...] = _STEP_LADDER,
                    seed: int = _RUN_SEED) -> List[Dict[str, Any]]:
    """n_steps ladder with the LEGACY floor — does finer dt ALONE tame the tail?"""
    rows: List[Dict[str, Any]] = []
    for ns in ladder:
        print(f"[ladder] {geometry} | n_steps={ns} (legacy floor) ...", flush=True)
        sp, box, a_start, ns_eff = _make_params(
            geometry, n_steps=ns, node_softening_gpc=0.0, seed=seed)
        r = _run_one(sp, box, a_start, ns_eff, pantheon_data)
        r.update({"geometry": geometry, "method": "legacy floor (ladder)"})
        rows.append(r)
        print(f"    n_steps={r['n_steps']:<5} max/median={r['max_over_median']:.2f}  "
              f"tail={r['tail_fraction']:.3f}  chi2/dof={r['chi2_dof']:.3f}")
    return rows


_CSV_COLS = ["geometry", "method", "n_steps", "max_over_median", "p99_over_median",
             "tail_fraction", "max_disp", "growth_factor", "chi2_dof", "n"]


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _fmt(x: float, prec: int = 3) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "inf" if (isinstance(x, float) and x == float("inf")) else "nan"
    return f"{x:.{prec}f}"


def _print_table(rows: List[Dict[str, Any]], title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    hdr = (f"  {'geometry':<11}{'method':<22}{'steps':>6}{'max/med':>10}"
           f"{'p99/med':>10}{'tailfrac':>10}{'growth':>9}{'chi2/dof':>10}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        print(f"  {r['geometry']:<11}{r['method']:<22}{r['n_steps']:>6}"
              f"{_fmt(r['max_over_median'],2):>10}{_fmt(r['p99_over_median'],2):>10}"
              f"{_fmt(r['tail_fraction'],3):>10}{_fmt(r['growth_factor'],2):>9}"
              f"{_fmt(r['chi2_dof'],3):>10}")
    print("=" * 100)


def _verdict(rows: List[Dict[str, Any]]) -> None:
    """Honest verdict: which law tames the tail with the least chi2 distortion."""
    print("\n" + "#" * 100)
    print("HONEST VERDICT — close-encounter law vs slingshot taming vs chi2 cost")
    print("#" * 100)
    for geom in _GEOMETRIES:
        grp = [r for r in rows if r["geometry"] == geom]
        base = next((r for r in grp if r["method"] == "legacy hard floor"), None)
        plummer = next((r for r in grp if r["method"] == "Plummer 1 Gpc"), None)
        bounded = next((r for r in grp if r["method"] == "bounded 1 Gpc"), None)
        if not (base and plummer and bounded):
            continue
        print(f"\n  [{geom}] baseline (legacy floor): "
              f"max/median={_fmt(base['max_over_median'],1)}, "
              f"chi2/dof={_fmt(base['chi2_dof'])}")
        for r in grp:
            if r["method"] == "legacy hard floor":
                continue
            tamed = (math.isfinite(r["max_over_median"]) and
                     math.isfinite(base["max_over_median"]) and
                     r["max_over_median"] < 0.5 * base["max_over_median"])
            print(f"    {r['method']:<20} max/median={_fmt(r['max_over_median'],1):>8}"
                  f"  chi2/dof={_fmt(r['chi2_dof']):>7}"
                  f"  {'TAMES' if tamed else 'partial/none'}")
        # Can the blunt 1 Gpc Plummer be replaced by the bounded law?
        if (math.isfinite(plummer["chi2_dof"]) and math.isfinite(bounded["chi2_dof"])
                and math.isfinite(plummer["max_over_median"])
                and math.isfinite(bounded["max_over_median"])):
            tame_ok = bounded["max_over_median"] <= 1.5 * plummer["max_over_median"]
            chi2_ok = bounded["chi2_dof"] <= plummer["chi2_dof"] + 0.05
            replace = tame_ok and chi2_ok
            print(f"    => bounded law {'CAN' if replace else 'does NOT clearly'} "
                  f"replace blunt 1 Gpc Plummer "
                  f"(tail {_fmt(bounded['max_over_median'],1)} vs "
                  f"{_fmt(plummer['max_over_median'],1)}; "
                  f"chi2 {_fmt(bounded['chi2_dof'])} vs {_fmt(plummer['chi2_dof'])}).")
    print("#" * 100)


def generate_figure(compare_rows: List[Dict[str, Any]],
                    ladder_rows: List[Dict[str, Any]]) -> str:
    """Tail metric + chi2 vs method (per geometry) and the step-ladder."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    method_labels = [m[0] for m in _METHODS]
    x = np.arange(len(method_labels))

    for col, geom in enumerate(_GEOMETRIES):
        grp = {r["method"]: r for r in compare_rows if r["geometry"] == geom}
        mm = [grp[m]["max_over_median"] if m in grp else float("nan")
              for m in method_labels]
        ch = [grp[m]["chi2_dof"] if m in grp else float("nan")
              for m in method_labels]
        # cap inf chi2 for display
        ch_disp = [c if math.isfinite(c) else float("nan") for c in ch]

        ax = axes[0, col]
        bars = ax.bar(x, [v if math.isfinite(v) else 0 for v in mm],
                      color="#d62728", alpha=0.85)
        ax.set_yscale("log")
        ax.set_title(f"{geom}: slingshot tail (max/median, log)", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("max/median (log)", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        for b, v in zip(bars, mm):
            if math.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                        ha="center", va="bottom", fontsize=8)

        axc = axes[1, col]
        bars2 = axc.bar(x, [c if math.isfinite(c) else 0 for c in ch_disp],
                        color="#1f77b4", alpha=0.85)
        axc.set_title(f"{geom}: Pantheon+ chi2/dof (lower=better fit)", fontsize=11)
        axc.set_xticks(x)
        axc.set_xticklabels(method_labels, rotation=30, ha="right", fontsize=8)
        axc.set_ylabel("chi2/dof", fontsize=9)
        axc.axhline(0.436, color="green", ls="--", lw=1.0, label="LCDM ~0.44")
        axc.axhline(0.843, color="gray", ls=":", lw=1.0, label="EdS null ~0.84")
        axc.legend(fontsize=7)
        axc.grid(True, alpha=0.3, axis="y")
        for b, c in zip(bars2, ch):
            label = f"{c:.2f}" if math.isfinite(c) else "inf"
            axc.text(b.get_x() + b.get_width() / 2,
                     (c if math.isfinite(c) else 0), label,
                     ha="center", va="bottom", fontsize=8)

    fig.suptitle(
        "WS8 — Close-encounter law + adaptive substepping vs slingshot AND chi2 "
        f"(runaway M={_RUN_M:.0f}, S={_RUN_S_GPC:.0f}, N={_RUN_N}, "
        f"t_start={_RUN_T_START})\n"
        "Bounded 'can't cross the midpoint' law + KDK substeps compared against the "
        "blunt Plummer floor on BOTH the tail and the fit.",
        fontsize=12, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = figure_path(_WS, "close_encounter_compare")
    fig.savefig(out, dpi=_DPI, bbox_inches=_BBOX)
    plt.close(fig)
    print(f"\n[fig] Saved: {out}")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=_RUN_SEED)
    p.add_argument("--n-steps", type=int, default=_BASE_N_STEPS)
    p.add_argument("--no-ladder", action="store_true",
                   help="Skip the (slow) n_steps ladder sub-table.")
    args = p.parse_args(argv)

    pantheon_data = load_pantheon()

    compare_rows = run_compare(pantheon_data, n_steps=args.n_steps, seed=args.seed)
    ladder_rows: List[Dict[str, Any]] = []
    if not args.no_ladder:
        ladder_rows = run_step_ladder(pantheon_data, seed=args.seed)

    _print_table(compare_rows, "CLOSE-ENCOUNTER COMPARISON (method x geometry)")
    if ladder_rows:
        _print_table(ladder_rows,
                     "STEP-LADDER (legacy floor; does finer dt ALONE tame the tail?)")
    _verdict(compare_rows)

    out_png = generate_figure(compare_rows, ladder_rows)
    csv_path = os.path.splitext(out_png)[0] + ".csv"
    _write_csv(csv_path, compare_rows + ladder_rows)
    print(f"[csv] Saved: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
