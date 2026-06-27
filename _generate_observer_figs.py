#!/usr/bin/env python3
"""
Observer-from-a-particle mu(z) PROTOTYPE figure + table  (items 5 / D)
=====================================================================

The user's big idea: "We're NOT in the centre. Compute mu(z) from the viewpoint
of EACH particle, not the cloud centre, and take the best one (and show the
distribution)."

The default a(t)->mu(z) pipeline measures expansion as the inner cloud's RMS
radius about its CENTRE OF MASS. This OPT-IN script (NOT on any default path)
runs a representative config, then — for EVERY observable particle as an observer
— builds an inferred a_p(t) from that particle's local frame and scores it
against the REAL Pantheon+ with the SAME authoritative chi2 the sweep uses
(cosmo.observer_distance.score_observer -> sim_to_distance_modulus +
evaluate_precomputed). It reports the DISTRIBUTION of per-particle chi2/dof, the
CENTRE-observer baseline, and the BEST observer — for BOTH observer definitions
(local_rms, hubble_flow) on BOTH a virialized config and a cube26 control.

Outputs (results/figures/ws8/, gitignored):
    observer_chi2_distribution.png  histogram per (config x definition), centre +
                                    best marked, LCDM/EdS reference lines.
    observer_chi2.csv               centre / best / median / p10 / p90 per row.
and prints the table + an HONEST verdict (does the best observer beat the centre,
and is the spread itself a PF2-style anisotropy signal, or is "take the best"
cherry-picking?).

INVARIANTS
----------
- Pure ADD-ON: does not change the centre-based a(t) path or any pinned number.
- Reuses the PRODUCT knobs + the authoritative scorer; the per-particle chi2 is
  comparable to the sweep CSV. Seeded -> reproducible. PNG/CSV only under
  results/figures/ws8/.
- Runtime: one sim per config (default 2 configs) + a cheap per-particle scoring
  loop. With particle_count=120 the whole script is a couple of minutes.
"""

from cosmo.encoding import configure_utf8_stdout
configure_utf8_stdout()

import argparse
import csv
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cosmo.constants import SimulationParameters
from cosmo.factories import run_external_node_simulation, setup_simulation_context
from cosmo.hubble_diagram import evaluate_model
from cosmo.observer_distance import (
    history_from_snapshots,
    observer_chi2_distribution,
    center_a_curve,
)
from cosmo.pantheon import load_pantheon
from cosmo.plots import figure_path, _DPI, _BBOX
from cosmo.sim_distance import sim_to_distance_modulus

_WS = "ws8"
_TODAY_GYR = 13.8
_T_START = 2.9
_SEED = 42

# Representative configs. The virialized_final HEADLINE config (force-balanced
# big virialized + doubly tamed) and a cube26 CONTROL at the SAME M/S/softening
# so the comparison isolates geometry (item 9). M/S from the near-LCDM band.
_M = 1000.0
_S_GPC = 30.0

_VIR_KWARGS = dict(
    vir_n_nodes=80, vir_extent=1.0, vir_mass_rule="massfunc",
    vir_mass_spread=0.8, vir_segregation=1.0, vir_s_metric="median",
    vir_relax_steps=1,
)


def _save_interval_for(n_steps: int, target_snaps: int = 12) -> int:
    """Largest save_interval that DIVIDES n_steps and yields >= target_snaps.

    The integrator only saves a final snapshot when (step+1) % save_interval == 0
    (cosmo/integrator.evolve), so save_interval MUST divide n_steps or the last
    snapshot lands before t=today and the mu(z) kernel's today-tolerance guard
    fires. We pick the largest divisor giving at least target_snaps intervals.
    """
    best = 1
    for d in range(1, n_steps + 1):
        if n_steps % d == 0 and (n_steps // d) >= target_snaps:
            best = d
    return best


def _make_params(
    geometry: str, *, n_particles: int, n_steps: int,
    node_softening_gpc: float = 1.0,
) -> Tuple[SimulationParameters, float, float, int]:
    t_dur = _TODAY_GYR - _T_START
    save_interval = _save_interval_for(n_steps)
    box, a_start, _ = setup_simulation_context(
        _T_START, t_dur, n_steps, save_interval=save_interval)
    vir = _VIR_KWARGS if geometry == "virialized" else {}
    sp = SimulationParameters(
        M_value=_M, S_value=_S_GPC, n_particles=n_particles, seed=_SEED,
        t_start_Gyr=_T_START, t_duration_Gyr=t_dur, n_steps=n_steps,
        damping_factor=None, center_node_mass=1.0, mass_randomize=0.0,
        node_mass_seed=_SEED, init_distribution="uniform_sphere",
        node_geometry=geometry, node_softening_gpc=node_softening_gpc,
        **vir,
    )
    return sp, box, a_start, n_steps


def _reference_chi2(pantheon_data, center_a, t_Gyr) -> Dict[str, float]:
    """LCDM and EdS chi2/dof on the SAME in-range SNe subset the centre observer
    uses, so the reference lines are comparable to the observer distribution."""
    dist = sim_to_distance_modulus(
        z_target=pantheon_data["z"], a=center_a, t_Gyr=t_Gyr,
        t_start_Gyr=_T_START)
    in_range = dist["in_range"]
    z_in = pantheon_data["z"][in_range]
    mu_in = pantheon_data["mu"][in_range]
    sig_in = pantheon_data["sigma"][in_range]
    out = {}
    for model in ("lcdm", "einstein_de_sitter"):
        ev = evaluate_model(z_in, mu_in, sig_in, model=model)
        out[model] = float(ev["chi2_dof"])
    return out


def run_config(
    geometry: str, pantheon_data, *, n_particles: int, n_steps: int,
) -> Dict[str, Any]:
    """Run ONE sim; return per-definition observer distributions + references."""
    print(f"\n[run] geometry={geometry}  M={_M}  S={_S_GPC}  "
          f"N={n_particles}  steps={n_steps} ...", flush=True)
    sp, box, a_start, ns = _make_params(
        geometry, n_particles=n_particles, n_steps=n_steps)
    ext = run_external_node_simulation(
        sp, box, a_start, save_interval=_save_interval_for(ns))
    sim = ext["sim"]
    mask = np.asarray(sim.particles.get_observable_mask(), dtype=bool)
    pos, vel, t = history_from_snapshots(sim.snapshots)

    # Reference lines (LCDM / EdS) on the centre observer's in-range subset.
    center_a = center_a_curve(pos, t, mask=mask)
    refs = _reference_chi2(pantheon_data, center_a, t)

    result: Dict[str, Any] = {"geometry": geometry, "refs": refs,
                              "distributions": {}}
    for definition in ("local_rms", "hubble_flow"):
        print(f"    scoring observers ({definition}) ...", flush=True)
        out = observer_chi2_distribution(
            pos, vel, t, sp.t_start_Gyr, pantheon_data,
            definition=definition, mask=mask)
        result["distributions"][definition] = out
        print(f"      centre={out['center_chi2_dof']:.3f}  "
              f"best={out['best_chi2_dof']:.3f} (obs#{out['best_observer']})  "
              f"median={out['median']:.3f}  "
              f"p10/p90={out['p10']:.3f}/{out['p90']:.3f}  "
              f"n_finite={out['n_finite']}/{out['n_observers']}")
    return result


# ---------------------------------------------------------------------------
# CSV + table + verdict
# ---------------------------------------------------------------------------

_CSV_COLS = ["geometry", "definition", "n_observers", "n_finite",
             "center_chi2_dof", "best_chi2_dof", "best_observer",
             "median", "p10", "p90", "center_growth",
             "chi2_lcdm", "chi2_eds"]


def _rows_from_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for r in results:
        for definition, out in r["distributions"].items():
            rows.append({
                "geometry": r["geometry"],
                "definition": definition,
                "n_observers": out["n_observers"],
                "n_finite": out["n_finite"],
                "center_chi2_dof": out["center_chi2_dof"],
                "best_chi2_dof": out["best_chi2_dof"],
                "best_observer": out["best_observer"],
                "median": out["median"],
                "p10": out["p10"],
                "p90": out["p90"],
                "center_growth": out["center_growth"],
                "chi2_lcdm": r["refs"]["lcdm"],
                "chi2_eds": r["refs"]["einstein_de_sitter"],
            })
    return rows


def _fmt(x, prec=3) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "inf" if (isinstance(x, float) and x == float("inf")) else "nan"
    return f"{x:.{prec}f}"


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _print_table(rows: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 104)
    print("OBSERVER-FROM-A-PARTICLE chi2/dof DISTRIBUTION (vs REAL Pantheon+)")
    print("=" * 104)
    hdr = (f"  {'geometry':<11}{'definition':<12}{'centre':>9}{'best':>9}"
           f"{'median':>9}{'p10':>9}{'p90':>9}{'best#':>7}{'finite':>8}"
           f"{'LCDM':>8}{'EdS':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        print(f"  {r['geometry']:<11}{r['definition']:<12}"
              f"{_fmt(r['center_chi2_dof']):>9}{_fmt(r['best_chi2_dof']):>9}"
              f"{_fmt(r['median']):>9}{_fmt(r['p10']):>9}{_fmt(r['p90']):>9}"
              f"{r['best_observer']:>7}"
              f"{r['n_finite']:>4}/{r['n_observers']:<3}"
              f"{_fmt(r['chi2_lcdm']):>8}{_fmt(r['chi2_eds']):>8}")
    print("=" * 104)


def _verdict(rows: List[Dict[str, Any]]) -> None:
    print("\n" + "#" * 104)
    print("HONEST VERDICT — does an off-centre observer materially improve the fit?")
    print("#" * 104)
    for r in rows:
        c = r["center_chi2_dof"]
        b = r["best_chi2_dof"]
        if not (math.isfinite(c) and math.isfinite(b)):
            print(f"  [{r['geometry']:<11} {r['definition']:<12}] "
                  f"non-finite centre/best — skip")
            continue
        improve = c - b
        improve_pct = 100.0 * improve / c if c > 0 else float("nan")
        spread = (r["p90"] - r["p10"]) if math.isfinite(r["p90"]) else float("nan")
        print(f"\n  [{r['geometry']:<11} {r['definition']:<12}]")
        print(f"    centre chi2/dof = {_fmt(c)}   best = {_fmt(b)}   "
              f"improvement = {_fmt(improve)} ({_fmt(improve_pct,1)}%)")
        print(f"    p10..p90 spread = {_fmt(spread)}  "
              f"(LCDM ref {_fmt(r['chi2_lcdm'])}, EdS ref {_fmt(r['chi2_eds'])})")
        # Honest framing:
        if math.isfinite(improve_pct) and improve_pct > 5.0 and b < r["chi2_lcdm"]:
            print("    => BEST observer BEATS the centre AND dips below the LCDM "
                  "reference — but this is the MIN over many observers (look at the "
                  "distribution, not just the min: 'best' is a cherry-pick).")
        elif math.isfinite(improve_pct) and improve_pct > 5.0:
            print("    => best observer improves on the centre, but does NOT beat "
                  "the LCDM reference. The improvement is a selection effect over "
                  "the observer ensemble, not a model win.")
        else:
            print("    => off-centre observers do NOT materially beat the centre "
                  "(the user's hypothesis is NOT supported for this config).")
        if math.isfinite(spread) and spread > 0.1:
            print("    => the LARGE per-observer spread is itself a PF2-style "
                  "anisotropy signal (different observers infer different "
                  "expansions).")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _make_figure(results: List[Dict[str, Any]], out_path: str) -> None:
    defs = ("local_rms", "hubble_flow")
    n_rows = len(results)
    n_cols = len(defs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.2 * n_cols, 4.0 * n_rows),
                             squeeze=False)
    for i, r in enumerate(results):
        refs = r["refs"]
        for j, definition in enumerate(defs):
            ax = axes[i][j]
            out = r["distributions"][definition]
            vals = out["chi2_dof"]
            finite = vals[np.isfinite(vals)]
            if finite.size > 0:
                # clip the upper tail for a readable histogram
                hi = np.percentile(finite, 98)
                hi = max(hi, out["center_chi2_dof"] * 1.5, refs["einstein_de_sitter"])
                clipped = np.clip(finite, None, hi)
                ax.hist(clipped, bins=30, color="#888888", alpha=0.75,
                        edgecolor="white", label=f"observers (n={finite.size})")
            ax.axvline(out["center_chi2_dof"], color="#d62728", lw=2.2,
                       label=f"centre ({out['center_chi2_dof']:.3f})")
            if math.isfinite(out["best_chi2_dof"]):
                ax.axvline(out["best_chi2_dof"], color="#2ca02c", lw=2.2, ls="--",
                           label=f"best ({out['best_chi2_dof']:.3f})")
            ax.axvline(refs["lcdm"], color="#1f77b4", lw=1.4, ls=":",
                       label=f"LCDM ({refs['lcdm']:.3f})")
            ax.axvline(refs["einstein_de_sitter"], color="#9467bd", lw=1.4, ls=":",
                       label=f"EdS ({refs['einstein_de_sitter']:.3f})")
            ax.set_title(f"{r['geometry']} — {definition}")
            ax.set_xlabel("per-observer chi2/dof vs Pantheon+")
            ax.set_ylabel("count")
            ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(
        "Observer-from-a-particle mu(z): per-particle chi2 distribution "
        f"(M={_M:.0f}, S={_S_GPC:.0f}, t_start={_T_START})",
        fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=_DPI, bbox_inches=_BBOX)
    plt.close(fig)
    print(f"\n[figure] wrote {out_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--particles", type=int, default=120,
                    help="inner particle count (default 120; prototype-sized)")
    ap.add_argument("--n-steps", type=int, default=273,
                    help="integration steps (default 273; dt~0.04 Gyr)")
    ap.add_argument("--geometries", nargs="+",
                    default=["virialized", "cube26"],
                    help="configs to run (default: virialized + cube26 control)")
    args = ap.parse_args(argv)

    try:
        pantheon_data = load_pantheon()
    except FileNotFoundError as exc:
        print(f"[error] Pantheon+ data absent: {exc}")
        return 2

    results = [
        run_config(geom, pantheon_data,
                   n_particles=args.particles, n_steps=args.n_steps)
        for geom in args.geometries
    ]

    rows = _rows_from_results(results)
    _print_table(rows)
    _verdict(rows)

    csv_path = figure_path(_WS, "observer_chi2").replace(".png", ".csv")
    _write_csv(csv_path, rows)
    print(f"\n[csv] wrote {csv_path}")

    png_path = figure_path(_WS, "observer_chi2_distribution")
    _make_figure(results, png_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
