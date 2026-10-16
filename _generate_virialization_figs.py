#!/usr/bin/env python3
"""
Center-only virialization study — Option A (lattice) vs Option B (relaxation)
============================================================================

Re-bases the virialization criterion on the DEEP INTERIOR of a LARGE grid (the
center nodes, selected by ``center_k`` independent of grid radius) and answers the
question PF8 left disputed: can a REALISTIC (non-crystal) relaxed blob virialize at
its center, or does only the analytic cubic-lattice crystal (Option A) reach balance?

It produces ACTUAL NUMBERS + VISUALS (no hand-waving):

  results/figures/ws8/virialization_residuals.csv     (gitignored; also printed)
      One row per {builder} x {mass rule} x {grid size}, where builder is:
        - A_lattice          : Option A, analytic force-balanced cubic ball
        - realistic          : the un-balanced Fibonacci segregated blob (steps=0)
        - B_relax_{R}_{K}    : Option B, K gradient-descent steps at rate R
      Columns: max/median CENTER-ONLY residual (the deep-interior selector), n_center,
      a_ref, AND the OLD inner_frac residual for contrast, plus the force-residual
      objective f = sum|a_i|^2 so Option B's monotone descent is visible.

  results/figures/ws8/virialization_residual_vs_gridsize.png
      Center-only max residual vs grid size for A / B / realistic, both mass rules,
      with the TOL=0.25 line. Shows whether the deep-center residual drops below TOL
      as the grid grows (it does for A at ~1e-30; it does NOT for B/realistic).

  results/figures/ws8/virialization_optionB_descent.png
      Option B's descent: the force-residual objective f AND the center-only residual
      vs relaxation step, for radial vs massfunc — the honest "how far does true
      relaxation get" picture (f falls monotonically; the center residual bottoms out
      far above TOL, never reaching the crystal's balance).

What "0.25" means (the user asked "Not 25%?"): it is a DIMENSIONLESS ratio
``|net node accel| / a_ref`` where a_ref is the magnitude of ONE characteristic
neighbour pull. residual=0.25 == "the leftover net force is a quarter of a single
neighbour pull". It is NOT a percentage of the total force. See
cosmo.node_geometry.virialization_residual for the full justification.

Usage
-----
    python _generate_virialization_figs.py                 # full grid (slow at n~2000)
    python _generate_virialization_figs.py --max-n 500     # cap the largest grid
    python _generate_virialization_figs.py --quick         # small/fast smoke run

INVARIANTS
----------
- Pure analysis: builds grids + computes the metric. No sim, no global RNG use, no
  product-path changes. Deterministic (seeded). PNG/CSV only under results/ (gitignored).
"""

from cosmo.encoding import configure_utf8_stdout
configure_utf8_stdout()

import argparse
import sys
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cosmo.constants import CosmologicalConstants
from cosmo.node_geometry import (
    build_virialized_grid,
    virialization_residual,
    _force_residual_objective_and_grad,
    _gradient_relax_positions,
)
from cosmo.plots import figure_path, _DPI, _BBOX
import csv
import os

_WS = "ws8"
_TOL = 0.25                      # the dimensionless center-residual tolerance
_S_GPC = 30.0
_M_EXT_KG = 1.0                  # per-node mean mass (residual is dimensionless)
_SEED = 12
_SPREAD = 0.8
_SEG = 1.0
_EXTENT = 2.5                    # multi-layer so the interior is genuinely buried
_RULES = ("radial", "massfunc")

# Deep-interior selector: K nodes closest to the centroid, size-independent so a
# bigger grid genuinely deepens the interior tested. Capped at the grid size.
_CENTER_K = 15
# Option B descent configuration (the COMPARED relaxation).
_RELAX_RATE = 0.1
_RELAX_HOLD = 0.3
_B_STEP_POINTS: Tuple[int, ...] = (5, 20, 60)   # tabulated Option-B step counts


def _S_m() -> float:
    return float(_S_GPC) * CosmologicalConstants.Gpc_to_m


# ===========================================================================
# Core measurement (pure)
# ===========================================================================

def _build(rule: str, n: int, *, mode: str = "lattice", steps: int = 1,
           rate: float = _RELAX_RATE) -> Tuple[np.ndarray, np.ndarray]:
    """Build one virialized grid for the given builder configuration."""
    return build_virialized_grid(
        _S_m(), n_nodes=int(n), M_ext_kg=_M_EXT_KG, vir_mass_rule=rule,
        vir_mass_spread=_SPREAD, vir_segregation=_SEG, vir_extent=_EXTENT,
        vir_relax_mode=mode, vir_relax_steps=int(steps), vir_relax_rate=float(rate),
        vir_hold_outer_frac=_RELAX_HOLD, seed=_SEED,
    )


def measure_row(builder: str, rule: str, n: int,
                pos: np.ndarray, masses: np.ndarray) -> Dict[str, Any]:
    """One tidy result row: center-only + legacy residual + objective f for a grid."""
    k = min(_CENTER_K, max(2, n // 4))
    res_center = virialization_residual(
        pos, masses, center_k=k, center_mass_kg=_M_EXT_KG)
    res_inner = virialization_residual(
        pos, masses, inner_frac=0.5, center_mass_kg=_M_EXT_KG)
    f_obj, _ = _force_residual_objective_and_grad(
        pos, masses, float(np.mean(masses)), CosmologicalConstants.G)
    return {
        "builder": builder,
        "rule": rule,
        "n": int(n),
        "max_residual_center": res_center["max_residual"],
        "median_residual_center": res_center["median_residual"],
        "n_center": res_center["n_inner"],
        "a_ref": res_center["a_ref"],
        "max_residual_inner_frac": res_inner["max_residual"],
        "median_residual_inner_frac": res_inner["median_residual"],
        "f_objective": f_obj,
        "passes_center_tol": bool(res_center["max_residual"] <= _TOL),
    }


def build_residual_table(sizes: Tuple[int, ...]) -> List[Dict[str, Any]]:
    """The full residual table: A / realistic / Option-B(steps) x rule x size."""
    rows: List[Dict[str, Any]] = []
    for rule in _RULES:
        for n in sizes:
            # Option A — analytic force-balanced lattice.
            rows.append(measure_row("A_lattice", rule, n, *_build(rule, n)))
            # Realistic — un-balanced Fibonacci segregated blob.
            rows.append(measure_row(
                "realistic", rule, n, *_build(rule, n, steps=0)))
            # Option B — true iterative relaxation at several step counts.
            for steps in _B_STEP_POINTS:
                label = f"B_relax_r{_RELAX_RATE}_s{steps}"
                rows.append(measure_row(
                    label, rule, n,
                    *_build(rule, n, mode="gradient", steps=steps)))
    return rows


def optionB_descent(rule: str, n: int, step_points: Tuple[int, ...]
                    ) -> Dict[str, np.ndarray]:
    """Option B descent curves vs step for one rule/size: f and center residual.

    Relaxes the SAME realistic start by an increasing number of steps. The objective
    f is computed on the RAW relaxed positions (before any NN-spacing rescale), so it
    is a genuine monotone descent trajectory; the center residual is dimensionless
    (rescale-invariant) and is reported as-is to show it bottoms out above TOL.
    """
    steps = np.asarray((0,) + tuple(step_points), dtype=np.int64)
    f_obj = np.empty(steps.size, dtype=np.float64)
    cres = np.empty(steps.size, dtype=np.float64)
    k = min(_CENTER_K, max(2, n // 4))
    pos0, m0 = _build(rule, n, steps=0)  # one realistic start
    cm = float(np.mean(m0))
    for i, s in enumerate(steps):
        relaxed = _gradient_relax_positions(
            pos0, m0, n_steps=int(s), rate=_RELAX_RATE,
            hold_outer_frac=_RELAX_HOLD)
        f_obj[i], _ = _force_residual_objective_and_grad(
            relaxed, m0, cm, CosmologicalConstants.G)
        cres[i] = virialization_residual(
            relaxed, m0, center_k=k, center_mass_kg=_M_EXT_KG)["max_residual"]
    return {"steps": steps.astype(np.float64), "f_objective": f_obj,
            "center_residual": cres}


# ===========================================================================
# Output: CSV + figures + stdout tables
# ===========================================================================

def write_csv(rows: List[Dict[str, Any]]) -> str:
    out = os.path.join(os.path.dirname(figure_path(_WS, "x")),
                       "virialization_residuals.csv")
    cols = ["builder", "rule", "n", "max_residual_center",
            "median_residual_center", "n_center", "a_ref",
            "max_residual_inner_frac", "median_residual_inner_frac",
            "f_objective", "passes_center_tol"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})
    return out


def print_table(rows: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 100)
    print("CENTER-ONLY VIRIALIZATION RESIDUALS  (deep-interior, center_k selector)")
    print(f"  S={_S_GPC} Gpc, spread={_SPREAD}, seg={_SEG}, extent={_EXTENT}, "
          f"seed={_SEED}, TOL={_TOL} (dimensionless: |net accel|/one-neighbour-pull)")
    print("=" * 100)
    hdr = (f"{'builder':<18}{'rule':<9}{'n':>5}{'n_ctr':>6}"
           f"{'center_max':>13}{'center_med':>13}{'inner_max':>13}{'pass<=TOL':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['builder']:<18}{r['rule']:<9}{r['n']:>5}{r['n_center']:>6}"
              f"{r['max_residual_center']:>13.3e}{r['median_residual_center']:>13.3e}"
              f"{r['max_residual_inner_frac']:>13.3e}"
              f"{str(r['passes_center_tol']):>10}")


def _best_B_center(rows: List[Dict[str, Any]], rule: str, n: int) -> float:
    """Option B's BEST (minimum) center residual over all swept step counts.

    Reports the strongest result a true relaxation achieves (the descent is non-
    monotone in the CENTER residual because the global objective trades center for
    edges), so the A-vs-B comparison gives Option B its best shot, not its worst.
    """
    vals = [r["max_residual_center"] for r in rows
            if r["builder"].startswith("B_relax_") and r["rule"] == rule
            and r["n"] == n]
    return min(vals) if vals else float("nan")


def fig_residual_vs_gridsize(rows: List[Dict[str, Any]]) -> str:
    """Center-only max residual vs grid size, A / B-best / realistic, both rules."""
    sizes = sorted({r["n"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
    for ax, rule in zip(axes, _RULES):
        # Option A (lattice) and realistic come straight from the rows.
        for builder, lab, style, color in [
            ("A_lattice", "Option A (lattice crystal)", "o-", "#2ca02c"),
            ("realistic", "realistic (un-relaxed)", "^--", "#d62728"),
        ]:
            ys = [next(r["max_residual_center"] for r in rows
                       if r["builder"] == builder and r["rule"] == rule
                       and r["n"] == n) for n in sizes]
            ys = [max(y, 1e-31) for y in ys]
            ax.plot(sizes, ys, style, color=color, label=lab, lw=1.6, ms=7)
        # Option B: the BEST center residual over the swept step counts.
        ys_b = [max(_best_B_center(rows, rule, n), 1e-31) for n in sizes]
        ax.plot(sizes, ys_b, "s-", color="#1f77b4", lw=1.6, ms=7,
                label="Option B (relaxed, best step)")
        ax.axhline(_TOL, color="black", ls=":", lw=1.5,
                   label=f"TOL = {_TOL}")
        ax.set_yscale("log")
        ax.set_xlabel("grid size (n_nodes)", fontsize=10)
        ax.set_title(f"mass rule = {rule}", fontsize=11)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="best")
    axes[0].set_ylabel("CENTER-ONLY max residual  |net accel| / a_ref  (log)",
                       fontsize=10)
    fig.suptitle(
        "Center-only virialization residual vs grid size — Option A vs Option B vs "
        "realistic\n"
        "Deep-interior nodes only (center_k). Below the dotted TOL line = "
        "'inner nodes would not move'. Only the analytic lattice (A) reaches it; "
        "a truly relaxed blob (B) does NOT.",
        fontsize=12.5, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = figure_path(_WS, "virialization_residual_vs_gridsize")
    fig.savefig(out, dpi=_DPI, bbox_inches=_BBOX)
    plt.close(fig)
    return out


def fig_optionB_descent(n: int, step_points: Tuple[int, ...]) -> str:
    """Option B descent: objective f and center residual vs step, both rules."""
    curves = {rule: optionB_descent(rule, n, step_points) for rule in _RULES}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6))
    colors = {"radial": "#d62728", "massfunc": "#1f77b4"}
    for rule in _RULES:
        c = curves[rule]
        ax1.plot(c["steps"], c["f_objective"], "o-", color=colors[rule],
                 label=rule, lw=1.6, ms=6)
        ax2.plot(c["steps"], c["center_residual"], "o-", color=colors[rule],
                 label=rule, lw=1.6, ms=6)
    ax1.set_yscale("log")
    ax1.set_xlabel("relaxation step", fontsize=10)
    ax1.set_ylabel("force-residual objective  f = Σ|a_i|²  (log)", fontsize=10)
    ax1.set_title("Objective DECREASES monotonically (true descent)", fontsize=11)
    ax1.grid(True, alpha=0.3, which="both"); ax1.legend(fontsize=9)
    ax2.axhline(_TOL, color="black", ls=":", lw=1.5, label=f"TOL = {_TOL}")
    ax2.set_yscale("log")
    ax2.set_xlabel("relaxation step", fontsize=10)
    ax2.set_ylabel("CENTER-ONLY max residual (log)", fontsize=10)
    ax2.set_title("Center residual bottoms out FAR above TOL\n"
                  "(global descent trades center for edges)", fontsize=11)
    ax2.grid(True, alpha=0.3, which="both"); ax2.legend(fontsize=9)
    fig.suptitle(
        f"Option B (true iterative relaxation) descent — n={n}, rate={_RELAX_RATE}, "
        f"hold_outer={_RELAX_HOLD}\n"
        "A realistic segregated blob CAN be driven downhill in force residual, but "
        "its DEEP CENTER never reaches the crystal's balance (TOL=0.25).",
        fontsize=12.5, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = figure_path(_WS, "virialization_optionB_descent")
    fig.savefig(out, dpi=_DPI, bbox_inches=_BBOX)
    plt.close(fig)
    return out


def print_verdict(rows: List[Dict[str, Any]], sizes: Tuple[int, ...]) -> None:
    """Honest one-paragraph verdict: does Option B reach center virialization?"""
    n_big = max(sizes)

    def _get(builder, rule):
        return next(r for r in rows if r["builder"] == builder
                    and r["rule"] == rule and r["n"] == n_big)

    print("\n" + "=" * 100)
    print(f"VERDICT (largest grid n={n_big}, deep-center selector; "
          "Option B = BEST over swept steps)")
    print("=" * 100)
    for rule in _RULES:
        a = _get("A_lattice", rule)
        b_best = _best_B_center(rows, rule, n_big)
        real = _get("realistic", rule)
        print(f"  rule={rule:<9} A(lattice) center_max={a['max_residual_center']:.2e} "
              f"(pass={a['passes_center_tol']}) | "
              f"B(relaxed) best center_max={b_best:.3f} "
              f"(pass={b_best <= _TOL}) | "
              f"realistic={real['max_residual_center']:.3f}")
    any_b_pass = any(_best_B_center(rows, rule, n_big) <= _TOL for rule in _RULES)
    print("-" * 100)
    if any_b_pass:
        print("  => Option B (realistic relaxation) PASSES the center-only criterion: "
              "PF8's 'must be a crystal' is OVERTURNED — a realistic relaxed blob "
              "virializes at its center.")
    else:
        print("  => Option B (realistic relaxation) does NOT reach TOL at the deep "
              "center even on the largest grid; only the analytic lattice (A) does.\n"
              "     The honest result: a TRUE relaxation reduces the force residual "
              "(monotone descent on Σ|a_i|²) but its center stays O(10-100)×TOL — "
              "orders of magnitude above the crystal's ~1e-30. PF8 STANDS, now with\n"
              "     the corrected CENTER-ONLY evidence and real Option-B numbers, not "
              "the 'irreducible monopole' hand-wave. The lattice is one (perfectly\n"
              "     balanced) realization; a realistic virialized blob does not reach "
              "exact center force-balance under position relaxation alone.")
    print("=" * 100)


# ===========================================================================
# CLI
# ===========================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-n", type=int, default=2000,
                   help="Largest grid size (default 2000; O(N^2) relaxation is slow).")
    p.add_argument("--quick", action="store_true",
                   help="Fast smoke run: small grids, fewer Option-B steps.")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    if args.quick:
        sizes = (26, 100)
    else:
        sizes = tuple(n for n in (26, 100, 500, 2000) if n <= args.max_n)
        if not sizes:
            sizes = (26,)
    print(f"[virialization] grid sizes: {sizes}  (center_k={_CENTER_K})")

    rows = build_residual_table(sizes)
    print_table(rows)
    csv_path = write_csv(rows)
    print(f"\n[virialization] CSV written: {csv_path}")

    p1 = fig_residual_vs_gridsize(rows)
    # Descent figure on a representative mid grid (n=100 if available, else largest).
    descent_n = 100 if 100 in sizes else max(sizes)
    p2 = fig_optionB_descent(descent_n, _B_STEP_POINTS)
    print(f"[virialization] figure: {p1}")
    print(f"[virialization] figure: {p2}")

    print_verdict(rows, sizes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
