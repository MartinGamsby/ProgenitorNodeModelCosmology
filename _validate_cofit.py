#!/usr/bin/env python3
"""
B4 — Co-fit VALIDATION (one-off, NOT a sweep axis)
==================================================

The user doubts the ternary co-fit works and reports the linear co-fit kept
PINNING S at the boundary. Rather than make the co-fit METHOD a sweep axis,
this one-off VALIDATES it cheaply on 1-2 representative cells: for each cell it
runs the S co-fit THREE ways on the SAME a(t)/scorer and compares the optimum.

  (a) linear   co-fit (the current sweep default, `linear_search_S`)
  (b) ternary  co-fit (`ternary_search_S`)
  (c) brute    — a FULL exhaustive S scan over `build_s_list(s_min, s_max)`
                 (the ground truth the other two approximate).

All three score with the AUTHORITATIVE chi2: the reconciled pantheon scorer
`compute_pantheon_metrics` reached through `worst_callback` (objective="pantheon",
the same path the real sweep uses). No new chi2 is invented here.

Representative cells (core S range s_min=3, s_max=35):
  * a NEAR-LCDM cell  : M=100  (weak tidal field — should not pin)
  * a STRONGER-field  : M=300  (stronger field — the case the user worries pins)
Both cube26 / uniform_sphere so this is a pure method check.

SPEED: this is a METHOD check, not the science run. particle_count is held at
PARTICLE_COUNT below (modest, documented) and n_steps at N_STEPS so the 3-way x
2-cell comparison runs in a couple of minutes. The science sweep uses 2000p/546;
the co-fit OPTIMUM location is insensitive to N in this band (PF4), so a modest N
is a faithful proxy for "which S does each method land on / does linear pin".

Decision rule (printed as the verdict):
  KEEP LINEAR for the core sweep IFF, for EVERY cell, linear AND ternary agree
  with brute force (best S within ONE grid step AND chi2/dof within ~0.01) AND
  linear does NOT pin at the s_min boundary. Otherwise recommend BRUTE-FORCE (or
  ternary) and report the exact discrepancy (especially a linear s_min pin).

Outputs (gitignored, under results/figures/ws1/):
  cofit_validation.csv   — per (cell, method): best_S, chi2/dof, pinned flags.
  cofit_validation.txt   — the same table + the verdict, as printed.

Usage:
  set PYTHONIOENCODING=utf-8 && python _validate_cofit.py
"""
from __future__ import annotations

from cosmo.encoding import configure_utf8_stdout
configure_utf8_stdout()

import csv
import math
import os
from typing import Dict, List, Optional, Tuple

import cosmo.parameter_sweep as ps
from cosmo.parameter_sweep import (
    build_s_list, linear_search_S, ternary_search_S, worst_callback,
    MatchWeights,
)
from cosmo.factories import setup_simulation_context
from cosmo.pantheon import load_pantheon
from cosmo.plots import figure_path

# Reuse the EXACT sweep machinery so the validated path == the production path.
from sweep import _make_sweep_config_for_cell, _make_sim_callback

# ---------------------------------------------------------------------------
# Validation settings (DOCUMENTED — this is a method check, not the science run)
# ---------------------------------------------------------------------------
PARTICLE_COUNT = 400     # modest for speed; the OPTIMUM-S location is N-insensitive
N_STEPS = 273            # matches the modest-N proxy; science run uses 2000p/546
T_START_GYR = 2.9
S_MIN, S_MAX = 3, 35     # the core S range (build_s_list(3,35) -> [3..30] step 1)

# Representative cells: near-LCDM (M=100) + stronger-field (M=300), cube26/uniform.
CELLS = [
    dict(label="near-LCDM (cube26, M=100)",
         cell=dict(M=100, amplitude=0.0, nm_seed=42, s_amplitude=0.0,
                   init="uniform_sphere", geometry="cube26")),
    dict(label="stronger-field (cube26, M=300)",
         cell=dict(M=300, amplitude=0.0, nm_seed=42, s_amplitude=0.0,
                   init="uniform_sphere", geometry="cube26")),
]

CHI2_TOL = 0.01          # "agree" if |chi2/dof - brute| <= this
GRID_STEP_TOL = 1        # "agree" if |best_S - brute_best_S| <= this (one grid step)


def _base_cfg() -> Dict:
    """Minimal run-config dict the sweep-config builder consumes."""
    return {
        "particle_count": PARTICLE_COUNT,
        "n_steps": N_STEPS,
        "t_start_Gyr": T_START_GYR,
        "s_min_gpc": S_MIN,
        "s_max_gpc": S_MAX,
        "centerM": 1,
        "objective": "pantheon",
        "s_cofit_method": "linear",
    }


def _brute_force_S(sweep_cfg, M, centerM, sim_cb, pantheon_data,
                   s_list: List[int]) -> Tuple[int, Dict[str, float]]:
    """Exhaustive S scan over s_list with the AUTHORITATIVE scorer (ground truth).

    Evaluates every S via worst_callback (objective="pantheon" ->
    compute_pantheon_metrics) and returns the S with the highest match_avg_pct
    (== lowest chi2/dof), plus that S's metrics. This is the same scoring
    linear/ternary use, just without any search shortcut.
    """
    best_S = None
    best_metrics = None
    for S in s_list:
        _, metrics = worst_callback(
            sim_cb, sweep_cfg, M, S, centerM, [42], None, MatchWeights(),
            pantheon_data=pantheon_data,
        )
        if best_metrics is None or metrics["match_avg_pct"] > best_metrics["match_avg_pct"]:
            best_metrics = metrics
            best_S = S
    return best_S, best_metrics


def _run_cell(label: str, cell: Dict, cfg: Dict,
              box_size_Gpc: float, a_start: float,
              pantheon_data: Dict, s_list: List[int]) -> List[Dict]:
    """Run linear / ternary / brute on ONE cell; return three result rows."""
    sweep_cfg = _make_sweep_config_for_cell(cell, cfg)
    sim_cb = _make_sim_callback(sweep_cfg, box_size_Gpc, a_start)
    centerM = cfg["centerM"]
    M = cell["M"]
    weights = MatchWeights()

    print(f"\n{'='*70}\nCELL: {label}\n{'='*70}")

    # (c) BRUTE FORCE — ground truth. Run first so its cache warms the others.
    print(f"[brute] exhaustive S scan over {s_list[0]}..{s_list[-1]} "
          f"({len(s_list)} values) ...")
    brute_S, brute_metrics = _brute_force_S(
        sweep_cfg, M, centerM, sim_cb, pantheon_data, s_list)
    brute_chi2 = brute_metrics.get("chi2_dof", float("inf"))
    print(f"[brute] best S={brute_S}  chi2/dof={brute_chi2:.4f}")

    # (a) LINEAR — the current default.
    print(f"[linear] linear_search_S [{S_MIN}..{S_MAX}] ...")
    lin_S, lin_dict, _, _ = linear_search_S(
        sweep_cfg, M, centerM, sim_cb, None, weights,
        S_MIN, S_MAX, prev_best_S=None, seeds=[42],
        pantheon_data=pantheon_data,
    )
    lin_chi2 = lin_dict.get("chi2_dof", float("inf"))
    print(f"[linear] best S={lin_S}  chi2/dof={lin_chi2:.4f}")

    # (b) TERNARY.
    print(f"[ternary] ternary_search_S [{S_MIN}..{S_MAX}] ...")
    tern_S, _, tern_dict, _ = ternary_search_S(
        sweep_cfg, M, centerM, sim_cb, None, weights,
        S_MIN, S_MAX, s_hint=None, seeds=[42],
        pantheon_data=pantheon_data,
    )
    tern_chi2 = tern_dict.get("chi2_dof", float("inf"))
    print(f"[ternary] best S={tern_S}  chi2/dof={tern_chi2:.4f}")

    rows = []
    for method, S, chi2 in (
        ("brute", brute_S, brute_chi2),
        ("linear", lin_S, lin_chi2),
        ("ternary", tern_S, tern_chi2),
    ):
        rows.append({
            "cell": label,
            "method": method,
            "best_S": S,
            "chi2_dof": chi2,
            "pinned_s_min": (S is not None and int(S) == S_MIN),
            "pinned_s_max": (S is not None and int(S) == s_list[-1]),
        })
    return rows


def _verdict(all_rows: List[Dict], s_list: List[int]) -> Tuple[str, List[str]]:
    """Apply the decision rule. Returns (RECOMMENDATION, detail lines)."""
    detail: List[str] = []
    keep_linear = True

    by_cell: Dict[str, Dict[str, Dict]] = {}
    for r in all_rows:
        by_cell.setdefault(r["cell"], {})[r["method"]] = r

    for cell, methods in by_cell.items():
        brute = methods["brute"]
        for name in ("linear", "ternary"):
            m = methods[name]
            ds = (abs(int(m["best_S"]) - int(brute["best_S"]))
                  if (m["best_S"] is not None and brute["best_S"] is not None)
                  else 999)
            dchi2 = (abs(m["chi2_dof"] - brute["chi2_dof"])
                     if (math.isfinite(m["chi2_dof"]) and math.isfinite(brute["chi2_dof"]))
                     else float("inf"))
            agrees = (ds <= GRID_STEP_TOL) and (dchi2 <= CHI2_TOL)
            detail.append(
                f"  {cell}: {name} S={m['best_S']} (brute S={brute['best_S']}, "
                f"dS={ds}) chi2/dof {m['chi2_dof']:.4f} vs {brute['chi2_dof']:.4f} "
                f"(dchi2={dchi2:.4f}) -> {'AGREE' if agrees else 'DISAGREE'}"
                + (" [PINS s_min]" if m["pinned_s_min"] else "")
            )
            if not agrees:
                keep_linear = False
        if methods["linear"]["pinned_s_min"]:
            keep_linear = False
            detail.append(f"  {cell}: linear PINS at s_min={S_MIN} "
                          f"(the user-expected failure mode)")

    if keep_linear:
        rec = ("KEEP LINEAR for the core sweep: linear & ternary agree with "
               "brute force (best S within one grid step, chi2/dof within "
               f"{CHI2_TOL}) and linear does NOT pin at s_min={S_MIN}.")
    else:
        # Prefer brute (exact) if it's affordable; ternary if it agreed everywhere.
        tern_ok = all(
            (abs(int(by_cell[c]["ternary"]["best_S"]) - int(by_cell[c]["brute"]["best_S"])) <= GRID_STEP_TOL
             and abs(by_cell[c]["ternary"]["chi2_dof"] - by_cell[c]["brute"]["chi2_dof"]) <= CHI2_TOL)
            for c in by_cell
            if by_cell[c]["ternary"]["best_S"] is not None
            and by_cell[c]["brute"]["best_S"] is not None
        )
        alt = "TERNARY" if tern_ok else "BRUTE-FORCE"
        rec = (f"SWITCH the core sweep co-fit to {alt}: linear disagreed with "
               "brute force (see DISAGREE / PINS lines above).")
    return rec, detail


def main() -> None:
    print("=" * 70)
    print("CO-FIT VALIDATION (B4) — linear vs ternary vs brute-force")
    print("=" * 70)
    print(f"  particle_count={PARTICLE_COUNT} (modest for speed), n_steps={N_STEPS}")
    print(f"  S range: build_s_list({S_MIN},{S_MAX})")

    s_list = build_s_list(S_MIN, S_MAX)
    print(f"  S grid ({len(s_list)}): {s_list}")

    t_dur = 13.8 - T_START_GYR
    box_size_Gpc, a_start, _ = setup_simulation_context(
        T_START_GYR, t_dur, N_STEPS, save_interval=10)
    pantheon_data = load_pantheon()
    print(f"  Loaded {pantheon_data['n']} SNe Ia")

    cfg = _base_cfg()
    all_rows: List[Dict] = []
    for spec in CELLS:
        all_rows += _run_cell(spec["label"], spec["cell"], cfg,
                              box_size_Gpc, a_start, pantheon_data, s_list)

    rec, detail = _verdict(all_rows, s_list)

    # ---- table ----
    header = f"{'cell':<32} {'method':<8} {'best_S':>6} {'chi2/dof':>9} {'pin?':>10}"
    lines = ["", "=" * 70, "CO-FIT VALIDATION TABLE", "=" * 70, header, "-" * len(header)]
    for r in all_rows:
        pin = ("s_min" if r["pinned_s_min"] else
               "s_max" if r["pinned_s_max"] else "no")
        chi2 = (f"{r['chi2_dof']:.4f}" if math.isfinite(r["chi2_dof"]) else "inf")
        lines.append(f"{r['cell']:<32} {r['method']:<8} {str(r['best_S']):>6} "
                     f"{chi2:>9} {pin:>10}")
    lines += ["", "VERDICT (decision rule):"] + detail
    lines += ["", "RECOMMENDATION:", "  " + rec, ""]
    report = "\n".join(lines)
    print(report)

    # ---- save (gitignored) ----
    csv_png = figure_path("ws1", "cofit_validation")
    out_dir = os.path.dirname(csv_png)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "cofit_validation.csv")
    txt_path = os.path.join(out_dir, "cofit_validation.txt")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "cell", "method", "best_S", "chi2_dof", "pinned_s_min", "pinned_s_max"])
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(f"[saved] {csv_path}")
    print(f"[saved] {txt_path}")


if __name__ == "__main__":
    main()
