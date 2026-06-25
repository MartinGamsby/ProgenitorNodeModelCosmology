#!/usr/bin/env python3
"""
Hubble Diagram — From-Sim N-body a(t) vs Pantheon+SH0ES  (Stage 1 gating)

Standalone script that:
  1. Runs ONE real N-body External-Node simulation to obtain a(t).
  2. Converts a(t) -> mu(z) via the simulation-based distance kernel
     (cosmo.sim_distance.sim_to_distance_modulus).
  3. Clips the Pantheon+SH0ES data to the sim-covered redshift range.
  4. Evaluates chi^2/R^2 for THREE curves on that clipped data:
       - From-sim External-Node N-body   (the real mechanism)
       - Analytic LCDM                   (comparison / reference)
       - Analytic matter-only            (comparison)
  5. DEVIATION DIAGNOSTIC: prints max and RMS of
     (mu_from_sim - mu_LCDM) offset-marginalized over the covered range.
     States plainly whether the deviation exceeds a typical data sigma.
  6. Saves a 2-panel PNG (top: data + curves; bottom: Delta-mu vs LCDM
     residuals with LCDM as zero reference).
  7. Writes a JSON sidecar next to the PNG with per-model stats + config.

Scientific purpose
------------------
Answer the gating question: does the real nonlinear N-body a(t) deviate
from LCDM within z <= ~2.3, or does it hug LCDM?  If the from-sim curve
hugs LCDM, the honest conclusion is "viable, currently indistinguishable
from LCDM with this data" — this script reports that faithfully.

Timing constraint
-----------------
t_start + t_duration MUST equal 13.8 Gyr so the last snapshot is z = 0.
t_duration is therefore DERIVED as 13.8 - t_start; --t-start controls it.
The full-coverage default (t_start=2.9) covers up to z ~ 2.3; at t_start=5.8
the sim only reaches z ~ 1.2.

Usage
-----
    python hubble_diagram_nbody.py                        # defaults (t_start=2.9)
    python hubble_diagram_nbody.py --t-start 5.8 --particles 80 --n-steps 300
    python hubble_diagram_nbody.py --from-best-config results/sweep_results_pantheon.csv
    python hubble_diagram_nbody.py --output-dir ./results --z-min 0.023

Requires the real Pantheon+SH0ES data file — see data/pantheon_plus/README.md.
"""

import argparse
import json
import math
import os
import sys

# ---------------------------------------------------------------------------
# UTF-8 stdout/stderr reconfiguration (Windows cp1252 guard).
# Must happen BEFORE any print() that might emit Greek characters (chi, Omega).
# ---------------------------------------------------------------------------
from cosmo.encoding import configure_utf8_stdout
configure_utf8_stdout()

# ---------------------------------------------------------------------------
# Headless backend — BEFORE pyplot is imported anywhere in this process.
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from cosmo.cli import add_common_arguments, args_to_sim_params
from cosmo.constants import SimulationParameters
from cosmo.distances import model_distance_modulus
from cosmo.factories import run_external_node_simulation, setup_simulation_context
from cosmo.parameter_sweep import expected_growth_factor, GROWTH_ANCHOR_TOL
import cosmo.hubble_diagram as hd_engine
import cosmo.pantheon as pantheon_loader
from cosmo.sim_distance import sim_to_distance_modulus
from cosmo.visualization import generate_output_filename


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TODAY_GYR: float = 13.8
_MIN_DT_GYR: float = 0.05      # dt must be < this; bump n_steps if not
_DEFAULT_T_START: float = 2.9  # Gyr — full-coverage default (z up to ~2.3)
_DEFAULT_PARTICLES: int = 2000  # full-coverage default
_DEFAULT_N_STEPS: int = math.ceil((_TODAY_GYR - _DEFAULT_T_START) / 0.04)
# dt = (13.8 - 2.9) / 273 ~ 0.040 Gyr < _MIN_DT_GYR = 0.05 ✓

# ---------------------------------------------------------------------------
# Color/style convention (mirrors hubble_diagram.py)
# ---------------------------------------------------------------------------
_MODEL_STYLES = {
    "external_node_nbody": {"color": "#ff7f0e", "ls": "-",  "lw": 2.5,
                             "label": "Ext-Node N-body (from sim)"},
    "lcdm":                {"color": "#1f77b4", "ls": "--", "lw": 2.0,
                             "label": "LCDM (analytic)"},
    "analytic_shortcut":   {"color": "#9467bd", "ls": ":",  "lw": 1.8,
                             "label": "Analytic shortcut (const-Omega_Lambda_eff)"},
    "einstein_de_sitter":  {"color": "#2ca02c", "ls": "-.", "lw": 1.8,
                             "label": "Matter-only, no Λ (Ωm=1)"},
}

_PLOT_ORDER = ("external_node_nbody", "lcdm", "analytic_shortcut", "einstein_de_sitter")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Hubble-diagram comparison: from-sim N-body a(t) vs Pantheon+SH0ES.\n"
            "Runs one N-body simulation, converts a(t)->mu(z), compares to real\n"
            "data clipped to the sim-covered z-range. Saves a 2-panel PNG and a\n"
            "JSON sidecar with per-model chi2/dof + config for reproducibility.\n"
            "\n"
            "Full-coverage default: t_start=2.9 Gyr, particles=2000,\n"
            f"n_steps={_DEFAULT_N_STEPS} (dt~0.040 Gyr, z coverage up to ~2.3).\n"
            "For a quick/cheap run use --t-start 5.8 --particles 80 --n-steps 300."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--output-dir", type=str, default="./results",
        help="Directory to save the output figure and JSON sidecar.",
    )
    parser.add_argument(
        "--pantheon-path", type=str, default=None,
        help=(
            "Path to Pantheon+SH0ES.dat. Defaults to the vendored location "
            "data/pantheon_plus/Pantheon+SH0ES.dat."
        ),
    )
    parser.add_argument(
        "--z-min", type=float, default=0.01,
        help="Minimum redshift cut applied to the data before z-range clipping.",
    )
    parser.add_argument(
        "--n-bins", type=int, default=20,
        help="Number of log-z bins for the binned data overlay.",
    )
    parser.add_argument(
        "--from-best-config", type=str, default=None, metavar="PATH",
        help=(
            "Path to a sweep results CSV (e.g. results/sweep_results_pantheon.csv). "
            "The best row (lowest chi2_dof) is loaded and its M, S, centerM are "
            "used as defaults. Explicit --M / --S / --center-node-mass flags override "
            "the loaded values."
        ),
    )
    add_common_arguments(parser)

    # Override add_common_arguments defaults for this script's physics.
    # Full-coverage defaults: t_start=2.9, particles=2000, n_steps=273.
    parser.set_defaults(
        M=855.0,
        S=37.8,
        particles=_DEFAULT_PARTICLES,
        n_steps=_DEFAULT_N_STEPS,
        t_start=_DEFAULT_T_START,
    )

    return parser


# ---------------------------------------------------------------------------
# Best-config loader
# ---------------------------------------------------------------------------

def load_best_config(csv_path: str) -> dict:
    """
    Load the best-fit configuration from a pantheon sweep results CSV.

    Reads the CSV produced by the pantheon parameter sweep, finds the row with
    the lowest chi2_dof (best fit), and returns a dict with keys:
        'M'       – float, M_factor (external node mass)
        'S'       – float, S_gpc (node separation in Gpc)
        'centerM' – float, centerM (central node mass multiplier)
        'chi2_dof' – float, reduced chi-squared of the best row
        'chi2'     – float, chi-squared of the best row (if present)
        'R2'       – float, R^2 of the best row (if present)

    Args:
        csv_path: Path to the sweep CSV file.

    Returns:
        Dict with at minimum 'M', 'S', 'centerM'.

    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError:        If required columns are missing or the file is empty.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"--from-best-config: sweep CSV not found: {csv_path!r}\n"
            "Run the pantheon parameter sweep first to generate this file."
        )

    try:
        import csv as _csv
        rows = []
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = _csv.DictReader(fh)
            for row in reader:
                rows.append(row)
    except Exception as exc:
        raise ValueError(
            f"--from-best-config: failed to read CSV {csv_path!r}: {exc}"
        ) from exc

    if not rows:
        raise ValueError(
            f"--from-best-config: sweep CSV is empty: {csv_path!r}"
        )

    required = {"M_factor", "S_gpc", "centerM"}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(
            f"--from-best-config: CSV {csv_path!r} is missing required columns: "
            f"{sorted(missing)}. "
            f"Available columns: {sorted(rows[0].keys())}"
        )

    # Find best row: prefer chi2_dof (pantheon sweep), fall back to diff_pct
    if "chi2_dof" in rows[0]:
        sort_key = "chi2_dof"
    elif "diff_pct" in rows[0]:
        sort_key = "diff_pct"
    else:
        sort_key = None

    if sort_key is not None:
        # Filter to rows whose sort key parses as a finite float, then take the
        # minimum. A single malformed/empty cell must NOT abort the whole sort
        # and silently fall back to rows[0] (which is not the best fit).
        def _sortable(r):
            try:
                v = float(r[sort_key])
            except (KeyError, ValueError, TypeError):
                return None
            return v if math.isfinite(v) else None

        scored = [(v, r) for r in rows if (v := _sortable(r)) is not None]
        if scored:
            best = min(scored, key=lambda vr: vr[0])[1]
        else:
            # No row has a parseable sort key — fall back to the first row.
            best = rows[0]
    else:
        best = rows[0]

    def _float(key: str, default=None):
        try:
            return float(best[key])
        except (KeyError, ValueError, TypeError):
            return default

    result = {
        "M":        _float("M_factor"),
        "S":        _float("S_gpc"),
        "centerM":  _float("centerM"),
        "chi2_dof": _float("chi2_dof"),
        "chi2":     _float("chi2"),
        "R2":       _float("R2"),
    }

    if result["M"] is None or result["S"] is None:
        raise ValueError(
            f"--from-best-config: could not parse M_factor / S_gpc from "
            f"best row in {csv_path!r}: {dict(best)}"
        )

    return result


# ---------------------------------------------------------------------------
# Core logic (callable for testing without argparse)
# ---------------------------------------------------------------------------

def run(
    sim_params: SimulationParameters,
    output_dir: str = "./results",
    pantheon_path=None,
    z_min: float = 0.01,
    n_bins: int = 20,
) -> dict:
    """
    Run the full Hubble-diagram comparison.

    Runs the N-body simulation, converts a(t)->mu(z), evaluates chi^2/R^2
    on the Pantheon+ data clipped to the sim-covered z-range, prints the
    deviation diagnostic table, saves a 2-panel PNG, and writes a JSON
    sidecar next to the PNG with per-model stats + config.

    Args:
        sim_params:    SimulationParameters with t_start_Gyr and t_duration_Gyr
                       set so that t_start + t_duration == 13.8 Gyr.
        output_dir:    Directory for the output PNG and JSON sidecar.
        pantheon_path: Path to Pantheon+SH0ES.dat (None = default location).
        z_min:         Minimum redshift cut before z-range clipping.
        n_bins:        Number of log-z bins for the plot overlay.

    Returns:
        dict with keys:
            'results_in_range' – per-model evaluate_* results on clipped data.
            'n_in_range'       – number of SNe in the sim-covered range.
            'n_dropped'        – number of SNe outside the sim-covered range.
            'z_cover'          – (z_min_cover, z_max_cover) tuple.
            'deviation_max'    – max |mu_sim - mu_lcdm| after offset marg.
            'deviation_rms'    – RMS of (mu_sim - mu_lcdm) after offset marg.
            'typical_sigma'    – median data sigma (for comparison).
            'growth_factor'    – model a(today)/a(t_start).
            'growth_target'    – physical (LCDM) a(today)/a(t_start).
            'anchor_ok'        – bool, whether growth anchor passes tolerance.
            'out_path'         – path to saved PNG.
            'sidecar_path'     – path to saved JSON sidecar.

    Raises:
        FileNotFoundError: If the Pantheon+ data file is absent.
        ValueError:        If timing constraint is violated or no in-range SNe.
    """
    t_start = sim_params.t_start_Gyr
    t_duration = sim_params.t_duration_Gyr

    # ------------------------------------------------------------------
    # 0. Pre-flight checks
    # ------------------------------------------------------------------
    t_end = t_start + t_duration
    if abs(t_end - _TODAY_GYR) > 0.05:
        raise ValueError(
            f"t_start ({t_start}) + t_duration ({t_duration}) = {t_end:.3f} Gyr, "
            f"but must equal {_TODAY_GYR} Gyr (today). "
            f"Set t_duration = {_TODAY_GYR} - t_start."
        )

    dt = t_duration / sim_params.n_steps
    if dt >= _MIN_DT_GYR:
        needed = int(np.ceil(t_duration / _MIN_DT_GYR)) + 1
        raise ValueError(
            f"dt = t_duration / n_steps = {t_duration:.2f} / {sim_params.n_steps} = "
            f"{dt:.4f} Gyr >= {_MIN_DT_GYR} Gyr. "
            f"Increase --n-steps to at least {needed} (currently {sim_params.n_steps})."
        )

    # ------------------------------------------------------------------
    # 1. Load Pantheon+SH0ES
    # ------------------------------------------------------------------
    data = pantheon_loader.load_pantheon(path=pantheon_path, z_min=z_min)
    print(
        f"Loaded {data['n']} SNe Ia  (z_min={z_min}, "
        f"z range: {data['z'].min():.4f} - {data['z'].max():.4f})"
    )

    # ------------------------------------------------------------------
    # 2. Set up and run the N-body simulation
    # ------------------------------------------------------------------
    print(
        f"\nRunning N-body simulation: "
        f"t_start={t_start} Gyr, t_duration={t_duration:.2f} Gyr, "
        f"n_steps={sim_params.n_steps}, particles={sim_params.n_particles}, "
        f"M={sim_params.M_value}, S={sim_params.S_value} ..."
    )

    box_size_Gpc, a_start, _baseline = setup_simulation_context(
        t_start, t_duration, sim_params.n_steps
    )
    ext = run_external_node_simulation(sim_params, box_size_Gpc, a_start)

    a_sim = ext["a"]
    t_Gyr_sim = ext["t_Gyr"]  # relative, starts at 0

    print(
        f"Simulation done. Snapshots: {len(a_sim)}, "
        f"a range: [{a_sim.min():.4f}, {a_sim.max():.4f}]"
    )

    # Physical expansion anchor: the model's total growth a(today)/a(t_start) must
    # match the real ~1+z(t_start), else its renormalized shape is physically
    # meaningless (a runaway config can fit the z-window while expanding absurdly).
    model_growth = float(a_sim[-1] / a_sim[0])
    target_growth = expected_growth_factor(t_start)
    growth_dev = abs(model_growth / target_growth - 1.0)
    anchor_ok = growth_dev <= GROWTH_ANCHOR_TOL
    print(
        f"Expansion anchor: model a(today)/a(t_start) = {model_growth:.3f}  "
        f"vs physical {target_growth:.3f}  "
        f"({growth_dev*100:.1f}% off -> {'PHYSICAL' if anchor_ok else 'UNPHYSICAL, would be rejected by sweep'})"
    )

    # ------------------------------------------------------------------
    # 3. Convert a(t) -> mu(z); clip data to sim-covered z-range
    # ------------------------------------------------------------------
    sim_dist = sim_to_distance_modulus(
        data["z"], a_sim, t_Gyr_sim, t_start_Gyr=t_start
    )

    in_range_mask = sim_dist["in_range"]
    z_in = data["z"][in_range_mask]
    mu_in = data["mu"][in_range_mask]
    sigma_in = data["sigma"][in_range_mask]
    mu_sim_in = sim_dist["mu"]
    z_cover = sim_dist["z_cover"]

    n_in_range = int(np.sum(in_range_mask))
    n_dropped = data["n"] - n_in_range

    print(
        f"\nSim covers z = [{z_cover[0]:.4f}, {z_cover[1]:.4f}]. "
        f"SNe in range: {n_in_range} / {data['n']} "
        f"({n_dropped} dropped outside sim z-range)."
    )

    # ------------------------------------------------------------------
    # 4. Evaluate models on the IN-RANGE data subset
    # ------------------------------------------------------------------
    results = {}

    # From-sim External-Node
    results["external_node_nbody"] = hd_engine.evaluate_precomputed(
        z_in, mu_in, sigma_in, mu_sim_in,
        model_name="external_node_nbody",
    )

    # Analytic LCDM (comparison / zero reference for residual panel)
    results["lcdm"] = hd_engine.evaluate_model(
        z_in, mu_in, sigma_in, model="lcdm"
    )

    # Analytic matter-only (comparison line)
    # Einstein-de Sitter (flat Omega_m=1): the meaningful "no dark energy" null
    # that the SN data decisively rule out. (The open Omega_m=0.3 "matter_only" is
    # nearly degenerate with LCDM in the Hubble diagram and is a misleading null.)
    results["einstein_de_sitter"] = hd_engine.evaluate_model(
        z_in, mu_in, sigma_in, model="einstein_de_sitter"
    )

    # Analytic constant-Omega_Lambda_eff shortcut (old circular approach)
    analytic_shortcut_ok = False
    try:
        results["analytic_shortcut"] = hd_engine.evaluate_model(
            z_in, mu_in, sigma_in,
            model="external_node",
            sim_params=sim_params,
        )
        # Rename the model label for display
        results["analytic_shortcut"]["model"] = "analytic_shortcut"
        analytic_shortcut_ok = True
    except ValueError as exc:
        print(
            f"\nAnalytic shortcut (const-Omega_Lambda_eff) undefined "
            f"(turnaround) for this config: {exc}"
        )
        results["analytic_shortcut"] = None

    # ------------------------------------------------------------------
    # 5. Deviation diagnostic
    # ------------------------------------------------------------------
    # Compute on a dense z grid within the sim-covered range for the plot.
    z_dense = np.linspace(z_cover[0] + 1e-4, z_cover[1] - 1e-4, 500)

    mu_sim_dense = _interp_sim_on_dense(sim_dist, z_dense)
    mu_lcdm_dense = model_distance_modulus(z_dense, "lcdm")

    # Offset-marginalize both on the same dense grid for a fair shape comparison.
    # Use the DeltaM values fit on the real data so the comparison is meaningful.
    deltaM_sim = results["external_node_nbody"]["DeltaM"]
    deltaM_lcdm = results["lcdm"]["DeltaM"]

    mu_sim_shifted = mu_sim_dense + deltaM_sim
    mu_lcdm_shifted = mu_lcdm_dense + deltaM_lcdm

    # Differences (NaN-safe)
    diff = mu_sim_shifted - mu_lcdm_shifted
    valid = np.isfinite(diff)
    if np.any(valid):
        dev_max = float(np.max(np.abs(diff[valid])))
        dev_rms = float(np.sqrt(np.mean(diff[valid] ** 2)))
    else:
        dev_max = float("nan")
        dev_rms = float("nan")

    typical_sigma = float(np.median(sigma_in))

    # ------------------------------------------------------------------
    # 6. Print results
    # ------------------------------------------------------------------
    _print_deviation_diagnostic(
        dev_max, dev_rms, typical_sigma, z_cover,
        n_in_range, n_dropped,
    )
    _print_table(results, sim_params, analytic_shortcut_ok)

    # ------------------------------------------------------------------
    # 7. Plot
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    fig = _make_figure(
        data, in_range_mask, results,
        z_dense, mu_sim_shifted, mu_lcdm_shifted,
        sim_params, z_cover, analytic_shortcut_ok,
        n_bins=n_bins,
    )

    out_path = generate_output_filename(
        "hubble_diagram_nbody", sim_params, "png", output_dir,
        include_timestamp=True, include_S=True, include_M=True, include_D=False,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out_path}")

    # ------------------------------------------------------------------
    # 8. JSON sidecar
    # ------------------------------------------------------------------
    sidecar_path = out_path + ".summary.json"
    summary = _build_sidecar(
        sim_params=sim_params,
        results=results,
        n_in_range=n_in_range,
        n_dropped=n_dropped,
        z_cover=z_cover,
        dev_max=dev_max,
        dev_rms=dev_rms,
        typical_sigma=typical_sigma,
        model_growth=model_growth,
        target_growth=target_growth,
        anchor_ok=anchor_ok,
    )
    with open(sidecar_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Sidecar written: {sidecar_path}")

    return {
        "results_in_range": results,
        "n_in_range": n_in_range,
        "n_dropped": n_dropped,
        "z_cover": z_cover,
        "deviation_max": dev_max,
        "deviation_rms": dev_rms,
        "typical_sigma": typical_sigma,
        "growth_factor": model_growth,
        "growth_target": target_growth,
        "anchor_ok": anchor_ok,
        "out_path": out_path,
        "sidecar_path": sidecar_path,
    }


# ---------------------------------------------------------------------------
# Sidecar builder
# ---------------------------------------------------------------------------

def _build_sidecar(
    sim_params: SimulationParameters,
    results: dict,
    n_in_range: int,
    n_dropped: int,
    z_cover: tuple,
    dev_max: float,
    dev_rms: float,
    typical_sigma: float,
    model_growth: float,
    target_growth: float,
    anchor_ok: bool,
) -> dict:
    """
    Build the JSON-serializable sidecar dict.

    Contains: config (M, S, centerM, t_start, particles, n_steps),
    coverage stats, per-model chi2/dof/R2/DeltaM, deviation diagnostics,
    and growth anchor.
    """
    def _model_entry(key: str) -> dict | None:
        r = results.get(key)
        if r is None:
            return None
        return {
            "chi2":     float(r["chi2"]),
            "dof":      int(r["dof"]),
            "chi2_dof": float(r["chi2_dof"]),
            "R2":       float(r["R2"]),
            "DeltaM":   float(r["DeltaM"]),
        }

    return {
        "config": {
            "M":          float(sim_params.M_value),
            "S":          float(sim_params.S_value),
            "centerM":    float(sim_params.center_node_mass),
            "t_start":    float(sim_params.t_start_Gyr),
            "particles":  int(sim_params.n_particles),
            "n_steps":    int(sim_params.n_steps),
        },
        "coverage": {
            "n_in_range": n_in_range,
            "n_dropped":  n_dropped,
            "z_cover":    list(z_cover),
        },
        "models": {
            "external_node_nbody": _model_entry("external_node_nbody"),
            "lcdm":                _model_entry("lcdm"),
            "analytic_shortcut":   _model_entry("analytic_shortcut"),
            "einstein_de_sitter":  _model_entry("einstein_de_sitter"),
        },
        "deviation": {
            "deviation_max":   dev_max,
            "deviation_rms":   dev_rms,
            "typical_sigma":   typical_sigma,
        },
        "growth_anchor": {
            "growth_factor": model_growth,
            "growth_target": target_growth,
            "anchor_ok":     anchor_ok,
        },
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _interp_sim_on_dense(sim_dist: dict, z_dense: np.ndarray) -> np.ndarray:
    """
    Interpolate the from-sim mu curve onto a dense z grid.

    sim_to_distance_modulus returns 'z' and 'mu' on the original data z_target
    grid (already clipped to the sim-covered range). For a dense plotting grid
    we simply re-interpolate from that curve; NaN outside the covered range.
    """
    z_src = sim_dist["z"]    # on z_target (data) grid, in range
    mu_src = sim_dist["mu"]  # corresponding mu
    return np.interp(z_dense, z_src, mu_src, left=np.nan, right=np.nan)


def _print_deviation_diagnostic(
    dev_max: float,
    dev_rms: float,
    typical_sigma: float,
    z_cover: tuple,
    n_in_range: int,
    n_dropped: int,
) -> None:
    """Print the Stage-1 gating deviation summary."""
    print("\n" + "=" * 68)
    print("DEVIATION DIAGNOSTIC  (from-sim N-body vs analytic LCDM)")
    print("=" * 68)
    print(f"  Redshift range compared : z = [{z_cover[0]:.4f}, {z_cover[1]:.4f}]")
    print(f"  SNe used                : {n_in_range}  ({n_dropped} outside sim range)")
    print(f"  Typical data sigma      : {typical_sigma:.4f} mag")
    print(f"  Max |mu_sim - mu_LCDM|  : {dev_max:.4f} mag  (offset-marginalized)")
    print(f"  RMS (mu_sim - mu_LCDM) : {dev_rms:.4f} mag  (offset-marginalized)")
    if np.isfinite(dev_max) and np.isfinite(typical_sigma) and typical_sigma > 0:
        ratio = dev_max / typical_sigma
        if ratio < 0.5:
            verdict = (
                "INDISTINGUISHABLE from LCDM within this data range. "
                "Deviation < 0.5 sigma — cannot be resolved with current data."
            )
        elif ratio < 1.0:
            verdict = (
                f"Marginal deviation ({ratio:.2f} x sigma). "
                "Borderline detectable — not conclusive."
            )
        else:
            verdict = (
                f"DETECTABLE deviation ({ratio:.2f} x sigma). "
                "From-sim curve differs from LCDM beyond the typical uncertainty."
            )
        print(f"\n  Verdict: {verdict}")
    print("=" * 68 + "\n")


def _print_table(results: dict, sim_params, analytic_shortcut_ok: bool) -> None:
    """Print a compact per-model statistics table."""
    Omega_eff = sim_params.external_params.Omega_Lambda_eff
    print(
        f"External-Node config: "
        f"M={sim_params.M_value}, S={sim_params.S_value}, "
        f"Omega_Lambda_eff={Omega_eff:.4f}\n"
    )

    display_order = [
        "external_node_nbody",
        "lcdm",
        "analytic_shortcut",
        "einstein_de_sitter",
    ]
    labels = {
        "external_node_nbody": "Ext-Node N-body",
        "lcdm":                "LCDM (analytic)",
        "analytic_shortcut":   "Analytic shortcut",
        "einstein_de_sitter":  "Matter-only (no L)",
    }

    header = f"{'Model':<22} {'chi2':>10} {'dof':>6} {'chi2/dof':>10} {'R2':>8}"
    print(header)
    print("-" * len(header))
    for key in display_order:
        r = results.get(key)
        if r is None:
            print(f"{labels[key]:<22} {'(turnaround — undefined)':>36}")
            continue
        print(
            f"{labels[key]:<22} {r['chi2']:>10.2f} {r['dof']:>6d} "
            f"{r['chi2_dof']:>10.4f} {r['R2']:>8.6f}"
        )


def _make_figure(
    data: dict,
    in_range_mask: np.ndarray,
    results: dict,
    z_dense: np.ndarray,
    mu_sim_shifted: np.ndarray,
    mu_lcdm_shifted: np.ndarray,
    sim_params,
    z_cover: tuple,
    analytic_shortcut_ok: bool,
    n_bins: int = 20,
) -> plt.Figure:
    """
    Build the 2-panel Hubble-diagram figure.

    Top panel  : data (scatter + binned errorbars) + model curves (mu vs z).
    Bottom panel: Delta-mu vs LCDM residuals. LCDM is the zero reference line;
                  the binned Ext-Node residual (mu_obs - mu_fit_LCDM) is
                  overlaid so deviations from LCDM are directly readable.
    """
    fig = plt.figure(figsize=(11, 8))
    gs = gridspec.GridSpec(
        2, 1, height_ratios=[3, 1], hspace=0.08, figure=fig
    )
    ax_top = fig.add_subplot(gs[0])
    ax_res = fig.add_subplot(gs[1], sharex=ax_top)

    # Full data (faint background)
    z_all = data["z"]
    mu_all = data["mu"]
    sigma_all = data["sigma"]
    ax_top.scatter(
        z_all, mu_all,
        s=1.5, alpha=0.08, color="gray", zorder=1, rasterized=True,
        label="_nolegend_",
    )

    # In-range data
    z_in = z_all[in_range_mask]
    mu_in = mu_all[in_range_mask]
    sigma_in = sigma_all[in_range_mask]
    n_in = int(np.sum(in_range_mask))

    ax_top.scatter(
        z_in, mu_in,
        s=2.0, alpha=0.20, color="#555555", zorder=2, rasterized=True,
        label="_nolegend_",
    )

    # Binned in-range data
    binned = pantheon_loader.bin_for_plot(z_in, mu_in, sigma_in, n_bins=n_bins)
    ax_top.errorbar(
        binned["z"], binned["mu"], yerr=binned["err"],
        fmt="o", color="black", ms=4, lw=1.2, capsize=2.5, zorder=3,
        label=f"Pantheon+SH0ES ({n_in} SNe in range, {len(binned['z'])} bins)",
    )

    # Vertical markers for sim z-range
    for zv in z_cover:
        ax_top.axvline(zv, color="gray", ls=":", lw=0.8, alpha=0.7)

    # --- From-sim N-body curve ---
    st_nbody = _MODEL_STYLES["external_node_nbody"]
    r_nbody = results["external_node_nbody"]
    ax_top.plot(
        z_dense, mu_sim_shifted,
        color=st_nbody["color"], ls=st_nbody["ls"], lw=st_nbody["lw"],
        label=f"{st_nbody['label']}  chi2/dof={r_nbody['chi2_dof']:.3f}, R2={r_nbody['R2']:.4f}",
        zorder=5,
    )

    # --- Analytic LCDM curve ---
    st_lcdm = _MODEL_STYLES["lcdm"]
    r_lcdm = results["lcdm"]
    ax_top.plot(
        z_dense, mu_lcdm_shifted,
        color=st_lcdm["color"], ls=st_lcdm["ls"], lw=st_lcdm["lw"],
        label=f"{st_lcdm['label']}  chi2/dof={r_lcdm['chi2_dof']:.3f}, R2={r_lcdm['R2']:.4f}",
        zorder=4,
    )

    # --- Analytic shortcut curve (if defined) ---
    if analytic_shortcut_ok and results.get("analytic_shortcut") is not None:
        r_sc = results["analytic_shortcut"]
        # Compute dense curve with the shortcut DeltaM
        mu_sc_dense = model_distance_modulus(z_dense, "external_node", sim_params=sim_params)
        mu_sc_shifted = mu_sc_dense + r_sc["DeltaM"]
        st_sc = _MODEL_STYLES["analytic_shortcut"]
        ax_top.plot(
            z_dense, mu_sc_shifted,
            color=st_sc["color"], ls=st_sc["ls"], lw=st_sc["lw"],
            label=f"{st_sc['label']}  chi2/dof={r_sc['chi2_dof']:.3f}, R2={r_sc['R2']:.4f}",
            zorder=3,
        )

    # --- Matter-only (Einstein-de Sitter) null curve ---
    r_mo = results["einstein_de_sitter"]
    mu_mo_dense = model_distance_modulus(z_dense, "einstein_de_sitter")
    mu_mo_shifted = mu_mo_dense + r_mo["DeltaM"]
    st_mo = _MODEL_STYLES["einstein_de_sitter"]
    ax_top.plot(
        z_dense, mu_mo_shifted,
        color=st_mo["color"], ls=st_mo["ls"], lw=st_mo["lw"],
        label=f"{st_mo['label']}  chi2/dof={r_mo['chi2_dof']:.3f}, R2={r_mo['R2']:.4f}",
        zorder=2,
    )

    ax_top.set_xscale("log")
    ax_top.set_ylabel("Distance modulus mu [mag]", fontsize=12)
    ax_top.legend(fontsize=8.5, loc="upper left")
    ax_top.set_title(
        f"Hubble Diagram — N-body a(t) vs Pantheon+SH0ES\n"
        f"M={sim_params.M_value}, S={sim_params.S_value}, "
        f"Omega_Lambda_eff={sim_params.external_params.Omega_Lambda_eff:.4f}, "
        f"t_start={sim_params.t_start_Gyr} Gyr  "
        f"[sim range: z={z_cover[0]:.3f}-{z_cover[1]:.3f}]",
        fontsize=10,
    )
    ax_top.grid(True, alpha=0.3, which="both")
    plt.setp(ax_top.get_xticklabels(), visible=False)

    # -----------------------------------------------------------------------
    # --- Residual panel: Delta-mu vs LCDM ---
    # LCDM is the zero reference. We plot:
    #   - a horizontal zero line (= LCDM)
    #   - binned (mu_obs - mu_fit_LCDM) with errorbars  (data residuals to LCDM)
    #   - binned (mu_obs - mu_fit_nbody) with errorbars (data residuals to N-body)
    # This makes the two models' shapes directly comparable against the data.
    # -----------------------------------------------------------------------
    ax_res.axhline(0.0, color=_MODEL_STYLES["lcdm"]["color"],
                   lw=1.5, ls="--", zorder=2,
                   label="LCDM (zero reference, Delta-mu = 0)")
    for zv in z_cover:
        ax_res.axvline(zv, color="gray", ls=":", lw=0.8, alpha=0.7)

    # Scatter: raw per-SN residuals for both models (faint, for context)
    for key, st_key in [
        ("external_node_nbody", "external_node_nbody"),
        ("lcdm", "lcdm"),
    ]:
        st = _MODEL_STYLES[st_key]
        ax_res.scatter(
            z_in, results[key]["residuals"],
            s=1.5, alpha=0.18, color=st["color"], rasterized=True, zorder=3,
        )

    # Binned residuals for the N-body curve (delta-mu vs LCDM visible as offset)
    bin_res_nbody = pantheon_loader.bin_for_plot(
        z_in, results["external_node_nbody"]["residuals"], sigma_in, n_bins=n_bins
    )
    ax_res.errorbar(
        bin_res_nbody["z"], bin_res_nbody["mu"], yerr=bin_res_nbody["err"],
        fmt="o", color=_MODEL_STYLES["external_node_nbody"]["color"],
        ms=3.5, lw=1.0, capsize=2.0, zorder=4,
        label="Ext-Node N-body (binned Delta-mu)",
    )

    # Binned residuals for LCDM (should sit on or near zero by construction,
    # but binning scatter makes this a useful visual check)
    bin_res_lcdm = pantheon_loader.bin_for_plot(
        z_in, results["lcdm"]["residuals"], sigma_in, n_bins=n_bins
    )
    ax_res.errorbar(
        bin_res_lcdm["z"], bin_res_lcdm["mu"], yerr=bin_res_lcdm["err"],
        fmt="s", color=_MODEL_STYLES["lcdm"]["color"],
        ms=3.0, lw=1.0, capsize=2.0, zorder=4, alpha=0.7,
        label="LCDM (binned Delta-mu, reference check)",
    )

    ax_res.set_xscale("log")
    ax_res.set_xlabel("Redshift z", fontsize=12)
    ax_res.set_ylabel("Delta-mu vs LCDM [mag]", fontsize=10)
    ax_res.set_ylim(-1.5, 1.5)
    ax_res.grid(True, alpha=0.3, which="both")
    ax_res.legend(fontsize=8, loc="upper right")

    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # --from-best-config: load M, S, centerM from sweep CSV (if provided).
    # Explicit CLI --M / --S / --center-node-mass override the loaded values.
    # ------------------------------------------------------------------
    if args.from_best_config is not None:
        try:
            cfg = load_best_config(args.from_best_config)
        except (FileNotFoundError, ValueError) as exc:
            print(f"\nERROR: {exc}", file=sys.stderr)
            sys.exit(1)

        print(
            f"\nLoaded best config from {args.from_best_config!r}: "
            f"M={cfg['M']}, S={cfg['S']}, centerM={cfg['centerM']}  "
            f"(chi2_dof={cfg['chi2_dof']})"
        )

        # Only apply the loaded config to flags the user did NOT explicitly
        # supply on the CLI. argparse stores the same value whether a flag was
        # passed or defaulted, so comparing against the default misfires when a
        # user explicitly passes a value equal to the default (e.g. --M 855.0).
        # Detect explicit flags directly from sys.argv instead.
        def _flag_given(*option_strings: str) -> bool:
            for tok in sys.argv[1:]:
                # Matches "--M", "--M=855", but not "--M-something".
                if tok in option_strings or any(
                    tok.startswith(opt + "=") for opt in option_strings
                ):
                    return True
            return False

        if not _flag_given("--M"):
            args.M = cfg["M"]
        if not _flag_given("--S"):
            args.S = cfg["S"]
        if not _flag_given("--center-node-mass") and cfg["centerM"] is not None:
            args.center_node_mass = cfg["centerM"]

    t_start = args.t_start
    t_duration = _TODAY_GYR - t_start

    if t_duration <= 0:
        print(
            f"ERROR: --t-start={t_start} Gyr >= {_TODAY_GYR} Gyr (today). "
            "t_duration would be <= 0.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Build sim_params with the derived t_duration (override what argparse set)
    sim_params = SimulationParameters(
        M_value=args.M,
        S_value=args.S,
        n_particles=args.particles,
        seed=args.seed,
        t_start_Gyr=t_start,
        t_duration_Gyr=t_duration,
        n_steps=args.n_steps,
        damping_factor=args.damping,
        center_node_mass=args.center_node_mass,
        mass_randomize=args.mass_randomize,
    )

    try:
        run(
            sim_params=sim_params,
            output_dir=args.output_dir,
            pantheon_path=args.pantheon_path,
            z_min=args.z_min,
            n_bins=args.n_bins,
        )
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
