#!/usr/bin/env python3
"""
Hubble Diagram — External-Node Model vs Pantheon+SH0ES

Standalone script that:
  1. Loads the Pantheon+SH0ES supernova distance-modulus compilation.
  2. Builds mu(z) curves for three cosmological models:
       - LCDM              (blue solid)
       - External-Node     (red dashed)
       - Matter-only       (green dotted)
  3. Fits an additive offset DeltaM per model (marginalizes M_B/H0 degeneracy).
  4. Computes and prints chi^2, chi^2/dof, and R^2 for each model.
  5. Saves a two-panel Hubble-diagram figure:
       Top:    data (binned + faint scatter) + three model curves.
       Bottom: residuals (mu_obs - mu_fit) for all three models.

This script is INDEPENDENT of run_simulation.py — it is purely semi-analytic
and does not call CosmologicalSimulation.run().

Default external-node config: --M 855 --S 37.8  (Omega_Lambda_eff ~ 0.70)
Override via CLI to explore other configurations.

Usage
-----
    python hubble_diagram.py                         # uses defaults above
    python hubble_diagram.py --M 855 --S 37.8        # explicit defaults
    python hubble_diagram.py --pantheon-path /path/to/Pantheon+SH0ES.dat
    python hubble_diagram.py --output-dir ./results --z-min 0.023

Requires the real Pantheon+SH0ES data file — see data/pantheon_plus/README.md.
"""

import argparse
import os
import sys

# The summary table and plot titles use Greek characters (Λ, Ω, χ²).  On
# Windows the console defaults to cp1252, which cannot encode them and would
# raise UnicodeEncodeError on the first print.  Reconfigure stdout/stderr to
# UTF-8 (with a safe fallback) so the script runs on any platform.
for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if _reconfigure is not None:
        try:
            _reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

# Headless backend — must be set BEFORE pyplot is imported anywhere in this
# process.  Matches the approach in the other plotting scripts (run_simulation.py
# uses plt.savefig without plt.show, so Agg is safe here too).
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from cosmo.cli import add_common_arguments, args_to_sim_params
from cosmo.distances import model_distance_modulus
import cosmo.hubble_diagram as hd_engine
import cosmo.pantheon as pantheon_loader
from cosmo.visualization import generate_output_filename


# ---------------------------------------------------------------------------
# Color/style convention (matches paper figures — lode/paper-reference.md)
# ---------------------------------------------------------------------------
_MODEL_STYLES = {
    "lcdm":          {"color": "#1f77b4", "ls": "-",  "lw": 2.0, "label": "ΛCDM"},
    "external_node": {"color": "#d62728", "ls": "--", "lw": 2.0, "label": "External-Node"},
    "matter_only":   {"color": "#2ca02c", "ls": ":",  "lw": 2.0, "label": "Matter-only"},
}

# Display order for residual panel (determines draw/legend order)
_MODEL_ORDER = ("lcdm", "external_node", "matter_only")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Hubble-diagram comparison: External-Node model vs Pantheon+SH0ES data.\n"
            "Produces a two-panel figure (data+model curves / residuals) and prints\n"
            "chi^2/dof and R^2 for each model."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Output
    parser.add_argument(
        "--output-dir", type=str, default="./results",
        help="Directory to save the output figure.",
    )

    # Pantheon+ data path override
    parser.add_argument(
        "--pantheon-path", type=str, default=None,
        help=(
            "Path to Pantheon+SH0ES.dat. Defaults to the vendored location "
            "data/pantheon_plus/Pantheon+SH0ES.dat."
        ),
    )

    # Redshift cut
    parser.add_argument(
        "--z-min", type=float, default=0.01,
        help="Minimum redshift cut applied to the data.",
    )

    # Number of bins for the binned overlay
    parser.add_argument(
        "--n-bins", type=int, default=20,
        help="Number of log-z bins for the binned data overlay.",
    )

    # External-node model parameters (M, S, …) — reuse shared CLI definitions.
    # Defaults here give Omega_Lambda_eff ~ 0.70, suitable for the full
    # Pantheon+ z range.  The raw SimulationParameters default (M=800, S=24)
    # gives Omega_Lambda_eff ~ 2.55 (closed universe, turnaround at z ~ 0.32).
    add_common_arguments(parser)

    # Override the add_common_arguments defaults so the script works out of the
    # box with a physically matched config (Omega_Lambda_eff ~ 0.70).
    # M=855, S=37.8 gives Omega_Lambda_eff = 0.6988 ≈ 0.70.
    # The raw SimulationParameters default (M=800, S=24) gives ~2.55 — a closed
    # universe that turns around at z~0.32, incompatible with Pantheon+ data.
    parser.set_defaults(M=855.0, S=37.8)

    return parser


# ---------------------------------------------------------------------------
# Core logic (callable for testing)
# ---------------------------------------------------------------------------

def run(
    sim_params,
    output_dir: str = "./results",
    pantheon_path=None,
    z_min: float = 0.01,
    n_bins: int = 20,
) -> dict:
    """
    Load data, evaluate models, produce plot, print table.

    Returns the results dict from compare_all_models (keys: model names).
    Raises FileNotFoundError (with actionable message) if data is absent.
    """
    # ------------------------------------------------------------------
    # 1. Load Pantheon+SH0ES
    # ------------------------------------------------------------------
    data = pantheon_loader.load_pantheon(path=pantheon_path, z_min=z_min)
    print(f"Loaded {data['n']} SNe Ia  (z_min={z_min}, "
          f"z range: {data['z'].min():.4f} – {data['z'].max():.4f})")

    # ------------------------------------------------------------------
    # 2. Evaluate all three models on the data grid
    # ------------------------------------------------------------------
    results = hd_engine.compare_all_models(data, sim_params=sim_params)

    # ------------------------------------------------------------------
    # 3. Build smooth model curves on a dense z grid for plotting
    # ------------------------------------------------------------------
    z_dense = np.linspace(z_min, float(data["z"].max()), 400)

    mu_dense = {}
    for model in _MODEL_ORDER:
        mu_raw = model_distance_modulus(
            z_dense, model, sim_params=sim_params
        )
        # Apply the same DeltaM that was fit on the data grid so the plotted
        # curve is offset-corrected and visually consistent with the data.
        DeltaM = results[model]["DeltaM"]
        mu_dense[model] = mu_raw + DeltaM

    # ------------------------------------------------------------------
    # 4. Print per-model summary table
    # ------------------------------------------------------------------
    _print_table(results, sim_params)

    # ------------------------------------------------------------------
    # 5. Plot
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    fig = _make_figure(data, results, mu_dense, z_dense, sim_params, n_bins=n_bins)

    out_path = generate_output_filename(
        "hubble_diagram", sim_params, "png", output_dir,
        include_timestamp=True, include_S=True, include_M=True, include_D=False
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out_path}")

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_table(results: dict, sim_params) -> None:
    """Print a compact per-model statistics table."""
    Omega_eff = sim_params.external_params.Omega_Lambda_eff
    print(
        f"\nExternal-Node config: "
        f"M={sim_params.M_value}, S={sim_params.S_value}, "
        f"Omega_Lambda_eff={Omega_eff:.4f}\n"
    )
    header = f"{'Model':<20} {'chi2':>10} {'dof':>6} {'chi2/dof':>10} {'R2':>8}"
    print(header)
    print("-" * len(header))
    for model in _MODEL_ORDER:
        r = results[model]
        label = _MODEL_STYLES[model]["label"]
        print(
            f"{label:<20} {r['chi2']:>10.2f} {r['dof']:>6d} "
            f"{r['chi2_dof']:>10.4f} {r['R2']:>8.6f}"
        )


def _make_figure(
    data: dict,
    results: dict,
    mu_dense: dict,
    z_dense: np.ndarray,
    sim_params,
    n_bins: int = 20,
) -> plt.Figure:
    """
    Build the two-panel Hubble-diagram figure.

    Top panel  : data (scatter + binned) + three model curves.
    Bottom panel: residuals (mu_obs - mu_fit) for each model.
    """
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(
        2, 1, height_ratios=[3, 1], hspace=0.08, figure=fig
    )
    ax_top = fig.add_subplot(gs[0])
    ax_res = fig.add_subplot(gs[1], sharex=ax_top)

    z_obs = data["z"]
    mu_obs = data["mu"]
    sigma_obs = data["sigma"]

    # -- Raw scatter (faint) -------------------------------------------
    ax_top.scatter(
        z_obs, mu_obs,
        s=1.5, alpha=0.15, color="gray", zorder=1, rasterized=True,
    )

    # -- Binned data points with error bars ----------------------------
    binned = pantheon_loader.bin_for_plot(z_obs, mu_obs, sigma_obs, n_bins=n_bins)
    ax_top.errorbar(
        binned["z"], binned["mu"], yerr=binned["err"],
        fmt="o", color="black", ms=4, lw=1.2, capsize=2.5, zorder=3,
        label=f"Pantheon+SH0ES ({data['n']} SNe, {len(binned['z'])} bins)",
    )

    # -- Model curves --------------------------------------------------
    for model in _MODEL_ORDER:
        st = _MODEL_STYLES[model]
        r = results[model]
        chi2_dof = r["chi2_dof"]
        R2 = r["R2"]
        lbl = f"{st['label']}  χ²/dof={chi2_dof:.3f}, R²={R2:.4f}"
        ax_top.plot(
            z_dense, mu_dense[model],
            color=st["color"], ls=st["ls"], lw=st["lw"],
            label=lbl, zorder=4,
        )

    ax_top.set_xscale("log")
    ax_top.set_ylabel(r"Distance modulus $\mu$ [mag]", fontsize=12)
    ax_top.legend(fontsize=9, loc="upper left")
    ax_top.set_title(
        f"Hubble Diagram — External-Node vs Pantheon+SH0ES\n"
        f"M={sim_params.M_value}, S={sim_params.S_value}, "
        f"Ω_Λ_eff={sim_params.external_params.Omega_Lambda_eff:.4f}",
        fontsize=11,
    )
    ax_top.grid(True, alpha=0.3, which="both")
    plt.setp(ax_top.get_xticklabels(), visible=False)

    # -- Residual panel ------------------------------------------------
    # Residuals for all three models; zero reference line
    ax_res.axhline(0.0, color="black", lw=0.8, ls="-", zorder=2)

    for model in _MODEL_ORDER:
        st = _MODEL_STYLES[model]
        residuals = results[model]["residuals"]  # mu_obs - mu_fit at data z
        ax_res.scatter(
            z_obs, residuals,
            s=1.5, alpha=0.25, color=st["color"], rasterized=True, zorder=3,
        )

    # Binned residuals for the External-Node model for clarity
    ext_res = results["external_node"]["residuals"]
    binned_res = pantheon_loader.bin_for_plot(z_obs, ext_res, sigma_obs, n_bins=n_bins)
    ax_res.errorbar(
        binned_res["z"], binned_res["mu"], yerr=binned_res["err"],
        fmt="o", color=_MODEL_STYLES["external_node"]["color"],
        ms=3.5, lw=1.0, capsize=2.0, zorder=4,
        label="Ext-Node (binned)",
    )

    ax_res.set_xscale("log")
    ax_res.set_xlabel("Redshift $z$", fontsize=12)
    ax_res.set_ylabel(r"$\mu_\mathrm{obs} - \mu_\mathrm{fit}$", fontsize=10)
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

    sim_params = args_to_sim_params(args)

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
