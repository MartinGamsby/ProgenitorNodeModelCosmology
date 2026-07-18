#!/usr/bin/env python3
"""
Section 6 artifact — vir_extent -> node-count coupling (item 10)
===============================================================

Demonstrates the item-10 coupling: with ``vir_extent_couples_nodes=True`` a larger
radial extent auto-raises the virialized node count (density-preserving N ~ extent^3),
so ``vir_extent`` becomes MEANINGFUL in the force-balanced lattice mode (where it was
a no-op) by driving how far the lattice ball reaches.

For a few ``vir_extent`` values it tabulates, for the DEFAULT force-balanced lattice
mode (vir_relax_steps=1):
  * effective node count (n_eff = round(base_n * extent^3)),
  * realized radial reach (r_max / S),
  * realized nearest-neighbour spacing (NN / S, must == 1 -> spacing == S),
  * ball density proxy (n_eff / reach^3, ~constant if density is held),
  * center-only virialization residual (must stay << 1 -> still virialized at core).

Outputs (gitignored, under results/figures/ws8/):
  vir_extent_nodecount.csv  — the table above.
  vir_extent_nodecount.png  — node-count & reach vs extent + density/residual panels.

Usage:
  set PYTHONIOENCODING=utf-8 && python _generate_extent_nodecount.py
"""
from __future__ import annotations

import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cosmo.node_geometry import (
    build_virialized_grid,
    nearest_neighbour_spacing,
    virialization_residual,
    extent_coupled_n_nodes,
)
from cosmo.plots import figure_path

BASE_N = 64          # base vir_n_nodes at extent == 1.0
S = 30.0             # target NN spacing (arbitrary units; everything reported / S)
EXTENTS = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
CENTER_K = 8         # deep-interior nodes the residual is measured on


def _row(extent: float) -> dict:
    pos, masses = build_virialized_grid(
        S, n_nodes=BASE_N, M_ext_kg=1.0, vir_extent=extent,
        vir_relax_steps=1, vir_extent_couples_nodes=True, seed=0,
    )
    n_eff = pos.shape[0]
    reach = float(np.linalg.norm(pos, axis=1).max()) / S
    nn = nearest_neighbour_spacing(pos, "median") / S
    res = virialization_residual(pos, masses, center_k=min(CENTER_K, n_eff))
    return {
        "vir_extent": extent,
        "n_eff": n_eff,
        "n_eff_law": extent_coupled_n_nodes(BASE_N, extent),  # cross-check
        "reach_over_S": reach,
        "nn_over_S": nn,
        "density_N_over_reach3": n_eff / reach ** 3,
        "center_max_residual": res["max_residual"],
        "center_median_residual": res["median_residual"],
        "n_center": res["n_inner"],
    }


def main() -> None:
    rows = [_row(e) for e in EXTENTS]

    csv_path = os.path.splitext(figure_path("ws8", "vir_extent_nodecount"))[0] + ".csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[csv] {csv_path}")

    # Console table (the user wants the numbers visible).
    print(f"\nbase vir_n_nodes={BASE_N}, coupling ON (N ~ extent^3), force-balanced lattice")
    print(f"{'extent':>7} {'n_eff':>6} {'reach/S':>8} {'NN/S':>7} "
          f"{'N/reach^3':>10} {'ctr_resid':>11}")
    for r in rows:
        print(f"{r['vir_extent']:>7} {r['n_eff']:>6} {r['reach_over_S']:>8.3f} "
              f"{r['nn_over_S']:>7.4f} {r['density_N_over_reach3']:>10.4f} "
              f"{r['center_max_residual']:>11.2e}")

    ext = np.array([r["vir_extent"] for r in rows])
    n_eff = np.array([r["n_eff"] for r in rows])
    reach = np.array([r["reach_over_S"] for r in rows])
    dens = np.array([r["density_N_over_reach3"] for r in rows])
    nn = np.array([r["nn_over_S"] for r in rows])
    resid = np.array([r["center_max_residual"] for r in rows])

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        "vir_extent -> node-count coupling (item 10): "
        f"base N={BASE_N}, force-balanced lattice, N = round(N0 * extent^3)",
        fontsize=12,
    )

    ax = axes[0, 0]
    ax.plot(ext, n_eff, "o-", label="effective node count")
    ax.plot(ext, BASE_N * ext ** 3, "k--", lw=1, label=r"$N_0\,\mathrm{extent}^3$")
    ax.set_xlabel("vir_extent"); ax.set_ylabel("node count")
    ax.set_title("A larger extent -> more nodes"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(ext, reach, "o-", color="C1")
    ax.set_xlabel("vir_extent"); ax.set_ylabel("radial reach / S")
    ax.set_title("Realized reach grows with extent\n(knob now meaningful in lattice mode)")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(ext, dens, "o-", color="C2", label=r"$N/\mathrm{reach}^3$ (density)")
    ax.axhline(dens.mean(), color="C2", ls=":", lw=1, label="mean density")
    ax.plot(ext, nn, "s-", color="C3", label="NN spacing / S (== 1)")
    ax.set_xlabel("vir_extent"); ax.set_ylabel("ratio")
    ax.set_title("Density ~constant; NN spacing == S"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.semilogy(ext, np.maximum(resid, 1e-30), "o-", color="C4")
    ax.axhline(0.25, color="r", ls="--", lw=1, label="0.25 tolerance")
    ax.set_xlabel("vir_extent"); ax.set_ylabel("center max residual (dimensionless)")
    ax.set_title(f"Still virialized at center (k={CENTER_K})"); ax.legend(); ax.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    png_path = figure_path("ws8", "vir_extent_nodecount")
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    print(f"[png] {png_path}")


if __name__ == "__main__":
    from cosmo.encoding import configure_utf8_stdout
    configure_utf8_stdout()
    main()
