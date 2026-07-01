"""Paper figure (Section 2.1) — the virialized meta-structure node points.

A SINGLE 3D scatter of the HMEA nodes, sized and coloured by MASS, with our
observable universe drawn to scale at the centre. Points only: no diagnostic
panels, no derived metrics. It illustrates where we sit inside the meta-structure.

Reuse: the 3D primitives come from cosmo.visualization (draw_universe_sphere,
setup_3d_axes) — the same helpers visualize_geometries.py / _generate_hero_figs.py
use — and the node grid from cosmo.node_geometry.build_virialized_grid. Nothing is
re-implemented here.

Re-runnable GENERATOR: every physical knob is a CLI argument (the headline config
is still moving). Crop to the local neighbourhood with --view-scale (a multiple of
S), or --view-scale 0 for the whole grid.

Usage
-----
    python _gen_fig_meta_structure.py                    # sensible default -> docs/
    python _gen_fig_meta_structure.py --sigma 3 --seg 0.4
    python _gen_fig_meta_structure.py --view-scale 0 --out results/figures/paper/x.png
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cosmo.encoding import configure_utf8_stdout
configure_utf8_stdout()
from cosmo.node_geometry import build_virialized_grid
from cosmo.constants import CosmologicalConstants
from cosmo.visualization import draw_universe_sphere, setup_3d_axes

_GPC_M = CosmologicalConstants.Gpc_to_m
_OBS_RADIUS_GPC = 14.0  # observable-universe radius (~ particle horizon), for scale


def nearest_indices(radius: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k nodes closest to the observer (smallest radius first).

    Args:
        radius: (N,) node distances from the origin.
        k:      how many to return (clamped to N; k<=0 -> empty).

    Returns:
        (<=k,) int index array, nearest first.
    """
    r = np.asarray(radius, dtype=np.float64)
    k = int(max(0, min(k, r.size)))
    if k == 0:
        return np.empty(0, dtype=np.int64)
    return np.argsort(r, kind="stable")[:k]


def generate(
    *, S_gpc: float, n_nodes: int, sigma: float, seg: float, mass_rule: str,
    seed: int, relax_mode: str, relax_steps: int, view_scale: float,
    n_stars: int, obs_radius_gpc: float, show_sphere: bool, out_path: str,
) -> str:
    """Build the meta-structure node-point figure and save it to out_path."""
    pos_m, masses = build_virialized_grid(
        S_gpc * _GPC_M, n_nodes=n_nodes, M_ext_kg=1.0, vir_mass_rule=mass_rule,
        vir_mass_spread=sigma, vir_segregation=seg, vir_s_metric="median",
        vir_relax_steps=relax_steps, vir_relax_mode=relax_mode, seed=seed,
    )
    pos = pos_m / _GPC_M                    # Gpc
    r = np.linalg.norm(pos, axis=1)        # Gpc
    mm = masses / masses.mean()            # mass in units of the mean

    # Optional crop to the local neighbourhood (view_scale <= 0 -> whole grid).
    if view_scale and view_scale > 0:
        lim = float(view_scale) * float(S_gpc)
        shown = r <= lim
    else:
        lim = float(r.max()) * 1.05
        shown = np.ones_like(r, dtype=bool)
    ps, ms_ = pos[shown], mm[shown]

    print(f"[meta-fig] S={S_gpc} Gpc, N={n_nodes}, sigma={sigma}, seg={seg}, "
          f"rule={mass_rule}, seed={seed}, mode={relax_mode}/{relax_steps}; "
          f"{int(shown.sum())} nodes shown (lim={lim:.0f} Gpc), "
          f"mass/mean range {mm.min():.3f}-{mm.max():.2f}")

    fig = plt.figure(figsize=(8.2, 7.4))
    ax = fig.add_subplot(111, projection="3d")

    # size ~ mass (with a floor so ~massless nodes still show); colour ~ mass.
    s_size = 20.0 + 200.0 * (ms_ / mm.max() if mm.max() > 0 else ms_)
    sc = ax.scatter(ps[:, 0], ps[:, 1], ps[:, 2], c=ms_, s=s_size, cmap="plasma",
                    edgecolors="k", linewidths=0.25, depthshade=True, alpha=0.92)
    cb = fig.colorbar(sc, ax=ax, fraction=0.030, pad=0.02)
    cb.set_label("node mass / mean", fontsize=9)

    if show_sphere:
        draw_universe_sphere(ax, obs_radius_gpc, alpha=0.14, color="#1f77b4")
    ax.scatter([0], [0], [0], color="#1f77b4", marker="o", s=40, zorder=6,
               edgecolors="k", linewidths=0.4)

    stars = nearest_indices(r, n_stars)
    if stars.size:
        ax.scatter(pos[stars, 0], pos[stars, 1], pos[stars, 2], marker="*",
                   s=320, facecolors="none", edgecolors="k", linewidths=1.3,
                   zorder=8)

    setup_3d_axes(ax, lim, title="")
    sphere_note = (f"; blue sphere = our observable universe (R$\\approx${obs_radius_gpc:.0f} Gpc)"
                   if show_sphere else "")
    fig.suptitle(
        f"The Virialized Meta-Structure (S = {S_gpc:.0f} Gpc, {n_nodes} nodes, "
        f"mass spread $\\sigma$ = {sigma:g}, segregation = {seg:g})\n"
        f"nodes sized & coloured by mass{sphere_note}",
        fontsize=12, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[meta-fig] wrote {out_path}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--S", type=float, default=20.0, help="grid spacing S [Gpc]")
    p.add_argument("--n-nodes", type=int, default=300, help="number of HMEA nodes")
    p.add_argument("--sigma", type=float, default=1.0,
                   help="vir_mass_spread (mass-function width)")
    p.add_argument("--seg", type=float, default=1.0, help="vir_segregation [0..1]")
    p.add_argument("--mass-rule", default="massfunc", choices=["massfunc", "radial"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--relax-mode", default="lattice", choices=["lattice", "gradient"])
    p.add_argument("--relax-steps", type=int, default=1)
    p.add_argument("--view-scale", type=float, default=2.5,
                   help="crop radius as a multiple of S (<=0 -> whole grid)")
    p.add_argument("--n-stars", type=int, default=0,
                   help="mark the N nearest nodes with a star (0 -> none)")
    p.add_argument("--obs-radius", type=float, default=_OBS_RADIUS_GPC,
                   help="observable-universe radius [Gpc]")
    p.add_argument("--no-sphere", action="store_true",
                   help="omit the observable-universe sphere (points only)")
    p.add_argument("--out", default="docs/fig_meta_structure.png")
    a = p.parse_args()
    generate(
        S_gpc=a.S, n_nodes=a.n_nodes, sigma=a.sigma, seg=a.seg,
        mass_rule=a.mass_rule, seed=a.seed, relax_mode=a.relax_mode,
        relax_steps=a.relax_steps, view_scale=a.view_scale, n_stars=a.n_stars,
        obs_radius_gpc=a.obs_radius, show_sphere=not a.no_sphere, out_path=a.out,
    )


if __name__ == "__main__":
    main()
