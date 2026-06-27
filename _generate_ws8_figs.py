#!/usr/bin/env python3
"""
WS8 Figure Generator — Virialized grid: positions, slingshot, mass-rule A/B
===========================================================================

Produces THREE figures under results/figures/ws8/:

  Fig 1 (node_geometries_by_mass.png)
      STATIC node positions for every geometry (cube26, cube_dense, fcc, bcc)
      AND the virialized grid (extent 1/2 × rule radial/massfunc), each 3D
      scatter coloured by NODE MASS so mass-segregation is visible. A red star
      marks the observable centre; reference rings mark the observable horizon
      (~14 Gpc) and the sim cloud scale.

  Fig 2 (particle_motion_slingshot.png)
      PARTICLE-MOTION / slingshot diagnostic. A SHORT sim is run for cube26 vs
      a virialized grid; per-particle initial->final displacement magnitude is
      shown (histogram + initial-position scatter coloured by displacement). A
      quantitative "slingshot tail" metric (max/median ratio, p99/median,
      tail-fraction beyond N× median) is annotated per geometry so the cube26
      vs virialized comparison is QUANTITATIVE, not just visual.
      DIAGNOSTIC ONLY — no softening/merge fix is implemented here.

  Fig 3 (massrule_a_vs_b.png)
      mass<->position A-vs-B comparison: vir_mass_rule="radial" (a) vs
      "massfunc" (b) at the same N/extent/spread/seed. VISUAL (node scatter +
      mass-vs-radius scatter) AND MATHEMATICAL (Pearson/Spearman mass-radius
      correlation, segregation slope, median-vs-mean NN spacing, mean-
      preservation check) side by side.

Usage
-----
    python _generate_ws8_figs.py                 # all 3 figures (runs 2 short sims)
    python _generate_ws8_figs.py --no-sim        # Fig 1 + Fig 3 only (no sim)
    python _generate_ws8_figs.py --n-particles 400 --n-steps 120

INVARIANTS
----------
- PNGs ONLY under results/figures/ws8/ (gitignored) via cosmo.plots.figure_path.
- Diagnostic only — does NOT implement the softening/merging runaway fix.
- Seeded -> reproducible. No change to existing figures (new ws8 dir).
- Uses existing sim.snapshots + get_observable_mask(); no new sim plumbing.
"""

from cosmo.encoding import configure_utf8_stdout
configure_utf8_stdout()

import argparse
import math
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from cosmo.constants import CosmologicalConstants, SimulationParameters
from cosmo.node_geometry import (
    build_node_positions,
    build_virialized_grid,
    nearest_neighbour_spacing,
)
from cosmo.factories import run_external_node_simulation, setup_simulation_context
from cosmo.plots import figure_path, _footer, _DPI, _BBOX

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_WS = "ws8"
_TODAY_GYR: float = 13.8
_OBS_HORIZON_GPC: float = 14.0  # observable-universe radius reference (~14 Gpc)
_MIN_DT_GYR: float = 0.05       # dt = t_duration / n_steps must stay BELOW this

# Small, fast defaults (Fig 2/3 sims must be SHORT).
_DEFAULT_N_PARTICLES = 400
# dt = (13.8 - 5.8) / 220 = 0.0364 Gyr — below BOTH the 0.05 hard ceiling and the
# 0.040 recommended-stability threshold. resolve_n_steps still guards user values.
_DEFAULT_N_STEPS = 220
_DEFAULT_SEED = 42
_DEFAULT_M = 5.0                # moderate field so slingshots, if any, are visible
_DEFAULT_S_GPC = 20.0
_DEFAULT_VIR_EXTENT = 1.0
_DEFAULT_VIR_N_NODES = 26
_DEFAULT_T_START = 5.8          # shorter (z<=~1.2) run keeps the diagnostic cheap
_DEFAULT_VIR_SPREAD = 0.6       # >0 so mass segregation is visible
_DEFAULT_VIR_SEGREGATION = 1.0

# Fig-1 geometry panels (label, kind, kwargs). kind: "static" or "vir".
_FIG1_PANELS: List[Tuple[str, str, Dict[str, Any]]] = [
    ("cube26", "static", {}),
    ("cube_dense", "static", {}),
    ("fcc", "static", {}),
    ("bcc", "static", {}),
    ("virialized\nextent=1 radial", "vir",
        {"vir_extent": 1.0, "vir_mass_rule": "radial"}),
    ("virialized\nextent=1 massfunc", "vir",
        {"vir_extent": 1.0, "vir_mass_rule": "massfunc"}),
    ("virialized\nextent=2 radial", "vir",
        {"vir_extent": 2.0, "vir_mass_rule": "radial"}),
    ("virialized\nextent=2 massfunc", "vir",
        {"vir_extent": 2.0, "vir_mass_rule": "massfunc"}),
]


# ===========================================================================
# PURE HELPERS (unit-testable, no sim, no I/O)
# ===========================================================================

def resolve_n_steps(t_duration_Gyr: float, n_steps: int,
                    min_dt_Gyr: float = _MIN_DT_GYR) -> int:
    """Return n_steps large enough that dt = t_duration/n_steps < min_dt_Gyr.

    The simulation validator rejects dt >= 0.05 Gyr. Bump the requested
    n_steps up to the minimum that keeps dt strictly below the ceiling.

    Args:
        t_duration_Gyr: Run duration in Gyr.
        n_steps:        Requested number of steps.
        min_dt_Gyr:     dt ceiling (exclusive).

    Returns:
        max(n_steps, ceil(t_duration/min_dt_Gyr)+1) — always yields dt<ceiling.
    """
    needed = int(math.ceil(t_duration_Gyr / float(min_dt_Gyr))) + 1
    return int(max(int(n_steps), needed))


def displacement_magnitudes(
    pos_initial: np.ndarray, pos_final: np.ndarray
) -> np.ndarray:
    """Per-particle |final - initial| displacement magnitudes.

    Args:
        pos_initial: (N, 3) initial positions.
        pos_final:   (N, 3) final positions (same N).

    Returns:
        (N,) array of Euclidean displacement magnitudes.

    Raises:
        ValueError: if the two arrays differ in shape.
    """
    pi = np.asarray(pos_initial, dtype=np.float64)
    pf = np.asarray(pos_final, dtype=np.float64)
    if pi.shape != pf.shape:
        raise ValueError(
            f"displacement_magnitudes: shape mismatch {pi.shape} vs {pf.shape}"
        )
    disp = pf - pi
    return np.sqrt(np.sum(disp * disp, axis=1))


def slingshot_metrics(disp: np.ndarray, tail_factor: float = 5.0) -> Dict[str, float]:
    """Quantitative slingshot-tail metrics from a displacement-magnitude array.

    A slingshot run is one where a FEW particles fly far while the bulk barely
    move — i.e. a heavy high-displacement tail. These scalars capture that:

    Args:
        disp:        (N,) per-particle displacement magnitudes (>= 0).
        tail_factor: A particle counts as a "slingshot" if its displacement
                     exceeds tail_factor × the median displacement.

    Returns:
        dict with:
          median, mean, max, p95, p99            — displacement stats,
          max_over_median  = max / median        — single worst slingshot,
          p99_over_median  = p99 / median        — tail heaviness,
          tail_fraction    = frac(disp > tail_factor*median) — runaway count,
          n                = particle count.
        Ratios are NaN when median == 0.
    """
    d = np.asarray(disp, dtype=np.float64)
    d = d[np.isfinite(d)]
    n = int(d.size)
    if n == 0:
        nan = float("nan")
        return {"median": nan, "mean": nan, "max": nan, "p95": nan, "p99": nan,
                "max_over_median": nan, "p99_over_median": nan,
                "tail_fraction": nan, "n": 0}
    median = float(np.median(d))
    mean = float(np.mean(d))
    dmax = float(np.max(d))
    p95 = float(np.percentile(d, 95))
    p99 = float(np.percentile(d, 99))
    if median > 0.0:
        max_over_median = dmax / median
        p99_over_median = p99 / median
        tail_fraction = float(np.mean(d > tail_factor * median))
    else:
        max_over_median = float("nan")
        p99_over_median = float("nan")
        tail_fraction = float("nan")
    return {
        "median": median, "mean": mean, "max": dmax, "p95": p95, "p99": p99,
        "max_over_median": max_over_median, "p99_over_median": p99_over_median,
        "tail_fraction": tail_fraction, "n": n,
    }


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation (NaN if either series is constant / too short)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (1..n), ties averaged — a tiny scipy.stats.rankdata."""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="stable")
    ranks = np.empty(a.size, dtype=np.float64)
    ranks[order] = np.arange(1, a.size + 1, dtype=np.float64)
    # Average tied ranks.
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size, dtype=np.float64)
    np.add.at(sums, inv, ranks)
    avg = sums / counts
    return avg[inv]


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (Pearson on ranks)."""
    return _pearson(_rankdata(x), _rankdata(y))


def mass_radius_stats(positions: np.ndarray, masses: np.ndarray,
                      M_ext_kg: float) -> Dict[str, float]:
    """Mass<->radius segregation + mass-distribution statistics for one grid.

    Args:
        positions: (N, 3) node positions.
        masses:    (N,) node masses (same order as positions).
        M_ext_kg:  Target per-node MEAN mass (mean-preservation check).

    Returns:
        dict with:
          pearson, spearman    — mass-vs-radius correlation,
          seg_slope            — slope of (mass/mean) vs (radius/mean radius),
          mass_min/max/mean/std,
          mean_preserved       — |mean(mass) - M_ext_kg| / M_ext_kg (≈0 = good),
          nn_median, nn_mean   — realized NN spacing under each metric,
          n.
    """
    pos = np.asarray(positions, dtype=np.float64)
    m = np.asarray(masses, dtype=np.float64)
    radius = np.linalg.norm(pos, axis=1)
    n = int(m.size)

    pearson = _pearson(radius, m)
    spearman = _spearman(radius, m)

    # Segregation slope: normalize both axes so the slope is dimensionless and
    # comparable across grids. slope of (m/mean_m) on (r/mean_r).
    mean_m = float(np.mean(m)) if n else float("nan")
    mean_r = float(np.mean(radius)) if n else float("nan")
    if n >= 2 and mean_m > 0 and mean_r > 0 and np.std(radius) > 0:
        xr = radius / mean_r
        ym = m / mean_m
        seg_slope = float(np.polyfit(xr, ym, 1)[0])
    else:
        seg_slope = float("nan")

    nn_median = nearest_neighbour_spacing(pos, "median") if n >= 2 else float("nan")
    nn_mean = nearest_neighbour_spacing(pos, "mean") if n >= 2 else float("nan")

    mean_preserved = (
        abs(mean_m - float(M_ext_kg)) / float(M_ext_kg)
        if M_ext_kg else float("nan")
    )

    return {
        "pearson": pearson,
        "spearman": spearman,
        "seg_slope": seg_slope,
        "mass_min": float(np.min(m)) if n else float("nan"),
        "mass_max": float(np.max(m)) if n else float("nan"),
        "mass_mean": mean_m,
        "mass_std": float(np.std(m)) if n else float("nan"),
        "mean_preserved": mean_preserved,
        "nn_median": float(nn_median),
        "nn_mean": float(nn_mean),
        "n": n,
    }


# ===========================================================================
# Fig 1 — static node positions coloured by mass
# ===========================================================================

def _panel_positions_masses(
    label: str, kind: str, kwargs: Dict[str, Any],
    S_gpc: float, vir_n_nodes: int, seed: int, M_ext_kg: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (positions_Gpc, masses) for one Fig-1 panel.

    Static lattices: positions from build_node_positions (units of S_gpc);
    masses uniform == M_ext_kg (constant colour, noted in title).
    Virialized: coupled positions+masses from build_virialized_grid.
    """
    if kind == "vir":
        pos, masses = build_virialized_grid(
            S_gpc, n_nodes=vir_n_nodes, M_ext_kg=M_ext_kg,
            vir_mass_rule=kwargs.get("vir_mass_rule", "radial"),
            vir_extent=kwargs.get("vir_extent", 1.0),
            vir_mass_spread=_DEFAULT_VIR_SPREAD,
            vir_segregation=_DEFAULT_VIR_SEGREGATION,
            vir_s_metric="median",
            seed=seed,
        )
        return pos, masses
    geom = label.split("\n")[0]
    pos = build_node_positions(geom, S_gpc, **kwargs)
    masses = np.full(pos.shape[0], float(M_ext_kg), dtype=np.float64)
    return pos, masses


def generate_fig1(S_gpc: float, vir_n_nodes: int, seed: int,
                  cloud_scale_gpc: Optional[float] = None) -> str:
    """Fig 1: static node positions for all geometries, coloured by mass."""
    panels = _FIG1_PANELS
    ncols = 4
    nrows = int(math.ceil(len(panels) / ncols))
    fig = plt.figure(figsize=(4.6 * ncols, 4.4 * nrows))

    print("[Fig 1] Building node geometries (positions coloured by mass) ...")
    for i, (label, kind, kwargs) in enumerate(panels, start=1):
        pos, masses = _panel_positions_masses(
            label, kind, kwargs, S_gpc, vir_n_nodes, seed
        )
        ax = fig.add_subplot(nrows, ncols, i, projection="3d")
        uniform = float(np.ptp(masses)) == 0.0
        sc = ax.scatter(
            pos[:, 0], pos[:, 1], pos[:, 2],
            c=masses, cmap="plasma", s=38, depthshade=True,
            edgecolors="k", linewidths=0.25,
        )
        # Observable centre.
        ax.scatter([0], [0], [0], c="red", marker="*", s=150, zorder=10)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
        cbar.set_label("node mass / M_ext" if not uniform else "mass (uniform)",
                       fontsize=7)
        cbar.ax.tick_params(labelsize=6)
        nn = (nearest_neighbour_spacing(pos, "median")
              if pos.shape[0] >= 2 else float("nan"))
        title = f"{label}\nN={pos.shape[0]}, NN≈{nn:.1f} Gpc"
        if uniform:
            title += "  (mass uniform)"
        ax.set_title(title, fontsize=9)
        lim = float(np.max(np.abs(pos))) * 1.08 if pos.size else 1.0
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_xlabel("x [Gpc]", fontsize=7)
        ax.set_ylabel("y [Gpc]", fontsize=7)
        ax.set_zlabel("z [Gpc]", fontsize=7)
        ax.tick_params(labelsize=6)

    cloud_note = (f", cloud R≈{cloud_scale_gpc:.2f} Gpc"
                  if cloud_scale_gpc else "")
    fig.suptitle(
        f"WS8 Fig 1 — HMEA node geometries coloured by NODE MASS  "
        f"(S={S_gpc:.0f} Gpc, red star = observable centre"
        f"; obs. horizon ≈{_OBS_HORIZON_GPC:.0f} Gpc{cloud_note})\n"
        "Mass-segregation (bigger nodes further out) is visible ONLY for the "
        "virialized panels; lattices have uniform per-node mass.",
        fontsize=12, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = figure_path(_WS, "node_geometries_by_mass")
    fig.savefig(out, dpi=_DPI, bbox_inches=_BBOX)
    plt.close(fig)
    print(f"[Fig 1] Saved: {out}")
    return out


# ===========================================================================
# Fig 2 — particle-motion / slingshot diagnostic
# ===========================================================================

def _run_motion_sim(
    geometry: str, M: float, S_gpc: float, n_particles: int, n_steps: int,
    t_start_Gyr: float, seed: int, vir_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, np.ndarray]:
    """Run ONE short sim and return inner-particle initial/final positions (Gpc).

    Returns dict: {'pos_initial', 'pos_final', 'disp', 'a_growth', 'geometry'}.
    Positions are in Gpc, restricted to the inner observable sub-region.
    """
    t_dur = _TODAY_GYR - t_start_Gyr
    box_size_Gpc, a_start, _ = setup_simulation_context(
        t_start_Gyr, t_dur, n_steps, save_interval=max(1, n_steps // 4)
    )
    vir_kwargs = vir_kwargs or {}
    sim_params = SimulationParameters(
        M_value=M, S_value=S_gpc, n_particles=n_particles, seed=seed,
        t_start_Gyr=t_start_Gyr, t_duration_Gyr=t_dur, n_steps=n_steps,
        damping_factor=None, center_node_mass=1.0, mass_randomize=0.0,
        node_mass_seed=seed, init_distribution="uniform_sphere",
        node_geometry=geometry,
        vir_n_nodes=vir_kwargs.get("vir_n_nodes", _DEFAULT_VIR_N_NODES),
        vir_extent=vir_kwargs.get("vir_extent", _DEFAULT_VIR_EXTENT),
        vir_mass_rule=vir_kwargs.get("vir_mass_rule", "radial"),
        vir_mass_spread=vir_kwargs.get("vir_mass_spread", _DEFAULT_VIR_SPREAD),
        vir_segregation=vir_kwargs.get("vir_segregation", _DEFAULT_VIR_SEGREGATION),
        vir_s_metric=vir_kwargs.get("vir_s_metric", "median"),
    )
    print(f"[Fig 2] Running short sim: geometry={geometry}, M={M}, "
          f"S={S_gpc}, N={n_particles}, n_steps={n_steps} ...")
    ext = run_external_node_simulation(
        sim_params, box_size_Gpc, a_start, save_interval=max(1, n_steps // 4)
    )
    sim = ext["sim"]
    mask = np.asarray(sim.particles.get_observable_mask(), dtype=bool)
    const = CosmologicalConstants()
    g = const.Gpc_to_m
    pos0 = sim.snapshots[0]["positions"][mask] / g
    pos1 = sim.snapshots[-1]["positions"][mask] / g
    disp = displacement_magnitudes(pos0, pos1)
    a_growth = float(ext["a"][-1] / ext["a"][0])
    return {"pos_initial": pos0, "pos_final": pos1, "disp": disp,
            "a_growth": a_growth, "geometry": geometry}


def _plot_motion_panel(ax_hist, ax_scatter, run: Dict[str, Any],
                       title: str) -> Dict[str, float]:
    """Plot a (histogram, initial-scatter) pair for one geometry; return metrics."""
    disp = run["disp"]
    metrics = slingshot_metrics(disp)

    ax_hist.hist(disp, bins=40, color="#1f77b4", alpha=0.8)
    ax_hist.axvline(metrics["median"], color="green", ls="--", lw=1.5,
                    label=f"median={metrics['median']:.3f} Gpc")
    ax_hist.axvline(metrics["p99"], color="orange", ls="-.", lw=1.3,
                    label=f"p99={metrics['p99']:.3f} Gpc")
    ax_hist.axvline(metrics["max"], color="red", ls=":", lw=1.3,
                    label=f"max={metrics['max']:.3f} Gpc")
    ax_hist.set_yscale("log")
    ax_hist.set_xlabel("displacement |final − initial| [Gpc]", fontsize=9)
    ax_hist.set_ylabel("particle count (log)", fontsize=9)
    ax_hist.set_title(title, fontsize=10)
    ax_hist.legend(fontsize=7, loc="upper right")
    ax_hist.grid(True, alpha=0.3)
    annot = (
        f"a-growth={run['a_growth']:.2f}\n"
        f"max/median={metrics['max_over_median']:.1f}\n"
        f"p99/median={metrics['p99_over_median']:.1f}\n"
        f"tail frac (>5×med)={metrics['tail_fraction']:.3f}\n"
        f"N={metrics['n']}"
    )
    ax_hist.text(0.02, 0.97, annot, transform=ax_hist.transAxes,
                 fontsize=7.5, va="top", ha="left",
                 bbox=dict(boxstyle="round", fc="white", alpha=0.7))

    # Initial-position scatter (x-y projection) coloured by displacement.
    pos0 = run["pos_initial"]
    sc = ax_scatter.scatter(pos0[:, 0], pos0[:, 1], c=disp, cmap="inferno",
                            s=10, alpha=0.85)
    cb = ax_scatter.figure.colorbar(sc, ax=ax_scatter, fraction=0.046, pad=0.02)
    cb.set_label("displacement [Gpc]", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax_scatter.set_xlabel("initial x [Gpc]", fontsize=9)
    ax_scatter.set_ylabel("initial y [Gpc]", fontsize=9)
    ax_scatter.set_title("initial positions (x-y), coloured by displacement",
                         fontsize=9)
    ax_scatter.set_aspect("equal", adjustable="datalim")
    ax_scatter.grid(True, alpha=0.3)
    return metrics


def generate_fig2(M: float, S_gpc: float, n_particles: int, n_steps: int,
                  t_start_Gyr: float, seed: int) -> Tuple[str, Dict[str, Dict[str, float]]]:
    """Fig 2: cube26 vs virialized particle-motion / slingshot diagnostic."""
    cube_run = _run_motion_sim(
        "cube26", M, S_gpc, n_particles, n_steps, t_start_Gyr, seed
    )
    vir_run = _run_motion_sim(
        "virialized", M, S_gpc, n_particles, n_steps, t_start_Gyr, seed,
        vir_kwargs={"vir_extent": _DEFAULT_VIR_EXTENT,
                    "vir_mass_rule": "radial"},
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    m_cube = _plot_motion_panel(
        axes[0, 0], axes[1, 0], cube_run,
        f"cube26 (M={M:.0f}, S={S_gpc:.0f} Gpc)",
    )
    m_vir = _plot_motion_panel(
        axes[0, 1], axes[1, 1], vir_run,
        f"virialized extent=1 radial (M={M:.0f}, S={S_gpc:.0f} Gpc)",
    )

    fig.suptitle(
        "WS8 Fig 2 — Particle-motion / SLINGSHOT diagnostic (cube26 vs virialized)\n"
        "Heavier high-displacement tail (larger max/median, p99/median, tail-frac) "
        "= more slingshot.  DIAGNOSTIC ONLY — no softening/merge fix applied.",
        fontsize=12, y=0.99,
    )
    _footer(axes[1, 0], f"M={M} S={S_gpc} N={n_particles} steps={n_steps} "
                        f"t_start={t_start_Gyr}")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = figure_path(_WS, "particle_motion_slingshot")
    fig.savefig(out, dpi=_DPI, bbox_inches=_BBOX)
    plt.close(fig)
    print(f"[Fig 2] Saved: {out}")
    return out, {"cube26": m_cube, "virialized": m_vir}


# ===========================================================================
# Fig 3 — mass-rule A (radial) vs B (massfunc)
# ===========================================================================

def generate_fig3(S_gpc: float, vir_n_nodes: int, seed: int,
                  vir_extent: float = 1.0) -> Tuple[str, Dict[str, Dict[str, float]]]:
    """Fig 3: A=radial vs B=massfunc mass<->position comparison (visual+math)."""
    M_ext_kg = 1.0
    common = dict(
        n_nodes=vir_n_nodes, M_ext_kg=M_ext_kg, vir_extent=vir_extent,
        vir_mass_spread=_DEFAULT_VIR_SPREAD,
        vir_segregation=_DEFAULT_VIR_SEGREGATION, seed=seed,
    )
    pos_a, mass_a = build_virialized_grid(S_gpc, vir_mass_rule="radial", **common)
    pos_b, mass_b = build_virialized_grid(S_gpc, vir_mass_rule="massfunc", **common)

    stats_a = mass_radius_stats(pos_a, mass_a, M_ext_kg)
    stats_b = mass_radius_stats(pos_b, mass_b, M_ext_kg)

    # Print the math to stdout (the headline numbers).
    print("\n[Fig 3] mass-rule A (radial) vs B (massfunc) — same N/extent/spread/seed")
    print(f"  {'metric':<22}{'A=radial':>14}{'B=massfunc':>14}")
    for key, lab in [
        ("pearson", "mass-r Pearson"), ("spearman", "mass-r Spearman"),
        ("seg_slope", "segregation slope"),
        ("nn_median", "NN spacing (median)"), ("nn_mean", "NN spacing (mean)"),
        ("mass_min", "mass min"), ("mass_max", "mass max"),
        ("mass_mean", "mass mean"), ("mass_std", "mass std"),
        ("mean_preserved", "mean-pres |Δ|/M"),
    ]:
        print(f"  {lab:<22}{stats_a[key]:>14.5f}{stats_b[key]:>14.5f}")

    fig = plt.figure(figsize=(14, 9))

    def _scatter3d(idx, pos, mass, label):
        ax = fig.add_subplot(2, 3, idx, projection="3d")
        sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=mass, cmap="plasma",
                        s=40, edgecolors="k", linewidths=0.25, depthshade=True)
        ax.scatter([0], [0], [0], c="red", marker="*", s=140, zorder=10)
        fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02).set_label(
            "mass / M_ext", fontsize=7)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("x [Gpc]", fontsize=7); ax.set_ylabel("y [Gpc]", fontsize=7)
        ax.set_zlabel("z [Gpc]", fontsize=7); ax.tick_params(labelsize=6)

    _scatter3d(1, pos_a, mass_a, "A = radial  (positions by mass)")
    _scatter3d(4, pos_b, mass_b, "B = massfunc  (positions by mass)")

    # Mass-vs-radius scatter (segregation signature).
    def _mr(idx, pos, mass, stats, label, color):
        ax = fig.add_subplot(2, 3, idx)
        r = np.linalg.norm(pos, axis=1)
        ax.scatter(r, mass, c=color, s=30, alpha=0.8)
        # Segregation fit line (un-normalized for display).
        if np.std(r) > 0:
            coeff = np.polyfit(r, mass, 1)
            xs = np.linspace(r.min(), r.max(), 50)
            ax.plot(xs, np.polyval(coeff, xs), "k--", lw=1.2)
        ax.set_xlabel("node radius [Gpc]", fontsize=9)
        ax.set_ylabel("node mass / M_ext", fontsize=9)
        ax.set_title(
            f"{label}\nPearson={stats['pearson']:.3f}, "
            f"Spearman={stats['spearman']:.3f}, slope={stats['seg_slope']:.3f}",
            fontsize=9,
        )
        ax.grid(True, alpha=0.3)

    _mr(2, pos_a, mass_a, stats_a, "A=radial: mass vs radius", "#d62728")
    _mr(5, pos_b, mass_b, stats_b, "B=massfunc: mass vs radius", "#1f77b4")

    # Math table panel.
    ax_tab = fig.add_subplot(1, 3, 3)
    ax_tab.axis("off")
    rows = [
        ("metric", "A=radial", "B=massfunc"),
        ("mass-r Pearson", f"{stats_a['pearson']:.3f}", f"{stats_b['pearson']:.3f}"),
        ("mass-r Spearman", f"{stats_a['spearman']:.3f}", f"{stats_b['spearman']:.3f}"),
        ("segregation slope", f"{stats_a['seg_slope']:.3f}", f"{stats_b['seg_slope']:.3f}"),
        ("NN spacing median", f"{stats_a['nn_median']:.2f}", f"{stats_b['nn_median']:.2f}"),
        ("NN spacing mean", f"{stats_a['nn_mean']:.2f}", f"{stats_b['nn_mean']:.2f}"),
        ("mass min", f"{stats_a['mass_min']:.3f}", f"{stats_b['mass_min']:.3f}"),
        ("mass max", f"{stats_a['mass_max']:.3f}", f"{stats_b['mass_max']:.3f}"),
        ("mass mean", f"{stats_a['mass_mean']:.4f}", f"{stats_b['mass_mean']:.4f}"),
        ("mass std", f"{stats_a['mass_std']:.3f}", f"{stats_b['mass_std']:.3f}"),
        ("mean-pres |Δ|/M", f"{stats_a['mean_preserved']:.1e}",
         f"{stats_b['mean_preserved']:.1e}"),
    ]
    tab = ax_tab.table(cellText=rows[1:], colLabels=rows[0],
                       cellLoc="center", loc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(8.5)
    tab.scale(1.0, 1.5)
    ax_tab.set_title("Mathematical comparison\n(same N/extent/spread/seed; "
                     "S target identical, both mean-preserving)", fontsize=10)

    fig.suptitle(
        f"WS8 Fig 3 — mass↔position A vs B  (N={vir_n_nodes}, extent={vir_extent}, "
        f"spread={_DEFAULT_VIR_SPREAD}, seed={seed}, S target={S_gpc:.0f} Gpc)\n"
        "A=radial: deterministic mass~f(r) (monotonic). "
        "B=massfunc: log-normal draw + segregation by mass rank.",
        fontsize=12, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = figure_path(_WS, "massrule_a_vs_b")
    fig.savefig(out, dpi=_DPI, bbox_inches=_BBOX)
    plt.close(fig)
    print(f"[Fig 3] Saved: {out}")
    return out, {"radial": stats_a, "massfunc": stats_b}


# ===========================================================================
# CLI
# ===========================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="WS8 figure generator — virialized grid positions, "
                    "slingshot diagnostic, mass-rule A/B comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--no-sim", action="store_true",
                   help="Skip Fig 2 (the only sim figure); emit Fig 1 + Fig 3.")
    p.add_argument("--n-particles", type=int, default=_DEFAULT_N_PARTICLES)
    p.add_argument("--n-steps", type=int, default=_DEFAULT_N_STEPS)
    p.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    p.add_argument("--M", type=float, default=_DEFAULT_M)
    p.add_argument("--S", type=float, default=_DEFAULT_S_GPC, dest="S_gpc")
    p.add_argument("--t-start", type=float, default=_DEFAULT_T_START)
    p.add_argument("--vir-extent", type=float, default=_DEFAULT_VIR_EXTENT)
    p.add_argument("--vir-n-nodes", type=int, default=_DEFAULT_VIR_N_NODES)
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    paths: List[str] = []

    # Fig 1 — static positions (no sim).
    cloud_scale = None
    try:
        _, _, _ = setup_simulation_context(
            args.t_start, _TODAY_GYR - args.t_start, 10
        )
    except Exception:
        pass
    p1 = generate_fig1(args.S_gpc, args.vir_n_nodes, args.seed, cloud_scale)
    paths.append(p1)

    # Fig 2 — slingshot diagnostic (runs 2 short sims).
    if not args.no_sim:
        t_dur = _TODAY_GYR - args.t_start
        n_steps = resolve_n_steps(t_dur, args.n_steps)
        if n_steps != args.n_steps:
            print(f"[ws8] Bumped n_steps {args.n_steps} -> {n_steps} "
                  f"to keep dt < {_MIN_DT_GYR} Gyr (dt={t_dur/n_steps:.4f}).")
        p2, fig2_metrics = generate_fig2(
            args.M, args.S_gpc, args.n_particles, n_steps, args.t_start, args.seed
        )
        paths.append(p2)
        _print_slingshot_verdict(fig2_metrics)
    else:
        print("[ws8] --no-sim: skipping Fig 2 (slingshot diagnostic).")

    # Fig 3 — mass-rule A vs B (no sim).
    p3, _ = generate_fig3(args.S_gpc, args.vir_n_nodes, args.seed, args.vir_extent)
    paths.append(p3)

    print("\n[ws8] Figures written:")
    for p in paths:
        print(f"  {p}")
    return 0


def _print_slingshot_verdict(metrics: Dict[str, Dict[str, float]]) -> None:
    """Print an honest cube26-vs-virialized slingshot comparison line."""
    c = metrics["cube26"]
    v = metrics["virialized"]
    print("\n" + "=" * 70)
    print("WS8 SLINGSHOT COMPARISON (inner observable particles)")
    print("=" * 70)
    print(f"  {'metric':<20}{'cube26':>14}{'virialized':>14}")
    for key, lab in [("max_over_median", "max/median"),
                     ("p99_over_median", "p99/median"),
                     ("tail_fraction", "tail frac (>5×med)"),
                     ("max", "max disp [Gpc]")]:
        print(f"  {lab:<20}{c[key]:>14.4f}{v[key]:>14.4f}")
    # Honest comparison on the headline tail-heaviness metric.
    cm = c["max_over_median"]
    vm = v["max_over_median"]
    if math.isfinite(cm) and math.isfinite(vm):
        if vm < cm * 0.95:
            verdict = (f"virialized REDUCES slingshot tail "
                       f"(max/median {vm:.1f} < cube26 {cm:.1f}).")
        elif vm > cm * 1.05:
            verdict = (f"virialized INCREASES slingshot tail "
                       f"(max/median {vm:.1f} > cube26 {cm:.1f}).")
        else:
            verdict = (f"virialized and cube26 have COMPARABLE slingshot tails "
                       f"(max/median {vm:.1f} vs {cm:.1f}).")
        print(f"\n  VERDICT: {verdict}")
    print("=" * 70)


if __name__ == "__main__":
    sys.exit(main())
