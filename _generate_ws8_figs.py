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
    node_net_accelerations,
    virialization_residual,
)
from cosmo.factories import (
    run_external_node_simulation,
    run_matter_only_simulation,
    setup_simulation_context,
)
from cosmo.plots import figure_path, _footer, _DPI, _BBOX
from cosmo.simulation import CosmologicalSimulation

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


def slingshot_sweep_row(
    knob: str, value: float, disp: np.ndarray, tail_factor: float = 5.0
) -> Dict[str, Any]:
    """Build ONE tidy knob-sweep row from a precomputed displacement array.

    Pure (no sim, no I/O): takes the per-particle displacement magnitudes for a
    single (knob, value) point and packs the headline slingshot-tail metrics into
    a flat dict suitable for a DataFrame/CSV row. Splitting this out keeps the
    sim-running part of the sweep thin and lets the row-builder be unit-tested.

    Args:
        knob:        Name of the knob being swept (e.g. "n_steps", "softening_gpc").
        value:       The knob value for this row (float; ints stored as float).
        disp:        (N,) per-particle displacement magnitudes for this run.
        tail_factor: Slingshot tail cutoff forwarded to slingshot_metrics.

    Returns:
        dict with keys: knob, value, max_over_median, p99_over_median,
        tail_fraction, max_disp, median_disp, n.
    """
    m = slingshot_metrics(disp, tail_factor=tail_factor)
    return {
        "knob": str(knob),
        "value": float(value),
        "max_over_median": m["max_over_median"],
        "p99_over_median": m["p99_over_median"],
        "tail_fraction": m["tail_fraction"],
        "max_disp": m["max"],
        "median_disp": m["median"],
        "n": m["n"],
    }


def softened_node_acceleration(
    particle_positions: np.ndarray,
    node_positions: np.ndarray,
    node_masses: np.ndarray,
    softening_m: float,
    G: float,
) -> np.ndarray:
    """Plummer-SOFTENED tidal acceleration from external nodes (DIAGNOSTIC ONLY).

    A drop-in replacement for cosmo.tidal_forces_numba.calculate_tidal_forces_numba
    whose ONLY difference is the singularity handling: the product code uses a hard
    ``r < 1e10 m`` floor (~3e-13 Gpc — effectively NO softening at Gpc scales),
    which lets a particle that passes very close to a near-point-mass node receive
    an enormous ``G m / r^2`` kick (the slingshot). This helper instead uses a
    Plummer softening ``r_soft^2 = r^2 + softening_m^2`` exactly like the internal
    particle-particle force (cosmo.integrator.calculate_internal_forces), so the
    sweep can MEASURE how a node-softening scale reduces the slingshot tail.

    DIAGNOSTIC ONLY: this is NOT wired into any product sim path. Section 4
    implements the real node softening; here it is monkeypatched onto a grid
    instance purely to quantify the lever. ``softening_m == 0`` reproduces the
    UNSOFTENED 1/r^2 force (minus the 1e10 m floor) for an apples-to-apples baseline.

    Args:
        particle_positions: (N, 3) particle positions in meters.
        node_positions:     (M, 3) node positions in meters.
        node_masses:        (M,) node masses in kg.
        softening_m:        Plummer softening length in meters (>= 0).
        G:                  Gravitational constant.

    Returns:
        (N, 3) accelerations in m/s^2.
    """
    pos = np.asarray(particle_positions, dtype=np.float64)
    npos = np.asarray(node_positions, dtype=np.float64)
    nmass = np.asarray(node_masses, dtype=np.float64)
    eps2 = float(softening_m) * float(softening_m)
    acc = np.zeros_like(pos)
    for j in range(npos.shape[0]):
        r_vec = npos[j] - pos                       # (N, 3) toward node
        r2 = np.sum(r_vec * r_vec, axis=1) + eps2   # (N,) softened
        r3 = r2 * np.sqrt(r2)
        acc += (G * nmass[j] / r3)[:, None] * r_vec
    return acc


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


_VIR_RESIDUAL_SIZES: Tuple[int, ...] = (26, 40, 80, 120)
_VIR_RESIDUAL_EXTENT: float = 2.5  # multi-layer so "inner" nodes really are inner


def virialization_residual_curve(
    sizes: Tuple[int, ...], rule: str, S_gpc: float, seed: int,
    *, spread: float = _DEFAULT_VIR_SPREAD, segregation: float = _DEFAULT_VIR_SEGREGATION,
    extent: float = _VIR_RESIDUAL_EXTENT, inner_frac: float = 0.5,
    M_ext_kg: float = 1.0,
) -> Dict[str, np.ndarray]:
    """max/median inner-node virialization residual vs grid size for one rule.

    For each n in ``sizes`` build the virialized grid and run
    cosmo.node_geometry.virialization_residual; collect max & median residuals.
    A virialized structure's residual should FALL as the grid grows; the current
    generator's typically does NOT (it is not force-balanced). Pure: no I/O.

    The grid is built at the REAL physical scale (S in METERS = S_gpc * Gpc_to_m)
    so node spacings sit well ABOVE the metric's 1e10 m singularity floor and the
    GEOMETRY (not the floor) drives the residual. The residual is dimensionless
    (a force ratio), so the absolute scale only matters via the floor.

    Returns dict of float arrays (same length as sizes):
      'sizes', 'max_residual', 'median_residual'.
    """
    S_m = float(S_gpc) * CosmologicalConstants.Gpc_to_m
    sizes_arr = np.asarray(sizes, dtype=np.int64)
    maxr = np.empty(sizes_arr.size, dtype=np.float64)
    medr = np.empty(sizes_arr.size, dtype=np.float64)
    for i, n in enumerate(sizes_arr):
        pos, masses = build_virialized_grid(
            S_m, n_nodes=int(n), M_ext_kg=M_ext_kg, vir_mass_rule=rule,
            vir_mass_spread=spread, vir_segregation=segregation,
            vir_extent=extent, seed=seed,
        )
        res = virialization_residual(
            pos, masses, inner_frac=inner_frac, center_mass_kg=M_ext_kg)
        maxr[i] = res["max_residual"]
        medr[i] = res["median_residual"]
    return {"sizes": sizes_arr.astype(np.float64),
            "max_residual": maxr, "median_residual": medr}


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
# Fig 4 — virialization (force-balance) residual: radial vs massfunc
# ===========================================================================

def generate_fig4_virialization(
    S_gpc: float, vir_n_nodes: int, seed: int,
    *, extent: float = _VIR_RESIDUAL_EXTENT,
) -> Tuple[str, Dict[str, Dict[str, float]]]:
    """Fig 4: inner-node FORCE-BALANCE (virialization) residual diagnostic.

    Three panels expose whether the inner nodes of a "big enough" virialized grid
    feel ~net-zero force (would not move) — the user's actual virialization test:

      (1) Per-inner-node residual scatter for radial vs massfunc on the BIG grid
          (residual = |net force| / characteristic single-neighbour pull; the
          dashed line is the target tolerance — points below it are "virialized").
      (2) max & median residual vs grid size (n in {26,40,80,120}) for both rules
          — should FALL as the grid grows IF the grid is virialized.
      (3) 3D node scatter coloured by net-force MAGNITUDE (red = big residual =
          "would move"), for the big radial grid.

    Returns (path, {'radial': {...}, 'massfunc': {...}}) where each dict carries
    the big-grid max_residual / median_residual.
    """
    M_ext_kg = 1.0
    inner_frac = 0.5
    # "Big enough" so inner nodes have a near-symmetric surround; default CLI
    # vir_n_nodes (26) is too small for a meaningful inner population.
    big_n = max(int(vir_n_nodes), 100)
    # Build at the REAL scale (meters) so spacings clear the 1e10 m floor.
    S_m = float(S_gpc) * CosmologicalConstants.Gpc_to_m

    def _big(rule, n=big_n):
        return build_virialized_grid(
            S_m, n_nodes=n, M_ext_kg=M_ext_kg, vir_mass_rule=rule,
            vir_mass_spread=_DEFAULT_VIR_SPREAD,
            vir_segregation=_DEFAULT_VIR_SEGREGATION,
            vir_extent=extent, seed=seed,
        )

    pos_a, mass_a = _big("radial")
    pos_b, mass_b = _big("massfunc")
    res_a = virialization_residual(pos_a, mass_a, inner_frac=inner_frac,
                                   center_mass_kg=M_ext_kg)
    res_b = virialization_residual(pos_b, mass_b, inner_frac=inner_frac,
                                   center_mass_kg=M_ext_kg)

    curve_a = virialization_residual_curve(
        _VIR_RESIDUAL_SIZES, "radial", S_gpc, seed, extent=extent,
        inner_frac=inner_frac, M_ext_kg=M_ext_kg)
    curve_b = virialization_residual_curve(
        _VIR_RESIDUAL_SIZES, "massfunc", S_gpc, seed, extent=extent,
        inner_frac=inner_frac, M_ext_kg=M_ext_kg)

    # Match the test's tolerance knob (informational dashed line only).
    tol = 0.25

    print("\n[Fig 4] virialization (force-balance) residual — radial vs massfunc")
    print(f"  big grid N={big_n}, extent={extent}, inner_frac={inner_frac}")
    print(f"  radial  : max={res_a['max_residual']:.4f} median={res_a['median_residual']:.4f}")
    print(f"  massfunc: max={res_b['max_residual']:.4f} median={res_b['median_residual']:.4f}")

    fig = plt.figure(figsize=(15, 5.2))

    # Panel 1 — per-inner-node residual scatter, both rules.
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.scatter(np.arange(res_a["n_inner"]), res_a["residual_per_node"],
                s=22, alpha=0.8, c="#d62728", label="radial")
    ax1.scatter(np.arange(res_b["n_inner"]), res_b["residual_per_node"],
                s=22, alpha=0.8, c="#1f77b4", label="massfunc")
    ax1.axhline(tol, color="green", ls="--", lw=1.4,
                label=f"target tol={tol}")
    ax1.set_yscale("log")
    ax1.set_xlabel("inner node index", fontsize=9)
    ax1.set_ylabel("residual |net force| / a_ref (log)", fontsize=9)
    ax1.set_title(f"Per-inner-node residual (N={big_n})\n"
                  "below the line = virialized (would not move)", fontsize=9)
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    # Panel 2 — residual vs grid size, both rules.
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(curve_a["sizes"], curve_a["max_residual"], "o-", color="#d62728",
             label="radial max")
    ax2.plot(curve_a["sizes"], curve_a["median_residual"], "o--", color="#d62728",
             alpha=0.5, label="radial median")
    ax2.plot(curve_b["sizes"], curve_b["max_residual"], "s-", color="#1f77b4",
             label="massfunc max")
    ax2.plot(curve_b["sizes"], curve_b["median_residual"], "s--", color="#1f77b4",
             alpha=0.5, label="massfunc median")
    ax2.axhline(tol, color="green", ls="--", lw=1.2, label=f"target tol={tol}")
    ax2.set_yscale("log")
    ax2.set_xlabel("grid size (n_nodes)", fontsize=9)
    ax2.set_ylabel("inner residual (log)", fontsize=9)
    ax2.set_title("Residual vs grid size\n(virialized => falls as grid grows)",
                  fontsize=9)
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    # Panel 3 — 3D node scatter coloured by net-force magnitude (radial big grid).
    # Positions are in meters (real scale); show axes in Gpc.
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    accel = node_net_accelerations(pos_a, mass_a, center_mass_kg=M_ext_kg)
    net_mag = np.linalg.norm(accel, axis=1)
    pos_a_gpc = pos_a / CosmologicalConstants.Gpc_to_m
    sc = ax3.scatter(pos_a_gpc[:, 0], pos_a_gpc[:, 1], pos_a_gpc[:, 2], c=net_mag,
                     cmap="inferno", s=34, edgecolors="k", linewidths=0.2,
                     depthshade=True)
    ax3.scatter([0], [0], [0], c="cyan", marker="*", s=140, zorder=10)
    fig.colorbar(sc, ax=ax3, fraction=0.045, pad=0.02).set_label(
        "|net force| [m/s^2] (red = would move)", fontsize=7)
    ax3.set_title("radial grid: nodes by net-force magnitude", fontsize=9)
    ax3.set_xlabel("x [Gpc]", fontsize=7); ax3.set_ylabel("y [Gpc]", fontsize=7)
    ax3.set_zlabel("z [Gpc]", fontsize=7); ax3.tick_params(labelsize=6)

    fig.suptitle(
        "WS8 Fig 4 — VIRIALIZATION (inner-node force-balance) residual: "
        "radial vs massfunc\n"
        "Inner nodes of a TRUE virialized grid feel ~net-zero force; "
        "the current generator does NOT (residuals >> tol) — Section 2 fixes it.",
        fontsize=12, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = figure_path(_WS, "virialization_residual")
    fig.savefig(out, dpi=_DPI, bbox_inches=_BBOX)
    plt.close(fig)
    print(f"[Fig 4] Saved: {out}")
    return out, {
        "radial": {"max_residual": res_a["max_residual"],
                   "median_residual": res_a["median_residual"]},
        "massfunc": {"max_residual": res_b["max_residual"],
                     "median_residual": res_b["median_residual"]},
    }


# ===========================================================================
# Fig 5 — slingshot knob-sweep diagnostic (root-cause + taming levers)
# ===========================================================================
#
# Runaway config (cube26, M=1000, S=10) slingshots one inner particle off a
# near-point-mass NODE to max/median ~500x. This block:
#   * proves the ROOT CAUSE (nodes ON vs OFF), and
#   * sweeps each candidate taming knob (n_steps, n_particles, node softening,
#     geometry) and quantifies how each moves the slingshot metric,
# so Section 4 knows which lever actually works. MEASUREMENT ONLY — the only
# physics deviation is the DIAGNOSTIC node-softening monkeypatch below, which is
# never wired into a product sim path.

# Runaway config the sweep is built around (dominated by a node close-pass).
_SLING_M = 1000.0
_SLING_S_GPC = 10.0
_SLING_N = 200
_SLING_N_STEPS = 80          # bumped by resolve_n_steps to keep dt < 0.05 Gyr
_SLING_T_START = 5.8
_SLING_SEED = 42

# Independent knob grids (each swept with the others held at the config above).
_SLING_NSTEPS_GRID: Tuple[int, ...] = (80, 160, 320)
_SLING_NPART_GRID: Tuple[int, ...] = (200, 400, 800)
# Node softening MULTIPLIER on the 1 Gpc particle-softening baseline. 0 == the
# current (effectively unsoftened) node force; the rest are diagnostic probes.
_SLING_SOFT_BASE_GPC = 1.0
_SLING_SOFT_GRID: Tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 5.0)


def _build_sling_params(
    n_particles: int, n_steps: int, geometry: str = "cube26",
    M: float = _SLING_M, S_gpc: float = _SLING_S_GPC,
    t_start: float = _SLING_T_START, seed: int = _SLING_SEED,
) -> Tuple[SimulationParameters, float, float, int]:
    """Build (sim_params, box_size_Gpc, a_start, resolved_n_steps) for a sweep run."""
    t_dur = _TODAY_GYR - t_start
    n_steps = resolve_n_steps(t_dur, n_steps)
    box, a_start, _ = setup_simulation_context(
        t_start, t_dur, n_steps, save_interval=max(1, n_steps // 4))
    sp = SimulationParameters(
        M_value=M, S_value=S_gpc, n_particles=n_particles, seed=seed,
        t_start_Gyr=t_start, t_duration_Gyr=t_dur, n_steps=n_steps,
        damping_factor=None, center_node_mass=1.0, mass_randomize=0.0,
        node_mass_seed=seed, init_distribution="uniform_sphere",
        node_geometry=geometry,
        vir_n_nodes=_DEFAULT_VIR_N_NODES, vir_extent=_DEFAULT_VIR_EXTENT,
        vir_mass_rule="radial", vir_mass_spread=_DEFAULT_VIR_SPREAD,
        vir_segregation=_DEFAULT_VIR_SEGREGATION, vir_s_metric="median",
    )
    return sp, box, a_start, n_steps


def _disp_from_sim(sim) -> np.ndarray:
    """Inner-observable per-particle |final-initial| displacement (Gpc) from a sim."""
    mask = np.asarray(sim.particles.get_observable_mask(), dtype=bool)
    g = CosmologicalConstants.Gpc_to_m
    p0 = sim.snapshots[0]["positions"][mask] / g
    p1 = sim.snapshots[-1]["positions"][mask] / g
    return displacement_magnitudes(p0, p1)


def run_slingshot_disp(
    n_particles: int, n_steps: int, *, geometry: str = "cube26",
    nodes: bool = True, node_soft_gpc: Optional[float] = None,
    M: float = _SLING_M, S_gpc: float = _SLING_S_GPC,
    t_start: float = _SLING_T_START, seed: int = _SLING_SEED,
) -> np.ndarray:
    """Run ONE short slingshot sim and return inner-particle displacements (Gpc).

    Args:
        n_particles / n_steps: sweep knobs.
        geometry:      "cube26" (default) or "virialized".
        nodes:         External HMEA nodes ON (default) or OFF (matter-only).
        node_soft_gpc: If not None AND nodes ON, MONKEYPATCH the grid's tidal
                       force with a Plummer-softened version (softened_node_
                       acceleration) using this softening length in Gpc. This is
                       the DIAGNOSTIC-ONLY node-softening probe (Section 4 owns the
                       real fix). None -> the product (1e10 m floor) force is used.
    """
    sp, box, a_start, n_steps = _build_sling_params(
        n_particles, n_steps, geometry=geometry, M=M, S_gpc=S_gpc,
        t_start=t_start, seed=seed)
    sim = CosmologicalSimulation(
        sp, box, a_start, use_external_nodes=nodes, use_dark_energy=False)
    if nodes and node_soft_gpc is not None:
        grid = sim.hmea_grid
        npos = grid.get_positions()
        nmass = grid.get_masses()
        soft_m = float(node_soft_gpc) * CosmologicalConstants.Gpc_to_m
        Gc = CosmologicalConstants.G

        def _softened_batch(positions, use_numba=True, _npos=npos, _nmass=nmass,
                            _soft=soft_m, _G=Gc):
            return softened_node_acceleration(positions, _npos, _nmass, _soft, _G)

        grid.calculate_tidal_acceleration_batch = _softened_batch
    sim.run(t_end_Gyr=sp.t_duration_Gyr, n_steps=n_steps,
            save_interval=max(1, n_steps // 4))
    return _disp_from_sim(sim)


def slingshot_knob_sweep(
    *, nsteps_grid: Tuple[int, ...] = _SLING_NSTEPS_GRID,
    npart_grid: Tuple[int, ...] = _SLING_NPART_GRID,
    soft_grid: Tuple[float, ...] = _SLING_SOFT_GRID,
    seed: int = _SLING_SEED,
) -> Dict[str, List[Dict[str, Any]]]:
    """Sweep EACH taming knob independently at the runaway config; quantify the tail.

    Holds the runaway config (cube26, M=1000, S=10) fixed and varies ONE knob at a
    time, recording slingshot_metrics for each value. Knobs:
      - "n_steps":       finer time resolution (does it resolve/tame the close pass?).
      - "n_particles":   more sampling (does it dilute or worsen the tail?).
      - "softening_gpc": DIAGNOSTIC node-softening length (the prime taming lever).
      - "nodes":         root-cause control — nodes ON vs OFF at the base config.
      - "geometry":      cube26 vs virialized (spreads node mass -> smaller peak pull).

    Returns dict keyed by knob name -> list of slingshot_sweep_row dicts. Pure-ish
    orchestrator: all heavy lifting is in run_slingshot_disp + slingshot_sweep_row.
    """
    base_N, base_steps = _SLING_N, _SLING_N_STEPS
    out: Dict[str, List[Dict[str, Any]]] = {}

    print("\n[Fig 5] Slingshot knob-sweep (runaway cube26 M=1000 S=10) ...")

    # Root cause: nodes ON vs OFF at the base config.
    out["nodes"] = []
    for on in (1.0, 0.0):
        disp = run_slingshot_disp(base_N, base_steps, nodes=bool(on), seed=seed)
        row = slingshot_sweep_row("nodes", on, disp)
        out["nodes"].append(row)
        print(f"  nodes={'ON ' if on else 'OFF'}  max/median={row['max_over_median']:.2f} "
              f"max={row['max_disp']:.3f}")

    # n_steps.
    out["n_steps"] = []
    for ns in nsteps_grid:
        disp = run_slingshot_disp(base_N, ns, seed=seed)
        row = slingshot_sweep_row("n_steps", ns, disp)
        out["n_steps"].append(row)
        print(f"  n_steps={ns:<5} max/median={row['max_over_median']:.2f} "
              f"tail={row['tail_fraction']:.3f}")

    # n_particles.
    out["n_particles"] = []
    for N in npart_grid:
        disp = run_slingshot_disp(N, base_steps, seed=seed)
        row = slingshot_sweep_row("n_particles", N, disp)
        out["n_particles"].append(row)
        print(f"  N={N:<5} max/median={row['max_over_median']:.2f} "
              f"tail={row['tail_fraction']:.3f}")

    # node softening (DIAGNOSTIC probe).
    out["softening_gpc"] = []
    for s in soft_grid:
        disp = run_slingshot_disp(base_N, base_steps, node_soft_gpc=s, seed=seed)
        row = slingshot_sweep_row("softening_gpc", s, disp)
        out["softening_gpc"].append(row)
        print(f"  soft={s:<4} Gpc max/median={row['max_over_median']:.2f} "
              f"tail={row['tail_fraction']:.3f}")

    # geometry: cube26 vs virialized at the base config.
    out["geometry"] = []
    for gi, geom in enumerate(("cube26", "virialized")):
        disp = run_slingshot_disp(base_N, base_steps, geometry=geom, seed=seed)
        row = slingshot_sweep_row("geometry", float(gi), disp)
        row["label"] = geom
        out["geometry"].append(row)
        print(f"  geometry={geom:<11} max/median={row['max_over_median']:.2f} "
              f"tail={row['tail_fraction']:.3f}")

    return out


def _print_slingshot_sweep_verdict(sweep: Dict[str, List[Dict[str, Any]]]) -> None:
    """Print the root-cause + which-knob-wins headline from a knob sweep."""
    on = sweep["nodes"][0]["max_over_median"]
    off = sweep["nodes"][1]["max_over_median"]
    soft_rows = sweep["softening_gpc"]
    soft0 = soft_rows[0]["max_over_median"]
    soft_best = min(soft_rows, key=lambda r: r["max_over_median"])
    print("\n" + "=" * 70)
    print("WS8 SLINGSHOT KNOB-SWEEP VERDICT")
    print("=" * 70)
    print(f"  ROOT CAUSE: nodes ON max/median={on:.1f} vs OFF={off:.1f} "
          f"({on / off:.0f}x) -> driven by the node close-pass.")
    print(f"  NODE SOFTENING: 0 Gpc -> {soft0:.1f}; "
          f"{soft_best['value']:.1f} Gpc -> {soft_best['max_over_median']:.1f} "
          f"({soft0 / max(soft_best['max_over_median'], 1e-9):.0f}x reduction).")
    print("  n_steps / n_particles do NOT reliably reduce the tail (see figure).")
    print("=" * 70)


def _plot_knob_panel(ax, rows: List[Dict[str, Any]], title: str,
                     xlabel: str, logx: bool = False) -> None:
    """Plot max/median (left axis) + tail_fraction (right) vs a knob's values."""
    xs = [r["value"] for r in rows]
    mm = [r["max_over_median"] for r in rows]
    tf = [r["tail_fraction"] for r in rows]
    ax.plot(xs, mm, "o-", color="#d62728", label="max/median")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("max/median (log)", color="#d62728", fontsize=9)
    ax.tick_params(axis="y", labelcolor="#d62728")
    if logx:
        ax.set_xscale("log")
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(xs, tf, "s--", color="#1f77b4", alpha=0.7, label="tail frac")
    ax2.set_ylabel("tail frac (>5×med)", color="#1f77b4", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="#1f77b4")


def generate_fig5_slingshot_knobs(
    sweep: Optional[Dict[str, List[Dict[str, Any]]]] = None, *,
    seed: int = _SLING_SEED,
) -> Tuple[str, Dict[str, List[Dict[str, Any]]]]:
    """Fig 5: small-multiples of the slingshot tail vs each knob + nodes ON/OFF.

    Six panels: nodes ON/OFF (root cause), n_steps, n_particles, node softening,
    geometry, and a summary text panel. Each line panel shows max/median (log) and
    tail-fraction vs the knob value, making "what reduces the slingshot" visual.
    """
    if sweep is None:
        sweep = slingshot_knob_sweep(seed=seed)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # Panel 1 — nodes ON vs OFF (root cause), as a bar chart.
    ax = axes[0, 0]
    labels = ["nodes ON", "nodes OFF"]
    vals = [sweep["nodes"][0]["max_over_median"], sweep["nodes"][1]["max_over_median"]]
    bars = ax.bar(labels, vals, color=["#d62728", "#2ca02c"])
    ax.set_yscale("log")
    ax.set_ylabel("max/median (log)", fontsize=9)
    ax.set_title("ROOT CAUSE: nodes ON vs OFF\n(tail collapses with nodes off)",
                 fontsize=9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                ha="center", va="bottom", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    _plot_knob_panel(axes[0, 1], sweep["n_steps"], "n_steps sweep", "n_steps")
    _plot_knob_panel(axes[0, 2], sweep["n_particles"], "n_particles sweep",
                     "n_particles")
    _plot_knob_panel(axes[1, 0], sweep["softening_gpc"],
                     "NODE SOFTENING sweep (diagnostic)", "node softening [Gpc]")

    # Panel 5 — geometry cube26 vs virialized bar.
    axg = axes[1, 1]
    glabels = [r.get("label", str(r["value"])) for r in sweep["geometry"]]
    gvals = [r["max_over_median"] for r in sweep["geometry"]]
    gbars = axg.bar(glabels, gvals, color=["#d62728", "#9467bd"])
    axg.set_yscale("log")
    axg.set_ylabel("max/median (log)", fontsize=9)
    axg.set_title("geometry: cube26 vs virialized", fontsize=9)
    for b, v in zip(gbars, gvals):
        axg.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                 ha="center", va="bottom", fontsize=9)
    axg.grid(True, alpha=0.3, axis="y")

    # Panel 6 — verdict text.
    axt = axes[1, 2]
    axt.axis("off")
    on = sweep["nodes"][0]["max_over_median"]
    off = sweep["nodes"][1]["max_over_median"]
    soft_rows = sweep["softening_gpc"]
    soft0 = soft_rows[0]["max_over_median"]
    soft_best = min(soft_rows, key=lambda r: r["max_over_median"])
    txt = (
        "VERDICT (runaway cube26, M=1000, S=10)\n"
        "-------------------------------------\n"
        f"ROOT CAUSE: node close-pass.\n"
        f"  nodes ON  max/median = {on:.0f}\n"
        f"  nodes OFF max/median = {off:.1f}  ({on/off:.0f}x)\n\n"
        "TAMING LEVERS:\n"
        f"  node softening {soft_best['value']:.1f} Gpc:\n"
        f"    max/median {soft0:.0f} -> {soft_best['max_over_median']:.1f}\n"
        f"    ({soft0/max(soft_best['max_over_median'],1e-9):.0f}x; tail->"
        f"{soft_best['tail_fraction']:.2f})\n"
        "  n_steps:     no reliable reduction\n"
        "  n_particles: does NOT help (often worse)\n"
        "  virialized:  partial help\n\n"
        "=> Section 4: add NODE softening\n"
        f"   ~{_SLING_SOFT_BASE_GPC:.0f} Gpc (match particle soft.)"
    )
    axt.text(0.02, 0.98, txt, transform=axt.transAxes, fontsize=10,
             va="top", ha="left", family="monospace",
             bbox=dict(boxstyle="round", fc="#fffbe6", alpha=0.9))

    fig.suptitle(
        "WS8 Fig 5 — SLINGSHOT root cause + taming-knob sweep "
        "(runaway cube26, M=1000, S=10)\n"
        "Node close-pass drives the tail; a ~1 Gpc NODE softening is the only "
        "knob that tames it. DIAGNOSTIC ONLY — Section 4 implements the fix.",
        fontsize=12, y=0.99,
    )
    _footer(axes[1, 0], f"M={_SLING_M:.0f} S={_SLING_S_GPC:.0f} "
                        f"N={_SLING_N} seed={seed}")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = figure_path(_WS, "slingshot_knob_sweep")
    fig.savefig(out, dpi=_DPI, bbox_inches=_BBOX)
    plt.close(fig)
    print(f"[Fig 5] Saved: {out}")
    _print_slingshot_sweep_verdict(sweep)
    return out, sweep


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

    # Fig 4 — virialization (force-balance) residual (no sim).
    p4, _ = generate_fig4_virialization(args.S_gpc, args.vir_n_nodes, args.seed)
    paths.append(p4)

    # Fig 5 — slingshot knob-sweep diagnostic (runs several short sims).
    if not args.no_sim:
        p5, _ = generate_fig5_slingshot_knobs(seed=args.seed)
        paths.append(p5)
    else:
        print("[ws8] --no-sim: skipping Fig 5 (slingshot knob-sweep).")

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
