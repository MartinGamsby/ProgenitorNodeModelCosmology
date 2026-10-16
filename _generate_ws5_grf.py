#!/usr/bin/env python3
"""
WS5 - GRF investigation: is init_distribution="grf" actually broken? (item 11)
==============================================================================

The handover records a swing at M=1500/S=30: uniform_sphere ~0.52 vs GRF ~1.56
chi2/dof, which CONTRADICTS the lode expectation ("isotropic chi2 is shape-driven,
not sampling-driven, so GRF and uniform give ~the same chi2"). This script
DIAGNOSES whether the GRF path is BROKEN or merely UNTESTED, with numbers and a
figure, decomposing the swing across H1 (real physics) / H2 (near-runaway) /
H3 (N-noise) / H4 (GRF-setup bug).

Two parts:

  PART A - GRF FIELD STATISTICS (no sim).  Inspect the raw GRF density field and
  the realized particle field directly: P(k) shape monotonicity, density-field
  mean/variance/NaN, Zel'dovich displacement RMS vs cell size, fraction of
  particles CLIPPED at the box edge (the H4 smoking gun), realized RMS/COM after
  the shared post-processing, and a clustering metric vs uniform_sphere.

  PART B - SWING DECOMPOSITION (small sims, authoritative chi2).  Run GRF vs
  uniform_sphere at the swing config (M=1500/S=30) AND a weak-field control,
  across seeds and N, scoring with the SAME compute_pantheon_metrics the sweep
  uses. Reports chi2/dof + growth_factor, the per-seed spread, and chi2 vs N
  (does GRF converge to uniform as N grows?).

Outputs (results/figures/ws5/, gitignored):
    grf_vs_uniform.csv   the full {part, config, seed, N} table
    grf_vs_uniform.png   the field-statistics + swing-decomposition panels
and prints everything to stdout with an explicit H1/H2/H3/H4 verdict.

Usage:
    PYTHONIOENCODING=utf-8 python _generate_ws5_grf.py            # default ladder
    PYTHONIOENCODING=utf-8 python _generate_ws5_grf.py --quick    # tiny smoke run
"""

from cosmo.encoding import configure_utf8_stdout
configure_utf8_stdout()

import argparse
import csv
import os
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cosmo.constants import CosmologicalConstants, SimulationParameters
from cosmo.particles import ParticleSystem
from cosmo.initial_distributions import (
    sample_grf, sample_uniform_sphere, _power_spectrum,
    grf_density_field, grf_field_stats,
)
from cosmo.factories import (
    run_external_node_simulation, setup_simulation_context, results_to_sim_result,
)
from cosmo.parameter_sweep import compute_pantheon_metrics
from cosmo.pantheon import load_pantheon
from cosmo.plots import figure_path, _DPI, _BBOX

_G = CosmologicalConstants.Gpc_to_m
_TODAY_GYR = 13.8
_T_START = 2.9


# --------------------------------------------------------------------------
# PART A - field statistics (pure, no sim)
# --------------------------------------------------------------------------

def _clip_fraction(positions: np.ndarray, box_size_m: float) -> float:
    """Fraction of particles sitting exactly on the bounding-box wall (clipped)."""
    limit = box_size_m / 2.0
    on_wall = np.any(np.isclose(np.abs(positions), limit, rtol=1e-9, atol=0.0), axis=1)
    return float(np.mean(on_wall))


def _nn_clustering(positions: np.ndarray, k: int = 1) -> float:
    """Median nearest-neighbour distance (smaller => more clustered). Normalised
    by the RMS radius so it is scale-free and comparable across distributions."""
    from scipy.spatial import cKDTree
    tree = cKDTree(positions)
    d, _ = tree.query(positions, k=k + 1)  # column 0 is self (0 distance)
    nn = d[:, 1]
    rms = np.sqrt(np.mean(np.sum(positions ** 2, axis=1)))
    return float(np.median(nn) / rms)


def part_a_field_stats(box_size_m: float, N: int = 2000, Ng: int = 64,
                       seed: int = 42) -> Dict[str, Any]:
    """Inspect the raw GRF density field + realized particle field."""
    print("=" * 74)
    print("PART A - GRF FIELD STATISTICS (no sim)")
    print(f"  box={box_size_m/_G:.2f} Gpc, N={N}, Ng={Ng}, seed={seed}")
    print("=" * 74)

    # --- raw density field delta(x) ---
    fs = grf_field_stats(box_size_m, seed=seed, Ng=Ng)
    print("\n[density field delta(x)]")
    print(f"  mean(delta)            = {fs['delta_mean']:+.3e}   (expect ~0)")
    print(f"  std(delta)             = {fs['delta_std']:.3e}")
    print(f"  any NaN/Inf in delta   = {fs['delta_has_nan']}")
    print(f"  P(k) low-k mean        = {fs['pk_low_mean']:.3e}")
    print(f"  P(k) high-k mean       = {fs['pk_high_mean']:.3e}")
    print(f"  P(k) monotone-decays   = {fs['pk_decays']}   (low-k power > high-k)")
    print(f"  displacement RMS / cell= {fs['disp_rms_over_cell']:.3f}   "
          f"(target 0.5; >1 => over-perturbed)")

    # --- realized particle field (raw, before post-processing) ---
    grf_raw = sample_grf(N, box_size_m, seed=seed, Ng=Ng)
    uni_raw = sample_uniform_sphere(N, (box_size_m / 2) / np.sqrt(3 / 5),
                                    np.random.default_rng(seed))
    clip_frac = _clip_fraction(grf_raw, box_size_m)
    print("\n[realized particle field, raw (pre post-processing)]")
    print(f"  N                      = {N}")
    print(f"  any NaN in GRF pos      = {bool(np.any(np.isnan(grf_raw)))}")
    print(f"  fraction clipped@wall   = {clip_frac:.3%}   "
          f"(H4 smoking gun if large)")

    # --- after the SHARED post-processing (center + RMS-norm), via ParticleSystem ---
    def _post(init, **kw):
        np.random.seed(seed)
        ps = ParticleSystem(
            n_particles=N, box_size_m=box_size_m, total_mass_kg=1e54,
            a_start=1.0, use_dark_energy=False, mass_randomize=0.0,
            init_distribution=init, init_kwargs=kw,
        )
        return ps.get_positions()

    grf_pp = _post("grf", Ng=Ng)
    uni_pp = _post("uniform_sphere")
    rms_grf = float(np.sqrt(np.mean(np.sum(grf_pp ** 2, axis=1))))
    rms_uni = float(np.sqrt(np.mean(np.sum(uni_pp ** 2, axis=1))))
    target = box_size_m / 2
    try:
        clus_grf = _nn_clustering(grf_pp)
        clus_uni = _nn_clustering(uni_pp)
    except Exception as exc:  # scipy missing: skip clustering, not fatal
        clus_grf = clus_uni = float("nan")
        print(f"  (clustering skipped: {exc})")

    print("\n[after shared post-processing (center + RMS-norm)]")
    print(f"  RMS GRF / target        = {rms_grf/target:.6f}   (must be 1.0)")
    print(f"  RMS uniform / target    = {rms_uni/target:.6f}   (must be 1.0)")
    print(f"  COM GRF / target        = {np.linalg.norm(np.mean(grf_pp,axis=0))/target:.2e}")
    print(f"  median NN/RMS  GRF      = {clus_grf:.4f}")
    print(f"  median NN/RMS  uniform  = {clus_uni:.4f}   "
          f"(GRF smaller => more clustered)")

    return {
        **fs,
        "clip_frac": clip_frac,
        "rms_grf_over_target": rms_grf / target,
        "rms_uni_over_target": rms_uni / target,
        "clus_grf": clus_grf,
        "clus_uni": clus_uni,
        "grf_raw": grf_raw,
        "uni_raw": uni_raw,
        "grf_pp": grf_pp,
        "uni_pp": uni_pp,
    }


# --------------------------------------------------------------------------
# PART B - swing decomposition (small sims, authoritative chi2)
# --------------------------------------------------------------------------

def _run_sim(M: float, S_gpc: float, N: int, seed: int, init: str,
             n_steps: int, pantheon_data, Ng: int = 64) -> Dict[str, float]:
    """One sim -> chi2/dof + growth via the SAME scorer the sweep uses.

    ``init`` is one of: "uniform_sphere", "grf" (sphere support, the new default),
    "grf_box" (legacy GRF-perturbed cube, for the before/after comparison)."""
    t_dur = _TODAY_GYR - _T_START
    box, a_start, _ = setup_simulation_context(
        _T_START, t_dur, n_steps, save_interval=max(1, n_steps // 8))
    if init == "grf":
        init_name, init_kwargs = "grf", {"Ng": Ng, "support": "sphere"}
    elif init == "grf_box":
        init_name, init_kwargs = "grf", {"Ng": Ng, "support": "box"}
    else:
        init_name, init_kwargs = init, {}
    sp = SimulationParameters(
        M_value=M, S_value=S_gpc, n_particles=N, seed=seed,
        t_start_Gyr=_T_START, t_duration_Gyr=t_dur, n_steps=n_steps,
        damping_factor=None, center_node_mass=1.0, mass_randomize=0.0,
        node_geometry="cube26",
        init_distribution=init_name, init_kwargs=init_kwargs,
    )
    ext = run_external_node_simulation(sp, box, a_start,
                                       save_interval=max(1, n_steps // 8))
    sim_result = results_to_sim_result(ext, sp)
    metrics = compute_pantheon_metrics(sim_result, pantheon_data, sp.t_start_Gyr)
    return {
        "chi2_dof": float(metrics.get("chi2_dof", float("nan"))),
        "growth_factor": float(metrics.get("growth_factor", float("nan"))),
    }


# Configs: the swing cell + a weak-field control.
_CONFIGS = [
    ("swing M1500/S30", 1500.0, 30.0),
    ("weak  M100/S60", 100.0, 60.0),
]


def part_b_swing(pantheon_data, *, n_steps: int,
                 seeds: Tuple[int, ...], n_list: Tuple[int, ...],
                 Ng: int = 64) -> List[Dict[str, Any]]:
    print("\n" + "=" * 74)
    print("PART B - SWING DECOMPOSITION (small sims, authoritative chi2)")
    print(f"  n_steps={n_steps}, seeds={seeds}, N={n_list}, Ng={Ng}")
    print("=" * 74)
    rows: List[Dict[str, Any]] = []
    for label, M, S in _CONFIGS:
        for N in n_list:
            for seed in seeds:
                for init in ("uniform_sphere", "grf", "grf_box"):
                    r = _run_sim(M, S, N, seed, init, n_steps, pantheon_data, Ng=Ng)
                    rows.append({
                        "config": label, "M": M, "S": S, "N": N,
                        "seed": seed, "init": init,
                        "chi2_dof": r["chi2_dof"],
                        "growth_factor": r["growth_factor"],
                    })
                    print(f"  {label:<16} N={N:<5} seed={seed} {init:<14} "
                          f"chi2/dof={r['chi2_dof']:.4f}  growth={r['growth_factor']:.4f}",
                          flush=True)
    return rows


def _summary(rows: List[Dict[str, Any]]) -> None:
    """Print per-config GRF-vs-uniform spread + N-convergence verdict."""
    print("\n" + "-" * 74)
    print("SWING SUMMARY (mean +/- std over seeds, per config/N/init)")
    print("-" * 74)
    configs = sorted({r["config"] for r in rows})
    n_list = sorted({r["N"] for r in rows})
    def _mean(cfg, N, init):
        vals = [r["chi2_dof"] for r in rows
                if r["config"] == cfg and r["N"] == N and r["init"] == init
                and np.isfinite(r["chi2_dof"])]
        return (np.mean(vals), np.std(vals)) if vals else (np.nan, np.nan)

    for cfg in configs:
        print(f"\n{cfg}")
        for N in n_list:
            line = f"  N={N:<5}"
            for init in ("uniform_sphere", "grf", "grf_box"):
                m, s = _mean(cfg, N, init)
                line += f" | {init}: {m:.4f}+/-{s:.4f}"
            mu, _ = _mean(cfg, N, "uniform_sphere")
            mg, _ = _mean(cfg, N, "grf")
            mb, _ = _mean(cfg, N, "grf_box")
            line += (f" || delta(grf-uni)={mg-mu:+.4f}"
                     f"  delta(box-uni)={mb-mu:+.4f}")
            print(line)


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------

def make_figure(field: Dict[str, Any], rows: List[Dict[str, Any]]) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (0,0) P(k) shape
    ax = axes[0, 0]
    k = np.logspace(-3, 1, 200)
    ax.loglog(k, _power_spectrum(k), color="tab:blue")
    ax.set_xlabel("k [h/Mpc]")
    ax.set_ylabel("P(k)  (arb.)")
    ax.set_title("GRF input power spectrum (BBKS LCDM)\n"
                 f"decays high-k: {field['pk_decays']}")

    # (0,1) particle slice x-y (GRF vs uniform) after post-processing
    ax = axes[0, 1]
    g = field["grf_pp"] / _G
    u = field["uni_pp"] / _G
    sel_g = np.abs(g[:, 2]) < 0.15 * np.std(g[:, 2]) + 1.0  # thin slice
    ax.scatter(g[:, 0], g[:, 1], s=4, alpha=0.5, color="tab:red", label="grf")
    ax.scatter(u[:, 0], u[:, 1], s=4, alpha=0.3, color="tab:gray", label="uniform")
    ax.set_aspect("equal")
    ax.set_xlabel("x [Gpc]"); ax.set_ylabel("y [Gpc]")
    ax.set_title(f"Particle field (post-proc)\nGRF clip@wall={field['clip_frac']:.1%}")
    ax.legend(loc="upper right", fontsize=8)

    # (1,0) chi2/dof vs N, GRF vs uniform, swing config
    ax = axes[1, 0]
    swing = "swing M1500/S30"
    n_list = sorted({r["N"] for r in rows if r["config"] == swing})
    _series = (("uniform_sphere", "tab:gray", "uniform_sphere"),
               ("grf", "tab:green", "grf (sphere, FIXED)"),
               ("grf_box", "tab:red", "grf (box, legacy)"))
    for init, col, lab in _series:
        means, errs = [], []
        for N in n_list:
            vals = [r["chi2_dof"] for r in rows if r["config"] == swing
                    and r["N"] == N and r["init"] == init and np.isfinite(r["chi2_dof"])]
            means.append(np.mean(vals) if vals else np.nan)
            errs.append(np.std(vals) if vals else 0.0)
        ax.errorbar(n_list, means, yerr=errs, marker="o", capsize=3,
                    color=col, label=lab)
    ax.axhline(0.436, ls="--", color="green", lw=1, label="LCDM 0.436")
    ax.axhline(0.843, ls=":", color="orange", lw=1, label="EdS 0.843")
    ax.set_xscale("log")
    ax.set_xlabel("N particles"); ax.set_ylabel("chi2/dof")
    ax.set_title("Swing cell M=1500/S=30: chi2/dof vs N\n(GRF sphere fix collapses the swing)")
    ax.legend(fontsize=7)

    # (1,1) growth_factor vs N, GRF vs uniform, swing config
    ax = axes[1, 1]
    for init, col, lab in _series:
        means = []
        for N in n_list:
            vals = [r["growth_factor"] for r in rows if r["config"] == swing
                    and r["N"] == N and r["init"] == init and np.isfinite(r["growth_factor"])]
            means.append(np.mean(vals) if vals else np.nan)
        ax.plot(n_list, means, marker="s", color=col, label=lab)
    # growth anchor target
    from cosmo.parameter_sweep import expected_growth_factor
    ax.axhline(expected_growth_factor(_T_START), ls="--", color="black", lw=1,
               label=f"anchor {expected_growth_factor(_T_START):.2f}")
    ax.set_xscale("log")
    ax.set_xlabel("N particles"); ax.set_ylabel("growth_factor a[-1]/a[0]")
    ax.set_title("Swing cell M=1500/S=30: growth vs N")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = figure_path("ws5", "grf_vs_uniform")
    plt.savefig(out, dpi=_DPI, bbox_inches=_BBOX)
    plt.close(fig)
    return out


def write_csv(field: Dict[str, Any], rows: List[Dict[str, Any]]) -> str:
    out = figure_path("ws5", "grf_vs_uniform").replace(".png", ".csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# PART A field statistics"])
        for k in ("delta_mean", "delta_std", "delta_has_nan", "pk_decays",
                  "disp_rms_over_cell", "clip_frac", "rms_grf_over_target",
                  "rms_uni_over_target", "clus_grf", "clus_uni"):
            w.writerow([k, field[k]])
        w.writerow([])
        w.writerow(["# PART B swing decomposition"])
        w.writerow(["config", "M", "S", "N", "seed", "init",
                    "chi2_dof", "growth_factor"])
        for r in rows:
            w.writerow([r["config"], r["M"], r["S"], r["N"], r["seed"],
                        r["init"], r["chi2_dof"], r["growth_factor"]])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="tiny smoke run (fast, not for headline numbers)")
    args = ap.parse_args()

    pantheon_data = load_pantheon()
    const = CosmologicalConstants()
    box = 10.0 * const.Gpc_to_m

    if args.quick:
        seeds = (42, 7)
        n_list = (300, 600)
        n_steps = 273
        Ng = 32
        N_field = 600
    else:
        seeds = (42, 7, 123)
        n_list = (1000, 2000, 4000)
        n_steps = 280
        Ng = 64
        N_field = 2000

    field = part_a_field_stats(box, N=N_field, Ng=Ng)
    rows = part_b_swing(pantheon_data, n_steps=n_steps, seeds=seeds,
                        n_list=n_list, Ng=Ng)
    _summary(rows)

    png = make_figure(field, rows)
    csv_out = write_csv(field, rows)
    print(f"\nFigure: {png}")
    print(f"CSV:    {csv_out}")


if __name__ == "__main__":
    main()
