"""Campaign-level paper figures from the sweep CSVs (NO sims — fast, CSV-only).

These show the MULTIPLICITY + robustness story (the paper's core claim):
  1. multiplicity_map.png    — best-observer chi2/dof over (M,S), one panel per sigma, with the
                               LCDM line; shows MANY configs reaching ~LCDM (localized_v4).
  2. step_convergence.png    — best-obs + centre vs n_steps (M=300/S=20/sigma=6); the result is
                               step-converged, not an artifact (localized_v4/v5 conv arms).
  3. seed_robustness.png     — best-obs across node-realization seeds {42,7,123} per cell; tight
                               spread => the ~LCDM match is generic, not seed-tuned (localized_v7).
  4. chi2_ladder.png         — External-Node (best) vs LCDM vs EdS null chi2/dof bar chart.

Run:  python _generate_campaign_figs.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import csv, glob, math, os
from collections import defaultdict
from cosmo.encoding import configure_utf8_stdout
configure_utf8_stdout()
from cosmo.plots import figure_path

OUT = "paper"
LCDM, EDS = 0.4360, 0.8430


def fn(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load(pattern):
    rows = []
    for f in glob.glob(os.path.join("results", pattern)):
        for r in csv.DictReader(open(f, newline="")):
            r["_f"] = os.path.basename(f)
            rows.append(r)
    return rows


def steps_of(tag):
    for k, v in {"loc4_sig": 1092, "loc4_conv_1638": 1638, "loc4_conv_2184": 2184,
                 "loc5_conv_2730": 2730, "loc5_conv_3276": 3276}.items():
        if k in tag:
            return v
    return None


def main():
    # ---- 1. multiplicity map (localized_v4) ----
    v4 = [r for r in load("ws1_sweep_loc4_sig*.csv")
          if r.get("anchor_ok") == "True" and math.isfinite(fn(r.get("best_observer_chi2")))]
    sigmas = sorted({fn(r["vir_mass_spread"]) for r in v4})
    fig, axes = plt.subplots(1, len(sigmas), figsize=(4 * len(sigmas), 4.2), squeeze=False)
    vmin, vmax = 0.43, 0.55
    for ax, sg in zip(axes[0], sigmas):
        cells = [r for r in v4 if fn(r["vir_mass_spread"]) == sg]
        M = np.array([fn(r["M_factor"]) for r in cells])
        Sg = np.array([fn(r["S_gpc"]) for r in cells])
        C = np.array([fn(r["best_observer_chi2"]) for r in cells])
        sc = ax.scatter(M, Sg, c=C, s=260, cmap="viridis_r", vmin=vmin, vmax=vmax,
                        edgecolors="k", linewidths=0.4)
        for m, s, c in zip(M, Sg, C):
            ax.annotate(f"{c:.3f}", (m, s), fontsize=6, ha="center", va="center",
                        color="white" if c < 0.49 else "black")
        ax.set_xscale("log"); ax.set_title(f"$\\sigma$ = {sg:g}")
        ax.set_xlabel("M"); ax.set_ylabel("S (Gpc)")
    fig.colorbar(sc, ax=axes[0].tolist(), label="best-observer $\\chi^2/dof$ (LCDM=0.436)")
    n_le = sum(1 for r in v4 if fn(r["best_observer_chi2"]) <= 0.45)
    fig.suptitle(f"Multiplicity: best-observer fit over (M, S, $\\sigma$) — "
                 f"{n_le} configs $\\leq$ 0.45 ($\\approx$LCDM)", y=1.04)
    p = figure_path(OUT, "multiplicity_map"); fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("  ->", p, f"({len(v4)} cells)")

    # ---- 2. step convergence (M=300/S=20/sigma=6) ----
    conv = [r for r in load("ws1_sweep_loc4_sig6.csv") + load("ws1_sweep_loc4_conv_*.csv")
            + load("ws1_sweep_loc5_conv_*.csv")
            if int(fn(r["M_factor"])) == 300 and int(fn(r["S_gpc"])) == 20
            and fn(r.get("vir_mass_spread")) == 6.0]
    pts = {}
    for r in conv:
        ns = steps_of(r["_f"])
        if ns:
            pts[ns] = (fn(r["best_observer_chi2"]), fn(r["center_chi2_dof"]), fn(r.get("frac_below_lcdm")))
    if pts:
        ns = sorted(pts)
        fig, ax = plt.subplots(figsize=(7.5, 5))
        ax.plot(ns, [pts[n][0] for n in ns], "o-", color="#1f77b4", label="best observer")
        ax.plot(ns, [pts[n][1] for n in ns], "s--", color="k", label="centre observer")
        ax.axhline(LCDM, color="#d62728", ls=":", label="$\\Lambda$CDM (0.436)")
        ax.set_xlabel("n_steps (time resolution)"); ax.set_ylabel("$\\chi^2/dof$ vs Pantheon+")
        ax.set_title("Step-convergence (M=300, S=20, $\\sigma$=6): converged, not an artifact")
        ax.legend()
        p = figure_path(OUT, "step_convergence"); fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
        print("  ->", p, f"({len(ns)} step counts)")

    # ---- 3. seed robustness (localized_v7) ----
    v7 = [r for r in load("ws1_sweep_loc7_seed*.csv")
          if r.get("anchor_ok") == "True" and math.isfinite(fn(r.get("best_observer_chi2")))]
    bycell = defaultdict(dict)
    for r in v7:
        bycell[(int(fn(r["M_factor"])), int(fn(r["S_gpc"])), fn(r["vir_mass_spread"]))][r["node_mass_seed"]] = fn(r["best_observer_chi2"])
    multi = {k: v for k, v in bycell.items() if len(v) >= 2}
    if multi:
        labels = [f"M{k[0]}\nS{k[1]}\n$\\sigma${k[2]:g}" for k in sorted(multi)]
        fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(multi)), 5))
        seed_colors = {"42": "#1f77b4", "7": "#ff7f0e", "123": "#2ca02c"}
        for xi, k in enumerate(sorted(multi)):
            for sd, c in multi[k].items():
                ax.scatter(xi, c, color=seed_colors.get(sd, "0.5"), s=40,
                           label=f"seed {sd}" if xi == 0 else None)
        ax.axhline(LCDM, color="#d62728", ls=":", label="$\\Lambda$CDM")
        ax.set_xticks(range(len(multi))); ax.set_xticklabels(labels, fontsize=6)
        ax.set_ylabel("best-observer $\\chi^2/dof$")
        spreads = [max(v.values()) - min(v.values()) for v in multi.values()]
        ax.set_title(f"Seed-robustness: best-observer across node realizations "
                     f"(mean spread {np.mean(spreads):.4f})")
        ax.legend(fontsize=8)
        p = figure_path(OUT, "seed_robustness"); fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
        print("  ->", p, f"({len(multi)} cells x seeds)")

    # ---- 4. chi2 ladder ----
    best = min((fn(r["best_observer_chi2"]) for r in v4), default=float("nan"))
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ["External-Node\n(best config)", "$\\Lambda$CDM", "EdS null\n(no dark energy)"]
    vals = [best, LCDM, EDS]
    cols = ["#1f77b4", "#d62728", "#2ca02c"]
    ax.bar(bars, vals, color=cols, alpha=0.85, edgecolor="k")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=11)
    ax.set_ylabel("$\\chi^2/dof$ vs Pantheon+ (lower = better)")
    ax.set_title("The model matches $\\Lambda$CDM, rejects the no-dark-energy null")
    p = figure_path(OUT, "chi2_ladder"); fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("  ->", p)
    print("[campaign-figs] done.")


if __name__ == "__main__":
    main()
