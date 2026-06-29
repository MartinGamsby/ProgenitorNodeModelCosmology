"""Hero figures from the saved 100k-particle runs (results/hero/*.npz). NO sims — pure
plotting, so it's fast and re-runnable. Produces, in results/figures/hero/:
  hero_particles_<tag>.png  — the 100k-particle cloud over time (median-centred, ONE fixed
                              scale across panels, runaway tail dropped) for each config.
  hero_hubble.png           — best-observer mu(z) of every hero config overlaid on Pantheon+
                              with LCDM + EdS references (chi2 from the result JSONs).
  hero_chi2_summary.png     — best-observer chi2/dof per config vs LCDM/EdS (multiplicity at 100k).

Run:  python _generate_hero_figs.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import glob, json, os
from cosmo.encoding import configure_utf8_stdout
configure_utf8_stdout()
from cosmo.plots import figure_path
from cosmo.visualization import setup_3d_axes

OUT = "hero"
LCDM, EDS = 0.4360, 0.8430
EXT_C, LCDM_C, EDS_C = "#1f77b4", "#d62728", "#2ca02c"


def load_all():
    """Load each results/hero/*.npz. The chi2 numbers are embedded in the npz by sweep.py
    (save_snapshots), so no separate result JSON is needed."""
    runs = []
    for npz in sorted(glob.glob("results/hero/*.npz")):
        tag = os.path.splitext(os.path.basename(npz))[0]
        d = np.load(npz)
        def g(k, default=float("nan")):
            return float(d[k]) if k in d.files else default
        res = {"best_observer_chi2": g("best_observer_chi2"),
               "center_chi2_dof": g("center_chi2_dof"),
               "frac_below_lcdm": g("frac_below_lcdm"),
               "frac_below_eds": g("frac_below_eds"),
               "growth_factor": g("growth_factor")}
        runs.append((tag, d, res))
    return runs


def particles_fig(tag, d, res):
    pos = d["pos_centred"]            # (ns, np, 3) Gpc, median-centred
    times = d["times_rel"]; t0 = float(d["t_start"])
    ns = pos.shape[0]
    sel = np.unique(np.linspace(0, ns - 1, 5).astype(int))
    # ONE fixed scale across panels = 97th pct bulk radius at the largest time (drop the tail).
    def bulkR(s):
        return float(np.percentile(np.linalg.norm(pos[s], axis=1), 97))
    lim = 1.12 * max(bulkR(s) for s in sel)
    fig = plt.figure(figsize=(20, 4.4))
    for j, s in enumerate(sel):
        ax = fig.add_subplot(1, 5, j + 1, projection="3d")
        P = pos[s]; P = P[np.linalg.norm(P, axis=1) <= lim]
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=1.2, color=EXT_C, alpha=0.4, edgecolors="none")
        setup_3d_axes(ax, lim, f"t = {t0 + float(times[s]):.1f} Gyr")
    fig.suptitle(f"External-Node cloud over time — {int(d['n_particles'])} particles "
                 f"(M={int(d['M'])}, S={int(d['S'])}, $\\sigma$={float(d['sigma']):g}; "
                 f"best-obs $\\chi^2/dof$={res['best_observer_chi2']:.3f})", y=1.03)
    p = figure_path(OUT, f"hero_particles_{tag}"); fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("  ->", p)


def hubble_fig(runs):
    try:
        from cosmo import pantheon as pl
        from cosmo.sim_distance import sim_to_distance_modulus
        from cosmo.distances import model_distance_modulus
        pan = pl.load_pantheon(); z = np.asarray(pan["z"]); mu = np.asarray(pan["mu"]); sg = np.asarray(pan["sigma"])
        fig, ax = plt.subplots(figsize=(9, 6.5))
        order = np.argsort(z)
        ax.plot(z[order], mu[order], ".", ms=2, color="0.7", alpha=0.35, label="Pantheon+ SNe", zorder=1)
        zd = np.linspace(max(1e-3, z.min()), z.max(), 200)
        ax.plot(zd, model_distance_modulus(zd, "lcdm"), color=LCDM_C, lw=2.2, ls="--",
                label=f"$\\Lambda$CDM ({LCDM:.3f})", zorder=4)
        ax.plot(zd, model_distance_modulus(zd, "einstein_de_sitter"), color=EDS_C, lw=2.0, ls="-.",
                label=f"EdS null ({EDS:.3f})", zorder=4)
        cmap = plt.cm.viridis(np.linspace(0, 0.85, len(runs)))
        for (tag, d, res), c in zip(runs, cmap):
            ab = d["a_best"]; ot = d["obs_t"]
            if ab.size < 2:
                continue
            sd = sim_to_distance_modulus(z_target=z, a=ab, t_Gyr=ot, t_start_Gyr=float(d["t_start"]))
            inr = sd["in_range"]; o = np.argsort(z[inr])
            ax.plot(z[inr][o], sd["mu"][o], color=c, lw=1.4, alpha=0.9,
                    label=f"M{int(d['M'])}/S{int(d['S'])}/$\\sigma${float(d['sigma']):g} "
                          f"({res['best_observer_chi2']:.3f})")
        ax.set_xlabel("Redshift z"); ax.set_ylabel("Distance modulus $\\mu$")
        ax.set_title("Best-observer $\\mu(z)$ of every 100k hero config vs Pantheon+")
        ax.legend(fontsize=7, loc="lower right", ncol=2)
        p = figure_path(OUT, "hero_hubble"); fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
        print("  ->", p)
    except Exception as e:
        print("  [warn] hero hubble fig skipped:", repr(e))


def chi2_summary(runs):
    labels = [f"M{int(d['M'])}\nS{int(d['S'])}\n$\\sigma${float(d['sigma']):g}" for _, d, _ in runs]
    best = [res["best_observer_chi2"] for _, _, res in runs]
    cen = [res["center_chi2_dof"] for _, _, res in runs]
    x = np.arange(len(runs))
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(runs)), 5))
    ax.scatter(x, best, s=70, color="#9467bd", zorder=3, label="best observer")
    ax.scatter(x, cen, s=50, color=EXT_C, marker="s", zorder=3, label="centre")
    ax.axhline(LCDM, color=LCDM_C, ls="--", label=f"$\\Lambda$CDM ({LCDM})")
    ax.axhline(EDS, color=EDS_C, ls="-.", label=f"EdS null ({EDS})")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("$\\chi^2/dof$ vs Pantheon+"); ax.set_ylim(0.40, max(0.9, max(cen) + 0.05))
    ax.set_title("100k-particle hero runs: best-observer $\\chi^2$ vs $\\Lambda$CDM / EdS")
    ax.legend(fontsize=9)
    p = figure_path(OUT, "hero_chi2_summary"); fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("  ->", p)


def main():
    runs = load_all()
    if not runs:
        print("no hero npz found in results/hero/"); return
    print(f"[hero-figs] {len(runs)} runs")
    for tag, d, res in runs:
        particles_fig(tag, d, res)
    hubble_fig(runs)
    chi2_summary(runs)
    print("[hero-figs] done.")


if __name__ == "__main__":
    main()
