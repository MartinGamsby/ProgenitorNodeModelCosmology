"""Best-vs-best comparison suite (PF25): best centerM=1 cell vs the paper-cloud candidate.

  A = cmxconv_s5_cm1_16k    (centerM=1,    seed 5, S=95 Gpc, 16000 particles, no outer shell)
  B = cmxconv_s0_cm3000_16k (centerM=3000, seed 0, S=89 Gpc, 16000 inner + 32000 outer capped)

Self-contained (runs at the end of the detached driver, no AI needed). All distances in
ABSOLUTE Gpc. Outputs to results/figures/centerm/best_compare/:
  expansion_compare.png, cloud_evolution_A.png, cloud_evolution_B.png,
  final_clouds_compare.png, nodes_compare.png, knot_trace_compare.png, verdict.txt
"""
import os, sys
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import argparse
_p = argparse.ArgumentParser(description="best-vs-best comparison suite (defaults = the original pair)")
for side, d in (("a", dict(tag="cmxconv_s5_cm1_16k", n=16000, cm=1, seed=5, S=95.0,
                           label="best centerM=1 (seed 5, S=95 Gpc)")),
                ("b", dict(tag="cmxconv_s0_cm3000_16k", n=16000, cm=3000, seed=0, S=89.0,
                           label="candidate centerM=3000 (seed 0, S=89 Gpc)"))):
    _p.add_argument(f"--{side}-tag", default=d["tag"]);   _p.add_argument(f"--{side}-n", type=int, default=d["n"])
    _p.add_argument(f"--{side}-cm", type=int, default=d["cm"]); _p.add_argument(f"--{side}-seed", type=int, default=d["seed"])
    _p.add_argument(f"--{side}-S", type=float, default=d["S"]); _p.add_argument(f"--{side}-label", default=d["label"])
_p.add_argument("--out", default="best_compare")
_a = _p.parse_args()

OUT = f"results/figures/centerm/{_a.out}"
os.makedirs(OUT, exist_ok=True)

CELLS = {
    "A": dict(tag=_a.a_tag, label=_a.a_label, n_inner=_a.a_n, centerM=_a.a_cm, seed=_a.a_seed, S=_a.a_S),
    "B": dict(tag=_a.b_tag, label=_a.b_label, n_inner=_a.b_n, centerM=_a.b_cm, seed=_a.b_seed, S=_a.b_S),
}

def load_cell(c):
    d = np.load(f"results/hero/{c['tag']}.npz")
    n_inner = c["n_inner"]
    n_total = n_inner + int(min((c["centerM"] - 1) * n_inner, 2 * n_inner))
    rng = np.random.default_rng(0)
    keep = rng.choice(n_total, size=min(6000, n_total), replace=False)
    inner_sel = keep < n_inner
    P_all = d["pos_centred"]
    P = P_all[:, inner_sel, :]                     # observable cloud only
    rf = np.linalg.norm(P[-1], axis=1); r90 = np.percentile(rf, 90)
    coreset = rf < 0.25 * r90
    med = np.array([np.median(np.linalg.norm(P[s][coreset], axis=1)) for s in range(len(P))])
    return dict(d=d, P=P, P_all=P_all, t=d["times_rel"], rf=rf, r90=r90,
                core=float(np.mean(coreset)), knot=float(med[-1] / med[0]), med=med,
                bo=float(d["best_observer_chi2"]), growth=float(d["growth_factor"]),
                a=np.asarray(d["a_curve"]), tG=np.asarray(d["t_Gyr"]),
                a_best=np.asarray(d["a_best"]), obs_t=np.asarray(d["obs_t"]))

cells = {k: load_cell(c) for k, c in CELLS.items()}

# 1. Expansion a(t): centre + best-observer + EdS analytic (a ~ t^{2/3}).
fig, ax = plt.subplots(figsize=(8.5, 5.6))
for k, sty in (("A", "-"), ("B", "--")):
    x = cells[k]
    a0 = x["a"][0]
    ax.plot(x["tG"], x["a"] / a0, "C0" + sty, label=f"{CELLS[k]['label']}: centre a(t), growth {x['growth']:.2f}")
    if x["a_best"].size:
        ax.plot(x["obs_t"], x["a_best"] / x["a_best"][0], "C1" + sty, alpha=0.8,
                label=f"   best-observer a(t) (bo chi2/dof {x['bo']:.4f})")
tG = cells["A"]["tG"]
ax.plot(tG, (tG / tG[0]) ** (2.0 / 3.0), "k:", label="Einstein-de Sitter a ~ t^(2/3) (no dark energy)")
ax.set_xlabel("t [Gyr]"); ax.set_ylabel("a(t) / a(t_start)")
ax.set_title("Expansion: best centerM=1 vs candidate centerM=3000 (both LCDM-quality vs Pantheon+)")
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(f"{OUT}/expansion_compare.png", dpi=115); plt.close(fig)

# 2. Cloud evolution (observable cloud, 5 frames) per cell.
for k in ("A", "B"):
    x = cells[k]; snaps = np.linspace(0, len(x["P"]) - 1, 5).astype(int)
    fig, axes = plt.subplots(1, 5, figsize=(19, 4.2))
    lim = np.percentile(np.linalg.norm(x["P"][-1], axis=1), 99) * 1.1
    for ax, s in zip(axes, snaps):
        Ps = x["P"][s]; r = np.linalg.norm(Ps, axis=1)
        ax.scatter(Ps[:, 0], Ps[:, 1], s=1.5, c=r, cmap="viridis", alpha=0.55, linewidths=0)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.set_title(f"t = {x['t'][s]:.1f} Gyr", fontsize=10)
        ax.set_xlabel("Gpc")
    axes[0].set_ylabel("Gpc")
    fig.suptitle(f"Observable-cloud evolution — {CELLS[k]['label']} (16000 particles)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(f"{OUT}/cloud_evolution_{k}.png", dpi=110); plt.close(fig)

# 3. Final clouds side by side (cloud-only + B's all-particle view with outer shell).
fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))
for ax, k in zip(axes[:2], ("A", "B")):
    x = cells[k]; Ps = x["P"][-1]; r = np.linalg.norm(Ps, axis=1)
    ax.scatter(Ps[:, 0], Ps[:, 1], s=1.5, c=r, cmap="viridis", alpha=0.55, linewidths=0)
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(0.25 * x["r90"] * np.cos(th), 0.25 * x["r90"] * np.sin(th), "r-", lw=1.1)
    lim = np.percentile(r, 99) * 1.1
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.set_xlabel("Gpc")
    ax.set_title(f"{CELLS[k]['label']}\ncore={x['core']:.3f}  knot={x['knot']:.2f}  "
                 f"bo={x['bo']:.4f}  growth={x['growth']:.2f}", fontsize=9.5)
xB = cells["B"]; PB = xB["P_all"][-1]; rB = np.linalg.norm(PB, axis=1)
axes[2].scatter(PB[:, 0], PB[:, 1], s=1.2, c=rB, cmap="viridis", alpha=0.5, linewidths=0)
limB = np.percentile(rB, 99.5) * 1.05
axes[2].set_xlim(-limB, limB); axes[2].set_ylim(-limB, limB); axes[2].set_aspect("equal")
axes[2].set_xlabel("Gpc")
axes[2].set_title("candidate, ALL particles:\nobservable cloud + outer progenitor mass shell", fontsize=9.5)
fig.suptitle("Final clouds (t = 13.8 Gyr): the observable universe in both models", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(f"{OUT}/final_clouds_compare.png", dpi=115); plt.close(fig)

# 4. External nodes: near neighbourhood per cell (absolute Gpc).
from cosmo.node_geometry import build_virialized_grid
from cosmo.constants import CosmologicalConstants
GPC = CosmologicalConstants.Gpc_to_m
fig = plt.figure(figsize=(13.5, 6.2))
for i, k in enumerate(("A", "B"), 1):
    c = CELLS[k]
    pos, mass = build_virialized_grid(
        c["S"] * GPC, n_nodes=300, M_ext_kg=1.0, vir_mass_spread=0.5, vir_s_metric="median",
        vir_relax_mode="medium", center_mass_frac=c["centerM"] / 3000.0, seed=c["seed"])
    p = pos / GPC; r = np.linalg.norm(p, axis=1); mm = mass / mass.mean()
    ax = fig.add_subplot(1, 2, i, projection="3d")
    view = 2.5 * c["S"]
    shown = r <= view
    ax.scatter(p[shown, 0], p[shown, 1], p[shown, 2], c=mm[shown],
               s=25 + 140 * (mm[shown] / mm.max()), cmap="plasma",
               edgecolors="k", linewidths=0.3, alpha=0.9)
    u, v = np.mgrid[0:2 * np.pi:24j, 0:np.pi:14j]; R0 = 14.26
    ax.plot_surface(R0 * np.cos(u) * np.sin(v), R0 * np.sin(u) * np.sin(v), R0 * np.cos(v),
                    alpha=0.2, color="#1f77b4", linewidth=0)
    ax.set_xlim(-view, view); ax.set_ylim(-view, view); ax.set_zlim(-view, view)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis): axis.set_tick_params(labelsize=6)
    ax.set_xlabel("Gpc", fontsize=8)
    ax.set_title(f"{c['label']}\nnearest HMEA {r.min():.0f} Gpc | {int(shown.sum())} nodes shown "
                 f"(view {view:.0f} Gpc)\nblue sphere = observable universe (14.26 Gpc)", fontsize=9)
fig.suptitle("The external HMEA nodes around us (medium structure, nodes sized/coloured by mass)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(f"{OUT}/nodes_compare.png", dpi=115); plt.close(fig)

# 5. Knot trace.
fig, ax = plt.subplots(figsize=(7.5, 5))
for k, sty in (("A", "-"), ("B", "--")):
    x = cells[k]
    ax.plot(x["t"], x["med"] / x["med"][0], sty,
            label=f"{CELLS[k]['label']}: knot {x['knot']:.2f} "
                  f"({'COLLAPSING' if x['knot'] < 0.9 else '~static' if x['knot'] <= 1.2 else 'expanding'})")
ax.axhline(1.0, color="gray", lw=0.8); ax.axhline(0.9, color="r", ls="--", lw=0.8)
ax.set_xlabel("t [Gyr]"); ax.set_ylabel("central-knot median radius / initial")
ax.set_title("Central-knot Lagrangian trace (final-core particles)")
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(f"{OUT}/knot_trace_compare.png", dpi=115); plt.close(fig)

lines = ["=== best-vs-best (self-measured) ==="]
for k in ("A", "B"):
    x = cells[k]
    lines.append(f"{CELLS[k]['label']}: bo={x['bo']:.4f} growth={x['growth']:.3f} "
                 f"core={x['core']:.3f} knot={x['knot']:.2f}")
open(f"{OUT}/verdict.txt", "w").write("\n".join(lines))
print("\n".join(lines)); print("figures ->", OUT)
