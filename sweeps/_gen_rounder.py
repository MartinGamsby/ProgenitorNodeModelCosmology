"""Generate the ROUNDER-cloud probe (idea A): per-seed co-fit -> rounder anchor-ok cloud.

Motivation (PF19 + the localized_v7 seed co-fit): the fig6 paper cloud (M200/S22/sig6, 75k)
is the roundest hero cloud (core-fraction 0.253), and core-fraction tracks GROWTH (weaker
traceless tidal compression -> rounder). All seeds match LCDM (~0.433) across the anchor-ok
band; the rounder route is a weaker field (lower M / higher S / lower sigma), NOT a special
seed. This probe MEASURES core-fraction at the rounder corner (which the flat best-observer
chi2 cannot discriminate) by saving snapshots.

Single-cell configs (one npz each via save_snapshots), 4000p / 1092 steps (core-fraction is
N-stable: 0.380@10k ~ 0.385@100k), virialized lattice 300 nodes, GRF sphere, Plummer 1 Gpc,
observer_k=-1. A control cell reproduces the fig6 config at the SAME probe N.

Run:  launch_sweep_detached.ps1 -ArmSet rounder -Parallel 6
Image/measure: read results/hero/rounder_*.npz (core = mean(r < 0.25*percentile(r,90))).
"""
import json, os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rounder")
os.makedirs(OUT, exist_ok=True)

COMMON = {
    "node_s_amplitudes": [0.0], "node_mass_amplitudes": [0.0],
    "t_start_Gyr": 2.9, "centerM": 1, "objective": "pantheon", "results_dir": "results",
    "init_distributions": ["grf"], "grf_support": "sphere",
    "node_geometries": ["virialized"], "vir_n_nodes": 300, "vir_mass_rule": "massfunc",
    "vir_segregation": 1.0, "vir_s_metric": "median", "vir_relax_steps": 1,
    "vir_relax_mode": "lattice", "node_softening_gpc": 1.0, "node_force_law": "plummer",
    "score_observers": True, "observer_definition": "local_rms",
    "observer_sample": 500, "observer_k": -1,
    "save_snapshots": True, "skip_figures": True, "skip_probe": True,
    "particle_count": 4000, "n_steps": 1092, "save_interval": 18,
}

# (M, S, sigma, seed, role)
CELLS = [
    (200, 22, 6.0, 42,  "ctrl",    "CONTROL == fig6 config at probe N (expect core ~0.25)"),
    (200, 25, 5.0, 42,  "primary", "roundest anchor-ok corner (growth ~2.93, best_obs ~0.438)"),
    (200, 25, 6.0, 42,  "sig",     "sigma lever: sigma5 vs sigma6 at S25"),
    (200, 22, 5.0, 42,  "S",       "S lever: S22 vs S25 at sigma5 (best-fit cell 0.4326)"),
    (200, 25, 5.0, 123, "seed",    "cross-seed roundness check at the rounder corner (seed 123)"),
    (200, 28, 5.0, 42,  "stretch", "STRETCH weaker field (S28, outside swept grid): maps roundness vs anchor boundary"),
]

arms = []
for i, (M, S, sig, seed, role, why) in enumerate(CELLS, start=1):
    cfg = dict(COMMON)
    cfg["M_values"] = [M]
    cfg["S_values"] = [S]
    cfg["vir_mass_spreads"] = [sig]
    cfg["node_mass_seeds"] = [seed]
    tag = f"rounder_{role}_M{M}_S{S}_sig{sig:.0f}_seed{seed}"
    cfg["tag"] = tag
    cfg["figures_dir"] = f"results/figures/{tag}"
    cfg["_comment"] = (f"rounder probe [{role}]: M={M} S={S} sigma={sig} seed={seed}. {why} "
                       f"4000p/1092 steps, save_snapshots -> results/hero/{tag}.npz. Measure core-"
                       f"fraction vs fig6 (0.253); pick the roundest that stays anchor-ok + best_obs ~0.44.")
    fn = f"{i:02d}_{role}.json"
    json.dump(cfg, open(os.path.join(OUT, fn), "w"), indent=2)
    arms.append(f"sweeps/rounder/{fn}")
    print(f"wrote {fn}: {tag}")

json.dump({"family": "rounder",
           "purpose": "Idea A: MEASURE core-fraction at the rounder anchor-ok corner (per-seed "
                      "co-fit already shows all seeds match LCDM ~0.433; roundness tracks growth, "
                      "not seed). Find a seed+config that is anchor-ok AND best_obs ~0.44 AND "
                      "rounder than fig6 (core 0.253), for a rounder paper cloud figure.",
           "arms": arms}, open(os.path.join(OUT, "_manifest.json"), "w"), indent=2)
print("arms:", *arms, sep="\n  ")
