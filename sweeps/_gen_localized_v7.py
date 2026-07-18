"""Generate localized_v7 — multiplicity + seed-robustness at the CLEAN observer_k=-1.

Findings so far: the ROBUST headline (PF16) is the well-sampled whole-cloud (observer_k=-1)
best-observer ~0.43 (= the model MATCHES LCDM-quality 0.436 from a generic vantage), converged
in n_steps, with ~7% of observers below LCDM at the best cell (M=300/S=20/sigma=6). Finite-k
LOCAL observers are noise artifacts at N=2000 (PF17) -> use k=-1.

v7 establishes the PAPER's core claim: MANY DISTINCT configs reach ~LCDM, ROBUST to the node
realization (seed). It refines M/S/sigma around the best AND sweeps the node_mass_seed (which,
for the virialized massfunc rule with spread>0, materially changes the grid, PF15) to show the
~0.43 match is generic, not seed-tuned. Plus one 4000-particle arm to confirm the headline cell
holds at higher N.

observer_k=-1, observer_sample=ALL, 2000p/1092 (convergence already confirmed). Plummer 1 Gpc,
virialized lattice 300 nodes, GRF sphere.
Run:  launch_sweep_detached.ps1 -ArmSet localized7 -Parallel 7
"""
import json, os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "localized_v7")
os.makedirs(OUT, exist_ok=True)

COMMON = {
    "node_s_amplitudes": [0.0], "node_mass_amplitudes": [0.0],
    "particle_count": 2000, "n_steps": 1092, "t_start_Gyr": 2.9, "centerM": 1,
    "objective": "pantheon", "results_dir": "results",
    "init_distributions": ["grf"], "grf_support": "sphere",
    "node_geometries": ["virialized"], "vir_n_nodes": 300, "vir_mass_rule": "massfunc",
    "vir_segregation": 1.0, "vir_s_metric": "median", "vir_relax_steps": 1,
    "vir_relax_mode": "lattice", "node_softening_gpc": 1.0, "node_force_law": "plummer",
    "score_observers": True, "observer_definition": "local_rms",
    "observer_sample": 2000, "observer_k": -1,
}

arms = []
# Seed-robustness arms: one node_mass_seed each, refined grid around the best.
for i, seed in enumerate([42, 7, 123], start=1):
    cfg = dict(COMMON)
    cfg["M_values"] = [200, 300, 400]
    cfg["S_values"] = [20, 22, 25]
    cfg["vir_mass_spreads"] = [5.0, 6.0, 7.0]
    cfg["node_mass_seeds"] = [seed]
    cfg["tag"] = f"loc7_seed{seed}"
    cfg["figures_dir"] = f"results/figures/{cfg['tag']}"
    cfg["_comment"] = (f"localized_v7 seed={seed}: multiplicity + seed-robustness at observer_k=-1 "
                       f"(clean, well-sampled). M{cfg['M_values']} x S{cfg['S_values']} x "
                       f"sigma{cfg['vir_mass_spreads']}. Does the ~0.43 LCDM-match hold across "
                       f"node realizations? Each (M,sigma,seed) is a distinct virialized grid (PF15).")
    fn = f"{i:02d}_seed{seed}.json"
    json.dump(cfg, open(os.path.join(OUT, fn), "w"), indent=2)
    arms.append(f"sweeps/localized_v7/{fn}")
    print(f"wrote {fn}")

# N-convergence arm: the headline cell at 4000 particles (observer_sample=4000 = all).
cfg = dict(COMMON)
cfg["particle_count"] = 4000
cfg["observer_sample"] = 4000
cfg["M_values"] = [300]
cfg["S_values"] = [20]
cfg["vir_mass_spreads"] = [6.0]
cfg["node_mass_seeds"] = [42]
cfg["tag"] = "loc7_N4000"
cfg["figures_dir"] = "results/figures/loc7_N4000"
cfg["_comment"] = ("localized_v7 N-convergence: the headline cell M=300/S=20/sigma=6 at 4000 "
                   "particles, observer_k=-1, observer_sample=4000. Confirms best-observer ~0.43 "
                   "holds at higher N (the k=-1 result is well-sampled, not N-limited like small-k).")
json.dump(cfg, open(os.path.join(OUT, "04_N4000.json"), "w"), indent=2)
arms.append("sweeps/localized_v7/04_N4000.json")
print("wrote 04_N4000.json")

json.dump({"family": "localized_v7",
           "purpose": "Multiplicity + seed-robustness at observer_k=-1 (the clean metric) + a "
                      "4000p N-convergence check of the headline cell. The paper's 'many configs "
                      "reach ~LCDM, robustly' claim.",
           "arms": arms}, open(os.path.join(OUT, "_manifest.json"), "w"), indent=2)
print("arms:", *arms, sep="\n  ")
