"""Generate the LOW-SIGMA landscape map (idea A, user: sigma<=2 is physically defensible; 5-6 is
not). Find (sigma<=2, M, S, node-count) that gives a good best-observer fit + anchor-ok.

Physics: the effective acceleration is the tidal ANISOTROPY. At high node count that anisotropy
must come from the mass-function WIDTH (high sigma). At LOW sigma the masses are near-uniform, so
the anisotropy must come from node PLACEMENT discreteness -> FEWER nodes. So we sweep node count
(separate config per vir_n_nodes) x sigma{1,1.5,2} x M x S and rank by best_observer_chi2.

Exploration resolution (1000p/273, matches explore_vir_spread), observer_k=-1, seed 42, GRF sphere,
Plummer 1 Gpc. Rank the anchor-ok best_observer cells; refine the winner at higher N + snapshot.
Run:  for each config: python sweep.py --config sweeps/lowsigma/<cfg>.json  (HMEA_CACHE_CONCURRENT=1 parallel)
"""
import json, os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lowsigma")
os.makedirs(OUT, exist_ok=True)

COMMON = {
    "node_mass_amplitudes": [0.0], "node_s_amplitudes": [0.0], "node_mass_seeds": [42],
    "particle_count": 1000, "n_steps": 273, "t_start_Gyr": 2.9, "centerM": 1,
    "objective": "pantheon", "results_dir": "results",
    "init_distributions": ["grf"], "grf_support": "sphere",
    "node_geometries": ["virialized"], "vir_mass_rule": "massfunc",
    "vir_segregation": 1.0, "vir_s_metric": "median", "vir_relax_steps": 1,
    "vir_relax_mode": "lattice", "node_softening_gpc": 1.0, "node_force_law": "plummer",
    "score_observers": True, "observer_definition": "local_rms",
    "observer_sample": 1000, "observer_k": -1,
    "M_values": [10, 50, 100, 300],
    "S_values": [10, 15, 22],
    "vir_mass_spreads": [1.0, 1.5, 2.0],
}

NODE_COUNTS = [26, 50, 150, 300, 600]

arms = []
for nn in NODE_COUNTS:
    cfg = dict(COMMON)
    cfg["vir_n_nodes"] = nn
    tag = f"lowsig_N{nn}"
    cfg["tag"] = tag
    cfg["figures_dir"] = f"results/figures/{tag}"
    cfg["_comment"] = (f"low-sigma map, vir_n_nodes={nn}: sigma{{1,1.5,2}} x M{{10,50,100,300}} x "
                       f"S{{10,15,22}} = 36 cells, 1000p/273, observer_k=-1, seed42. Does a physically "
                       f"defensible sigma<=2 reach a good best-observer fit at THIS node count? "
                       f"(low sigma -> anisotropy from placement -> fewer nodes may fit better.)")
    fn = f"N{nn}.json"
    json.dump(cfg, open(os.path.join(OUT, fn), "w"), indent=2)
    arms.append(f"sweeps/lowsigma/{fn}")
    print(f"wrote {fn}: vir_n_nodes={nn} (36 cells)")

json.dump({"family": "lowsigma",
           "purpose": "Find a physically defensible sigma<=2 config (M,S,node-count) with a good "
                      "best-observer fit. Sweep node count x sigma{1,1.5,2} x M x S; rank anchor-ok "
                      "by best_observer_chi2; refine + snapshot the winner.",
           "node_counts": NODE_COUNTS, "arms": arms},
          open(os.path.join(OUT, "_manifest.json"), "w"), indent=2)
print("arms:", *arms, sep="\n  ")
