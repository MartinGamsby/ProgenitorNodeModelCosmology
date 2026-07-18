"""Seed hunt at the DEFENSIBLE high-S/M ridge config (idea A): does a different node_mass_seed
give a rounder cloud at M1000 / sigma1.5 / N600 / S~41-48, while keeping best_obs ~0.44?

Earlier (M200/S25/sigma5) seed 1 broke the roundness<->fit tradeoff (growth 2.87, core 0.087 at
LCDM fit). This checks whether a seed similarly rounds the defensible sigma1.5 high-S/M config.
Stage 1: sweep 16 seeds x S{41,48} at fixed M1000/sigma1.5/N600, 2000p/546, observer_k=-1, NO
snapshots. Rank anchor-ok best_obs<=0.46 by GROWTH (proxy for core-fraction). Stage 2 snapshots
the roundest winners at 4000p. Split 4 arms (4 seeds each) for parallelism.
"""
import json, os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seedhunt2")
os.makedirs(OUT, exist_ok=True)

COMMON = {
    "node_mass_amplitudes": [0.0], "node_s_amplitudes": [0.0],
    "particle_count": 2000, "n_steps": 546, "t_start_Gyr": 2.9, "centerM": 1,
    "objective": "pantheon", "results_dir": "results",
    "init_distributions": ["grf"], "grf_support": "sphere",
    "node_geometries": ["virialized"], "vir_n_nodes": 600, "vir_mass_rule": "massfunc",
    "vir_segregation": 1.0, "vir_s_metric": "median", "vir_relax_steps": 1,
    "vir_relax_mode": "lattice", "node_softening_gpc": 1.0, "node_force_law": "plummer",
    "score_observers": True, "observer_definition": "local_rms",
    "observer_sample": 2000, "observer_k": -1,
    "M_values": [1000], "S_values": [41, 48], "vir_mass_spreads": [1.5],
}

SEED_ARMS = {"a": [1, 2, 3, 5], "b": [7, 11, 17, 23], "c": [42, 55, 77, 99], "d": [123, 314, 500, 777]}

arms = []
for name, seeds in SEED_ARMS.items():
    cfg = dict(COMMON)
    cfg["node_mass_seeds"] = seeds
    cfg["tag"] = f"seedhunt2_{name}"
    cfg["figures_dir"] = f"results/figures/seedhunt2_{name}"
    cfg["_comment"] = (f"seed-hunt2 arm {name}: seeds {seeds} at M1000/sigma1.5/N600 x S{{41,48}}, "
                       f"2000p/546, observer_k=-1, NO snapshots. Does a seed round the defensible "
                       f"high-S/M config while keeping best_obs~0.44? Rank by growth; snapshot winners.")
    fn = f"{name}.json"
    json.dump(cfg, open(os.path.join(OUT, fn), "w"), indent=2)
    arms.append(f"sweeps/seedhunt2/{fn}")
    print(f"wrote {fn}: seeds {seeds}")

json.dump({"family": "seedhunt2",
           "purpose": "Does a different seed round the defensible sigma1.5 high-S/M config "
                      "(M1000/S~45/N600) at fixed LCDM-quality fit? 16 seeds x S{41,48}.",
           "arms": arms}, open(os.path.join(OUT, "_manifest.json"), "w"), indent=2)
print("arms:", *arms, sep="\n  ")
