"""Generate the SEED-HUNT (idea A, more seeds): does a DIFFERENT node_mass_seed give a
rounder good-fit cloud than seed 42?

Method (respecting the co-fit correction): sweep MANY seeds x a small rounder-band grid
(M200 x S{22,25} x sigma{5,6}) so each seed is judged on ITS OWN best anchor-ok cell, not
another seed's optimum. 2000p/1092 (growth + best_obs are N-stable for ranking), observer_k=-1.
NO snapshots (a multi-cell config would clobber the single tag-based npz) -> Stage 1 RANKS by
GROWTH, the validated cross-seed proxy for core-fraction (pooled probe cells are perfectly
monotonic: lower growth -> rounder). Stage 2 snapshots only the roundest winners at 4000p.

Split into 4 arms (4 seeds each) for parallelism. One CSV per arm: results/ws1_sweep_seedhunt_*.csv.
Run:  for a in a b c d: python sweep.py --config sweeps/seedhunt/<arm>.json  (HMEA_CACHE_CONCURRENT=1 if parallel)
"""
import json, os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seedhunt")
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
    "M_values": [200], "S_values": [22, 25], "vir_mass_spreads": [5.0, 6.0],
}

# 16 seeds (42/7/123 are the known ones; the rest are new). Split 4 per arm.
SEED_ARMS = {
    "a": [1, 2, 3, 5],
    "b": [7, 11, 17, 23],
    "c": [42, 55, 77, 99],
    "d": [123, 314, 500, 777],
}

arms = []
for name, seeds in SEED_ARMS.items():
    cfg = dict(COMMON)
    cfg["node_mass_seeds"] = seeds
    cfg["tag"] = f"seedhunt_{name}"
    cfg["figures_dir"] = f"results/figures/seedhunt_{name}"
    cfg["_comment"] = (f"seed-hunt arm {name}: seeds {seeds} x M200 x S{{22,25}} x sigma{{5,6}} "
                       f"(4 seeds x 4 cells = 16 cells), 2000p/1092, observer_k=-1, NO snapshots. "
                       f"Rank anchor-ok best_obs<=0.46 cells by GROWTH (proxy for core-fraction) "
                       f"to find a seed rounder than 42. Stage 2 snapshots the winners at 4000p.")
    fn = f"{name}.json"
    json.dump(cfg, open(os.path.join(OUT, fn), "w"), indent=2)
    arms.append(f"sweeps/seedhunt/{fn}")
    print(f"wrote {fn}: seeds {seeds}")

json.dump({"family": "seedhunt",
           "purpose": "Idea A (more seeds): find whether a different node_mass_seed yields a "
                      "rounder (lower-growth) anchor-ok Pantheon-matching cloud than seed 42. "
                      "Stage 1 ranks 16 seeds by growth over a rounder-band co-fit grid; Stage 2 "
                      "snapshots the roundest winners at 4000p to measure core-fraction.",
           "seeds": sum(SEED_ARMS.values(), []), "arms": arms},
          open(os.path.join(OUT, "_manifest.json"), "w"), indent=2)
print("arms:", *arms, sep="\n  ")
