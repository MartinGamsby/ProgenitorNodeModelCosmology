"""Generate localized_v6 — the CORRECTED observer_k experiment.

v5's observer_k arms were INVALID: build_cache_name did not encode observer params, so all
four observer_k arms read v4's cached (observer_k=-1) best_observer_chi2 — observer_k had no
effect (identical results). FIXED in cosmo/parameter_sweep.py (observer params now slug the
metrics cache key when score_observers is on; regression test
TestObserverInSweep::test_observer_params_are_keyed_equals_run_in_cache).

v6 re-runs the observer_k sweep with the fix -> each observer_k now recomputes REAL observer
scoring (distinct cache keys ...64obsk / ...allobsk / ...). Same band as v5's obsk arms so it
is the clean corrected version. observer_sample=2000 (ALL particles -> true best vantage +
exact frac_below_lcdm). Question: does a genuinely LOCAL observer (finite k) find a vantage
that beats LCDM by MORE / at a larger fraction than whole-cloud (k=-1)?

2000p / 1092 steps, virialized lattice 300 nodes, Plummer 1 Gpc, GRF sphere (matched to v4/v5).
Run:  launch_sweep_detached.ps1 -ArmSet localized6 -Parallel 7
"""
import json, os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "localized_v6")
os.makedirs(OUT, exist_ok=True)

COMMON = {
    "node_mass_amplitudes": [0.0], "node_s_amplitudes": [0.0], "node_mass_seeds": [42],
    "particle_count": 2000, "n_steps": 1092, "t_start_Gyr": 2.9, "centerM": 1,
    "objective": "pantheon", "results_dir": "results",
    "init_distributions": ["grf"], "grf_support": "sphere",
    "node_geometries": ["virialized"], "vir_n_nodes": 300, "vir_mass_rule": "massfunc",
    "vir_segregation": 1.0, "vir_s_metric": "median", "vir_relax_steps": 1,
    "vir_relax_mode": "lattice", "node_softening_gpc": 1.0, "node_force_law": "plummer",
    "score_observers": True, "observer_definition": "local_rms", "observer_sample": 2000,
    "M_values": [100, 300, 500], "S_values": [20, 25, 30], "vir_mass_spreads": [4.0, 6.0, 8.0],
}

arms = []
for i, k in enumerate([64, 128, 256, -1], start=1):
    ktag = "all" if k == -1 else str(k)
    cfg = dict(COMMON)
    cfg["observer_k"] = k
    cfg["tag"] = f"loc6_obsk{ktag}"
    cfg["figures_dir"] = f"results/figures/{cfg['tag']}"
    cfg["_comment"] = (f"localized_v6 CORRECTED observer_k={k} "
                       f"({'whole cloud' if k==-1 else 'LOCAL k-NN'}) with the cache-key fix: "
                       f"observer scoring now REALLY recomputes per k. M{cfg['M_values']} x "
                       f"S{cfg['S_values']} x sigma{cfg['vir_mass_spreads']}, observer_sample=2000.")
    fn = f"{i:02d}_obsk{ktag}.json"
    json.dump(cfg, open(os.path.join(OUT, fn), "w"), indent=2)
    arms.append(f"sweeps/localized_v6/{fn}")
    print(f"wrote {fn} (observer_k={k})")

json.dump({"family": "localized_v6",
           "purpose": "CORRECTED observer_k sweep (post cache-key fix): does a local finite-k "
                      "observer beat LCDM by more / at a larger fraction than whole-cloud k=-1?",
           "arms": arms}, open(os.path.join(OUT, "_manifest.json"), "w"), indent=2)
print("arms:", *arms, sep="\n  ")
