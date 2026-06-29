"""Generate the 7 hero configs: known-good cells (v8-confirmed best-observer < LCDM at
4000p) at HIGH resolution (100k particles / 3000 steps). One config each -> run in
parallel via launch_hero_detached.ps1. Steps stay at the CONVERGED ~3000 (best-observer
plateaus by ~2730; 100k steps would be days for zero physics gain)."""
import json, os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hero")
os.makedirs(OUT, exist_ok=True)

# (M, S, sigma) — v8-confirmed below-LCDM, spread across the grid for variety.
CELLS = [
    (300, 20, 6.0),   # documented headline
    (400, 22, 5.0),   # v8 best (0.4336)
    (200, 20, 7.0),   # (0.4337)
    (300, 20, 5.0),   # (0.4341)
    (400, 20, 5.0),   # (0.4349)
    (300, 22, 7.0),   # (0.4348)
    (200, 22, 6.0),   # (0.4338)
]

COMMON = {
    "node_mass_amplitudes": [0.0], "node_s_amplitudes": [0.0], "node_mass_seeds": [42],
    "particle_count": 100000, "n_steps": 3000, "t_start_Gyr": 2.9, "centerM": 1,
    "objective": "pantheon", "results_dir": "results",
    "init_distributions": ["grf"], "grf_support": "sphere",
    "node_geometries": ["virialized"], "vir_n_nodes": 300, "vir_mass_rule": "massfunc",
    "vir_segregation": 1.0, "vir_s_metric": "median", "vir_relax_steps": 1,
    "vir_relax_mode": "lattice", "node_softening_gpc": 1.0, "node_force_law": "plummer",
    "score_observers": True, "observer_definition": "local_rms", "observer_sample": 500,
    "observer_k": -1,
}

names = []
for i, (M, Sv, sg) in enumerate(CELLS, start=1):
    sgt = str(sg).replace(".0", "").replace(".", "p")
    cfg = dict(COMMON)
    cfg["M_values"] = [M]; cfg["S_values"] = [Sv]; cfg["vir_mass_spreads"] = [sg]
    cfg["tag"] = f"hero_M{M}_S{Sv}_sig{sgt}"
    cfg["_comment"] = (f"HERO run: M={M} S={Sv} sigma={sg}, 100k particles / 3000 steps "
                       f"(converged), virialized lattice 300 nodes, GRF sphere, Plummer 1 Gpc. "
                       f"v8-confirmed best-observer < LCDM. For great images + converged chi2.")
    fn = f"{i:02d}_M{M}_S{Sv}_sig{sgt}.json"
    json.dump(cfg, open(os.path.join(OUT, fn), "w"), indent=2)
    names.append(f"sweeps/hero/{fn}")
    print("wrote", fn)

print("\nconfigs:", *names, sep="\n  ")
