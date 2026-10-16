"""Generate localized_v8 — 4000-particle MULTIPLICITY confirmation at converged N.

The v7 4000p N-check (M=300/S=20/σ6) showed the headline cell converges to centre ~0.44 /
best ~0.435 (= MATCHES LCDM 0.436), with frac_below_lcdm shrinking 7%->0.2% at higher N (the
2000p fraction was partly a sampling tail). v8 confirms the MULTIPLICITY at this converged
resolution: several best-CENTRE configs (centre ~0.43-0.45 at 2000p) re-run at 4000 particles,
observer_k=-1 (well-sampled), observer_sample=4000. Goal: show that MULTIPLE distinct configs
match LCDM-quality at converged N (not just the one N-check cell) — the paper's claim, honestly.

Split into 3 arms by M for parallelism. virialized lattice 300 nodes, GRF sphere, Plummer 1 Gpc,
1092 steps (converged), seed 42.
Run:  launch_sweep_detached.ps1 -ArmSet localized8 -Parallel 7
"""
import json, os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "localized_v8")
os.makedirs(OUT, exist_ok=True)

COMMON = {
    "node_s_amplitudes": [0.0], "node_mass_amplitudes": [0.0], "node_mass_seeds": [42],
    "particle_count": 4000, "n_steps": 1092, "t_start_Gyr": 2.9, "centerM": 1,
    "objective": "pantheon", "results_dir": "results",
    "init_distributions": ["grf"], "grf_support": "sphere",
    "node_geometries": ["virialized"], "vir_n_nodes": 300, "vir_mass_rule": "massfunc",
    "vir_segregation": 1.0, "vir_s_metric": "median", "vir_relax_steps": 1,
    "vir_relax_mode": "lattice", "node_softening_gpc": 1.0, "node_force_law": "plummer",
    "score_observers": True, "observer_definition": "local_rms",
    "observer_sample": 4000, "observer_k": -1,
    "S_values": [20, 22], "vir_mass_spreads": [5.0, 6.0, 7.0],
}

arms = []
for i, M in enumerate([200, 300, 400], start=1):
    cfg = dict(COMMON)
    cfg["M_values"] = [M]
    cfg["tag"] = f"loc8_M{M}_N4000"
    cfg["figures_dir"] = f"results/figures/{cfg['tag']}"
    cfg["_comment"] = (f"localized_v8 M={M} at 4000 particles, observer_k=-1, observer_sample=4000: "
                       f"converged-N multiplicity confirmation. S{cfg['S_values']} x sigma"
                       f"{cfg['vir_mass_spreads']}. Do multiple configs match LCDM (~0.44) at 4000p?")
    fn = f"{i:02d}_M{M}_N4000.json"
    json.dump(cfg, open(os.path.join(OUT, fn), "w"), indent=2)
    arms.append(f"sweeps/localized_v8/{fn}")
    print(f"wrote {fn}")

json.dump({"family": "localized_v8",
           "purpose": "4000-particle converged-N multiplicity confirmation: several best-centre "
                      "configs match LCDM-quality at observer_k=-1 / 4000p (the paper's multiplicity "
                      "claim at proper resolution).",
           "arms": arms}, open(os.path.join(OUT, "_manifest.json"), "w"), indent=2)
print("arms:", *arms, sep="\n  ")
