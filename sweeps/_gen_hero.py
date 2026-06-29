"""Generate the 7 hero configs: a RESOLUTION LADDER scaling particles AND steps together
(steps matter — finer dt resolves the denser structure high N exposes). Run via the EXISTING
launcher: launch_sweep_detached.ps1 -ArmSet hero. Each is a single-config sweep with
save_snapshots=true (-> results/hero/<tag>.npz for images) + skip_figures=true (no figure
re-run) + save_interval scaled to ~60 snapshots. Imaged by _generate_hero_figs.py."""
import json, os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hero")
os.makedirs(OUT, exist_ok=True)

# (M, S, sigma, particles, steps) — known-good cells (v8-confirmed best-observer < LCDM),
# scaling particles 10k->100k with steps 5k->12k (the user's ladder: 10k/5k, 50k/10k, ...).
LADDER = [
    (300, 20, 6.0,  10000,  5000),
    (400, 22, 5.0,  20000,  6000),
    (200, 20, 7.0,  30000,  7000),
    (300, 20, 5.0,  50000,  8000),
    (300, 22, 7.0,  50000, 10000),
    (200, 22, 6.0,  75000, 11000),
    (300, 20, 6.0, 100000, 12000),   # the hero: headline config at full resolution
]

COMMON = {
    "node_mass_amplitudes": [0.0], "node_s_amplitudes": [0.0], "node_mass_seeds": [42],
    "t_start_Gyr": 2.9, "centerM": 1, "objective": "pantheon", "results_dir": "results",
    "init_distributions": ["grf"], "grf_support": "sphere",
    "node_geometries": ["virialized"], "vir_n_nodes": 300, "vir_mass_rule": "massfunc",
    "vir_segregation": 1.0, "vir_s_metric": "median", "vir_relax_steps": 1,
    "vir_relax_mode": "lattice", "node_softening_gpc": 1.0, "node_force_law": "plummer",
    "score_observers": True, "observer_definition": "local_rms", "observer_sample": 500,
    "observer_k": -1,
    "save_snapshots": True, "skip_figures": True, "skip_probe": True,
}

names = []
for i, (M, Sv, sg, npart, nstep) in enumerate(LADDER, start=1):
    sgt = str(sg).replace(".0", "").replace(".", "p")
    cfg = dict(COMMON)
    cfg["M_values"] = [M]; cfg["S_values"] = [Sv]; cfg["vir_mass_spreads"] = [sg]
    cfg["particle_count"] = npart; cfg["n_steps"] = nstep
    cfg["save_interval"] = max(1, nstep // 60)        # ~60 snapshots saved
    cfg["tag"] = f"hero_M{M}_S{Sv}_sig{sgt}_{npart//1000}k_{nstep//1000}ksteps"
    cfg["_comment"] = (f"HERO ladder rung {i}: M={M} S={Sv} sigma={sg}, {npart} particles / "
                       f"{nstep} steps (dt~{(13.8-2.9)/nstep*1000:.1f} Myr). Barnes-Hut, "
                       f"virialized lattice 300 nodes, GRF sphere, Plummer 1 Gpc. "
                       f"save_snapshots + skip_figures.")
    fn = f"{i:02d}_M{M}_S{Sv}_sig{sgt}_{npart//1000}k_{nstep//1000}ksteps.json"
    json.dump(cfg, open(os.path.join(OUT, fn), "w"), indent=2)
    names.append(f"sweeps/hero/{fn}")
    print(f"wrote {fn}  ({npart}p / {nstep} steps, save_interval={cfg['save_interval']})")

print("\nconfigs:", *names, sep="\n  ")
