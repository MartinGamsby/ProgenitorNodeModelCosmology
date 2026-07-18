"""Generate the localized_v4 sweep configs.

localized_v4 = the PRODUCTION-resolution refinement of the high-sigma ridge the
exploration found. explore_vir_spread_hi (1000p / 273 steps, Plummer 1 Gpc) located
the best HONEST centre chi2/dof ~0.46 at high mass-spread sigma~5-8, M~300, S~20-30
(e.g. M=300/S=20/sigma=6 -> center 0.459, near LCDM 0.436, decisively below EdS 0.843).
This family re-runs and refines that region at 2000 particles / 1092 steps (dt~10 Myr,
>=1000 per the user's request) to check whether the ~0.46 centre survives convergence
and to pin the (sigma, M, S) minimum.

Treatment = Plummer 1 Gpc (node_softening_gpc=1.0, node_force_law=plummer), MATCHING the
explore run that found the region. Justified: core arms 07/08/09 (soft0/bounded/plummer)
gave IDENTICAL centre chi2 (0.498) at M=500/S=18 -- the close-encounter treatment does not
move the centre a(t) at these non-extreme configs, and Plummer is ~8x faster than
bounded+substep. A dedicated step-convergence pair validates n_steps=1092.

Run via:  powershell -File launch_sweep_detached.ps1 -ArmSet localized -Parallel 7
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "localized_v4")
os.makedirs(OUT, exist_ok=True)

# Common knobs shared by every localized_v4 arm.
COMMON = {
    "node_mass_amplitudes": [0.0],          # isotropic headline (PF2: anisotropy is the separate signal)
    "node_s_amplitudes": [0.0],
    "node_mass_seeds": [42],
    "particle_count": 2000,                 # >= 2000 (user)
    "n_steps": 1092,                        # >= 1000 (user); dt ~10 Myr, 2x the 546-step core
    "t_start_Gyr": 2.9,
    "centerM": 1,
    "objective": "pantheon",
    "results_dir": "results",
    "init_distributions": ["grf"],
    "grf_support": "sphere",
    "node_geometries": ["virialized"],
    "vir_n_nodes": 300,
    "vir_mass_rule": "massfunc",
    "vir_segregation": 1.0,
    "vir_s_metric": "median",
    "vir_relax_steps": 1,
    "vir_relax_mode": "lattice",
    "node_softening_gpc": 1.0,
    "node_force_law": "plummer",
    "score_observers": True,
    "observer_sample": 256,
}

M_GRID = [100, 200, 300, 400, 500]
S_GRID = [15, 18, 20, 25, 30]
SIGMAS = [4.0, 5.0, 6.0, 7.0, 8.0]

arms = []

def write_arm(fname, cfg, comment):
    cfg = dict(cfg)
    cfg["figures_dir"] = f"results/figures/{cfg['tag']}"
    cfg["_comment"] = comment
    path = os.path.join(OUT, fname)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    arms.append(f"sweeps/localized_v4/{fname}")
    print("wrote", fname, "->", len(cfg["M_values"]) * len(cfg["S_values"]) *
          len(cfg.get("vir_mass_spreads", [1])), "cells")

# --- 5 sigma arms: one mass-spread each, full M x S grid (EXPLICIT S, no co-fit) ---
for i, sig in enumerate(SIGMAS, start=1):
    sigtag = str(sig).replace(".0", "").replace(".", "p")
    cfg = dict(COMMON)
    cfg["M_values"] = M_GRID
    cfg["S_values"] = S_GRID
    cfg["vir_mass_spreads"] = [sig]
    cfg["tag"] = f"loc4_sig{sigtag}"
    write_arm(
        f"{i:02d}_sigma{sigtag}.json", cfg,
        f"localized_v4 sigma={sig}: refine the high-spread ridge at PRODUCTION res "
        f"(2000p/1092 steps, Plummer 1 Gpc, virialized lattice 300 nodes). "
        f"M{M_GRID} x S{S_GRID}. Honest metric = center_chi2_dof; also best_observer + "
        f"frac_below_lcdm/eds. explore_vir_spread_hi found ~0.46 here at 1000p/273.",
    )

# --- node-count sensitivity: best corners at 500 nodes ---
cfg = dict(COMMON)
cfg["vir_n_nodes"] = 500
cfg["M_values"] = [300, 500]
cfg["S_values"] = [20, 30]
cfg["vir_mass_spreads"] = [6.0, 8.0]
cfg["tag"] = "loc4_nodes500"
write_arm(
    "06_nodes500.json", cfg,
    "localized_v4 node-count check: the best corners (M{300,500} x S{20,30} x sigma{6,8}) "
    "at vir_n_nodes=500 vs the 300-node main grid -- does a richer mass function move the "
    "centre chi2? 2000p/1092, Plummer 1 Gpc.",
)

# --- step-convergence: the headline-ish cell at higher n_steps (1092 comes from sig6 arm) ---
for steps in (1638, 2184):
    cfg = dict(COMMON)
    cfg["n_steps"] = steps
    cfg["M_values"] = [300]
    cfg["S_values"] = [20]
    cfg["vir_mass_spreads"] = [6.0]
    cfg["tag"] = f"loc4_conv_{steps}"
    write_arm(
        f"07_conv_{steps}steps.json", cfg,
        f"localized_v4 step-convergence: M=300/S=20/sigma=6 at n_steps={steps} "
        f"(dt~{10.9/steps*1000:.1f} Myr). Compare to the sig6 arm's 1092-step value to "
        f"confirm n_steps>=1092 is converged (validates the user's >=1000 floor).",
    )

# --- manifest ---
manifest = {
    "family": "localized_v4",
    "purpose": "Production-resolution (2000p/1092 steps) refinement of the high-sigma "
               "ridge (sigma 4-8, M 100-500, S 15-30) where the exploration found the "
               "best honest centre chi2/dof ~0.46. Plummer 1 Gpc treatment.",
    "arms": arms,
    "main_grid": {"M": M_GRID, "S": S_GRID, "sigma": SIGMAS,
                  "particle_count": 2000, "n_steps": 1092, "vir_n_nodes": 300},
}
with open(os.path.join(OUT, "_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

print("\narms (in run order):")
for a in arms:
    print("  ", a)
total = 5 * len(M_GRID) * len(S_GRID) + 2 * 2 + 1 + 1
print(f"\ntotal cells ~= {total}")
