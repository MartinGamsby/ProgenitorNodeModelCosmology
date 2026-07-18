"""Generate the localized_v5 sweep configs.

v4 (2000p/1092) BEAT LCDM by best-observer: 12 cells < 0.436, frac_below_lcdm up to 5.1%
(M=300/S=20/sigma=6), strong multiplicity (78 cells <= 0.45 across M{100-500} S{15-30}
sigma{4-8}). v4 used observer_k=-1 (WHOLE-CLOUD RMS) and observer_sample=256.

v5 does two things, per the user's "best observer, not center; iterate to beat LCDM":
 (A) OBSERVER_K SWEEP at observer_sample=2000 (ALL particles -> true best vantage + EXACT
     frac_below_lcdm). Arms isolate observer_k in {64,128,256, -1(whole cloud)} on v4's
     winning band M{100,300,500} x S{20,25,30} x sigma{4,6,8}. Finite k = a genuinely LOCAL
     observer (more physical) -> wider vantage spread -> hunt LOWER best-observer + a bigger
     fraction below LCDM. observer_k is the ONLY axis that differs across these 4 arms
     (clean attribution).
 (B) STEP-CONVERGENCE EXTENSION: M{100,300}/S20/sigma{6,8} at n_steps {2730,3276} (k=-1 to
     extend v4's 1092/1638/2184 ladder). v4 showed best-obs DECREASING with steps but
     DECELERATING (0.4396->0.4339->0.4317); confirm it plateaus ~0.43 so "beats LCDM" is
     not a resolution artifact (1092 is the CONSERVATIVE end).

2000 particles, virialized lattice 300 nodes, Plummer 1 Gpc, GRF sphere (matched to v4).
Run:  launch_sweep_detached.ps1 -ArmSet localized5 -Parallel 7
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "localized_v5")
os.makedirs(OUT, exist_ok=True)

COMMON = {
    "node_mass_amplitudes": [0.0],
    "node_s_amplitudes": [0.0],
    "node_mass_seeds": [42],
    "particle_count": 2000,
    "n_steps": 1092,
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
    "observer_definition": "local_rms",
    "observer_sample": 2000,          # ALL particles -> true min + exact frac_below_lcdm
}

BAND_M = [100, 300, 500]
BAND_S = [20, 25, 30]
BAND_SIG = [4.0, 6.0, 8.0]

arms = []
def write_arm(fname, cfg, comment):
    cfg = dict(cfg)
    cfg["figures_dir"] = f"results/figures/{cfg['tag']}"
    cfg["_comment"] = comment
    json.dump(cfg, open(os.path.join(OUT, fname), "w"), indent=2)
    arms.append(f"sweeps/localized_v5/{fname}")
    nM = len(cfg.get("M_values", [])); nS = len(cfg.get("S_values", [])); nsig = len(cfg.get("vir_mass_spreads", [1]))
    print(f"wrote {fname:24} -> {nM*nS*nsig} result rows (obs_k={cfg.get('observer_k')}, steps={cfg['n_steps']})")

# (A) observer_k sweep on the winning band, observer_sample=2000
for i, k in enumerate([64, 128, 256, -1], start=1):
    ktag = "all" if k == -1 else str(k)
    cfg = dict(COMMON)
    cfg["M_values"] = BAND_M
    cfg["S_values"] = BAND_S
    cfg["vir_mass_spreads"] = BAND_SIG
    cfg["observer_k"] = k
    cfg["tag"] = f"loc5_obsk{ktag}"
    write_arm(f"{i:02d}_obsk{ktag}.json", cfg,
              f"localized_v5 observer_k={k} ({'whole cloud' if k==-1 else 'LOCAL k-NN'}): "
              f"best-observer hunt on v4's winning band M{BAND_M} x S{BAND_S} x sigma{BAND_SIG}, "
              f"observer_sample=2000 (true min + exact frac_below_lcdm). Isolates observer_k.")

# (B) step-convergence extension (k=-1, matched to v4 ladder)
for i, steps in enumerate([2730, 3276], start=5):
    cfg = dict(COMMON)
    cfg["n_steps"] = steps
    cfg["observer_k"] = -1
    cfg["M_values"] = [100, 300]
    cfg["S_values"] = [20]
    cfg["vir_mass_spreads"] = [6.0, 8.0]
    cfg["tag"] = f"loc5_conv_{steps}"
    write_arm(f"{i:02d}_conv_{steps}steps.json", cfg,
              f"localized_v5 step-convergence n_steps={steps} (dt~{10.9/steps*1000:.1f} Myr): "
              f"extend v4's 1092/1638/2184 ladder on M{{100,300}}/S20/sigma{{6,8}} to confirm "
              f"best-observer plateaus (the sub-LCDM result is not a step artifact). observer_k=-1.")

json.dump({"family": "localized_v5",
           "purpose": "observer_k (local-observer) hunt to push best-observer below LCDM + "
                      "exact frac_below_lcdm at observer_sample=2000, plus step-convergence "
                      "confirmation. Built on v4's sub-LCDM winning band.",
           "arms": arms},
          open(os.path.join(OUT, "_manifest.json"), "w"), indent=2)
print("\narms:", *arms, sep="\n  ")
