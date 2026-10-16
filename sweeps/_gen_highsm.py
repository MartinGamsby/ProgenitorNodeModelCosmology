"""Generate the HIGH-S / HIGH-M low-sigma sweep (idea A, user: S was too small; higher S, higher M).

The prior low-sigma map capped at S22/M300. The effective dark energy Omega_Lambda_eff ~ M/S^3, so
the good-fit RIDGE runs to much higher S with correspondingly higher M (M ~ S^3). This sweep walks
that ridge at PHYSICALLY DEFENSIBLE sigma<=2 using the sweep's per-M S CO-FIT (ternary, the
validated method): for each M in {500,1000,3000,9000} it searches the matching S in [20,80].

Split into 6 arms = node-count {300,600} x sigma {1,1.5,2} for parallelism. 2000p/546 (the 1000p/273
map over-promised on the collapse boundary). observer_k=-1, seed 42, GRF sphere, Plummer 1 Gpc.
Rank anchor-ok by best_observer_chi2; refine + snapshot the winner (does higher S/M give BOTH a
good fit AND a rounder cloud, since a far node shell is smoother across the observable sphere?).
Run:  per arm: python sweep.py --config sweeps/highsm/<cfg>.json  (HMEA_CACHE_CONCURRENT=1 parallel)
"""
import json, os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "highsm")
os.makedirs(OUT, exist_ok=True)

COMMON = {
    "node_mass_amplitudes": [0.0], "node_s_amplitudes": [0.0], "node_mass_seeds": [42],
    "particle_count": 2000, "n_steps": 546, "t_start_Gyr": 2.9, "centerM": 1,
    "objective": "pantheon", "results_dir": "results",
    "init_distributions": ["grf"], "grf_support": "sphere",
    "node_geometries": ["virialized"], "vir_mass_rule": "massfunc",
    "vir_segregation": 1.0, "vir_s_metric": "median", "vir_relax_steps": 1,
    "vir_relax_mode": "lattice", "node_softening_gpc": 1.0, "node_force_law": "plummer",
    "score_observers": True, "observer_definition": "local_rms",
    "observer_sample": 2000, "observer_k": -1,
    "M_values": [500, 1000, 3000, 9000],
    "S_values": "co-fit", "s_min_gpc": 20, "s_max_gpc": 80, "s_cofit_method": "ternary",
}

arms = []
for nn in [300, 600]:
    for sig in [1.0, 1.5, 2.0]:
        cfg = dict(COMMON)
        cfg["vir_n_nodes"] = nn
        cfg["vir_mass_spreads"] = [sig]
        tag = f"highsm_N{nn}_sig{sig:.1f}".replace(".", "p")
        cfg["tag"] = tag
        cfg["figures_dir"] = f"results/figures/{tag}"
        cfg["_comment"] = (f"high-S/M ridge, N={nn}, sigma={sig}: M{{500,1000,3000,9000}} with per-M "
                           f"S CO-FIT ternary [20,80]. 2000p/546, observer_k=-1, seed42. Does a higher "
                           f"S/M ridge point at defensible sigma<=2 fit LCDM AND round the cloud "
                           f"(far node shell = smoother tidal field across the observable sphere)?")
        fn = f"{tag}.json"
        json.dump(cfg, open(os.path.join(OUT, fn), "w"), indent=2)
        arms.append(f"sweeps/highsm/{fn}")
        print(f"wrote {fn}: N={nn} sigma={sig}, M{{500..9000}} S co-fit[20,80]")

json.dump({"family": "highsm",
           "purpose": "Walk the M/S^3 ridge to HIGHER S/M at defensible sigma<=2 (per-M S co-fit). "
                      "Find the best-observer fit + check if a far/high-S node shell gives a rounder "
                      "cloud than the low-S configs. 6 arms = N{300,600} x sigma{1,1.5,2}.",
           "arms": arms}, open(os.path.join(OUT, "_manifest.json"), "w"), indent=2)
print("arms:", *arms, sep="\n  ")
