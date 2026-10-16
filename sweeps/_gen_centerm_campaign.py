"""Generate the LARGE-centerM CAMPAIGN (PF24, user-directed, 2026-07-13).

centerM = the PROGENITOR NODE'S MASS (settled: the progenitor was a typical node of the
virialized meta-structure; the Big Bang split it into the particle cloud + outer mass). It drives
BOTH epochs with ONE knob: post-BB the WS4 outer particles (outer_particle_cap=2 keeps N feasible,
outer TOTAL exact), pre-BB the medium relaxation's us-node mass vir_center_mass_frac =
centerM / M_value (at centerM == M_value == 3000 the progenitor is literally a typical node).

A REAL sweep this time: centerM log-spaced {1..3000} (9 values) x 8 medium seeds = 72 cells,
each with its OWN ternary S co-fit [40, 160] Gpc (the structure differs per (centerM, seed) so a
fixed S would conflate). Fixed: medium geometry, sigma 0.5, grfmass init, M=3000, 2000p/546,
best-observer scoring. GOALS (per the user): NOT the isotropic fit (PF6: outer mass inert) but the
STRUCTURE - a meta-structure + cloud that looks like the universe (no big central knot), with the
observational gate "no HMEA within ~15 Gpc" (checked after co-fit: d_near_Gpc = ratio x S_cofit).
Tidal stretching is the sought dark-energy effect; only the growth anchor disqualifies.

Phase A (warm_medium_cache.py) pre-relaxes all 72 (seed, vcm) mediums serially per worker so the
parallel Phase-B sims hit a warm disk cache (avoids concurrent-relaxation races).
Run: results/logs/cmx_driver.sh (nohup, 4-way pool, 8k-cheap first not applicable - all 2000p).
"""
import json, os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "centerm_campaign")
os.makedirs(OUT, exist_ok=True)

M_VALUE = 3000
CENTERMS = [1, 2, 3, 10, 30, 100, 300, 1000, 3000]
SEEDS = [0, 1, 2, 3, 5, 7, 9, 11]

COMMON = {
    "node_mass_amplitudes": [0.0], "node_s_amplitudes": [0.0],
    "particle_count": 2000, "n_steps": 546, "t_start_Gyr": 2.9,
    "objective": "pantheon", "results_dir": "results",
    "init_distributions": ["grfmass"], "grf_support": "sphere",
    "node_geometries": ["virialized"], "vir_n_nodes": 300, "vir_mass_rule": "massfunc",
    "vir_mass_spreads": [0.5], "vir_segregation": 1.0, "vir_s_metric": "median",
    "vir_relax_steps": 1, "vir_relax_mode": "medium",
    "node_softening_gpc": 1.0, "node_force_law": "plummer",
    "score_observers": True, "observer_definition": "local_rms",
    "observer_sample": 2000, "observer_k": -1,
    "M_values": [M_VALUE],
    "S_values": "co-fit", "s_min_gpc": 40, "s_max_gpc": 160, "s_cofit_method": "ternary",
    "outer_particle_cap": 2.0,
}

arms = []
for seed in SEEDS:
    for cM in CENTERMS:
        cfg = dict(COMMON)
        cfg["node_mass_seeds"] = [seed]
        cfg["centerM"] = float(cM)
        cfg["vir_center_mass_frac"] = cM / M_VALUE   # the epochs identity
        tag = f"cmx_s{seed}_cm{cM}"
        cfg["tag"] = tag
        cfg["figures_dir"] = f"results/figures/{tag}"
        cfg["_comment"] = (f"centerM campaign: progenitor mass centerM={cM} (vcm={cM/M_VALUE:.5f}), "
                           f"medium seed {seed}, sigma0.5, grfmass, M{M_VALUE}, S co-fit [40,160], "
                           f"outer cap 2x. Gate AFTER co-fit: nearest HMEA > 15 Gpc.")
        json.dump(cfg, open(os.path.join(OUT, f"{tag}.json"), "w"), indent=2)
        arms.append(tag)

json.dump({"family": "centerm_campaign", "M_value": M_VALUE, "centerMs": CENTERMS,
           "seeds": SEEDS, "arms": arms,
           "purpose": "PF24 large-centerM sweep: 9 progenitor masses x 8 medium seeds, per-cell "
                      "S co-fit. Structure goal (no central knot; no HMEA within 15 Gpc), not chi2."},
          json_file := open(os.path.join(OUT, "_manifest.json"), "w"), indent=2)
print(f"wrote {len(arms)} configs")
