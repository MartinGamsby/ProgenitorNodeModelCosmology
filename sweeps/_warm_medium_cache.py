"""Phase-A cache warmer for the centerM campaign: pre-relax every (seed, vcm) medium so the
parallel Phase-B sweeps hit a warm data/vir_medium cache (the disk cache is best-effort and
unlocked; concurrent first-relaxations of the SAME cell could race). Each worker gets a disjoint
seed subset, so no two workers ever build the same (n, sigma, seed, cm) file.

Usage: python sweeps/_warm_medium_cache.py <seed> [<seed> ...]
"""
import os, sys, time
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
import numpy as np
from cosmo.node_geometry import build_virialized_grid
from cosmo.constants import CosmologicalConstants

GPC = CosmologicalConstants.Gpc_to_m
M_VALUE = 3000
CENTERMS = [1, 2, 3, 10, 30, 100, 300, 1000, 3000]

for seed in [int(a) for a in sys.argv[1:]]:
    for cM in CENTERMS:
        vcm = cM / M_VALUE
        t = time.time()
        pos, mass = build_virialized_grid(
            55.0 * GPC, n_nodes=300, M_ext_kg=1.0, vir_mass_spread=0.5,
            vir_s_metric="median", vir_relax_mode="medium",
            center_mass_frac=vcm, seed=seed)
        r_gpc = np.linalg.norm(pos, axis=1) / GPC
        print(f"seed {seed} centerM {cM} (vcm {vcm:.5f}): {time.time()-t:5.1f}s  "
              f"nearest HMEA at S=55: {r_gpc.min():.1f} Gpc", flush=True)
