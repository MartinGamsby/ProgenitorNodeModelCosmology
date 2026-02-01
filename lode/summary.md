# ProgenitorNodeModelCosmology Summary

N-body cosmology simulator testing whether External-Node Model (classical tidal forces from trans-observable massive structures) can replicate ΛCDM expansion without dark energy. Python codebase uses Leapfrog integration to evolve 200-300 particles under internal gravity + tidal forces from 26 HMEA (Hyper-Massive External Attractor) nodes in 3×3×3-1 cubic lattice. Draft paper (docs/paper.tex) proposes Progenitor Hypothesis: Big Bang was destabilization of a node in virialized meta-structure, explaining isotropy. Parameter exploration (via parameter_sweep.py) finds MULTIPLE close matches (M=855 S=25, M=97000 S=65, M=69 S=15) achieving >99% endpoint match with R²>0.89 for expansion rate dynamics.

Matter-only comparison validates mechanism: N-body matter-only NEVER exceeds LCDM (physics constraint enforced via velocity calibration). Velocity calibration scales initial velocities by ~0.72x to compensate for N-body's ~65% deceleration compared to Friedmann; this ensures matter-only stays below LCDM at all timesteps regardless of t_start value. Without external-nodes, matter-only reaches ~75-85% of LCDM endpoint, proving external-nodes provide genuine acceleration.

Toy model scope: late-time acceleration (t=3.8→13.8 Gyr, 10 Gyr period); doesn't address CMB, BAO, structure formation, or early universe. Starts at t=3.8 Gyr (not Big Bang) to focus on late-universe expansion. Code purpose: test mechanism viability, explore parameters, generate data for ongoing draft refinement.

Key technical insights:
1. Initial velocities: model-appropriate Hubble parameter (H_lcdm for ΛCDM, H_matter for matter-only); COM removal; RMS radius normalization ensures identical starting size
2. Velocity calibration for matter-only: one-time scaling (~0.72x) based on predicted final expansion, compensates for N-body deceleration deficit
3. Leapfrog pre-kick eliminates "initial bump" artifact by properly initializing velocity staggering at t=-dt/2
4. solve_friedmann_at_times computes ΛCDM baseline at exact N-body snapshot times for precise alignment
5. Timestep validation enforces dt_s < 0.05 Gyr to prevent leapfrog instability
6. Unit-aware variable naming (_m, _s, _kg, _si, _mps2 suffixes) throughout codebase
7. Three force methods: 'direct' (NumPy O(N²)), 'numba_direct' (Numba JIT O(N²), 14-17x speedup), 'barnes_hut' (real octree O(N log N))
8. R² metric for statistical rigor; last-half R² (5 Gyr) isolates late-time acceleration behavior
9. Validation: 231 tests including matter-only never-exceeds-LCDM, Numba verification, reproducibility checks
10. Paper predictions: dipole anisotropy ΔH₀/H₀ ≈ 4.4% (comparable to Hubble Tension 8.6%)
