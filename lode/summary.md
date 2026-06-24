# ProgenitorNodeModelCosmology Summary

N-body cosmology simulator testing whether External-Node Model (classical tidal forces from trans-observable massive structures) can replicate LCDM expansion without dark energy. Python codebase uses Leapfrog integration to evolve 2000 particles under internal gravity + tidal forces from 26 HMEA (Hyper-Massive External Attractor) nodes in 3x3x3-1 cubic lattice. Draft paper (docs/VirializedMetaStructure.tex) proposes Progenitor Hypothesis: Big Bang was destabilization of a node in virialized meta-structure, explaining isotropy.

Parameter exploration finds MULTIPLE balanced-optimization configs across wide mass range:
- M=9000 S=38: 98.02% endpoint, R²_size=0.9954, R²_rate=0.9630 (primary)
- M=875 S=24: 97.92% endpoint, R²_size=0.9951, R²_rate=0.9603
- M=92 S=15: 98.67% endpoint, R²_size=0.9966, R²_rate=0.9565
Size-only optimization can reach R²_size=0.9991 (M=800 S=22) but distorts expansion rate—expected tradeoff: size is integrated quantity, rate is derivative (Section 4.3).

Matter-only comparison validates mechanism: N-body matter-only NEVER exceeds LCDM (physics constraint enforced via velocity calibration at sim.run()). Velocity calibration scales initial velocities to compensate for N-body's ~65-80% deceleration compared to Friedmann; auto-calculated from t_start or passed as explicit damping parameter. Without external-nodes, matter-only has R²_rate=0.835 vs external-node 0.963, and R²_size=0.989 vs 0.995.

The project also includes a separate, data-anchored Hubble-diagram test (hubble_diagram.py + cosmo/distances.py, cosmo/pantheon.py, cosmo/hubble_diagram.py): it turns each model's semi-analytic H(z) into a distance-modulus mu(z) curve, overlays REAL Pantheon+SH0ES supernovae, and reports per-model chi^2/R^2 after marginalizing an additive magnitude offset. This is model-vs-real-data (distinct from the N-body model-vs-LCDM-theory R^2) and leaves the N-body path untouched. Open issue: the (M,S) matching real SNe (Omega_Lambda_eff~=0.70, near M=855,S=37.8) differs from the paper's primary N-body config (M=9000,S=38 -> Omega_Lambda_eff~=7.24, a closed universe) — see lode/physics/hubble-diagram.md.

A genuine FROM-SIM, NON-circular version of this test now exists (cosmo/sim_distance.py, hubble_diagram_nbody.py, and an objective="pantheon" parameter-sweep mode): instead of the semi-analytic Omega_Lambda_eff curve — which at 0.70 is mathematically == LCDM by construction (circular) — it integrates mu(z) directly from the REAL N-body a(t) via D_C = c*integral(dt/a) (no differentiation, so it avoids the expansion-rate edge artifacts). Stage-1 gating (t_start=5.8, z<=~0.96) finds the from-sim curve currently INDISTINGUISHABLE from LCDM (~0.77 sigma, slightly worse than LCDM — real, not identical-by-construction). Stage-2 validated a safe earlier start floor t_start=2.9 Gyr (z_max~2.23, full Pantheon+ range) with no calibration change. See lode/physics/hubble-diagram-nbody.md.

The from-sim comparison is packaged as a reproducible publication tool (hubble_diagram_nbody.py): t_start=2.9 full-coverage default, a Delta-mu-vs-LCDM residual panel, an Einstein-de Sitter (Omega_m=1) "no dark energy" null, --from-best-config to load a sweep's best (M,S,centerM), and a machine-readable JSON sidecar. CANONICAL result (uniform default config, real Pantheon+): from-sim chi2/dof~=0.50, LCDM 0.44, EdS null 0.84, growth anchor 3.10 vs physical 3.30 (PHYSICAL). Honest verdict: the model gives effective dark energy (sits with LCDM, far from the EdS null) but does NOT beat LCDM and is degenerate in M/S^3. See lode/physics/pantheon-comparison-results.md.

Two model-realism levers add anisotropy without changing the isotropic background: (1) deterministic seed-driven per-node HMEA masses (SimulationParameters.node_mass_seed/node_mass_amplitude — mean-preserving log-normal, sweepable; amplitude=0 default is the legacy uniform lattice); (2) a selectable realistic particle init (init_distribution="grf": Gaussian random field with BBKS LCDM P(k) + Zel'dovich displacement; default stays uniform_sphere). cosmo/anisotropy.py + anisotropy_report.py measure the resulting directional shear (inertia-tensor eigenvalue spread) and Hubble dipole ΔH/H; convergence_check.py verifies the GRF observable a(t)/mu(z) converges over N. A UTF-8 stdout guard (cosmo/encoding.py, configure_utf8_stdout) lets the Greek-glyph banners print on Windows cp1252 without PYTHONIOENCODING.

Toy model scope: late-time acceleration (t=5.8->13.8 Gyr, 8 Gyr period); doesn't address CMB, BAO, structure formation, or early universe. Starts at t=5.8 Gyr (not Big Bang) to focus on late-universe expansion. Code purpose: test mechanism viability, explore parameters, generate data for ongoing draft refinement.

Key technical insights:
1. Initial velocities: model-appropriate Hubble parameter (H_lcdm for LCDM, H_matter for matter-only); COM removal; RMS radius normalization ensures identical starting size
2. Velocity calibration at sim.run(damping=None): scales initial velocities for non-LCDM models, auto-calculated from t_start via formula (t_start/13.8)^0.135
3. Leapfrog pre-kick eliminates "initial bump" artifact by properly initializing velocity staggering at t=-dt/2
4. solve_friedmann_at_times computes LCDM baseline at exact N-body snapshot times for precise alignment
5. Timestep validation enforces dt_s < 0.05 Gyr to prevent leapfrog instability
6. Unit-aware variable naming (_m, _s, _kg, _si, _mps2 suffixes) throughout codebase
7. Three force methods: 'direct' (NumPy O(N^2)), 'numba_direct' (Numba JIT O(N^2), 14-17x speedup), 'barnes_hut' (real octree O(N log N))
8. R² metrics for both size and expansion rate; balanced configs achieve R²_size>0.99 R²_rate>0.95
9. Validation: 232 tests including matter-only never-exceeds-LCDM, Numba verification, reproducibility checks
10. Paper predictions: dipole anisotropy deltaH0/H0 ~ 4.6-11.3% (comparable to Hubble Tension 8.6%); dark flow ~320-790 km/s; predictions robust across M/S configs
11. Sweep CSV metric note: match_curve_rmse_pct = 100−RMSE×100; actual RMSE = 1−(match_curve_rmse_pct/100)
