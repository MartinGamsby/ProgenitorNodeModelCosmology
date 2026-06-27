# Lode Map

## Core Files
- [summary.md](./summary.md) - One-paragraph project overview
- [terminology.md](./terminology.md) - Domain vocabulary (HMEA, Ω_Λ_eff, damping factor, etc.)
- [practices.md](./practices.md) - Development patterns and conventions
- [practices/unit-conventions.md](./practices/unit-conventions.md) - Unit naming conventions (_m, _s, _kg, _si, _mps2 suffixes)
- [paper-reference.md](./paper-reference.md) - docs/VirializedMetaStructure.tex ground truth: what code tests, scope, success criteria

## Architecture
- [architecture/module-structure.md](./architecture/module-structure.md) - Code organization and dependencies
- [architecture/data-flow.md](./architecture/data-flow.md) - How data flows through run_simulation.py
- [architecture/testing.md](./architecture/testing.md) - Unit test structure, status, philosophy

## Physics
- [physics/theoretical-framework.md](./physics/theoretical-framework.md) - External-Node Model, Progenitor Hypothesis, predictions, scope/limitations
- [physics/force-calculations.md](./physics/force-calculations.md) - Internal gravity, tidal forces, dark energy, Hubble drag; per-node MASS (node_mass_amplitude) + POSITION (node_s_amplitude) anisotropy knobs; LEVER EXPERIMENT honest verdict (symmetry-breaking drives shear/dipole ~10x but does NOT reach LCDM isotropically — anisotropy is the discriminating signal)
- [physics/barnes-hut-optimization.md](./physics/barnes-hut-optimization.md) - Force methods: direct, numba_direct (O(N²) JIT), barnes_hut (real octree O(N log N))
- [physics/initial-conditions.md](./physics/initial-conditions.md) - SELF-CONSISTENT EdS ICs (default eds_consistent=True): v=H_EdS*r with H_EdS=2/(3 t_start) + cloud mass = EdS critical mass => M_ext=0 reproduces Einstein-de Sitter by construction (THE invariant, ~0.2% growth / 0.02 mag). Legacy velocity-calibration fudge now skipped; t_start floor lifted. PLUS pre_start_tidal_boost (default True): physical pre-t_start HMEA velocity boost dv_r=g_r·(3/5)·t_start, vanishes at M=0, small effect (~0.35% growth). Honest capability+runaway tables (best M=1500/S=30 chi2/dof 0.51, between EdS null 0.85 and LCDM 0.43; runaway S_crit∝M^(1/3)). Future steps noted: centerM>1, node_s_amplitude. Selectable init_distribution (uniform_sphere default + grf)
- [physics/realistic-initial-conditions.md](./physics/realistic-initial-conditions.md) - GRF+Zel'dovich C1 (IMPLEMENTED) and C2 galaxy-catalog future direction; convergence numbers; C2 caveats (mask contamination, security)
- [physics/observable-mask-and-outer-mass.md](./physics/observable-mask-and-outer-mass.md) - WS4 IMPLEMENTED: centerM repurposed as OUTER-MASS multiplier (extra Big-Bang matter outside the observable sphere; N linear; R_sim=R_obs·centerM^(1/3)); observable inner-region mask on a(t)/H(z)/mu(z)/growth-anchor (the hard invariant); softening FROZEN at 1 Gpc (decoupled from centerM); outer_density_ceiling (cap 2.0); cache v3 + float centerM slug; honest result = outer mass barely moves isotropic chi2 (~0.012)
- [physics/node-placement-vs-perturbation.md](./physics/node-placement-vs-perturbation.md) - WHY the virialized grid exists: multi-layer lattices (fcc/bcc) park most nodes 4-6x beyond the ~14 Gpc horizon where 1/r^3 makes them inert, so "S" is NOT comparable across geometries (cube26 = clean single shell). The VIRIALIZATION CRITERION (virialization_residual metric) + the finding that net-null inner force requires LATTICE symmetry (random blob can't force-balance; vir_relax_steps balance level). HONEST section-1 finding: the node_mass_amplitude/node_s_amplitude PERTURBATION machinery is CORRECT on all geometries (no multi-layer bug) — it's the PLACEMENT, not the perturbation
- [physics/slingshot-and-softening.md](./physics/slingshot-and-softening.md) - Particle slingshot root cause (close pass to a near-point HMEA node, NOT particle-particle; n_steps/n_particles don't help) + the node_softening_gpc Plummer fix (default 0 byte-identical) + "doubly tamed" (force-balanced geometry + softening); M=0==EdS preserved. ALSO Section-4 opt-in alternatives (default OFF/byte-identical): node_force_law="bounded" ("can't cross the midpoint" capped close-range accel, distinct from the blunt Plummer floor) + adaptive KDK sub-stepping (node_substep_threshold/node_substeps). Measured trade table (_generate_ws8_close_encounter.py): bounded+substep tames cube26 (844x->2.4x, growth into anchor) ; substep ALONE does not (PF9 re-confirmed); virialized runaway is geometric, untamed by any law
- [physics/integration.md](./physics/integration.md) - Leapfrog algorithm implementation
- [physics/hubble-diagram.md](./physics/hubble-diagram.md) - SEMI-ANALYTIC mu(z) Hubble-diagram test vs real Pantheon+ SNe (H(z) from Omega_Lambda_eff; CIRCULAR == LCDM at 0.70); open M,S discrepancy
- [physics/hubble-diagram-nbody.md](./physics/hubble-diagram-nbody.md) - FROM-SIM N-body mu(z) test (D_C=c∫dt/a from real a(t); NON-circular). Stage-1 gating (~indistinguishable from LCDM at z<=0.96), Stage-2 safe floor t_start=2.9, Stage-3 pantheon sweep objective
- [physics/pantheon-comparison-results.md](./physics/pantheon-comparison-results.md) - CANONICAL from-sim vs Pantheon+ numbers (chi2/dof: model 0.50, LCDM 0.44, EdS null 0.84; growth anchor 3.10 vs 3.30; honest "with-LCDM, doesn't beat it, M/S^3 degenerate" verdict); comparison tool + JSON sidecar + --from-best-config
- [physics/observer-from-particle.md](./physics/observer-from-particle.md) - PROTOTYPE (items 5/D, NOT pinned, off the default path): mu(z) from EACH particle's local frame, not the cloud centre. cosmo/observer_distance.py (pure: local_rms k-NN RMS growth + hubble_flow local-Hubble-integral defs; centre observer reproduces centre a(t); authoritative chi2 via sim_to_distance_modulus+evaluate_precomputed). HONEST verdict: best observer beats centre but is a cherry-pick (report the DISTRIBUTION median/p10/p90); the large per-observer spread is a PF2 anisotropy signal, not an isotropic-fit win. _generate_observer_figs.py -> results/figures/ws8/observer_chi2_{distribution.png,.csv}

## Numerics
- [numerics/timestep-stability.md](./numerics/timestep-stability.md) - Timestep requirements, instability symptoms, energy monitoring
- [numerics/lcdm-baseline.md](./numerics/lcdm-baseline.md) - ΛCDM baseline computation standardization, reference values, bug fixes
- [numerics/leapfrog-staggering.md](./numerics/leapfrog-staggering.md) - Velocity staggering, pre-kick fix, initial bump elimination
- [numerics/expansion-rate-calculation.md](./numerics/expansion-rate-calculation.md) - Hubble parameter H(t) from numerical derivatives, edge artifacts from smoothing, diagnostic tools

## Scripts
- [scripts/parameter-sweep.md](./scripts/parameter-sweep.md) - The sweep LIBRARY (cosmo/parameter_sweep.py): search algorithms, match metrics, objectives, build_cache_name. sweep.py is the single DRIVER; root parameter_sweep.py + pantheon_knob_sweep.py are DELETED (now JSON configs)
- [scripts/visualization.md](./scripts/visualization.md) - 3D visualization pipeline, comparison mode, animation
- hubble_diagram.py - Standalone SEMI-ANALYTIC Hubble-diagram-vs-Pantheon+ script (documented in [physics/hubble-diagram.md](./physics/hubble-diagram.md))
- hubble_diagram_nbody.py + cosmo/sim_distance.py - FROM-SIM N-body Hubble-diagram Stage-1 gating script + a(t)->mu(z) kernel (documented in [physics/hubble-diagram-nbody.md](./physics/hubble-diagram-nbody.md))
- sweep.py - THE single config-driven sweep driver (M/S/amplitudes/seed/init/particles/geometry/vir_*/node_softening/start-size; objective pantheon|lcdm; per-M S co-fit; resumable+cached; CSV+figures). Replaces the DELETED root parameter_sweep.py + pantheon_knob_sweep.py. Documented in [plans/overarching-sweep.md](./plans/overarching-sweep.md)
- sweeps/*.json - config catalog: lcdm_example.json (objective=lcdm), knob_grf.json (GRF 2-knob factorial), virialized_final.json (WS1/W6 HEADLINE: force-balanced virialized + doubly-tamed + near-LCDM band). Every config is smoke-tested (loads+expands)
- anisotropy_report.py + cosmo/anisotropy.py - Directional shear / Hubble-dipole anisotropy diagnostic (documented in [physics/anisotropy-diagnostic.md](./physics/anisotropy-diagnostic.md))
- convergence_check.py - GRF particle-count N-ladder convergence harness (N ∈ {1 000, 10 000, [100 000]}; barnes_hut auto; three-invariant gate; documented in [physics/realistic-initial-conditions.md](./physics/realistic-initial-conditions.md))
- _generate_ws8_close_encounter.py - Section-4 close-encounter comparison: slingshot tail AND Pantheon+ chi2/dof for {legacy floor, Plummer 1/0.1 Gpc, bounded law, adaptive substep, bounded+substep} x {cube26, virialized} + n_steps ladder. Uses the PRODUCT knobs (node_force_law/node_substep_*) and the authoritative compute_pantheon_metrics scorer. Output results/figures/ws8/close_encounter_compare.{csv,png} (gitignored)
- _generate_ws8_figs.py - WS8 virialized-grid figures (node positions colored by mass incl. virialized variants; cube26-vs-virialized particle-motion/slingshot diagnostic; radial-vs-massfunc a/b mass-rule compare). Output results/figures/ws8/ (gitignored). PURE helpers tested in tests/test_ws8_figs.py. Documented in [scripts/visualization.md](./scripts/visualization.md#ws8--virialized-grid-figures-_generate_ws8_figspy)

## Plotting
- [plans/graphs-from-scripts.md](./plans/graphs-from-scripts.md) - WS2: shared `cosmo/plots.py` (IMPLEMENTED); figure set F1-F12; output `results/figures/<ws>/<name>.png`; regeneration script `_generate_ws2_figs.py`; 21 tests in `tests/test_plots.py`

## Plans
- [plans/](./plans/) - Future enhancements and TODOs
- [plans/hubble-diagram-followups.md](./plans/hubble-diagram-followups.md) - Reconcile (M,S) discrepancy, optional paper edit, full-covariance chi^2 (N-body-derived d_L now DONE)
- [plans/deeper-exploration-roadmap.md](./plans/deeper-exploration-roadmap.md) - HUB for the next deeper-exploration phase: pinned findings, Mermaid phase diagram, sequencing (cheap exploration -> mid-phase concepts -> heavy convergence/GPU). Links the 7 workstream sub-files below
- [plans/pinned-findings.md](./plans/pinned-findings.md) - PF1 M=0==EdS; PF2 anisotropy=discriminating signal; PF3 M/S^3 degeneracy; PF4 corrected near-LCDM/far-from-null framing; PF5 runaway boundary; PF6 outer-mass barely moves isotropic fit; PF7 placement-not-perturbation; PF8 force-balance-requires-lattice-symmetry; PF9 slingshot-cause-and-softening-fix; PF10 start-size-is-a-real-a(t)-lever
- [plans/overarching-sweep.md](./plans/overarching-sweep.md) - WS1: ONE config-driven multi-param sweep tool (M/S/amplitudes/seed/init/particles/geometry; per-M S co-fit on pantheon; resumable+cached; CSV+figures) superseding the ad-hoc scripts
- [plans/graphs-from-scripts.md](./plans/graphs-from-scripts.md) - WS2: every claim gets a SAVED PNG (results/figures/); enumerated figure set F1-F12; shared cosmo/plots.py helper
- [plans/node-geometries.md](./plans/node-geometries.md) - WS3: node-geometry factory (cube26/cube_dense/fcc/bcc, all volume-filling/virialized; hollow shells excluded) threaded through HMEAGrid+sweep+anisotropy; vacuum-traceless honesty. IMPLEMENTED: cosmo/node_geometry.py + visualize_geometries.py + passing tests.
- [plans/centerm-reconception.md](./plans/centerm-reconception.md) - WS4 DESIGN RATIONALE (IMPLEMENTED): why centerM was repurposed from a softening artifact into an outer-mass multiplier; mechanism now lives in physics/observable-mask-and-outer-mass.md; honest result in pinned-findings PF6
- [plans/grf-vs-uniform.md](./plans/grf-vs-uniform.md) - WS5: explain the GRF chi2 swing (0.51->1.56 at M=1500/S=30); H1 real / H2 near-runaway / H3 N-noise / H4 GRF-setup, with graphs
- [plans/particle-convergence.md](./plans/particle-convergence.md) - WS6 (LAST/slowest): high-N Barnes-Hut convergence of chi2/dof+growth+shear/dipole for headline configs
- [plans/scale-out-gpu-hf.md](./plans/scale-out-gpu-hf.md) - WS7 (FUTURE, not now): Numba-CUDA GPU sweep + HuggingFace dataset results store, triggered only when scripts+numbers are good

## Temporary
- [tmp/](./tmp/) - Session scraps (git-ignored)
