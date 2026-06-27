# Terminology

**HMEA** - Hyper-Massive External Attractors. Massive structures beyond observable universe providing tidal forces.

**External-Node Model** - Alternative cosmology replacing dark energy with classical gravity from HMEA nodes.

**ΛCDM** - Lambda Cold Dark Matter. Standard cosmology with dark energy (cosmological constant Λ).

**M_value** - External node mass as multiple of observable universe mass. Parameter being explored; balanced-optimization configs at M=92-9000 give R²_size>0.99.

**S_value** - HMEA grid spacing in Gpc. Parameter being explored; balanced-optimization configs at S=15-38 Gpc.

**Ω_Λ_eff** - Effective dark energy density from tidal forces. Ω_Λ_eff = 2G×M_ext/(S³×H₀²).

**Tidal acceleration** - Force from external nodes: a = GM_ext × r / |r_node - r|³

**Damping factor** - Coefficient multiplying initial Hubble flow velocities. With damping=1.0, each model starts with v = H × r where H is model-appropriate (H_lcdm for ΛCDM, H_matter for matter-only). Auto-calculated from deceleration parameter unless explicitly overridden. For physics tests, use damping=1.0.

**Deceleration parameter (q)** - Measure of expansion deceleration: q = 0.5×Ω_m(a)/[Ω_m(a)+Ω_Λ] - 1.0. Positive q means deceleration (matter-dominated), negative q means acceleration (Λ-dominated). Used to auto-calculate damping factor.

**Progenitor Node** - Hypothesized source of Big Bang: a node in virialized meta-structure that destabilized, explaining isotropy without fine-tuning. (Draft theory in VirializedMetaStructure.tex)

**Virialized Meta-Structure** - Hypothesized static, gravitationally balanced lattice of HMEAs beyond observable universe. Analogous to relaxed galaxy cluster. (Draft theory in VirializedMetaStructure.tex)

**Great Metabolism Hypothesis** - Speculative cyclical cosmology where universes are transient accretion events feeding meta-structure nodes. (Draft theory in VirializedMetaStructure.tex)

**Toy Model** - Simplified proof-of-concept. Addresses late-time acceleration only; doesn't address CMB, BAO, structure formation, inflation, or early universe. Code supports draft paper development.

**Draft Paper** - docs/VirializedMetaStructure.tex. Working document describing External-Node Model theoretical framework. Under development, not published.

**Scale factor (a)** - Cosmic expansion parameter. a=1 at present, a<1 in past, a>1 in future.

**Hubble parameter (H)** - Expansion rate H = ȧ/a. Current: H₀ ≈ 70 km/s/Mpc.
  - H_lcdm(a) = H₀√(Ω_m/a³ + Ω_Λ) - includes dark energy
  - H_matter(a) = H₀√(Ω_m/a³) - matter-only, no dark energy
  - At a=0.839: H_lcdm ≈ 2.57e-18 s⁻¹, H_matter ≈ 2.02e-18 s⁻¹ (21% lower)

**Leapfrog** - Symplectic integrator using Kick-Drift-Kick algorithm. Energy-conserving.

**Softening** - Minimum gravitational interaction distance preventing singularities. Base value frozen at 1.0 Gpc per M_obs (WS4: independent of centerM; the integrator still scales it by mean_particle_mass^(1/3)). The legacy centerM->softening coupling was removed so centerM>1 changes a(t) only via real added outer-mass gravity, not a resolution artifact.

**Matter-only** - Cosmology with Ω_Λ=0 (no dark energy, no external nodes). Pure matter deceleration.

**Observable universe** - Region within particle horizon. Modeled as N tracer particles.

**Friedmann equation** - Differential equation governing cosmic expansion: ȧ = H₀√(Ω_m/a³ + Ω_Λ)×a

**RMS radius** - Root-mean-square distance of particles from center of mass. Proxy for universe size.

**center_node_mass / centerM** - REPURPOSED (WS4). Outer-MASS multiplier = total simulated mass / inner observable mass (>= 1.0, default 1.0). >1.0 adds extra Big-Bang matter OUTSIDE the inner observable sphere at the same density + per-particle mass; N grows LINEARLY (centerM=2 -> 2x particles), R_sim = R_obs·centerM^(1/3). The inner observable region (its density, mass, a(t)) is UNCHANGED; only outer particles are appended. a(t)/H(z)/mu(z)/growth-anchor are measured on the inner observable subset ONLY (observable mask). centerM=1.0 -> byte-identical to pre-WS4. NO LONGER a softening knob (softening is now frozen, independent of centerM). centerM>1 requires eds_consistent + uniform_sphere. See lode/physics/observable-mask-and-outer-mass.md.

**Observable mask** - Boolean per-particle mask (True=inner observable, False=outer shell) set in ParticleSystem._initialize_particles; applied in CosmologicalSimulation._calculate_expansion_history so a(t)/mu(z)/growth-anchor use only the inner observable sub-region. All-True (byte-identical) at centerM=1. The integrator is untouched: outer particles still exert gravity. See lode/physics/observable-mask-and-outer-mass.md.

**outer_density_ceiling** - Multiplier on the inner EdS-critical density for centerM>1 outer particles (default 1.0 = same density). Clipped to SimulationParameters.MAX_OUTER_DENSITY_CEILING = 2.0 with a UserWarning. No effect at centerM=1.

**Node-mass amplitude / seed** - `node_mass_amplitude` and `node_mass_seed` on SimulationParameters/ExternalNodeParameters. Make the 26 HMEA node masses log-normal: m_i = M_ext_kg·w_i/mean(w), w_i = exp(amplitude·g_i), g_i = default_rng(seed) standard normals. amplitude=0 (default) => all nodes uniform = M_ext_kg (backward compatible). MEAN-PRESERVING (mean == M_ext_kg exactly), so total external mass + Omega_Lambda_eff stay fixed (both linear in masses). CAVEAT: growth is NOT fully fixed — at strong tidal field (small S / large M) amplitude>0 nudges the realized growth factor up a few % (second-order, nonlinear a(t)); the seed selects shear/dipole ORIENTATION. For the Pantheon chi2 this makes amplitude degenerate with M/S, NOT an independent fit knob. Sweepable. See lode/physics/pantheon-comparison-results.md + force-calculations.md.

**Shear index / inertia-tensor spread** - Anisotropy diagnostic: eigenvalue spread of the particle cloud's inertia tensor (axis-dependent expansion). ~0 for an isotropic cloud; nonzero under anisotropic node masses. See lode/physics/anisotropy-diagnostic.md.

**Hubble dipole (ΔH/H)** - Directional expansion-rate asymmetry from a v·r̂-vs-r fit by hemisphere. Predicted dark-flow / Hubble-tension-scale signal of the External-Node Model; measured by cosmo/anisotropy.py. See lode/physics/anisotropy-diagnostic.md.

**Pre-start tidal boost** - `pre_start_tidal_boost` on SimulationParameters (default True). For M_ext>0 the cloud arrives at t_start with a radial velocity slightly ABOVE pure EdS Hubble flow because the HMEA tidal field has pulled on it from the Big Bang to t_start. Added on top of v=H_EdS*r as dv_r = g_r(t_start)·(3/5)·t_start_seconds, where g_r is the radial component of the SAME node-sum tidal accel the integrator uses. Vanishes as M_ext→0 (linear in node mass) AND only applied with external nodes on, so M=0==EdS is preserved exactly. NOT a fit-to-LCDM knob. Effect on growth is small (~0.35% at M=3000/S=30). Implemented in CosmologicalSimulation._apply_pre_start_tidal_boost. See lode/physics/initial-conditions.md.

**GRF / Zel'dovich** - Gaussian Random Field initial particle distribution (init_distribution="grf"): density field with approximate LCDM P(k), displaced by the Zel'dovich approximation (linear-order particle displacement from the density field). Deterministic per seed. Alternative to the default uniform_sphere. See lode/physics/realistic-initial-conditions.md.

**BBKS** - Bardeen-Bond-Kaiser-Szalay transfer function: the approximate LCDM matter transfer function shaping P(k) for the GRF init. See lode/physics/realistic-initial-conditions.md.

**Node geometry** - `node_geometry` on ExternalNodeParameters/SimulationParameters; selects the HMEA node layout. Volume-filling only: `cube26` (default, 26-node 3×3×3-1 shell), `cube_dense`, `fcc`, `bcc` (lattices, positions-only via `build_node_positions`), and `virialized` (coupled positions+masses). Hollow shells excluded. See lode/plans/node-geometries.md.

**Virialized geometry** - `node_geometry="virialized"`: the ONLY geometry returning COUPLED (positions, masses) already paired (mass i ↔ radius i). Models a relaxed cluster with MASS SEGREGATION (bigger node → larger radius). Reached via `build_virialized_grid(...)` or an HMEAGrid with this geometry (which takes the coupled branch in `_create_grid`); `build_node_positions("virialized")` RAISES (positions-only would lose the coupling). On this branch `node_masses()`/`node_mass_amplitude` are IGNORED (vir_mass_spread owns the mass distribution); `node_s_amplitude` still composes on top. Mean-preserving (mean==M_ext_kg), M=0==EdS preserved, seeded, NO PHYSICS_CACHE_VERSION bump (new `virializedgeo`+vir_* cache sub-slugs). See lode/plans/node-geometries.md.

**Mass segregation** - Coupling where more massive nodes sit FURTHER from the centre and smaller nodes cluster near it (as in a relaxed galaxy cluster). The defining property of the virialized geometry; tested as a POSITIVE mass-radius correlation. Falsified by `vir_segregation=0` (mass/radius decoupled). See lode/plans/node-geometries.md.

**vir_* parameters** - Virialized-grid knobs on ExternalNodeParameters/SimulationParameters/SweepConfig/CLI (consumed only when node_geometry=="virialized"): `vir_n_nodes` (exact node count, default 26), `vir_extent` (continuous radial-RANGE multiplier; radii span [0.5S,(0.5+extent)·S], range ratio 1+2·extent; default 1.0), `vir_mass_rule` ("radial" mass~f(r) default, or "massfunc" log-normal draw + segregate by rank), `vir_mass_spread` (mass-distribution amplitude; 0 → uniform = THE falsifiable knob; default 0.0), `vir_segregation` (mass↔radius coupling strength [0,1]; 0 → decoupled; default 1.0), `vir_s_metric` ("median" default or "mean" NN-spacing statistic targeted as S). Seeded by node_mass_seed. See lode/plans/node-geometries.md.

**Nearest-neighbour spacing (NN spacing)** - `nearest_neighbour_spacing(positions, metric)` in cosmo/node_geometry.py: for each node the distance to its CLOSEST other node, reduced by `metric` ("median" or "mean"). The virialized generator rescales its layout by a single global factor so the realized NN spacing equals the target S exactly — so S is the characteristic spacing per the chosen metric (this is also why vir_extent must widen the radial RANGE, since a pure global factor cancels under the rescale). See lode/plans/node-geometries.md, lode/physics/node-placement-vs-perturbation.md.

**vir_relax_steps** - Virialized-grid BALANCE LEVEL on ExternalNodeParameters/SimulationParameters/SweepConfig/CLI (DEFAULT 1). 0 = REALISTIC mode = the legacy Fibonacci-sphere segregated layout (NOT force-balanced; big-grid inner residual O(20-30)). >=1 = FORCE-BALANCED mode = an exact cubic-lattice ball with a node AT the origin and masses assigned by radius shell (antipodes share a mass) so the inner-node net force is at machine precision (~1e-30) for BOTH mass rules => satisfies the virialization criterion. NOT a count of iterative relaxation steps: a CONTINUOUS position relaxation provably cannot reach the tolerance on a finite canvas (irreducible central monopole), so the name is a balance LEVEL switch, not a step count. Cache sub-slug `{vir_relax_steps}vrx` (virialized geometry only). See lode/plans/node-geometries.md.

**Virialization residual** - `virialization_residual(positions, masses, *, inner_frac, center_mass_kg, G)` in cosmo/node_geometry.py: builds the per-INNER-node net gravitational acceleration (via `node_net_accelerations`, which mirrors the sim's tidal law incl. the 1e10 m floor, plus a central node of mass `center_mass_kg`=centerM) and divides by a characteristic single-neighbour pull `a_ref` => DIMENSIONLESS residual per inner node (`inner_frac` = innermost fraction by radius). Returns dict: residual_per_node, inner_idx, max_residual, median_residual, a_ref, n_inner. Criterion: inner nodes are "virialized" when max_residual <= VIRIALIZATION_TOL (=0.25; "inner net force under 25% of one neighbour's pull"). The user's physical criterion made measurable: "the inner nodes of a big enough virialized grid should not move; if they do, it's not virialized." See lode/physics/node-placement-vs-perturbation.md.

**Force-balanced lattice** - The default (vir_relax_steps>=1) virialized realization: a cubic-lattice ball, node at origin, masses by radius shell. Net force on inner nodes ~0 by lattice SYMMETRY (opposing pulls cancel). PHYSICS FINDING: a random mass-segregated blob CANNOT be force-balanced (the central monopole is irreducible); only lattice symmetry reaches ~0 net inner force. HMEA nodes are STATIC boundary conditions (a frozen virialized meta-structure), so an exact symmetric lattice is the right realization of "virialized". See lode/physics/node-placement-vs-perturbation.md.

**node_softening_gpc / node_softening_m** - Plummer NODE-softening length (Gpc on the param; derived `node_softening_m` in meters) on ExternalNodeParameters/SimulationParameters/SweepConfig/CLI, applied on the TIDAL force path (cosmo.tidal_forces_numba.calculate_tidal_forces_numba + the numpy fallback in HMEAGrid). DEFAULT 0.0 = the LEGACY hard `r<1e10 m` floor (~3e-13 Gpc, effectively no softening) = BYTE-IDENTICAL tidal force, so cube26 a(t) and all existing caches are unchanged and PHYSICS_CACHE_VERSION stays v3. >0.0 = `r_soft^2 = r^2 + (node_softening_gpc*Gpc_to_m)^2` (same Plummer convention as the internal particle force), which caps the close-pass node kick. Geometry-agnostic. Cache slug `{node_softening_gpc}nsoft` only when !=0.0. See "slingshot / taming" and lode/physics/slingshot-and-softening.md.

**Slingshot / taming** - SLINGSHOT = a particle gets a runaway displacement (heavy-tailed; max/median displacement ~514x on cube26 at M=1000/S=10) from a near-point-NODE close pass. ROOT CAUSE pinned by UT: with external nodes OFF the tail collapses to ~1.3x; it is the node close-pass, NOT particle-particle encounters; and n_steps / n_particles do NOT tame it (only node softening does). TAMING = node_softening_gpc>0 caps the close-pass kick. "DOUBLY TAMED" = the virialized (force-balanced) geometry already lowers the tail (~23x vs cube26's ~514x) AND node_softening_gpc=1.0 further collapses it (cube26 514->~2.8, virialized ~23->~7). Vanishes at M_ext=0 (M=0==EdS preserved). See lode/physics/slingshot-and-softening.md.

**start_size_scale** - Multiplier on the LCDM-implied INITIAL cloud size (applied in CosmologicalSimulation.__init__: `box_size_Gpc *= start_size_scale`) on SimulationParameters/SweepConfig/CLI. DEFAULT 1.0 = BYTE-IDENTICAL a(t)/positions/masses (no slug, no cache bump). A FALSIFIABLE initial-size/density lever, NOT a normalization offset: a(t) is an RMS RATIO so a uniform rescale cancels at M=0 (M=0==EdS holds at ANY size because the EdS-critical cloud mass scales with volume, keeping density critical), but at M_ext>0 the nodes keep their UNSCALED spacing S so a bigger/smaller cloud spans a different fraction of S => different differential tidal shear => the a(t) SHAPE moves (growth ~3.07/3.38/4.73 at scale 0.8/1.0/1.2 in one test cell). <=0 raises ValueError. Cache slug `{start_size_scale}ssz` only when !=1.0. See lode/physics/initial-conditions.md.
