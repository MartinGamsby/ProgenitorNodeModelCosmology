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

**Softening** - Minimum gravitational interaction distance preventing singularities. Default: 1 Mpc.

**Matter-only** - Cosmology with Ω_Λ=0 (no dark energy, no external nodes). Pure matter deceleration.

**Observable universe** - Region within particle horizon. Modeled as N tracer particles.

**Friedmann equation** - Differential equation governing cosmic expansion: ȧ = H₀√(Ω_m/a³ + Ω_Λ)×a

**RMS radius** - Root-mean-square distance of particles from center of mass. Proxy for universe size.

**center_node_mass** - Central progenitor node mass as multiple of M_observable_kg. Default 1.0. Controls total_mass_kg for particle system and softening_m scaling in CosmologicalSimulation. Larger values model more massive central structures.

**Node-mass amplitude / seed** - `node_mass_amplitude` and `node_mass_seed` on SimulationParameters/ExternalNodeParameters. Make the 26 HMEA node masses log-normal: m_i = M_ext_kg·w_i/mean(w), w_i = exp(amplitude·g_i), g_i = default_rng(seed) standard normals. amplitude=0 (default) => all nodes uniform = M_ext_kg (backward compatible). MEAN-PRESERVING (mean == M_ext_kg exactly), so total external mass + Omega_Lambda_eff stay fixed (both linear in masses). CAVEAT: growth is NOT fully fixed — at strong tidal field (small S / large M) amplitude>0 nudges the realized growth factor up a few % (second-order, nonlinear a(t)); the seed selects shear/dipole ORIENTATION. For the Pantheon chi2 this makes amplitude degenerate with M/S, NOT an independent fit knob. Sweepable. See lode/physics/pantheon-comparison-results.md + force-calculations.md.

**Shear index / inertia-tensor spread** - Anisotropy diagnostic: eigenvalue spread of the particle cloud's inertia tensor (axis-dependent expansion). ~0 for an isotropic cloud; nonzero under anisotropic node masses. See lode/physics/anisotropy-diagnostic.md.

**Hubble dipole (ΔH/H)** - Directional expansion-rate asymmetry from a v·r̂-vs-r fit by hemisphere. Predicted dark-flow / Hubble-tension-scale signal of the External-Node Model; measured by cosmo/anisotropy.py. See lode/physics/anisotropy-diagnostic.md.

**Pre-start tidal boost** - `pre_start_tidal_boost` on SimulationParameters (default True). For M_ext>0 the cloud arrives at t_start with a radial velocity slightly ABOVE pure EdS Hubble flow because the HMEA tidal field has pulled on it from the Big Bang to t_start. Added on top of v=H_EdS*r as dv_r = g_r(t_start)·(3/5)·t_start_seconds, where g_r is the radial component of the SAME node-sum tidal accel the integrator uses. Vanishes as M_ext→0 (linear in node mass) AND only applied with external nodes on, so M=0==EdS is preserved exactly. NOT a fit-to-LCDM knob. Effect on growth is small (~0.35% at M=3000/S=30). Implemented in CosmologicalSimulation._apply_pre_start_tidal_boost. See lode/physics/initial-conditions.md.

**GRF / Zel'dovich** - Gaussian Random Field initial particle distribution (init_distribution="grf"): density field with approximate LCDM P(k), displaced by the Zel'dovich approximation (linear-order particle displacement from the density field). Deterministic per seed. Alternative to the default uniform_sphere. See lode/physics/realistic-initial-conditions.md.

**BBKS** - Bardeen-Bond-Kaiser-Szalay transfer function: the approximate LCDM matter transfer function shaping P(k) for the GRF init. See lode/physics/realistic-initial-conditions.md.
