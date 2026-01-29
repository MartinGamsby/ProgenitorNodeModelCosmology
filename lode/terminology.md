# Terminology

**HMEA** - Hyper-Massive External Attractors. Massive structures beyond observable universe providing tidal forces.

**External-Node Model** - Alternative cosmology replacing dark energy with classical gravity from HMEA nodes.

**ΛCDM** - Lambda Cold Dark Matter. Standard cosmology with dark energy (cosmological constant Λ).

**M_value** - External node mass as multiple of observable universe mass. Parameter being explored; M≈800 gives 99.4% ΛCDM match.

**S_value** - HMEA grid spacing in Gpc. Parameter being explored; S≈24 Gpc gives 99.4% ΛCDM match.

**Ω_Λ_eff** - Effective dark energy density from tidal forces. Ω_Λ_eff = G×M_ext/(S³×H₀²).

**Tidal acceleration** - Force from external nodes: a = GM_ext × r / |r_node - r|³

**Damping factor** - Coefficient multiplying initial Hubble flow velocities. Compensates for no ongoing Hubble drag in matter-only evolution. Auto-calculated from deceleration parameter (d=0.4-0.25×q, clipped to 0.1-0.7) unless explicitly overridden. Empirical best-fit: 0.91.

**Deceleration parameter (q)** - Measure of expansion deceleration: q = 0.5×Ω_m(a)/[Ω_m(a)+Ω_Λ] - 1.0. Positive q means deceleration (matter-dominated), negative q means acceleration (Λ-dominated). Used to auto-calculate damping factor.

**Progenitor Node** - Hypothesized source of Big Bang: a node in virialized meta-structure that destabilized, explaining isotropy without fine-tuning. (Draft theory in paper.tex)

**Virialized Meta-Structure** - Hypothesized static, gravitationally balanced lattice of HMEAs beyond observable universe. Analogous to relaxed galaxy cluster. (Draft theory in paper.tex)

**Great Metabolism Hypothesis** - Speculative cyclical cosmology where universes are transient accretion events feeding meta-structure nodes. (Draft theory in paper.tex)

**Toy Model** - Simplified proof-of-concept. Addresses late-time acceleration only; doesn't address CMB, BAO, structure formation, inflation, or early universe. Code supports draft paper development.

**Draft Paper** - docs/paper.tex. Working document describing External-Node Model theoretical framework. Under development, not published.

**Scale factor (a)** - Cosmic expansion parameter. a=1 at present, a<1 in past, a>1 in future.

**Hubble parameter (H)** - Expansion rate H = ȧ/a. Current: H₀ ≈ 70 km/s/Mpc.

**Leapfrog** - Symplectic integrator using Kick-Drift-Kick algorithm. Energy-conserving.

**Softening** - Minimum gravitational interaction distance preventing singularities. Default: 1 Mpc.

**Matter-only** - Cosmology with Ω_Λ=0 (no dark energy, no external nodes). Pure matter deceleration.

**Observable universe** - Region within particle horizon. Modeled as N tracer particles.

**Friedmann equation** - Differential equation governing cosmic expansion: ȧ = H₀√(Ω_m/a³ + Ω_Λ)×a

**RMS radius** - Root-mean-square distance of particles from center of mass. Proxy for universe size.

**center_node_mass** - Central progenitor node mass as multiple of M_observable_kg. Default 1.0. Controls total_mass_kg for particle system and softening_m scaling in CosmologicalSimulation. Larger values model more massive central structures.
