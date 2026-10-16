# Terminology

## Model
- **HMEA** — Hyper-Massive External Attractor: ancient, EM-silent black-hole node beyond the horizon.
- **External-Node Model** — tides from HMEAs replace dark energy; Newtonian, flat space.
- **Progenitor Hypothesis** — the Big Bang was one typical node of the meta-structure destabilising.
- **Progenitor Node Model** — paper's name for External-Node Model + Progenitor Hypothesis.
- **Virialized meta-structure / medium** — periodic self-gravitating node medium at 2K/|U| ≈ 1 (`vir_relax_mode="medium"`).
- **"us" node** — node 0 of the medium relaxation (the progenitor); dropped from the HMEAs because in the sim it IS the cloud.
- **Traceless tides** — vacuum node rearrangements change shear, not isotropic expansion (to linear order).
- **Ω_Λ,eff** — G·M_ext/(S³H0²): order-of-magnitude label only.
- **Dark giant principle** — why HMEAs are unseen (no accretion, negligible Hawking radiation).
- **Great Metabolism** — speculative cyclic cosmology in the paper; untested by code.

## Parameters
- **M_value / M** — mean node mass in observable-universe masses (M_obs = 1e53 kg); `M_ext_kg` in SI.
- **S / S_value** — node nearest-neighbour spacing, Gpc (always quoted in absolute Gpc).
- **centerM** (`center_node_mass`) — progenitor mass / total matter in M_obs; outer shell post-BB.
- **vcm** (`vir_center_mass_frac`) — "us" node mass in mean-node units; = centerM / M_value.
- **σ** (`vir_mass_spread`) — width of the log-normal node-mass function (production 0.5).
- **node_mass_amplitude / node_s_amplitude / node_mass_seed** — mean-preserving log-normal node mass / radial-position perturbations and their seed.
- **node_softening_gpc / node_force_law** — Plummer node softening; `plummer` or `bounded` close-range law.
- **node_substep_threshold / node_substeps** — adaptive KDK sub-stepping near nodes.
- **force_method** — internal gravity: auto (BH ≥ 1000, numba_direct ≥ 100, direct), or explicit incl. `cuda_direct`.
- **start_size_scale** — initial cloud size multiplier (shape lever at M > 0).
- **outer_particle_cap / outer_density_ceiling** — cap on outer-shell tracers (heavier ones) / outer density multiplier.
- **init_distribution** — `uniform_sphere`, `grf` (Zel'dovich-displaced GRF), `grfmass` (GRF contrast in masses); `grf_support` sphere/box.
- **BBKS** — ΛCDM transfer function shaping the GRF power spectrum.
- **mass_randomize** — uniform particle-mass scatter (0 in sweeps).

## Simulation and measurement
- **EdS** — Einstein–de Sitter, Ω_m = 1: the no-dark-energy null (χ²/dof 0.843).
- **PF1 / M=0 == EdS** — the invariant: zero node mass reproduces EdS exactly.
- **EdS-consistent ICs** — v = H_EdS r with H_EdS = 2/(3 t_start) and cloud mass = critical mass.
- **Pre-start tidal boost** — dv_r = g_r(t_start)·(3/5)·t_start: the tides' pull before t_start.
- **Legacy calibration** — old velocity rescaling toward ΛCDM (off by default; pre-2026-06-24 results).
- **a(t) / scale_factor** — inner-cloud RMS radius ratio rms(t)/rms(0).
- **Observable mask** — boolean marking inner (observable) particles; a(t) uses only them.
- **Box size** — initial cloud DIAMETER (RMS radius = box/2).
- **Softening** — internal-gravity ε = 1 Gpc × (m̄/M_obs)^{1/3} (small-N boost).
- **KDK / force reuse** — kick-drift-kick leapfrog; one force evaluation per step.
- **Growth / growth anchor** — a(end)/a(0) must be within 20% of ΛCDM's (≈3.30 at t_start 2.9).
- **Runaway** — growth far above the anchor (node pull wins); rejected by the anchor.
- **Slingshot** — heavy-tailed particle kick from a close pass to a near-point node.
- **Knot / knot ratio** — central-core collapse; median radius of final-core particles, last/first (< 0.9 collapsing).

## Scoring
- **Pantheon+** — Pantheon+SH0ES SN Ia compilation (z ≥ 0.01, no calibrators, diagonal errors).
- **μ(z)** — distance modulus from D_C = c∫dt/a; **offset marginalization** fits ΔM analytically.
- **chi2_dof** — χ²/(n−1) on the in-range SNe (`n_sne_used`); ΛCDM reference 0.436.
- **Best observer** — the particle vantage whose μ(z) fits best: the headline (we are not central).
- **Centre observer** — COM-based a(t); `center_chi2_dof`, context only.
- **frac_below_lcdm / frac_below_eds** — share of observers beating the reference (genericity).
- **observer_k** — neighbours per observer; −1 = whole cloud (the only well-sampled choice).
- **local_rms / hubble_flow** — the two observer a(t) definitions.
- **Universe-like (composite criterion)** — best_obs ≤ ~0.46 + anchor-ok + knot ≥ 0.9 + no HMEA within 15 Gpc.
- **Knot guard** (`s_knot_guard`) — co-fit moves S up to avoid a knot within a χ² budget.
- **Shear index / Hubble dipole / bulk drift** — anisotropy diagnostics (the discriminating signal).
- **M/S³ degeneracy** — isotropic fit constrains only the effective expansion.

## Sweep machinery
- **Cell** — one parameter combination; **arm** — one config file; **campaign** — a family (e.g. cmx2 = guarded 2k, cmx3 = 8k, cmx4 = 20k).
- **Co-fit** — per-cell search of S (ternary in production).
- **keyed == run** — every value in the cache key reaches the sim and vice versa.
- **Slug** — key token added only for a non-default knob (e.g. `1.0nsoft`).
- **PHYSICS_CACHE_VERSION** — manual token (v3) bumped when default physics changes.
- **Hero run** — very high-N showcase sim saved as `results/hero/*.npz`.
- **PF##** — pinned finding ID (stable; cited in code), see findings/summary.md.
- **WS1–WS7** — historical workstream labels in code comments (WS7 = GPU/HF, still open).
