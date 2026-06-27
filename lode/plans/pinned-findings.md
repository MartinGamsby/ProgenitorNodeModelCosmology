# Pinned Findings (anchors for the deeper-exploration phase)

Back to [deeper-exploration-roadmap.md](./deeper-exploration-roadmap.md).

The verified, committed results every downstream workstream must respect. These are
the load-bearing conclusions — do NOT re-derive or contradict them without evidence.
Numbers flagged "to be re-pinned" are the ones the deeper sweep (WS1) exists to nail
down with finer grids + graphs.

## PF1 — M=0 == Einstein-de Sitter (THE invariant)

With external node mass M=0 the sim reproduces matter-only / EdS expansion BY
CONSTRUCTION (self-consistent ICs: v=H_EdS·r with H_EdS=2/(3 t_start) + cloud mass =
EdS critical mass). Validated to ~0.2-0.3% growth, ~0.02 mag in mu(z); the mu(z) sits
ON the EdS null and ~10x farther from LCDM (correct ordering). t_start-independent.
The pre-start tidal boost VANISHES at M=0, so M=0==EdS holds with the boost ON or OFF.
Source of truth: [../physics/initial-conditions.md](../physics/initial-conditions.md).
**Every new geometry / knob / centerM rework MUST preserve this.**

## PF2 — Anisotropy is the discriminating, falsifiable signal

The external vacuum masses are TRACELESS for any arrangement, so symmetry-breaking
(node_mass_amplitude, node_s_amplitude) drives strong shear / Hubble-dipole (~10x) but
does NOT improve the isotropic Hubble-diagram fit. The lever experiment confirmed this
empirically. The model's distinguishing observable is the ANISOTROPY tensor
(shear_index, ΔH/H), NOT the isotropic mu(z) chi2. Measured by `cosmo/anisotropy.py` /
`anisotropy_report.py`. Source: [../physics/anisotropy-diagnostic.md](../physics/anisotropy-diagnostic.md),
[../physics/force-calculations.md](../physics/force-calculations.md).
**Implication for WS3 (geometries): a new geometry can change the shear/dipole pattern
and can change the isotropic effect only via the same 2nd-order growth coupling — judge
geometries primarily on the anisotropy story and the net-effect honesty, not on a
hoped-for isotropic-chi2 win.**

## PF3 — Isotropic M/S³ degeneracy

SNe constrain the effective expansion (~effective Ω_Λ), not M and S separately. Once
each M is paired with its growth-anchored S, chi2/dof is essentially FLAT across M from
~20 to ~200000. So a single "best M" is not meaningful for the isotropic fit; report
the BEST ISOTROPIC config and the degeneracy band, not a point estimate. node_mass_
amplitude is degenerate with M/S (a growth nudge), not an independent isotropic fit knob.
Source: [../scripts/parameter-sweep.md](../scripts/parameter-sweep.md) Stage-3 table,
[../physics/pantheon-comparison-results.md](../physics/pantheon-comparison-results.md).

## PF4 — Corrected framing: near-LCDM, far-from-null (NOT a failure)

**RE-PINNED from WS1 targeted_near_lcdm sweep (2026-06-25), 400p/273 steps, t_start=2.9,
one consistent kernel/anchor.**

Honest isotropic (amp=0) result at the best S per M:
- Best isotropic chi2/dof ≈ 0.52 (M=200/S=20, M=1500/S=30, degeneracy band ~0.52-0.54)
- LCDM reference: 0.436
- EdS null: 0.843

So the model sits between LCDM and EdS at ~0.52, roughly halfway in log-chi2 space.
It lands closer to LCDM than to EdS, confirming genuine effective dark energy.
It does NOT match LCDM (gap ~0.08 chi2/dof units is not noise).

With node_mass_amplitude=0.5 (symmetry-breaking amplitude), best chi2/dof ≈ 0.487
(M=1500/S=55) — a modest improvement driven by the amplitude growth nudge (PF2/PF3).

Particle-count stability at low-S best configs (amp=0):
- M=200/S=20: 400p→0.52, 1000p→0.53, 2000p→0.53 (stable within 0.01)
- M=1500/S=30: 400p→0.52, 1000p→0.54, 2000p→0.53 (stable within 0.01)
- CONCLUSION: low-S near-LCDM band is NOT sensitive to N=400→2000 at these configs.

The ~0.50 chi2 cited in prior Lode entries came from cache entries with a different
key format (n_sne_used=1339 vs 1425, different Pantheon cut or amplitude runs), not from
current amp=0 runs on the same kernel. They should NOT be used as isotropic baselines.

Near-LCDM band (amp=0): M=200-3000 at S=20-35 gives chi2/dof 0.52-0.54 (flat band).
S must be LOW (20-35 Gpc); high-S (80-90) gives 0.69 (wrong part of landscape).

The M/S^3 degeneracy is confirmed: chi2/dof is essentially flat across M when S is
co-varied to maintain constant effective Ω_Λ. Report "near-LCDM isotropic band ~0.52-0.54"
not a single "best M" number.

**Runaway boundary** (PF5): S_crit ∝ M^(1/3). At M=3000, all S≤45 amp=0.5 runaway.
At M=1500, S=20-25 runaway for both amp=0 and amp=0.5. The boundary is real.

GRF-vs-uniform sensitivity (at M=1500/S=30): uniform_sphere chi2~0.526; the swing is now
RESOLVED (PF12) — grf-sphere (fixed default) 0.84, grf-box (legacy cube) 1.14. The
sphere-support fix cuts ~half the swing; the residual Δ~+0.32 is real clustering physics.
Do NOT claim 0.52 for GRF init (it reads ~0.84 in this strong-field cell).

## PF5 — Runaway boundary is real physics

Small-S configs run away (nearest-node (S−R)⁻² attraction overwhelms bound expansion);
boundary scales roughly S_crit ∝ M^(1/3) (constant Ω_Λ_eff contour). The growth anchor
(GROWTH_ANCHOR_TOL=0.20) rejects runaway configs in the sweep. Non-monotonic: matching
total growth with the WRONG a(t) SHAPE is penalized by SNe, so the best SHAPE is at
INTERMEDIATE strength (undershooting the anchor), not at the strongest bound config.
Source: [../physics/initial-conditions.md](../physics/initial-conditions.md) runaway map.
**WS1 must map this boundary as a figure (runaway-boundary map) so every claimed config
is visibly on the bound side.**

## PF6 — WS4 centerM (outer mass) barely moves the isotropic fit (honest result)

centerM is now the OUTER-MASS multiplier (extra Big-Bang matter OUTSIDE the inner
observable sphere; a(t) measured on the inner region only; softening frozen). Source:
[../physics/observable-mask-and-outer-mass.md](../physics/observable-mask-and-outer-mass.md).

Reduced sweep (M∈{1,2,5,20}, centerM∈{1.0,1.5,2.0,3.0}, S co-fit 18–40 Gpc, 400p,
273 steps, t_start=2.9). Best chi2/dof vs Pantheon+ per centerM (all at M=20, S=39):

| centerM | best chi2/dof |
|---------|---------------|
| 1.0     | 0.6921        |
| 1.5     | 0.7030        |
| 2.0     | **0.6801** (best) |
| 3.0     | 0.6844        |
| LCDM ref| 0.436         |
| EdS ref | 0.843         |

- Outer matter helped only MARGINALLY (~0.012 chi2/dof, centerM=1→2). Outer mass
  alone did NOT move the isotropic fit toward LCDM.
- The small-M (M=1–2) hypothesis corner produced NO anchor_ok rows (too weak to reach
  the growth anchor) — the small-M + outer-mass hypothesis is NOT supported by this grid.
- **CAVEAT — not the global best.** This reduced grid OMITTED the prior known corner
  ~M=50/S=20 (~0.52, see PF4). So "best=0.68" is the best of the SMALL-M hypothesis
  grid, not the model's global best; a fuller sweep including M~50/S~20 is still OPEN.
  Do NOT overstate — and do not use 0.68 as the model's isotropic baseline (PF4's
  ~0.52 low-S band remains the near-LCDM headline).

## PF7 — Node PLACEMENT, not perturbation; multi-layer "S" is not comparable

Tidal stretch ~1/r³ is near-field dominated. The dense multi-layer lattices (fcc/bcc)
park MOST of their nodes 4–6× beyond the ~14 Gpc horizon (verified: fcc out to ~86 Gpc,
bcc ~386 nodes out to ~83 Gpc at S=20), where they are effectively inert. So a single
parameter "S" is NOT comparable across geometries; `cube26` is the clean single-shell
case where S means what it says. This is the placement reason no volume-filling lattice
beats the cube on the isotropic fit (consistent with PF2/PF3), and the motivation for the
`virialized` geometry (exact node count + controlled radial extent + NN-spacing-pinned S
+ mass segregation).

HONEST corollary (section-1 generalization, `tests/test_node_geometry_anisotropy.py`,
128 tests): the per-node `node_mass_amplitude` / `node_s_amplitude` PERTURBATION machinery
is CORRECT on EVERY geometry — mean-preserving for any N, ray-preserving, seeded,
separate-RNG, global-RNG-independent. **There is NO multi-layer perturbation bug.** What
differs between geometries is the unperturbed PLACEMENT (and hence the 1/r³-weighted
near-field), not how the perturbation acts. Source:
[../physics/node-placement-vs-perturbation.md](../physics/node-placement-vs-perturbation.md),
[node-geometries.md](./node-geometries.md).

## PF8 — Only the analytic lattice reaches center force-balance; iterative relaxation (Option B) does NOT

RE-GROUNDED on the CENTER-ONLY metric of a LARGE grid + REAL Option-B numbers (the
disputed "irreducible central monopole" hand-wave is DROPPED; the conclusion is now
empirical, not asserted).

The user's criterion ("the DEEP-INTERIOR nodes of a big-enough virialized grid should
not move") is measurable via `virialization_residual`, now with a CENTER-ONLY selector:
`center_k` (the K nodes closest to the centroid, size-INDEPENDENT so a bigger grid
genuinely deepens the interior) or `center_frac`, in addition to the legacy `inner_frac`
(byte-identical default 0.5). The residual is DIMENSIONLESS = |net node accel| / a_ref
(a_ref = one characteristic neighbour pull, `reference="mean_pairwise"`); criterion
`max_residual <= VIRIALIZATION_TOL = 0.25` — NOT 25% of total force (the user asked).

MEASURED at the deep center (S=30 Gpc, both mass rules, grid sizes n=26/100/500):
- **Option A (analytic lattice, `vir_relax_mode="lattice"`, DEFAULT):** center max
  residual ~1e-28..1e-31 at EVERY size → PASSES TOL=0.25 (opposing lattice pulls cancel).
- **Realistic un-relaxed (Fibonacci segregated):** residual 25..275, GROWING with grid
  size → fails.
- **Option B (TRUE iterative relaxation, `vir_relax_mode="gradient"`):** genuinely
  descends the force-residual objective f=Σ|a_i|² (analytic gradient validated vs finite
  diff ~3e-6, monotone backtracking) — radial n=500: 275→**95**; massfunc n=500: 96→**32**.
  It REDUCES the residual but BOTTOMS OUT ~O(100×) ABOVE TOL.

FINDING: Option B does NOT reach center force-balance at the deep interior of even the
largest grid, for BOTH mass rules — a realistic blob relaxed by position descent gets
~95/32 vs the lattice's ~1e-29. So the analytic lattice is the one perfectly-balanced
realization. HMEA nodes are STATIC boundary conditions (a frozen virialized
meta-structure), so the symmetric lattice is the physically correct realization of
"virialized". `vir_relax_mode` selects A (lattice) vs B (gradient), and `vir_relax_steps`
is the gradient step COUNT in mode B (a balance LEVEL in mode A). Source:
[../physics/node-placement-vs-perturbation.md](../physics/node-placement-vs-perturbation.md),
[node-geometries.md](./node-geometries.md); tests `test_virialization_validation.py`,
`test_virialized_grid.py`, `test_virialization_figs.py`.

## PF9 — Slingshot cause is the node close-pass; node softening is the only lever

The runaway particle "slingshot" (heavy-tailed displacement, max/median ~514x on cube26
at M=1000/S=10) is caused by a particle's close pass to a near-point HMEA NODE (the
unsoftened 1/r^3 tidal force diverges), NOT by particle-particle encounters: turning the
external nodes OFF collapses the tail to ~1.3x (UT-pinned). n_steps and n_particles do
NOT tame it. The fix is `node_softening_gpc` (Plummer node softening on the tidal path):
default 0.0 = legacy hard floor = BYTE-IDENTICAL (no PHYSICS_CACHE_VERSION bump);
=1.0 caps the close-pass kick. "DOUBLY TAMED": the force-balanced virialized geometry
already lowers the tail (~23x vs ~514x) AND softening collapses it further (cube26
514→~2.8, virialized ~23→~7). Vanishes at M_ext=0 (PF1 preserved); far-field < 5%
change.

CLOSE-ENCOUNTER LAW COMPARISON (Section 4, all opt-in / default OFF / byte-identical, no
PHYSICS_CACHE_VERSION bump): two alternatives to the blunt Plummer floor were built and
measured on a deliberately runaway config (cube26, M=1000/S=10, N=300, 273 steps),
chi2/dof from the authoritative `compute_pantheon_metrics` (max/median displacement,
growth, chi2/dof):

| close-range treatment | max | median | growth | chi2/dof |
|-----------------------|-----|--------|--------|----------|
| legacy hard floor | 844× | 6927 | — | inf |
| Plummer 1 Gpc | 2.8× | 3.04 | — | 29.0 |
| bounded "can't cross midpoint" 1 Gpc | 6.7× | 5.23 | — | inf |
| adaptive KDK substep alone | 93× | 1559 | — | inf |
| bounded + substep | 2.4× | 3.38 | in-anchor | **48.7** |

`node_force_law="bounded"` CAPS the per-node accel at its value at the softening radius
(G·m/soft²) below the softening length (distinct from Plummer softening force → 0);
adaptive substep (`node_substep_threshold`/`node_substeps`, KDK subdivision near a node)
refines dt during close passes. VERDICT: substep ALONE does NOT tame the tail
(re-confirms the "more steps won't help" part of this PF); the bounded law tames it and
pulls growth toward LCDM; bounded+substep is the only cube26 combo landing growth in the
anchor window with finite chi2. The VIRIALIZED runaway at this extreme config is
GEOMETRIC (mass spread, no single near-point node) and is NOT tamed by any law — an
honest negative result. The mostly-inf chi2 are the growth anchor correctly rejecting
deliberately extreme runaways. Source:
[../physics/slingshot-and-softening.md](../physics/slingshot-and-softening.md);
tests `tests/test_slingshot.py`.

## PF10 — start_size_scale is a REAL a(t)-shape lever, not a normalization offset

`start_size_scale` multiplies the initial cloud size. It is NOT a divided-out offset:
a(t) is an RMS RATIO so a uniform rescale cancels at M=0 (M=0==EdS holds at ANY size
because the EdS-critical cloud mass scales with volume, keeping density critical), but at
M_ext>0 the nodes keep their UNSCALED spacing S, so a different-size cloud spans a
different fraction of S ⇒ different differential tidal shear ⇒ the a(t) SHAPE moves
(total growth ~3.07/3.38/4.73 at scale 0.8/1.0/1.2 in one test cell). Default 1.0 is
byte-identical (no slug, no cache bump). It is the density/size counterpart to M/S — a
way to vary the tidal-to-self-gravity ratio without changing M or S. Source:
[../physics/initial-conditions.md](../physics/initial-conditions.md); tests
`tests/test_start_size.py`.

## PF11 — The figure↔CSV chi2 conflict was a real keyed-but-not-run bug; ONE authoritative chi2 now

The reported 0.903 (CSV) vs ~0.52 (figure) for `virialized_final` was NOT a normalization
nuance — it was a BUG in the FIGURE path. `_generate_mu_z_panel` (sweep.py) hand-rolled a
`SimulationParameters` that OMITTED `node_geometry`, `geometry_kwargs`, all `vir_*`,
`node_softening_gpc`, and `start_size_scale`, so the figure re-ran a cube26/no-softening
sim (a DIFFERENT a(t) → ~0.52) while the CSV held the true virialized score (~0.90). The
classic keyed-but-not-run bug, in the figure path.

FIX: a single source of truth — `_build_sim_params(sweep_cfg, M, S, centerM, seed)` is now
called by BOTH the sim-callback and the panel; the panel rebuilds the cell from the CSV row
(`_cell_from_best_row` → `_make_sweep_config_for_cell` → `_build_sim_params`), the IDENTICAL
machinery the real run uses, so the panel can never drift from the cell.
`_emit_chi2_reconciliation` prints/writes CSV chi2_dof (AUTHORITATIVE) vs figure-recomputed
chi2_dof + |diff|, requiring <0.01. Verified end-to-end on the real virialized_final knobs:
CSV = figure = 0.824771, |diff| = 0.000000. No physics changed → no PHYSICS_CACHE_VERSION
bump; cube26-default figures byte-identical. CONSEQUENCE: every chi2 quoted before this fix
for a NON-default-geometry cell must use the CSV (authoritative) value, not the figure
annotation. Source: [../scripts/parameter-sweep.md](../scripts/parameter-sweep.md),
[overarching-sweep.md](./overarching-sweep.md); tests
`tests/test_overarching_sweep.py::TestMuZPanelParamsMatchSim`.

## PF12 — GRF is NOT broken; the swing was half a cube-vs-sphere setup bug + half real clustering

The GRF density field is healthy (mean δ~5e-22, no NaN/Inf, P(k) decays large→small scale,
Zel'dovich displacement RMS-controlled to 0.5 cell, RMS-norm exact, COM~0). The chi2 swing
was real and decomposed (authoritative `compute_pantheon_metrics`, 3 seeds, N∈{1000,2000,4000})
into H4 (setup, ~half) + H1 (real physics, residual) — NOT near-runaway (growth ~2.76 vs
anchor 3.30), NOT N-noise (seed spread ~0.01-0.04).

ROOT CAUSE (H4): the GRF sampler filled a CUBE (linspace³ Lagrangian grid), not a sphere, so
the cloud had a fat radial tail (~11% of particles beyond 1.3R vs uniform's 0.1%); those
corner particles changed the bulk a(t) in the strong tidal field. FIX: `sample_grf(support=
"sphere")` (new DEFAULT) masks the Lagrangian grid to the same `(box/2)/sqrt(3/5)` radius
uniform_sphere uses, so the ONLY remaining difference vs uniform is clustering. Legacy
`support="box"` kept for comparison; uniform_sphere byte-identical; GRF entries are separate
cache keys (no PHYSICS_CACHE_VERSION bump).

KEY NUMBERS (chi2/dof): swing cell M=1500/S=30 — uniform 0.526, grf-sphere (fixed) 0.84
(Δ=+0.32 = genuine clustering), grf-box (legacy) 1.14 (Δ=+0.61): the sphere fix cuts ~half
the swing; the residual +0.32 is REAL clustering physics (flat across N). Weak cell
M=100/S=60 — uniform 0.61, grf-sphere 0.61 (Δ=+0.002): GRF == uniform in the isotropic
regime, confirming the clustering-insensitive expectation there. M=0==EdS holds for GRF.
**A Section-7 GRF row reads ~0.84 (not ~0.53) in the strong-field band because that residual
is real clustering, not a bug — do NOT claim 0.52 for GRF init.** Source:
[../physics/realistic-initial-conditions.md](../physics/realistic-initial-conditions.md),
[grf-vs-uniform.md](./grf-vs-uniform.md); tests `tests/test_realistic_init.py` (GRF field
stats, support, GRF-EdS).

## PF13 — Observer-from-a-particle: best observer is a cherry-pick, NOT a model win (but reveals anisotropy spread)

The user's hypothesis ("we're not in the centre — compute mu(z) from EACH particle and take
the best") was prototyped (`cosmo/observer_distance.py`, two observer defs: `local_rms` k-NN
RMS growth, `hubble_flow` local-Hubble integral; centre-observer limit reproduces the centre
a(t) exactly; scored with the SAME authoritative chi2). It is a PURE ADD-ON off the default
path and is NOT pinned as a fit improvement.

NUMBERS (M=1000, S=30, t_start=2.9, N=120, real Pantheon+): cube26 — centre 0.515, BEST
observer 0.436 (= LCDM floor), MEDIAN 0.535, p10/p90 0.45/0.74. virialized — centre 15.22
(centre runs away here, growth ~32× vs physical ~3.2×), best observer 0.90-0.93, MEDIAN
8.7-9.6, p10/p90 ~5/42.

VERDICT: the BEST observer beats the centre in every config (cube26 +15%, virialized +94%),
but this is the MIN over 120 observers — a SELECTION EFFECT, not a model win. The fair
statistic is the DISTRIBUTION, and the MEDIAN is WORSE than the centre. So "take the best
particle" does NOT honestly improve the fit. What it DOES reveal is a large per-observer
SPREAD = a PF2-style anisotropy signal (off-centre observers infer materially different
expansions). Prototype scale N=120. Source:
[../physics/observer-from-particle.md](../physics/observer-from-particle.md); tests
`tests/test_observer_distance.py`.

## PF14 — vir_extent can drive node count (opt-in) to keep the ball density constant

`vir_extent_couples_nodes` (default False = byte-identical) makes `vir_extent` DRIVE the
virialized node count to hold the ball density constant — which also makes `vir_extent`
meaningful again in the force-balanced lattice mode (where it was a NO-OP, since the lattice
ball radius derives from node count). Density law: volume-filling ball (ρ=N/V, V=4/3·π·R³);
realized reach R grows ~linearly with extent, so holding ρ constant under R~extent requires
N~extent³ (`extent_coupled_n_nodes(N0,extent)=max(1,round(N0·extent³))`). Reference extent
1.0 → factor 1.0 → node count UNCHANGED even when the flag is on.

NUMBERS (base N0=64, lattice mode, coupling ON): extent 1.0→n 64 (reach/S 2.45, density 4.35,
center residual 2.4e-07); 1.5→n 216 (reach/S 3.74, density 4.12); 2.0→n 512 (reach/S 5.10,
density 3.86); 3.0→n 1728 (reach/S 7.81, density 3.63). Node count rises as extent³, reach
grows (knob now meaningful), NN spacing stays == S, density holds in a ~3.6-4.4 band (vs a
~27× collapse a fixed-count grid would suffer over 1→3), center stays force-balanced
(residual << 0.25). Default OFF byte-identical (no slug); ON at extent=1 is a byte-identical
no-op; keyed==run through the sweep path. Source:
[node-geometries.md](./node-geometries.md); tests
`tests/test_virialized_grid.py::TestExtentNodeCoupling`,
`tests/test_overarching_sweep.py`.

## PF-PENDING — "virialized fits worse/better than cube26": NOT YET PINNED (awaiting the comparison_v2 sweep)

The claim that the virialized grid "fits worse" than cube26 is UNVERIFIED and must stay so
until the detached `sweeps/comparison_v2/` sweep runs cube26 as a control AT THE SAME
softening / force-law / start-size in the SAME sweep (item 9 attribution). The config + the
detached, resumable launcher (`launch_sweep_detached.ps1`, Start-Process, 19 arms isolating
one variable each, 222 cells) are BUILT and the keyed==run wiring is verified (Section 7
threaded four previously keyed-but-not-run axes: `node_force_law`, `node_substep_threshold`,
`node_substeps`, and especially `vir_relax_mode` — Option B was ENTIRELY UNREACHABLE from any
sweep config before this). But the multi-day RESULTS are a flagged follow-on. Until that
sweep COMPLETES, do NOT pin: cube26-vs-virialized attribution, the low/fine M/S landscape,
start-size/co-fit/convergence results, or any final headline chi2 band for virialized. Smoke
(4 cells) confirmed the pipeline (best cube26 M=100/S=19 chi2/dof 0.4993, anchor_ok; chi2
reconciliation passed) but is NOT a result. Source:
[overarching-sweep.md](./overarching-sweep.md),
[../scripts/parameter-sweep.md](../scripts/parameter-sweep.md).

## What is NOT yet pinned (the job of this phase)

- The cube26-vs-virialized attribution and the final headline chi2 band for virialized
  (PENDING the detached `comparison_v2` sweep — see PF-PENDING above). The chi2 DEFINITION
  is now authoritative (PF11), but the multi-day comparison numbers are not in yet.
- Whether ANY node geometry yields a larger net isotropic effect (WS3).
- Whether all conclusions survive high N (WS6) — the comparison_v2 convergence ladder
  (1000p/2000p/4000p) addresses this once it runs.
- A fuller WS4 centerM sweep INCLUDING the M~50/S~20 corner (the reduced grid that
  produced PF6 omitted it). centerM-as-outer-mass semantics are now IMPLEMENTED (PF6);
  whether outer mass helps at the model's actual best-fit corner is still open.

(RESOLVED since the last edit: the GRF chi2 swing is now attributed — PF12, the cube-vs-
sphere setup bug + real clustering; and the figure↔CSV chi2 conflict is fixed — PF11.)
