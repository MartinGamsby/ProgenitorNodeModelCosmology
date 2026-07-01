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

## PF13 — Observer-from-a-particle: a RANDOM observer CAN match Pantheon (correct inference, not cherry-pick) — report best EXISTS + FRACTION viable

EPISTEMOLOGY (the framing the user insisted on, replacing the old "cherry-pick" read): we are
a RANDOM observer, NOT at the centre, and Pantheon+ (the data) TELLS us WHERE we are. So
SELECTING the best-matching observer is CORRECT INFERENCE — the inference the data licenses —
NOT cheating. The honest question is therefore TWO statistics, not "is the best a fluke?":
  (i)  EXISTENCE — does a Pantheon-matching observer EXIST? (the best-observer chi2/dof). If
       yes ⇒ the model is VIABLE from a real, random vantage.
  (ii) GENERICITY — what FRACTION of observers see Pantheon-like expansion (chi2/dof below the
       LCDM reference AND below the EdS-null reference)? A LARGE fraction = a Pantheon-like
       view is a GENERIC vantage; a SMALL fraction = our vantage is FINE-TUNED. We REPORT the
       fraction so fine-tuning stays visible — we do NOT overclaim.
The median / p10 / p90 remain the "how typical are we" context.

The scorer (`cosmo/observer_distance.py`, two observer defs: `local_rms` k-NN RMS growth,
`hubble_flow` local-Hubble integral; centre-observer limit reproduces the centre a(t) exactly;
scored with the SAME authoritative chi2). `fraction_at_or_below(chi2_dof, threshold)` is the pure
helper for stat (ii); `observer_chi2_distribution(..., lcdm_ref=, eds_ref=)` emits
`frac_below_lcdm` / `frac_below_eds`.

WIRED INTO THE SWEEP (the user's "always get the best one, for sweeps"): `SweepConfig.score_observers`
(default False; set True in all the real sweep configs — core_v3, satellites, explore) makes
`compute_pantheon_metrics` ALSO score a strided sample of `observer_sample` (default 128) per-particle
observers and make the BEST observer the HEADLINE `chi2_dof` (so the S co-fit + the best-cell
selection optimize on it), keeping the centre value as `center_chi2_dof`. New CSV columns:
`best_observer_chi2`, `center_chi2_dof`, `observer_median_chi2`, `frac_below_lcdm`, `frac_below_eds`.
It needs the per-cell SNAPSHOTS (now carried on `SimResult.snapshots` from the sim, populated in
`results_to_sim_result`) — so a sweep must RE-RUN to gain observer columns (the cache-hit check also
requires `best_observer_chi2`). Pure post-sim analysis: no a(t)/cache-KEY change, no
PHYSICS_CACHE_VERSION bump; default-off is byte-identical. Verified end-to-end on a 1-cell virialized
sweep: chi2_dof == best_observer (0.477) vs center_chi2_dof 0.718, frac_below_eds 0.66, frac_below_lcdm
0 (500p probe). Tests: `tests/test_overarching_sweep.py::TestObserverInSweep`.

NUMBERS (M=1000, S=30, t_start=2.9, N=120, real Pantheon+): cube26 — centre 0.515, BEST
observer 0.436 (= the LCDM reference: a Pantheon-matching observer EXISTS), MEDIAN 0.535,
p10/p90 0.45/0.74; a SIZEABLE fraction of observers sit below the EdS null (sub-EdS is generic)
and a smaller fraction reach sub-LCDM. virialized — centre 15.22 (the centre RUNS AWAY here,
growth ~32× vs physical ~3.2×), best observer 0.90-0.93, MEDIAN 8.7-9.6, p10/p90 ~5/42 (only a
small fraction viable: this config's vantage is fine-tuned). Re-run `_generate_observer_figs.py`
to refresh the exact fraction-viable numbers per (config × definition).

VERDICT: a Pantheon-matching observer EXISTS in cube26 (best = LCDM reference) — the model is
viable from a random vantage, which is the correct inference once we accept we are NOT in a
special place. How fine-tuned that vantage is, is read off the FRACTION viable (reported, not
hidden): the larger the fraction below LCDM/EdS, the more generic the match. The large
per-observer SPREAD is additionally a PF2-style anisotropy signal (off-centre observers infer
materially different expansions). Prototype scale N=120. Source:
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

## PF-PENDING — core_v3 + localized_v4 sweeps COMPLETED (2026-06-28/29); headline now PF16

UPDATE: the detached `core_v3` family (13 arms) AND the `localized_v4` follow-up COMPLETED.
By BEST-OBSERVER (the canonical metric, PF13/PF16): the virialized lattice at HIGH mass-spread
(σ4–8) reaches and BEATS LCDM (PF16). core_v3 cube26 control arms gave best-observer ~0.44–0.45
too (so cube26 is NOT clearly better — the old "virialized fits worse" worry does not hold by
best-observer); a precise matched cube-vs-virialized attribution can still be extracted from
`results/ws1_sweep_core3_*.csv` if needed. The original text below is retained for context.

The claim that the virialized grid "fits worse" than cube26 is UNVERIFIED and must stay so
until the detached `sweeps/core_v3/` sweep runs cube26 as a control AT THE SAME softening /
force-law in the SAME family (item 9 attribution). The 13-arm family (FEWER but HIGHER quality
than the deleted comparison_v2: 2000p/546, M{1,5,10,50,100,500} (regular ×5/×2 grid), S co-fit
[3..35] ternary; virialized arms now vir_n_nodes=300 + vir_mass_spread=2.0 (a WIDER mass
function — see PF15); 12 core arms = 3 GRF geometries × 3 MATCHED treatments + cube26-uniform
control × 3 = 72 cells; arm 13 = the B3a seed sweep, 15 cells) + the secondary satellites
(start_size 6/conv 3/extent 3) + the detached, resumable, PARALLEL-capable launcher
(`launch_sweep_detached.ps1`, Start-Process, core-only default / `-IncludeSatellites` /
`-Parallel N` with the concurrency-safe shared cache) are BUILT and the keyed==run wiring is
verified. The runtime CALIBRATOR (`_calibrate_runtime.py`, NO blind launch) measured (at the OLD
vir_n_nodes=150): cube26 bounded+substep ~88 s/sim, virA bounded ~243 s, virB ~65 s; none/plummer
~22-36 s. NOTE those are now STALE — the bumped 300-node virialized arms are ~2× slower; re-run
the calibrator. `-Parallel N` divides wall-time by ~N.
But the multi-day RESULTS are a flagged follow-on. Until that sweep COMPLETES, do NOT pin:
cube26-vs-virialized attribution, the low/fine M/S landscape, start-size/convergence/extent
results, or any final headline chi2 band for virialized. Source:
[overarching-sweep.md](./overarching-sweep.md),
[../scripts/parameter-sweep.md](../scripts/parameter-sweep.md).

## PF15 — Node mass-function WIDTH (vir_mass_spread) is a key lever: a wide distribution avoids the small-S collapse and reaches the accelerating corner

`vir_mass_spread` is the σ of the per-shell log-normal node mass draw (`exp(σ·N(0,1))`,
mean-preserving). It is now a SWEEPABLE axis (`vir_mass_spreads` list in a sweep config →
`expand_grid` emits one cell per spread; keyed==run via the `<>vsp` cache slug).

**The growth anchor assumes nothing about node placement or the mass distribution.** It
(`cosmo/parameter_sweep.py::compute_pantheon_metrics`, `GROWTH_ANCHOR_TOL=0.20`) gates ONLY the
total realized expansion `a[-1]/a[0]` against the ΛCDM growth (`expected_growth_factor` ≈ 3.30×
at t_start=2.9; admissible window ≈ [2.64, 3.97]). So a config is admissible iff its expansion
lands there — nothing forbids small or nearby or sub-M nodes.

**Findings (virialized lattice, GRF, 600p probe, single seed — probe-scale, NOT converged):**
- NARROW spread (σ=0.8, the OLD default): at small S the dense shell of comparable-mass nodes
  sits inside the observable cloud and its net gravity OVERWHELMS expansion → the CLOUD collapses
  (growth → ~1; e.g. M=10/S≤10 → growth 0.97–1.5, anchor-rejected). This is the real failure
  mode — NOT "M too weak." **Corrects the stale "M=1–2 → no anchor_ok" claim** (the narrow
  outer-mass grid, see §the small-M note above): M=1 IS admissible at S≥10 (growth ~2.68); the
  nodes are static (the GRID never collapses) — it is the particle cloud whose expansion is
  suppressed.
- WIDE spread (σ=3, 500 nodes): a few HUGE nodes (segregated to large radius) + many tiny ones →
  negligible nearby pull → NO collapse (rejections 6/12 → 1/12); M=10/50/100/300/500 become
  admissible. **M=100 / S=10 / σ=3 → growth 3.03 (genuinely ACCELERATING) and chi2/dof ≈ 0.510 —
  the model's best-known floor (~0.52), now in a REALISTIC wide-mass FORCE-BALANCED virialized
  structure** (lattice per-shell antipodal cancellation holds at any σ). A band M~100–500 / S~10–20
  sits at 0.51–0.61; small S (≤5) still runs away (the few giant nodes too close).

**Default bumped accordingly:** virialized core/seed/startsize arms now `vir_n_nodes=300` +
`vir_mass_spread=2.0` (a "bigger distribution, more nodes" baseline — deliberately NOT the
extreme), with `sweeps/explore_vir_spread.json` sweeping σ∈{0.8,1.5,2,3,4} × M{10..500} × S{10..30}
to FIND the best (σ,M,S) region for the focused final sweep. **CAVEAT:** probe-scale (600–1000p,
1 seed); the converged result + best defaults await the exploration sweep. Source:
`cosmo/parameter_sweep.py` (expand_grid `vir_mass_spreads`; build_cache_name `vsp`),
`sweeps/explore_vir_spread.json`; tests `TestVirMassSpreadAxis`.

## PF16 — At production resolution the model MATCHES LCDM-quality, robustly across MANY configs + seeds (localized_v4/v5/v7)

NET (after the v7 4000p N-check, honest): the External-Node model REACHES chi2/dof ≈ 0.43–0.44
vs Pantheon+ — essentially EQUAL to LCDM (0.436) — from the central AND best vantage, ROBUST
across M/S/σ and node-realization SEED, decisively rejecting the EdS null (0.843). It MATCHES
LCDM; it does NOT robustly BEAT it (a vantage marginally below LCDM exists but is RARE, ~0.2% at
4000p — the 2000p "~5–7% below" was partly a sampling-tail effect, see the N-check below).
This is a gravity-only toy model producing LCDM-equivalent effective dark energy — and it
SHARPENS the old PF4 "near-LCDM ~0.52" band down to ≈LCDM via the wide-mass-function (high-σ)
virialized structure.

FRAMING (user-insisted, see memory observer-not-center.md + PF13): we are NOT at the centre
(P(centre)~0), so the HEADLINE metric is the **best-observer chi2/dof** (`best_observer_chi2`),
NOT `center_chi2_dof` and NOT a mean — Pantheon localises us to the best-matching vantage.
Showing MANY configs reach a good vantage is the paper's MULTIPLICITY argument, not overfitting.

`sweeps/localized_v4/` (2000 particles, 1092 steps ~10 Myr, virialized lattice 300 nodes, GRF
sphere, Plummer 1 Gpc, vir_mass_spread σ∈{4,5,6,7,8}, M{100–500}, S{15–30}, observer_sample=256,
observer_k=-1; LCDM ref 0.436, EdS null 0.843):
- **12 cells with best_observer_chi2 < LCDM (0.436)**; best **0.4317** (M=300/S=20/σ6, 2184-step
  arm), with **frac_below_lcdm up to 0.051** — i.e. a real ~5% of sampled observers beat LCDM,
  not merely the single best touching it.
- **Strong multiplicity:** 78 cells ≤0.45, 81 ≤0.46, spanning M{100–500} × S{15–30} × σ{4–8}.
  Many DISTINCT configs reach ~LCDM-quality from some vantage.
- Decisively below the EdS null everywhere (frac_below_eds ~0.7–0.98).
- At the standout cell M=300/S=20/σ6 even the CENTRE chi2 is ~0.434 (≈LCDM) — the wide mass
  function makes the central observer itself Pantheon-like (a NEW high-σ regime, distinct from
  PF4's 0.52 centre floor which was σ=0/400p/273-step at different M/S — not a contradiction).

HONEST CAVEATS:
- best-observer = EXISTENCE; the reported `frac_below_lcdm` (≤~5%) is the GENERICITY (PF13). A
  Pantheon-beating vantage EXISTS and is non-fine-tuned at the few-% level, but is not yet typical.
- STEP-CONVERGENCE CONFIRMED (localized_v5 conv arms, M=300/S=20/σ6): best-observer
  1092→0.4396, 1638→0.4339, 2184→0.4317, 2730→0.4314, 3276→**0.4313** (decrements shrink to
  ~1e-4 ⇒ CONVERGED ~0.431, below LCDM 0.436). The CENTRE converges in lockstep: 0.4417→…→
  **0.4335** (≈LCDM) — at this cell even the central observer is Pantheon-like. And
  frac_below_lcdm RISES with resolution (0.00→0.020→0.051→0.063→**0.070**) — the sub-LCDM
  result gets MORE generic with steps, NOT a resolution artifact. So 1092 was the conservative
  end; the converged headline is best-obs ~0.431 / centre ~0.434 / 7% of observers below LCDM.
- observer_k=-1 (whole-cloud RMS, well-sampled = robust). FINITE local observer_k is a NOISE
  artifact at N=2000 (PF17) — NOT a real improvement; use k=-1.
- SEED-ROBUST (localized_v7, seeds {42,7,123} × M{200,300,400} × S{20,22,25} × σ{5,6,7},
  observer_k=-1): best-observer is STABLE across node realizations — per-cell spread 0.0007–0.0041
  (mean 0.0023). 45 configs ≤0.45 across the 3 seeds → the ~0.44 LCDM-match is GENERIC, not
  seed-tuned. (CAVEAT: most cells' CENTRE is 0.46–0.61 and frac_below_lcdm ~0.1–0.3% — the best
  ~0.433 vantage is the favourable tail; M=300/S=20/σ6 is the special cell whose CENTRE itself
  ≈LCDM.)
- N-CHECK (localized_v7, M=300/S=20/σ6 at 4000p, observer_k=-1): best-obs 0.4347, CENTRE 0.4414
  (≈LCDM), growth 3.64, frac_below_lcdm 0.002. vs 2000p (0.4313 / 0.4338 / 0.070): the converged
  values are ~0.435 best / ~0.44 centre, and frac_below_lcdm SHRINKS 7%→0.2% at higher N (the
  2000p fraction was partly the noise tail). So the HONEST converged claim is "MATCHES LCDM
  (~0.44), rare vantage marginally below" — not "7% beat LCDM".

Source: `sweeps/localized_v4/` + `_gen_localized_v4.py`, `results/ws1_sweep_loc4_*.csv`;
[overarching-sweep.md](./overarching-sweep.md); analysis scratchpad/analyze_obs.py. PF-PENDING
(core_v3) is now COMPLETE — see below.

## PF17 — Small-k LOCAL observers are noise-dominated; the near-zero best-observer chi2 are overfitting artifacts (use well-sampled observers)

Tested whether a genuinely LOCAL observer (finite `observer_k` k-NN, more physical than the
whole-cloud k=-1) finds a vantage further below LCDM (`sweeps/localized_v6/`, the CORRECTED
observer_k sweep after the PF-cache fix below; observer_sample=2000 = ALL particles, k∈
{64,128,256,-1}, on the σ4-8 / M{100-500} / S{20-30} band).

RESULT — NEGATIVE / a methodology guardrail. Finite-k arms produce a FEW absurdly low
best_observer_chi2 (0.0038, 0.0105, 0.053 — chi2/dof far below LCDM 0.436 AND below the SNe
error floor), but these are STATISTICAL ARTIFACTS, not real vantages:
- `best << median` while median/center stay normal (e.g. M=500/S=30/σ6, k=128: best 0.0038,
  MEDIAN 0.510, CENTRE 0.509). The bulk distribution did NOT improve — only the noisy MIN tail
  extended.
- `frac_below_lcdm` stays TINY (~0.4-1.3%) and does NOT rise with finite k; no smooth trend
  across k (e.g. 0.093 / 0.0038 / 0.433 for k=64/128/256) — the signature of NOISE, not a lever.
- Mechanism: k=64-256 of 2000 tracer particles is a SPARSE local sample, so the local a(t) is
  sampling-noise dominated; the min over ~2000 noisy curves catches a lucky fit (a real observer
  measures millions of local galaxies, not ~64 tracers). chi2/dof ~0.004 over ~1500 SNe = the
  curve threading inside the error bars = overfitting to tracer noise.

HONEST CONSEQUENCE: best-observer is only trustworthy when WELL-SAMPLED. Use observer_k=-1
(whole cloud) or large k, and ALWAYS sanity-check best vs MEDIAN (best<<median = fluke) + the
fraction. The ROBUST headline stays the k=-1 result (PF16): best-observer ~0.43 (≈LCDM, the
model MATCHES LCDM-quality from a generic vantage), median ~0.5, ~7% of observers below LCDM,
decisively below the EdS null — across MANY configs (the multiplicity). The small-k near-zero
values must NOT be quoted. To test local observers legitimately needs far higher N (so a local
neighbourhood is well-sampled) — a future check. Source: `sweeps/localized_v6/`,
`results/ws1_sweep_loc6_*.csv`; memory observer-not-center.md; PF13.

## PF-OBSCACHE — observer params were keyed != run in the metrics cache (FIXED)

`build_cache_name` (cosmo/parameter_sweep.py) did not encode `observer_k` / `observer_sample` /
`observer_definition`, yet the metrics cache STORES best_observer_chi2 / frac_below_* / observer_
median_chi2 (which depend on them). So a re-run with a different observer_k SERVED the stale
cached observer score — observer_k had NO effect (the first localized_v5 obsk arms all returned
IDENTICAL results, == the v4 k=-1 values). FIX: append `{def}obsdef_{sample}obssamp_{k}obsk`
(k=-1 → "all") to the cache key ONLY when score_observers is True (score_observers=False stays
byte-identical; no PHYSICS_CACHE_VERSION bump). Observer scoring is post-sim, so distinct
observer params now map to distinct cache entries (keyed == run); the sim is recomputed because
the metrics cache stores sim + observer metrics together. Regression test
`tests/test_overarching_sweep.py::TestObserverInSweep::test_observer_params_are_keyed_equals_run_in_cache`;
full file 117 pass. Source: cosmo/parameter_sweep.py build_cache_name.

## PF18 — 4000p multiplicity confirmed; the center/COM/RMS are slingshot-OUTLIER-inflated (best-observer is the robust metric)

CONVERGED MULTIPLICITY (localized_v8, 4000 particles, observer_k=-1, M{200,300,400}×S{20,22}×
σ{5,6,7}): **11/15 anchor-ok configs have best-observer chi2/dof < LCDM (0.436)**, all clustered
~0.433-0.434, growth 3.0-3.9. So MANY distinct configs match/slightly-beat LCDM at proper
resolution — the paper's multiplicity claim, confirmed at high N. frac_below_lcdm stays small
(~0.000-0.002) → the sub-LCDM vantage is rare (matches, doesn't robustly beat).

COHERENT BULK DRIFT (figure diagnostics, _generate_paper_figs.py — CORRECTED): the headline
config's cloud COM drifts ~30 Gpc over 10.9 Gyr. This is NOT an outlier artifact — the
MEDIAN-centre drift (28.9 Gpc) ≈ the MEAN-COM drift (30.8 Gpc), so the WHOLE cloud coherently
translates (the net-force / "dark-flow" signal from the asymmetric trans-horizon node field).
The diagnostic across σ shows large drift/RMS (2.3-4.4) and >100% pre-start tidal boost are
MODEL-WIDE (every viable config). Read literally the drift is ~9c, but a BULK translation of the
whole observable cloud is UNOBSERVABLE from inside (no absolute frame) and does NOT affect a(t)
(COM-relative) or the observer chi2 (LOCAL k-NN) — so the chi2 result is unaffected; the drift is
a separate, large anisotropy/peculiar-flow feature to report, not hide. SEPARATELY, a residual
PF9 slingshot tail (Plummer 1 Gpc tamed to ~7×, not eliminated) does inflate the mean-COM RMS
"size" somewhat, so the CENTER-based size/growth are mildly outlier-sensitive; the BEST-OBSERVER
chi2 (local) is the robust headline (and the user's chosen metric). Paper figures centre on the
MEDIAN, drop the runaway tail, use one fixed scale. The large bulk drift + the superluminal-if-
literal velocity are an OPEN physical question for the model (is the net-force regime realistic?).
Source: _generate_paper_figs.py, results/figures/paper/, results/logs/diag_drift.out; PF9, PF16/PF17.

## PF19 — HERO ladder: the ~0.44 best-observer result is converged in BOTH particles AND steps (10k->100k, 5k->12k)

The high-resolution hero ladder (sweeps/hero/, run via sweep.py save_snapshots + -ArmSet hero;
results/hero/*.npz, imaged by _generate_hero_figs.py) scales particles AND time-steps together
(steps matter — finer dt resolves the denser structure high N exposes):

| particles | steps | M/S/sigma | best-obs | centre | growth |
|-----------|-------|-----------|----------|--------|--------|
| 10k | 5k | 300/20/6 | 0.4424 | 0.453 | 3.63 |
| 20k | 6k | 400/22/5 | 0.4476 | 0.453 | 3.50 |
| 30k | 7k | 200/20/7 | 0.4464 | 0.466 | 3.33 |
| 50k | 8k | 300/20/5 | 0.4449 | 0.457 | 3.39 |
| 50k | 10k | 300/22/7 | 0.4458 | 0.452 | 3.58 |
| 75k | 11k | 200/22/6 | 0.4491 | 0.541 | 3.09 |
| **100k** | **12k** | **300/20/6** | **0.4465** | **0.453** | **3.64** |

SEED SENSITIVITY (cloud-visual investigation): at the wide-sigma headline cell M300/S20/σ6,
node_mass_seed is NOT freely swappable for a rounder cloud — only a MINORITY of seeds are
anchor-ok (seed 42 growth 3.64), while others (7/99/777) RUN AWAY (growth 5.7-6.2, anchor-fail,
even at 20k/8000 so not a dt artefact). The runaway seeds only LOOK uniform because they
over-expand (diffuse → low central concentration). So PF19's "seed-robust fit ±0.002" holds for
the anchor-ok SUBSET, not all seeds. The anchor-ok realizations are centrally concentrated
(traceless tidal COMPRESSION; core-fraction 0.38 at M300 vs 0.016 uniform). The route to a
rounder Pantheon-matching cloud is LOWER external mass (weaker compression): M200/S22/σ6 (75k)
is anchor-ok (growth 3.09), matches (best-obs 0.449), AND is the least concentrated (core 0.25)
— used for the paper cloud figure (fig6). Not the seed.

Best-observer chi2/dof is STABLE at **~0.442-0.449 ≈ LCDM (0.436)** across the entire ladder
(10x particles, 2.4x steps) — the match is CONVERGED, not a low-resolution artifact, and the
many distinct (M,S,sigma) configs all land on Pantheon+ ≈ LCDM (multiplicity; hero_hubble.png).
All decisively reject the EdS null (frac_below_eds 0.76-0.94). frac_below_lcdm=0 at these
well-sampled (k=-1) vantages -> MATCHES LCDM, does not robustly beat it (PF16/PF18 consistent).
Infra: the timing probe (5 full cube26 sims at config size) is bypassed by skip_probe (it was
lethal at 100k); energy history retained at high N via chunked PE. Figures: results/figures/hero/
(7 cloud-evolution panels + hero_hubble + hero_chi2_summary). Source: sweeps/hero/, sweep.py.

## PF20 — Wide-σ mass segregation wastes mass on gravitationally-inert FAR nodes (near nodes go massless); pantheon sweep never checks SIZE

Building the ACTUAL headline virialized geometry (`build_virialized_grid`, 300 nodes,
massfunc, seg=1.0, force-balanced lattice, seed 42 — the `sweeps/hero/*.json` config)
and ranking nodes by their differential-tidal proxy `m/d³` exposes a pathology in the
wide-σ mass function:

| σ (vir_mass_spread), seg=1.0 | near-node mass/mean | crop (2.5·S=50 Gpc) tidal share | max mass/mean @ r |
|---|---|---|---|
| 0.5 | 0.43–0.83 | 61% | 1.5× @ 87 Gpc |
| 1.0 | 0.18–0.64 | 45% | 2.0× @ 87 Gpc |
| 2.0 | 0.03–0.33 | 20% | 3.2× @ 87 Gpc |
| 3.0 | 0.003–0.15 | ~3% | 4.6× @ 87 Gpc |
| **6.0 (headline)** | **~0.00** | **0.4%** | **9.5× @ 87 Gpc** |

Mechanism: massfunc SORTS a log-normal draw and assigns the biggest masses to the OUTERMOST
shells (mass segregation, seg=1). With a wide σ the mean-preserving normalization then drives
the inner shells to ~0 mass. But the tide falls as `m/d³`, so the massive far nodes are
gravitationally INERT — at σ=6 the nodes within 50 Gpc carry 0.4% of Σ(m/d³); the field is a
handful of ~9.5× nodes at ~87 Gpc (≈4× the 14 Gpc horizon). i.e. the headline config sources its
"dark energy" almost entirely from mass placed where 1/d³ can't use it, and the near nodes we'd
actually feel are ~massless. HONEST reading: a wide mass function is only viable at MILD
segregation (or low σ); "virialized AND physically near-sourced" lives at σ≲1, seg=1 (near nodes
order-M, near-dominated). σ=0 (uniform) is force-balanced but NOT a relaxed/virialized cluster
(no segregation). This is the tension the §2.1 figure now makes visible (see below).

SWEEP GAP (user-flagged, TASK-A-ADJACENT — NOT yet implemented): `compute_pantheon_metrics`
(cosmo/parameter_sweep.py) selects the best config by `chi2_dof` ALONE (match_avg_pct=100/(1+chi2_dof)),
with the ONLY size/expansion constraint a ±20% GATE on the total growth factor a[-1]/a[0]
(GROWTH_ANCHOR_TOL, `expected_growth_factor`). It does NOT match the size CURVE or final size to
LCDM — that machinery (match_curve_pct/match_end_pct/size R²) exists ONLY for objective="lcdm".
So a wide-σ config can pass the coarse growth gate and win on best-OBSERVER chi2 while its a(t)
SHAPE drifts from LCDM. A size-agreement metric alongside chi2 in the pantheon objective would
likely penalize exactly these mass-wasting configs. Proposed but pending (touches shared sweep
code / task A).

FIGURE: `_gen_fig_meta_structure.py` (parameterized paper generator; every physical knob is a CLI
arg since the config is still moving) renders a POINTS-ONLY single 3D scatter of the local HMEA
neighbourhood (cropped by --view-scale), nodes sized + coloured by MASS, with our observable sphere
to scale — REUSING cosmo.visualization.draw_universe_sphere + setup_3d_axes (no duplicated 3D code).
Wired into paper §2.1 (fig:meta-structure) at σ=1 (docs/fig_meta_structure.png). The m/d³/crop-share
numbers above are a diagnostic of the mass function (this PF), NOT plotted — the figure is a clean
geometry illustration. The size-agreement sweep gap is TASK-A's to implement, not done here.

## What is NOT yet pinned (the job of this phase)

- The cube26-vs-virialized attribution and the final headline chi2 band for virialized
  (PENDING the detached `core_v3` sweep — see PF-PENDING above). The chi2 DEFINITION
  is now authoritative (PF11), but the multi-day comparison numbers are not in yet.
- Whether ANY node geometry yields a larger net isotropic effect (WS3).
- Whether all conclusions survive high N (WS6) — the `satellite_convergence`
  (1000p/2000p/4000p on one cell, N IS the variable) check addresses this once it runs.
- A fuller WS4 centerM sweep INCLUDING the M~50/S~20 corner (the reduced grid that
  produced PF6 omitted it). centerM-as-outer-mass semantics are now IMPLEMENTED (PF6);
  whether outer mass helps at the model's actual best-fit corner is still open.

(RESOLVED since the last edit: the GRF chi2 swing is now attributed — PF12, the cube-vs-
sphere setup bug + real clustering; and the figure↔CSV chi2 conflict is fixed — PF11.)
