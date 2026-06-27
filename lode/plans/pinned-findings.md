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

GRF-vs-uniform sensitivity (from prior work at M=1500/S=30): uniform_sphere gives
chi2~0.52, GRF gives chi2~1.56 — large swing. Not re-run here; still an open question
(WS5). Do NOT claim 0.52 for GRF init.

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

## What is NOT yet pinned (the job of this phase)

- The single consistent chi2/dof for the nominal + best configs on one kernel/anchor.
- Whether ANY node geometry yields a larger net isotropic effect (WS3).
- Whether the GRF chi2 swing is real physics, near-runaway sensitivity, particle-count
  noise, or a GRF setup issue (WS5).
- Whether all conclusions survive high N (WS6).
- A fuller WS4 centerM sweep INCLUDING the M~50/S~20 corner (the reduced grid that
  produced PF6 omitted it). centerM-as-outer-mass semantics are now IMPLEMENTED (PF6);
  whether outer mass helps at the model's actual best-fit corner is still open.
