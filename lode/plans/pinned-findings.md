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

The honest read of the isotropic fit (corrected from an earlier over-negative framing):

- On chi2/dof, the model's best physical configs (~0.48-0.51) are CLOSE to LCDM (~0.43)
  and FAR from the Einstein-de Sitter no-dark-energy null (~0.85, which the SNe
  decisively disfavor).
- So the toy model lands NEAR LCDM and DECISIVELY REJECTS no-dark-energy — a genuine
  success for a toy model. It produces real effective dark energy.
- It is NOT an exact LCDM match (does not beat LCDM) and is M/S³-degenerate.
- The landscape is config/init-sensitive: the symmetric base at the paper-nominal
  M=855/S=37.8 can sit ON the EdS null (~0.67-0.85 depending on kernel/anchor), and
  GRF-vs-uniform swung the chi2 (0.51 → 1.56) at M=1500/S=30 — see
  [grf-vs-uniform.md](./grf-vs-uniform.md).

**Therefore the exact numbers must be PINNED DOWN with finer sweeps + graphs (WS1+WS2)
before any claim is published. Reframe to fair "near-LCDM / far-from-null" wording, but
do NOT over-claim an LCDM match either.** Several chi2 numbers currently in the Lode
predate the current kernel/anchor and disagree with each other (0.50 vs 0.67 vs 0.85 at
nominal config); WS1 re-runs them on ONE consistent kernel/anchor and graphs them.

## PF5 — Runaway boundary is real physics

Small-S configs run away (nearest-node (S−R)⁻² attraction overwhelms bound expansion);
boundary scales roughly S_crit ∝ M^(1/3) (constant Ω_Λ_eff contour). The growth anchor
(GROWTH_ANCHOR_TOL=0.20) rejects runaway configs in the sweep. Non-monotonic: matching
total growth with the WRONG a(t) SHAPE is penalized by SNe, so the best SHAPE is at
INTERMEDIATE strength (undershooting the anchor), not at the strongest bound config.
Source: [../physics/initial-conditions.md](../physics/initial-conditions.md) runaway map.
**WS1 must map this boundary as a figure (runaway-boundary map) so every claimed config
is visibly on the bound side.**

## What is NOT yet pinned (the job of this phase)

- The single consistent chi2/dof for the nominal + best configs on one kernel/anchor.
- Whether ANY node geometry yields a larger net isotropic effect (WS3).
- Whether the GRF chi2 swing is real physics, near-runaway sensitivity, particle-count
  noise, or a GRF setup issue (WS5).
- Whether all conclusions survive high N (WS6).
- The correct centerM semantics (extend the sim sphere OUTSIDE the observable region) —
  currently centerM only sets softening under eds_consistent (WS4).
