# WS5 — GRF-vs-Uniform Shape Investigation

Back to [deeper-exploration-roadmap.md](./deeper-exploration-roadmap.md). Phase 2.
Uses WS1 (sweep) + WS2 (figures) + WS6 (convergence) tools.

## The anomaly

The isotropic background was EXPECTED to be clustering-insensitive: chi2/dof is shape-
driven, not sampling-driven, so `init_distribution="grf"` and `"uniform_sphere"` should
give ~the same chi2/dof at fixed (M,S). The convergence ladder and the 400p knob sweep
both reported them indistinguishable (see
[../physics/realistic-initial-conditions.md](../physics/realistic-initial-conditions.md),
[../physics/pantheon-comparison-results.md](../physics/pantheon-comparison-results.md)).

BUT at M=1500/S=30 the isotropic chi2/dof SWUNG from ~0.51 (uniform) to ~1.56 (grf).
That contradicts the clustering-insensitive expectation and must be explained before any
GRF-based number is trusted.

## Candidate explanations to test

| Hypothesis | How to test | Figure |
|-----------|-------------|--------|
| H1: Real physics — clustering genuinely changes a(t) at strong tidal field | Sweep grf-vs-uniform across (M,S); is the swing localized to strong-field / near-runaway cells, or everywhere? | F10 |
| H2: Near-runaway sensitivity — M=1500/S=30 is close to the bound/runaway edge (PF5), where small IC differences amplify | Overlay the grf/uniform chi2 swing on the runaway-boundary map (F9); is the swing where S→S_crit? | F9+F10 |
| H3: Particle-count noise — 400p GRF realization is noisy; the swing shrinks with N | Convergence ladder (WS6) grf vs uniform at M=1500/S=30: does the swing collapse as N grows? | F11 |
| H4: GRF setup issue — Zel'dovich displacement at this box/scale over-perturbs, or RMS-norm interacts badly with clustered positions | Inspect realized growth + initial RMS + a few seeds; compare GRF density variance to expectation | F10 + a diagnostic panel |

## Plan

1. Run a grf-vs-uniform DELTA sweep over the full (M,S) grid on the WS1 tool (same
   kernel/anchor), emitting F10 (the chi2/dof and growth delta maps) and overlaying F9
   (runaway boundary). This immediately separates H1/H2 (localized vs global; on the
   runaway edge or not).
2. At the worst-swing cell(s), run the convergence ladder (WS6) for BOTH init
   distributions across N. If the swing collapses with N → H3 (noise). If it persists →
   H1 or H2 or H4.
3. If it persists and is NOT on the runaway edge, audit the GRF setup: check that the
   GRF run's INITIAL RMS radius matches uniform (the shared RMS-norm should force this),
   that the realized growth anchor is comparable, and that multiple GRF seeds agree.
   This isolates H4 (a setup bug) from H1 (real clustering physics).

## Honesty note

If the GRF chi2 differs from uniform ONLY near the runaway edge and ONLY at low N, the
defensible headline stays the best ISOTROPIC config (PF3/PF4) and GRF is reported as a
realism/anisotropy variant, not a fit knob. If the difference is real and N-converged
away from the edge, that is a NEW finding (clustering affects the bulk a(t)) and must be
characterized, not buried.

## Files this workstream touches

- WS1 sweep tool (grf-vs-uniform delta mode + F10).
- `convergence_check.py` (already does the N-ladder; run for both inits at the swing cell).
- `cosmo/initial_distributions.py` — only if H4 (GRF setup) is confirmed; otherwise
  read-only inspection.
- The runaway-boundary figure F9 (WS1) for the overlay.

## Deliverables

- F10 grf-vs-uniform delta maps + the F9 overlay.
- A definitive attribution of the 0.51→1.56 swing to one of H1-H4, with the supporting
  figures.
- Updated wording in pantheon-comparison-results.md / realistic-initial-conditions.md to
  match the verdict (currently they assert "indistinguishable" which the swing
  contradicts — resolve the contradiction here).
