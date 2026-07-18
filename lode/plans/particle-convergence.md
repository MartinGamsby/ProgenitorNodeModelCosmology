# WS6 — Particle-Count Convergence (LAST / slowest)

Back to [deeper-exploration-roadmap.md](./deeper-exploration-roadmap.md). Phase 3.
Runs AFTER the cheap exploration (WS1-WS3) and the mid-phase concepts (WS4-WS5) so heavy
compute is spent only on conclusions already established at modest N.

## Why last

Phase 1/2 deliberately run at modest N (400-800) for speed. Those conclusions (the
re-pinned chi2 band, geometry verdict, centerM reconception, GRF attribution) must then
be shown to SURVIVE higher resolution. Convergence is the slowest workstream (finer
grain, higher N), so it is sequenced at the end — once the cheap work has decided WHAT is
worth converging.

## Current state to build on

`convergence_check.py` already sweeps N ∈ {1 000, 10 000} (+ 100 000 with `--full`),
auto-selects `barnes_hut` (real octree O(N log N)) for large N, and gates on three
invariants (growth anchor, ≤LCDM-analytic, dt<0.05). At N=1k and 10k the isotropic
growth was fully converged (0.000% spread). The ≤LCDM reference is ANALYTIC, so the gate
reflects physics, not shot noise. Force methods available: `direct`, `numba_direct`,
`barnes_hut` (see [../physics/barnes-hut-optimization.md](../physics/barnes-hut-optimization.md)).

## Plan

- Extend the N-ladder to the largest N feasible on CPU via Barnes-Hut (e.g. 10⁵, 10⁶ if
  tractable), for the HEADLINE configs only (best isotropic, the anisotropy-showcase
  config, the best geometry from WS3, the centerM-reconception inner region from WS4,
  and the GRF swing cell from WS5) — not the whole grid.
- For each headline config, converge THREE quantities, not just growth:
  1. chi2/dof vs Pantheon+ (the isotropic claim, PF3/PF4),
  2. growth factor vs the physical anchor (PF1/PF5),
  3. shear_index + Hubble dipole (the PF2 discriminating signal — these are the noisiest
     at low N; `anisotropy_report.py` already warns N<2000 is noisy).
- Confirm Barnes-Hut θ (opening angle) is tight enough that the tidal + internal forces
  match `numba_direct` at small N before trusting it at large N (an accuracy cross-check,
  not just speed).
- Emit F11: chi2/dof, growth, and shear/dipole vs N, each with a visible converged
  plateau, per headline config.

## Honesty gate

A Phase-1/2 conclusion is only FINAL once F11 shows its quantity plateaus before the
largest N — i.e. the modest-N number was already converged. If a quantity is STILL
moving at the largest feasible N (likely for shear/dipole), say so and bound the
residual; do not quote a low-N anisotropy number as converged.

## Files this workstream touches

- `convergence_check.py` — extend N-ladder; add chi2/dof + shear/dipole tracks (not just
  growth + ≤LCDM); route figures through the shared plot helper (WS2, F11).
- `cosmo/anisotropy.py` callers — measure shear/dipole at each N.
- WS1 sweep tool — a "headline configs only" high-N mode reusing the cache.
- Barnes-Hut path (`cosmo/...` octree) — θ accuracy cross-check vs numba_direct.

## Deliverables

- F11 convergence figures (chi2/dof, growth, shear/dipole vs N) for each headline config.
- A converged-or-bounded statement for every Phase-1/2 headline number.
- The high-N runs are exactly the workload that motivates the GPU/HF scale-out
  ([scale-out-gpu-hf.md](./scale-out-gpu-hf.md)) — sequenced immediately after this.
