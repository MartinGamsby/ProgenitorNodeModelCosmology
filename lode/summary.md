# ProgenitorNodeModelCosmology — summary

Newtonian N-body toy model testing whether tidal forces from trans-horizon Hyper-Massive External
Attractors (HMEAs), arranged as a virialized meta-structure (a periodic self-gravitating medium
at 2K/|U| ≈ 1), can replace dark energy for the late universe (t = 2.9 → 13.8 Gyr). A cloud of
tracer particles starts from EdS-consistent initial conditions (M = 0 reproduces Einstein–de
Sitter exactly), evolves under internal gravity plus the node tides with a kick-drift-kick
leapfrog, and its measured a(t) is turned into μ(z) and scored against real Pantheon+ supernovae
(offset-marginalized χ², growth anchor, best-matching observer because we are not at the
centre). Under the Progenitor Hypothesis the Big Bang was one typical node destabilising, and
its mass centerM acts in both epochs. Current evidence: best-observer χ²/dof ≈ 0.44 (ΛCDM
0.436, EdS null 0.843) across many configurations, converged to 10⁵ particles — it matches but
does not beat ΛCDM and is degenerate in M/S³; with node spacing chosen jointly on fit and core
stability, 68 of 72 (centerM × realization) cells are universe-like at 8000 particles; the
discriminating prediction is anisotropy (shear, H0 dipole, bulk flow). Everything runs through
`sweep.py` + JSON configs with a keyed==run cache; the draft paper is
`docs/VirializedMetaStructure.tex`; next compute step is a self-hosted GPU (20k+ particles).

Start with [lode-map.md](./lode-map.md), [findings/summary.md](./findings/summary.md),
[practices.md](./practices.md).
