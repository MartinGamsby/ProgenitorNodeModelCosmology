# Observer-from-a-Particle mu(z) (PROTOTYPE — items 5 / D)

The user's hypothesis, framed as CORRECT INFERENCE: "We are a RANDOM observer, NOT
in the centre. Pantheon+ (the data) TELLS us WHERE we are, so the best-matching
observer is the inference the data licenses, NOT a cherry-pick. Compute mu(z) from
EACH particle and ask: does a Pantheon-matching observer EXIST, and how GENERIC is
that vantage?" This is a PROTOTYPE answering it (PF13); a pure ADD-ON, NOT on any
default path. Related: PF2 (anisotropy is the discriminating signal) in
[../plans/pinned-findings.md](../plans/pinned-findings.md),
[./hubble-diagram-nbody.md](./hubble-diagram-nbody.md) (the centre-based a(t)->mu(z)),
[./anisotropy-diagnostic.md](./anisotropy-diagnostic.md).

## What it is

The default pipeline measures a(t) as the inner cloud's RMS radius about its CENTRE
OF MASS ([../physics/observable-mask-and-outer-mass.md](./observable-mask-and-outer-mass.md)).
An observer sitting ON a particle p, off-centre, sees a DIFFERENT, generally
ANISOTROPIC expansion. `cosmo/observer_distance.py` (pure functions, mirrors
`cosmo/sim_distance.py` / `cosmo/anisotropy.py` style — no I/O, no sim) builds an
inferred a_p(t) from a chosen observer particle's local frame, then scores it with
the SAME AUTHORITATIVE chi2 the sweep uses (`sim_to_distance_modulus` +
`evaluate_precomputed`). Pure ADD-ON: the centre-based result and every pinned
number are untouched.

## The two observer definitions (BOTH implemented + compared)

```mermaid
flowchart LR
  snaps[sim snapshots\npos,vel,t] --> hist[history_from_snapshots]
  hist --> rms[local_rms\nRMS of p's k-NN about p]
  hist --> hub[hubble_flow\nfit H_p of p's k-NN,\na_p=exp(int H_p dt)]
  rms --> score[score_observer\nsim_to_distance_modulus+evaluate_precomputed]
  hub --> score
  score --> dist[observer_chi2_distribution\ncentre / best / median / p10 / p90]
```

- `local_rms`: a_p(t) = RMS distance of p's neighbour SET about p, set FIXED once at
  the today snapshot (k-NN or all) and TRACKED by identity (Lagrangian local growth),
  normalized a_p[0]=1. **Centre-observer limit:** COM "observer" + neighbour set=ALL
  reproduces the centre-based RMS a(t) EXACTLY (the reproduction invariant the tests
  pin; `center_a_curve` re-implements `simulation._calculate_expansion_history`).
- `hubble_flow`: per snapshot fit the local Hubble slope of p's neighbours about p,
  `H_p = sum(v_r*r)/sum(r^2)` (the GLOBAL-slope formula from
  `anisotropy.hubble_dipole`, centred on p and its peculiar velocity), then
  `a_p = exp(integral H_p dt)`. Closest to how a real observer infers expansion;
  ties to PF2. Agrees with `local_rms` on an isotropic cloud to integration tol.

`score_observer` is the per-observer scorer; it OMITS the growth anchor (an
off-centre observer legitimately has a different total growth than the centre, so
anchoring to the centre's growth would wrongly reject every off-centre observer) but
REPORTS `growth_factor` so runaway observers are visible. `observer_chi2_distribution`
scores every observable particle and returns the distribution + centre baseline +
best observer; pass `lcdm_ref=` / `eds_ref=` (the per-run in-range LCDM / EdS-null
chi2/dof) and it also returns `frac_below_lcdm` / `frac_below_eds`.

`fraction_at_or_below(chi2_dof, threshold)` is the PURE helper behind the genericity
statistic: the fraction of FINITE per-observer chi2/dof at/below a reference. It is
monotone non-decreasing in `threshold`, 0 below the min finite chi2, 1 at/above the
max, inclusive at the boundary, and NaN if no observer is finite. This is the HONEST
counterpart to "the best observer": a LARGE fraction below LCDM/EdS = a Pantheon-like
vantage is GENERIC; a SMALL fraction = our vantage is FINE-TUNED. We report it so
fine-tuning is visible.

## Prototype numbers (M=1000, S=30, t_start=2.9, N=120, seed=42; `_generate_observer_figs.py`)

| config | def | centre | best | median | p10 / p90 | LCDM | EdS | f<LCDM | f<EdS |
|--------|-----|--------|------|--------|-----------|------|-----|--------|-------|
| virialized | local_rms | 15.22 | **0.93** | 8.66 | 5.1 / 43.0 | 0.43 | 0.86 | small | small |
| virialized | hubble_flow | 15.22 | **0.90** | 9.55 | 5.8 / 41.6 | 0.43 | 0.86 | small | small |
| cube26 | local_rms | 0.515 | **0.436** | 0.535 | 0.45 / 0.74 | 0.42 | 0.75 | (re-run) | (re-run) |
| cube26 | hubble_flow | 0.515 | **0.436** | 0.536 | 0.45 / 0.74 | 0.42 | 0.75 | (re-run) | (re-run) |

`f<LCDM` / `f<EdS` = fraction of observers at/below the LCDM / EdS-null reference
(the genericity statistic; large = generic vantage, small = fine-tuned). The exact
fractions are emitted to the CSV (`frac_below_lcdm` / `frac_below_eds`) and annotated
on the figure — re-run `_generate_observer_figs.py` to refresh them. Figure
`results/figures/ws8/observer_chi2_distribution.png` (+ `observer_chi2.csv`), both
gitignored.

## Verdict — a RANDOM observer CAN match Pantheon (correct inference)

The framing is EXISTENCE + GENERICITY, not "is the best a fluke" (we are a random
observer; Pantheon localises us, so the best-matching observer is the inference the
data licenses — see PF13):

- **(i) A Pantheon-matching observer EXISTS** in cube26: the BEST observer lands right
  at the LCDM reference (0.436). So from a real, random vantage the model is VIABLE —
  this is the correct read once we accept we are NOT in a special place.
- **(ii) How GENERIC is that vantage?** Read off `frac_below_lcdm` / `frac_below_eds`
  (fraction of observers at/below the LCDM / EdS-null reference). A LARGE fraction =
  a Pantheon-like view is a generic vantage; a SMALL fraction = fine-tuned. We REPORT
  the fraction (figure annotation + CSV columns) so fine-tuning stays VISIBLE and we
  do not overclaim. The median / p10/p90 are the "how typical are we" context (cube26
  median ~0.535 sits between LCDM and EdS; virialized median ~9 because its centre runs
  away — `center_growth`~32x vs the physical ~3.2x — while individual particle frames
  still find near-LCDM a_p).
- **The large per-observer SPREAD is additionally a PF2-style anisotropy signal** —
  different observers infer materially different expansions (cube26 spread ~0.3;
  virialized spread ~38). Whether to promote this to a Section-7 axis depends on the
  comparison sweep (it is also a PF2 diagnostic).

## Tests / regenerate

- `tests/test_observer_distance.py`: centre reproduces centre-based a(t);
  both defs agree on isotropic cloud + hubble_flow integration converges; off-centre
  observer in anisotropic cloud differs measurably; best<=centre; determinism; mask
  restriction; `fraction_at_or_below` (monotone in threshold, 0 below min, 1 above
  max, inclusive boundary, inf-ignored, NaN if none) + distribution emits
  `frac_below_lcdm`/`frac_below_eds` when refs are given; 2 SLOW real-sim+real-Pantheon
  smokes (skip if data absent).
  `set PYTHONIOENCODING=utf-8 && python -m pytest tests/test_observer_distance.py -q`
- `python _generate_observer_figs.py [--particles N --n-steps K --geometries ...]`
  writes the PNG + CSV and prints the table + verdict. NOTE: `save_interval` MUST
  divide `n_steps` (the integrator only saves a final snapshot on an exact multiple,
  else the today-tolerance guard fires) — `_save_interval_for` enforces this.
