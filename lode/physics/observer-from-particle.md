# Observer-from-a-Particle mu(z) (PROTOTYPE — items 5 / D)

The user's hypothesis: "We're NOT in the centre — compute mu(z) from the viewpoint
of EACH particle, not the cloud centre, and take the best one (and show the
distribution). This will change a lot." This is a PROTOTYPE answering it; it is NOT
a pinned finding and NOT on any default path. Related: PF2 (anisotropy is the
discriminating signal) in [../plans/pinned-findings.md](../plans/pinned-findings.md),
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
best observer.

## Prototype numbers (M=1000, S=30, t_start=2.9, N=120, seed=42; `_generate_observer_figs.py`)

| config | def | centre | best | median | p10 / p90 | LCDM | EdS |
|--------|-----|--------|------|--------|-----------|------|-----|
| virialized | local_rms | 15.22 | **0.93** | 8.66 | 5.1 / 43.0 | 0.43 | 0.86 |
| virialized | hubble_flow | 15.22 | **0.90** | 9.55 | 5.8 / 41.6 | 0.43 | 0.86 |
| cube26 | local_rms | 0.515 | **0.436** | 0.535 | 0.45 / 0.74 | 0.42 | 0.75 |
| cube26 | hubble_flow | 0.515 | **0.436** | 0.536 | 0.45 / 0.74 | 0.42 | 0.75 |

Figure `results/figures/ws8/observer_chi2_distribution.png` (+ `observer_chi2.csv`),
both gitignored.

## HONEST verdict (does it "change a lot"?)

- **Best observer DOES beat the centre** in every config (cube26 15%, virialized 94%),
  and the cube26 best observer lands right at the LCDM floor (0.436). BUT this is the
  MIN over 120 observers — a SELECTION EFFECT / cherry-pick, not a model win. The
  median and p10/p90 (reported alongside) are the fair statistic, and the median is
  WORSE than the centre for cube26 and far worse for virialized.
- **The large per-observer SPREAD is itself a PF2-style anisotropy signal** — different
  observers infer materially different expansions (cube26 spread ~0.3; virialized
  spread ~38 because its centre runs away: `center_growth`~32x vs the physical ~3.2x at
  this N/config, while individual particle frames still find near-LCDM a_p).
- So "compute from a particle and take the best" does NOT honestly improve the fit as a
  model claim; the right read is the DISTRIBUTION. Whether to promote this to a Section-7
  axis depends on the comparison sweep (it is a PF2 diagnostic, not an isotropic-chi2 win).

## Tests / regenerate

- `tests/test_observer_distance.py` (19): centre reproduces centre-based a(t);
  both defs agree on isotropic cloud + hubble_flow integration converges; off-centre
  observer in anisotropic cloud differs measurably; best<=centre; determinism; mask
  restriction; 2 SLOW real-sim+real-Pantheon smokes (skip if data absent).
  `set PYTHONIOENCODING=utf-8 && python -m pytest tests/test_observer_distance.py -q`
- `python _generate_observer_figs.py [--particles N --n-steps K --geometries ...]`
  writes the PNG + CSV and prints the table + verdict. NOTE: `save_interval` MUST
  divide `n_steps` (the integrator only saves a final snapshot on an exact multiple,
  else the today-tolerance guard fires) — `_save_interval_for` enforces this.
