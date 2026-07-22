# Memory budget & sweep parallelism

Back to [lode-map](../lode-map.md). Related: [timestep-stability](./timestep-stability.md),
[parameter-sweep](../scripts/parameter-sweep.md).

## The binding constraint on a parallel sweep is RAM, not cores

Sweep cells are independent single-threaded `python sweep.py --config` processes, so
throughput scales with how many run at once — until they collectively exhaust free RAM
and die with `numpy._core._exceptions._ArrayMemoryError`. Size a pool against PEAK
transient memory per sim, never steady-state working set (a 24k-particle cell sits at
~270 MB but SPIKES far higher).

## `Integrator.potential_energy` was the spike (FIXED)

PE is a diagnostic (energy-conservation history, sampled ~10x/run at N<=20000, ~5x
above — see `evolve`). It is chunked to avoid the full (N,N,3) array, but the block
heuristic counted ELEMENTS only:

```python
block = max(1, int(1e7 // N))        # OLD -- under-counted ~9x
```

Each iteration actually allocates `diff` (b,N,3) **plus ~6 (b,N) arrays** (`r2`,
`r_soft`, `mp`, `pe`, `mask`, and the `pe[mask]` copy) => `b*N*8*(3+6)` bytes. The
heuristic ignored both the xyz factor and the temporaries, so at N=24000 it picked
b=416 => **~719 MB per PE call**.

FIX (`cosmo/integrator.py`): budget PEAK BYTES via the module constant
`_PE_BLOCK_BYTES = 64 MB`:

```python
block = max(1, int(_PE_BLOCK_BYTES // (max(1, N) * 8 * 9)))
```

| N | block old -> new | peak per PE call |
|---|---|---|
| 2 000 | 5000 -> 466 | 720 MB -> 67 MB |
| 8 000 | 1250 -> 116 | 720 MB -> 67 MB |
| 24 000 | 416 -> 38 | 719 MB -> **66 MB** |
| 100 000 | 100 -> 9 | 720 MB -> 65 MB |

Peak is now ~flat in N. **The PE VALUE is unchanged** — same pairs, same
upper-triangle mask, only block boundaries move (verified vs a dense reference:
rel diff 1.2e-16). Pure memory fix, no physics change, no cache-key change.

## Pool driver rules (learned the hard way on the cmx3 campaign)

- **Never a `wait`-barrier batch.** The original `cmx3_driver.sh` launched 4 then
  `wait`ed for ALL, so 3 slots idled behind each batch's slowest cell — measured 1 sim
  running at ~8% CPU on 8 logical threads.
- **Do not hand-roll a pool with `wait -n`.** It returns immediately for already-reaped
  jobs, so a `while`-spin inflates the counter (140 phantom "completions" for 72
  configs). Use `xargs -P N`.
- **A cell that writes no data row is a FAILURE, not a completion.** Log real OK/FAIL,
  delete the header-only stub CSV so a rerun recomputes it, and **refuse to write the
  verdict / done-marker unless every cell has data** — otherwise a partially-crashed
  run masquerades as a finished campaign.
- **Beware self-matching kills.** `kill` loops that match a process cmdline against a
  substring (e.g. `cmx3_pool`) also match the *agent's own shell*, which runs that
  string in its command line. Match the exact script path and exclude `$$`/`$PPID`.

Current cmx3 driver: `results/logs/cmx3_pool_driver.sh` (xargs pool, default 4 ==
physical cores on the i7-7700, resumable, OK/FAIL logging, guarded verdict).
