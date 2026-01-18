# Timestep Stability Requirements

## Problem

Leapfrog integrator (kick-drift-kick) is symplectic and conserves energy well **IF** timestep is small enough. With too large timesteps, the integrator becomes unstable and spuriously injects energy into the system, causing matter-only simulations to expand faster than ΛCDM (physically impossible).

## Failure Mode Observed

**Configuration**: 20 Gyr simulation, 50 particles, damping=0.9, 150 steps
- Timestep: dt = 0.133 Gyr = 4.21e15 s
- Crossing time: t_cross ≈ 15.5 Gyr
- Steps per crossing time: 116 (marginally stable)

**Symptoms**:
1. **Steps 1-76 (t=3.93-13.93 Gyr)**: Normal behavior
   - Velocities decrease: 0.995x → 0.888x initial
   - Radial acceleration negative (inward): ~-4e-10 to -7e-11 m/s²
   - Energy drift: ~7.3% (acceptable)

2. **Step 91 (t=15.93 Gyr)**: SUDDEN CATASTROPHIC FAILURE
   - Radial velocity: 368,425 → 495,837 km/s (+35% in ONE step!)
   - Tangential velocity: 43,099 → 306,625 km/s (7x increase!)
   - Radial acceleration STILL NEGATIVE (-6.091e-11 m/s²)
   - Energy drift: 7.3% → 214.9%

3. **Steps 91-150**: Runaway continues
   - Velocity increases to 3.15x initial
   - Energy drift reaches 1627% (16x initial energy)
   - Final expansion: 54.64 Gpc (vs LCDM's 27.74 Gpc)

**Physical impossibility**: Inward acceleration CANNOT cause outward velocity increase. This is spurious energy injection from integrator instability.

## Root Cause

Leapfrog assumes forces are smooth over the timestep. When dt is too large:
1. Particles move significant distances during one step
2. Force changes significantly between start and end of step
3. The "kick-drift-kick" approximation breaks down
4. Energy is not conserved; system gains kinetic energy artificially

The instability manifests when particles reach configurations where local dynamical time < timestep. At step ~91, some particle pair likely had close approach with acceleration changing rapidly, but timestep too large to resolve it accurately.

## Solution: More Timesteps

**Same configuration with 500 steps instead of 150**:
- Timestep: dt = 0.040 Gyr = 1.26e15 s (3.3x smaller)
- Steps per crossing time: 387 (well-resolved)

**Result**: Matter-only = 27.29 Gpc (+1.6% vs LCDM) ✓ Correct behavior!

Energy remains stable, no spurious injection, physically sensible deceleration.

## Timestep Requirements

**Rule of thumb**: Need dt << t_dyn where t_dyn is shortest dynamical timescale in system.

For N-body cosmology:
- Crossing time: t_cross = box_size / v_rms
- Close encounter time: t_enc = min_separation / relative_velocity
- Dynamical time: t_dyn = min(t_cross, t_enc)

**Recommended**:
- Minimum 250-500 steps per crossing time
- For 20 Gyr simulations: dt < 0.05 Gyr
- For safety: Always use n_steps ≥ 500 for production runs

**Warning signs of insufficient timesteps**:
- Energy drift > 10% (should be < 1% for well-resolved)
- Sudden velocity jumps between consecutive steps
- Matter-only expanding faster than ΛCDM
- Particles accelerating opposite to net force direction

## Energy Monitoring

The integrator tracks energy at 10 points during evolution. Energy drift formula:

```
dE/E0 = (E - E_initial) / |E_initial|
```

Expected behavior:
- **Well-resolved**: |dE/E0| < 1% (leapfrog conserves energy to machine precision)
- **Marginally stable**: |dE/E0| ~ 5-10% (acceptable for exploratory runs)
- **Unstable**: |dE/E0| > 50% or sudden jumps (INVALID, increase n_steps)

## Diagnostic Tools

Created diagnostic scripts in project root:
- `diagnose_matter_only.py` - Tracks RMS radius, velocity, acceleration, energy every 15 steps
- `diagnose_distances.py` - Monitors minimum inter-particle separation vs softening length
- `diagnose_velocity_components.py` - Splits velocity into radial/tangential components

These reveal the step where instability onset occurs and whether it's due to close encounters, large-scale dynamics, or timestep errors.

## Implementation

**Automatic validation added** (cosmo/simulation.py:91-140): The `_validate_timestep()` method checks timestep before running simulation.

**Behavior**:
- **dt > 0.05 Gyr**: ERROR - exits with detailed message, refuses to run
- **0.04 Gyr < dt ≤ 0.05 Gyr**: WARNING - shows recommendation but allows run
- **dt ≤ 0.04 Gyr**: Silent - proceeds normally

**Error message format**:
```
======================================================================
ERROR: INSUFFICIENT TIMESTEPS FOR NUMERICAL STABILITY
======================================================================
Simulation duration: 20.0 Gyr
Requested steps:     150
Timestep (dt):       0.1333 Gyr

The leapfrog integrator becomes unstable with timesteps > 0.05 Gyr.
This causes spurious energy injection, making matter-only simulations
expand faster than LCDM (physically impossible).

MINIMUM steps required:    400 (dt < 0.050 Gyr)
RECOMMENDED steps:         500 (dt < 0.040 Gyr)

Example: For a 20 Gyr simulation, use --n-steps 500 or more
======================================================================
```

This prevents users from accidentally running unstable simulations and getting invalid results.

## Related Files

- cosmo/integrator.py:230-274 - Leapfrog evolve() function
- tests/test_model_comparison.py - Uses n=20 steps (short test duration, acceptable)
- run_simulation.py - User specifies --n-steps parameter
- lode/architecture/testing.md - Documents this fix
