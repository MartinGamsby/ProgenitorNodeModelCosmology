"""
Particle-count convergence check for GRF initial conditions.

Sweeps N in {1_000, 10_000, 100_000} (or a subset bounded by time/memory),
using the 'grf' init_distribution and the 'barnes_hut' force method (via the
auto-selector in integrator.py which chooses barnes_hut for N >= 1000).

For each N it records:
  - growth: final RMS / initial RMS  (proxy for a_final/a_initial)
  - growth_anchor: pass/fail vs. expected_growth_factor(t_start) ± 20 %
  - dt_ok: dt < 0.05 Gyr
  - never_exceed_lcdm: max( size_grf / size_lcdm ) <= 1.001, where size_lcdm is
    the ANALYTIC LambdaCDM Friedmann a(t) scaled to the run's own initial RMS
    (particle-count independent — the gate reflects physics, not shot noise).
  - chi2_dof: Pantheon+ score (from compute_pantheon_metrics if available)

Invariants checked per N:
  1. never-exceed-LCDM  (vs analytic Friedmann a(t), not an N-body proxy)
  2. growth anchor (a[-1]/a[0] within 20 % of analytic LCDM reference)
  3. dt < 0.05 Gyr

Usage:
  python convergence_check.py            # N in {1000, 10000}; extend N_LIST for 1e5
  python convergence_check.py --full     # also run N=100000 (very slow)
  python convergence_check.py --png      # save results/convergence_check.png

PYTHONIOENCODING=utf-8 python convergence_check.py
"""

import sys
import argparse
import time
import numpy as np

# UTF-8 stdout/stderr guard (section-1 requirement). Use the shared helper so
# BOTH stdout and stderr are converted (a hand-rolled stdout-only wrapper would
# still crash on a non-ASCII traceback written to stderr on a cp1252 console).
from cosmo.encoding import configure_utf8_stdout
configure_utf8_stdout()

from cosmo.constants import CosmologicalConstants, LambdaCDMParameters, SimulationParameters
from cosmo.particles import ParticleSystem
from cosmo.integrator import LeapfrogIntegrator
from cosmo.analysis import solve_friedmann_at_times


def _rms(ps: ParticleSystem) -> float:
    pos = ps.get_positions()
    return float(np.sqrt(np.mean(np.sum(pos ** 2, axis=1))))


def run_one(N: int, t_start_Gyr: float = 5.8,
            t_duration_Gyr: float = 2.0, n_steps: int = 100,
            seed: int = 42, Ng: int = 64) -> dict:
    """Run a single GRF simulation at particle count N and return metrics."""
    const = CosmologicalConstants()

    # Derive a_start from Friedmann
    res0 = solve_friedmann_at_times(np.array([t_start_Gyr]))
    a_start = float(res0['a'][0])
    box_size_m = 10.0 * const.Gpc_to_m  # ~14 Gpc box

    dt_Gyr = t_duration_Gyr / n_steps
    dt_s = dt_Gyr * 1e9 * 365.25 * 24 * 3600

    # ---- GRF matter-only at N ----
    np.random.seed(seed)
    ps_grf = ParticleSystem(
        n_particles=N,
        box_size_m=box_size_m,
        total_mass_kg=const.M_observable_kg,
        a_start=a_start,
        use_dark_energy=False,
        mass_randomize=0.0,
        init_distribution="grf",
        init_kwargs={"Ng": Ng},
    )
    # force_method='auto' will choose barnes_hut for N >= 1000
    integ_grf = LeapfrogIntegrator(ps_grf, use_dark_energy=False, use_external_nodes=False)

    rms0_grf = _rms(ps_grf)
    grf_sizes = [rms0_grf]
    for _ in range(n_steps):
        integ_grf.step(dt_s)
        grf_sizes.append(_rms(ps_grf))
    grf_sizes = np.array(grf_sizes)

    growth = float(grf_sizes[-1] / grf_sizes[0])

    # ---- Analytic LCDM reference a(t) at the snapshot time grid ----
    # Canonical "never-exceed-LCDM" reference: the analytic LambdaCDM Friedmann
    # a(t) (Omega_Lambda=0.7), NOT an N-body cloud. This makes the gate depend on
    # physics, not on particle count / shot noise. Mirrors the established pattern
    # in tests/test_early_time_behavior.py (test_lcdm_nbody_vs_analytic_lcdm,
    # test_matter_only_decelerates_correctly): solve_friedmann_at_times with the
    # LCDM Omega_Lambda, then scale by the run's OWN initial RMS so both series
    # start at the same physical size.
    t_grid_Gyr = t_start_Gyr + np.linspace(0.0, t_duration_Gyr, n_steps + 1)
    a_lcdm = solve_friedmann_at_times(t_grid_Gyr)['a']   # Omega_Lambda defaults to LCDM (0.7)
    lcdm_sizes = grf_sizes[0] * (a_lcdm / a_lcdm[0])

    # Growth anchor (LCDM expansion over the window; particle-count independent)
    analytic_growth = float(a_lcdm[-1] / a_lcdm[0])
    growth_frac_err = abs(growth / analytic_growth - 1.0)
    growth_anchor_ok = growth_frac_err <= 0.20

    # Never-exceed-LCDM: GRF size must not exceed the analytic LCDM size at any step.
    ratio = grf_sizes / lcdm_sizes
    max_ratio = float(np.max(ratio))
    never_exceed_ok = max_ratio <= 1.001

    return {
        'N': N,
        'growth': growth,
        'analytic_growth': analytic_growth,
        'growth_frac_err': growth_frac_err,
        'growth_anchor_ok': growth_anchor_ok,
        'max_lcdm_ratio': max_ratio,
        'never_exceed_ok': never_exceed_ok,
        'dt_Gyr': dt_Gyr,
        'dt_ok': dt_Gyr < 0.05,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Particle-count convergence check (GRF + Barnes-Hut)')
    parser.add_argument('--full', action='store_true',
                        help='Include N=100_000 (very slow, requires ~hours)')
    parser.add_argument('--png', action='store_true',
                        help='Save convergence plot to results/convergence_check.png')
    parser.add_argument('--t-start', type=float, default=5.8,
                        help='Simulation start time in Gyr (default 5.8)')
    parser.add_argument('--t-duration', type=float, default=2.0,
                        help='Simulation duration in Gyr (default 2.0)')
    parser.add_argument('--n-steps', type=int, default=100,
                        help='Number of timesteps (dt = duration/n_steps; must keep dt < 0.05 Gyr)')
    parser.add_argument('--Ng', type=int, default=64,
                        help='GRF grid resolution (default 64)')
    args = parser.parse_args()

    # Validate dt constraint
    dt = args.t_duration / args.n_steps
    if dt >= 0.05:
        print(f"WARNING: dt={dt:.4f} Gyr >= 0.05 Gyr. "
              f"Increase --n-steps to at least {int(args.t_duration / 0.05) + 1}.")

    N_LIST = [1_000, 10_000]
    if args.full:
        N_LIST.append(100_000)

    print("=" * 72)
    print("GRF Particle-Count Convergence Check")
    print(f"  t_start={args.t_start} Gyr, duration={args.t_duration} Gyr, "
          f"n_steps={args.n_steps}, dt={dt:.4f} Gyr, Ng={args.Ng}")
    print(f"  N values: {N_LIST}")
    print("=" * 72)
    print()

    results = []
    for N in N_LIST:
        print(f"--- N = {N:,} ---")
        t0 = time.time()
        r = run_one(N,
                    t_start_Gyr=args.t_start,
                    t_duration_Gyr=args.t_duration,
                    n_steps=args.n_steps,
                    Ng=args.Ng)
        elapsed = time.time() - t0
        r['wall_s'] = elapsed
        results.append(r)
        print(f"  growth:           {r['growth']:.6f}  (analytic {r['analytic_growth']:.6f},"
              f" err={r['growth_frac_err']:.3%})")
        print(f"  growth_anchor:    {'PASS' if r['growth_anchor_ok'] else 'FAIL'}")
        print(f"  max LCDM ratio:   {r['max_lcdm_ratio']:.6f}  "
              f"({'PASS' if r['never_exceed_ok'] else 'FAIL'})")
        print(f"  dt={r['dt_Gyr']:.4f} Gyr  {'PASS' if r['dt_ok'] else 'FAIL'}")
        print(f"  wall time:        {elapsed:.1f} s")
        print()

    # ---- Summary table ----
    print("-" * 72)
    print(f"{'N':>10}  {'growth':>10}  {'anchor':>8}  {'<=LCDM':>8}  {'dt<0.05':>8}  {'time(s)':>8}")
    print("-" * 72)
    for r in results:
        print(f"{r['N']:>10,}  {r['growth']:>10.4f}  "
              f"{'PASS' if r['growth_anchor_ok'] else 'FAIL':>8}  "
              f"{'PASS' if r['never_exceed_ok'] else 'FAIL':>8}  "
              f"{'PASS' if r['dt_ok'] else 'FAIL':>8}  "
              f"{r['wall_s']:>8.1f}")
    print("-" * 72)
    print()

    # Convergence check: growth should stabilise
    if len(results) >= 2:
        growths = [r['growth'] for r in results]
        spread = max(growths) - min(growths)
        mean_g = np.mean(growths)
        print(f"Isotropic growth spread over N-ladder: {spread / mean_g:.3%}")
        if spread / mean_g < 0.05:
            print("CONVERGENCE: isotropic expansion is INSENSITIVE to N (expected).")
        else:
            print("WARNING: spread > 5 % — may need more N-steps or larger Ng.")
        print()
        print("Note: The isotropic background (growth, chi2_dof) is EXPECTED to be")
        print("insensitive to clustering. The value of GRF realism is in structure")
        print("(shear/dipole diagnostics in anisotropy_report.py), not in chi2_dof.")

    # ---- Optional PNG ----
    if args.png:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import os

            os.makedirs('results', exist_ok=True)
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))

            Ns = [r['N'] for r in results]
            growths = [r['growth'] for r in results]
            ratios = [r['max_lcdm_ratio'] for r in results]

            axes[0].semilogx(Ns, growths, 'o-', label='GRF growth')
            axes[0].axhline(results[0]['analytic_growth'], ls='--', color='gray',
                            label=f"analytic ({results[0]['analytic_growth']:.4f})")
            axes[0].set_xlabel('N particles')
            axes[0].set_ylabel('a_final / a_initial (RMS proxy)')
            axes[0].set_title('Isotropic growth vs N')
            axes[0].legend()

            axes[1].semilogx(Ns, ratios, 's-', color='tab:red', label='max(GRF/LCDM)')
            axes[1].axhline(1.0, ls='--', color='gray')
            axes[1].set_xlabel('N particles')
            axes[1].set_ylabel('max size ratio')
            axes[1].set_title('Never-exceed-LCDM invariant')
            axes[1].legend()

            plt.tight_layout()
            out = 'results/convergence_check.png'
            plt.savefig(out, dpi=100)
            print(f"Plot saved to {out}")
        except ImportError:
            print("matplotlib not available; --png skipped.")

    # Exit non-zero if any invariant failed
    failures = [r for r in results
                if not r['growth_anchor_ok'] or not r['never_exceed_ok'] or not r['dt_ok']]
    if failures:
        print(f"\nFAILURE: {len(failures)} N value(s) violated an invariant.")
        sys.exit(1)
    else:
        print("\nAll invariants passed.")
        sys.exit(0)


if __name__ == '__main__':
    main()
