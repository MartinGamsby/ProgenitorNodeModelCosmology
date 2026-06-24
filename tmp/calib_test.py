"""Diagnostic: is the velocity calibration masking the external-node effect?
Run M=20 vs M=100000 (huge node-mass difference), with calibration ON (damping=None,
auto) and OFF (damping=1.0, fixed full Hubble flow). Compare the resulting a(t).
If a(t) diverges strongly only with calibration OFF, the calibration was absorbing
the node effect. If a(t) barely moves either way, the nodes contribute little.
"""
import sys, os, io, re
sys.path.insert(0, os.getcwd())
import numpy as np
from cosmo.constants import SimulationParameters
from cosmo.factories import setup_simulation_context, run_external_node_simulation

t_start, t_dur, n_steps, npart, S = 2.9, 10.9, 300, 800, 37.8
BOX, A_START, _ = setup_simulation_context(t_start, t_dur, n_steps, 10)

def run(M, damping):
    sp = SimulationParameters(M_value=M, S_value=S, n_particles=npart, seed=42,
                              t_start_Gyr=t_start, t_duration_Gyr=t_dur, n_steps=n_steps,
                              damping_factor=damping, center_node_mass=1, mass_randomize=0.0)
    buf = io.StringIO(); real = sys.stdout; sys.stdout = buf
    ext = run_external_node_simulation(sp, BOX, A_START, 10)
    sys.stdout = real
    log = buf.getvalue()
    m = re.search(r"[Vv]elocity scale factor:\s*([0-9.]+)", log)
    vscale = float(m.group(1)) if m else float('nan')
    a = np.asarray(ext["a"])
    return a[-1] / a[0], vscale   # growth factor a_final/a_start

print(f"{'M':>8} {'calibration':>12} {'a_final/a_start':>16} {'vel_scale':>10}")
print("-"*50)
res = {}
for damping, name in [(None, "ON(auto)"), (1.0, "OFF(fixed)")]:
    for M in [20, 100000]:
        g, vs = run(M, damping)
        res[(name, M)] = g
        print(f"{M:>8} {name:>12} {g:>16.4f} {vs:>10.4f}")

print()
for name in ["ON(auto)", "OFF(fixed)"]:
    lo, hi = res[(name, 20)], res[(name, 100000)]
    print(f"calibration {name:>11}: growth(M=20)={lo:.4f}  growth(M=100000)={hi:.4f}  "
          f"ratio={hi/lo:.4f}  ({(hi/lo-1)*100:+.1f}% from 5000x more node mass)")
