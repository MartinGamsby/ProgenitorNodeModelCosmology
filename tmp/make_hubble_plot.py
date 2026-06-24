"""One-off: Hubble-diagram residuals (vs LCDM) for an inline plot.
Curves: real Pantheon+ (binned), LCDM, Einstein-de Sitter (Om=1) matter null,
and the FROM-SIM external-node N-body curve. Offsets marginalized per model.
Outputs JSON between markers.
"""
import sys, os, io, json
sys.path.insert(0, os.getcwd())
import numpy as np

from cosmo.constants import SimulationParameters
from cosmo.pantheon import load_pantheon, bin_for_plot
from cosmo.distances import distance_modulus
from cosmo.hubble_diagram import fit_offset
from cosmo.sim_distance import sim_to_distance_modulus
from cosmo.factories import setup_simulation_context, run_external_node_simulation

H0 = 70.0  # offset marginalization absorbs absolute scale; only shape matters

data = load_pantheon()
zd, mud, sigd = data["z"], data["mu"], data["sigma"]

def mu_lcdm(z): return distance_modulus(np.asarray(z, float), 0.3, 0.7, H0)
def mu_eds(z):  return distance_modulus(np.asarray(z, float), 1.0, 0.0, H0)  # flat Einstein-de Sitter

# --- run ONE external-node N-body config (Omega_Lambda_eff ~ 0.7), full z ---
t_start, t_dur, n_steps = 2.9, 10.9, 300
BOX, A_START, _ = setup_simulation_context(t_start, t_dur, n_steps, 10)
sp = SimulationParameters(M_value=855.0, S_value=37.8, n_particles=2000, seed=42,
                          t_start_Gyr=t_start, t_duration_Gyr=t_dur, n_steps=n_steps,
                          damping_factor=None, center_node_mass=1, mass_randomize=0.0)
_real = sys.stdout
sys.stdout = io.StringIO()
ext = run_external_node_simulation(sp, BOX, A_START, 10)
sys.stdout = _real
a_sim = np.asarray(ext["a"]); t_Gyr_sim = np.asarray(ext["t_Gyr"])

a_today = a_sim / a_sim[-1]
z_sim = 1.0 / a_today - 1.0
z_lo, z_hi = float(z_sim.min()), float(z_sim.max())

def mu_ext_at(zq):
    res = sim_to_distance_modulus(np.asarray(zq, float), a_sim, t_Gyr_sim, t_start_Gyr=t_start)
    return np.asarray(res["z"]), np.asarray(res["mu"])

# --- offsets fit to data (same nuisance marginalization as chi2) ---
def fit_analytic(model_fn):
    dM, _ = fit_offset(mud, model_fn(zd), sigd)
    return dM

dM_lcdm = fit_analytic(mu_lcdm)
dM_eds  = fit_analytic(mu_eds)

# kernel clips z_target to the sim-covered range (contiguous); align data to that range
ze_fit, mue_fit = mu_ext_at(zd)
lo, hi = float(ze_fit[0]), float(ze_fit[-1])
m2 = (zd >= lo - 1e-9) & (zd <= hi + 1e-9)
assert int(m2.sum()) == len(mue_fit), (int(m2.sum()), len(mue_fit))
dM_ext, _ = fit_offset(mud[m2], mue_fit, sigd[m2])

# --- residuals vs LCDM ---
grid = np.geomspace(0.012, min(2.1, z_hi) * 0.999, 60)
r_lcdm = np.zeros_like(grid)
r_eds  = (mu_eds(grid) + dM_eds) - (mu_lcdm(grid) + dM_lcdm)
zeg, mueg = mu_ext_at(grid)
r_ext = (mueg + dM_ext) - (mu_lcdm(zeg) + dM_lcdm)

# --- binned data residual vs LCDM ---
b = bin_for_plot(zd, mud, sigd, 16)
zb, mub, sigb = b["z"], b["mu"], b["err"]
r_data = mub - (mu_lcdm(zb) + dM_lcdm)

out = {
    "z_grid": [float(x) for x in grid],
    "r_lcdm": [float(x) for x in np.atleast_1d(r_lcdm)],
    "r_eds":  [float(x) for x in np.atleast_1d(r_eds)],
    "ext_z":  [float(x) for x in np.atleast_1d(zeg)],
    "r_ext":  [float(x) for x in np.atleast_1d(r_ext)],
    "data_z": [float(x) for x in zb],
    "data_r": [float(x) for x in np.atleast_1d(r_data)],
    "data_e": [float(x) for x in np.atleast_1d(sigb)],
    "z_sim_range": [z_lo, z_hi],
    "n_data": int(len(zd)),
    "dM": {"lcdm": float(dM_lcdm), "eds": float(dM_eds), "ext": float(dM_ext)},
}
print("JSON_START"); print(json.dumps(out)); print("JSON_END")
