"""Generate the replacement fig4_pantheon: the BEST-OBSERVER mu(z) of the headline 100k
config (M300/S20/sigma6) vs Pantheon+SH0ES, LCDM, and the EdS null, with a residual panel.
Replaces the old 0.501 central-observer figure with the ~0.44 best-observer result."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from cosmo.encoding import configure_utf8_stdout
configure_utf8_stdout()
from cosmo import pantheon as pl
from cosmo.sim_distance import sim_to_distance_modulus
from cosmo.distances import model_distance_modulus
from cosmo.hubble_diagram import evaluate_precomputed

LCDM_C, EXT_C, EDS_C = "#1f77b4", "#ff7f0e", "#2ca02c"
d = np.load("results/hero/hero_M300_S20_sig6_100k_12ksteps.npz")
t_start = float(d["t_start"])
pan = pl.load_pantheon(); z = np.asarray(pan["z"]); mu = np.asarray(pan["mu"]); sg = np.asarray(pan["sigma"])

# best-observer sim mu(z)
sd = sim_to_distance_modulus(z_target=z, a=d["a_best"], t_Gyr=d["obs_t"], t_start_Gyr=t_start)
inr = sd["in_range"]; zin, muin, sgin = z[inr], mu[inr], sg[inr]
ev = evaluate_precomputed(zin, muin, sgin, sd["mu"])
mu_l = model_distance_modulus(zin, "lcdm"); mu_e = model_distance_modulus(zin, "einstein_de_sitter")
ev_l = evaluate_precomputed(zin, muin, sgin, mu_l); ev_e = evaluate_precomputed(zin, muin, sgin, mu_e)
zd = np.linspace(max(1e-3, zin.min()), zin.max(), 240)

fig, (ax, axr) = plt.subplots(2, 1, figsize=(7.2, 7.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
o = np.argsort(zin)
ax.errorbar(zin, muin, yerr=sgin, fmt=".", ms=2.5, color="0.6", alpha=0.35, ecolor="0.85",
            label="Pantheon+SH0ES", zorder=1)
ax.plot(zin[o], sd["mu"][o], color=EXT_C, lw=2.4, zorder=4,
        label=f"External-Node (best observer)  $\\chi^2/dof={ev['chi2_dof']:.3f}$")
ax.plot(zd, model_distance_modulus(zd, "lcdm"), color=LCDM_C, lw=1.9, ls="--",
        label=f"$\\Lambda$CDM  $\\chi^2/dof={ev_l['chi2_dof']:.3f}$")
ax.plot(zd, model_distance_modulus(zd, "einstein_de_sitter"), color=EDS_C, lw=1.9, ls="-.",
        label=f"Einstein--de Sitter null  $\\chi^2/dof={ev_e['chi2_dof']:.3f}$")
ax.set_ylabel("Distance modulus $\\mu$"); ax.legend(loc="lower right", fontsize=9)
ax.set_title("From-simulation External-Node $\\mu(z)$ vs Pantheon+SH0ES (best observer)")
mu_l_at = model_distance_modulus(zin, "lcdm")
axr.axhline(0, color=LCDM_C, ls="--", lw=1.4)
axr.scatter(zin, sd["mu"] - mu_l_at, s=5, color=EXT_C, alpha=0.5)
axr.set_xlabel("Redshift z"); axr.set_ylabel("$\\Delta\\mu$ vs $\\Lambda$CDM")
p = "docs/fig4_pantheon.png"; fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"wrote {p}: best-obs chi2/dof={ev['chi2_dof']:.4f}, LCDM={ev_l['chi2_dof']:.4f}, EdS={ev_e['chi2_dof']:.4f}, n={len(zin)}")
