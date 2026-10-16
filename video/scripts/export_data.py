"""Export REAL repo data for the explainer video -> video/public/data/*.json.

Every number the video shows on a chart comes from here (or from video/src/script.ts for
narrated round numbers). Re-run after new results:  npm run data   (from video/)

Sources (all repo-local, nothing is fetched):
  data/pantheon_plus/Pantheon+SH0ES.dat          real SNe (cosmo.pantheon.load_pantheon)
  results/hero/<HERO>.npz                         best-observer a(t) + particle snapshots
  results/hero/<KNOT>.npz                         a collapsing-core (knot) run, for contrast
  data/vir_medium/<MEDIUM>.npz                    relaxed virialized medium (node positions)
  results/figures/centerm/cmx3_verdict.csv        the 72-cell universe-like verdict (8k)
"""
import json
import os
import sys

import numpy as np

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)
os.chdir(REPO)

from cosmo.pantheon import load_pantheon, bin_for_plot  # noqa: E402
from cosmo.distances import model_distance_modulus  # noqa: E402
from cosmo.sim_distance import sim_to_distance_modulus  # noqa: E402
from cosmo.hubble_diagram import evaluate_precomputed, fit_offset  # noqa: E402
from cosmo.node_geometry import nearest_neighbour_spacing  # noqa: E402

OUT = os.path.join(REPO, "video", "public", "data")
HERO = "results/hero/cmxconv_s0_cm3000_16k.npz"     # converged universe-like cell (PF26)
KNOT = "results/hero/medium_best.npz"               # best-chi2 but collapsing core (PF22)
MEDIUM = "data/vir_medium/medium_N301_sig0.5000_seed0_cm1.0000.npz"
VERDICT = "results/figures/centerm/cmx3_verdict.csv"
T0 = 13.8
CLOUD_POINTS = 2200


def r(x, nd=4):
    return [round(float(v), nd) for v in np.asarray(x).ravel()]


def dump(name, obj):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, separators=(",", ":"))
    print(f"wrote {name} ({os.path.getsize(os.path.join(OUT, name)) // 1024} KB)")


def inner_selection(d, centerM):
    """Rows of pos_centred that are inner (observable) particles; mirrors sweep.save_snapshots."""
    N = int(d["pos_centred"].shape[1])
    n_total_saved = int(d["n_particles"])
    n_inner = int(round(n_total_saved / (1 + min(max(centerM - 1, 0), 2))))
    rng = np.random.default_rng(0)
    keep = rng.choice(n_total_saved, size=min(6000, n_total_saved), replace=False)
    assert len(keep) == N
    return keep < n_inner, n_inner


def knot_ratio(P):
    rf = np.linalg.norm(P[-1], axis=1)
    cs = rf < 0.25 * np.percentile(rf, 90)
    return float(np.median(np.linalg.norm(P[-1][cs], axis=1)) / np.median(np.linalg.norm(P[0][cs], axis=1))), cs


def cloud_json(path, centerM, label):
    d = np.load(path)
    sel, n_inner = inner_selection(d, centerM)
    P = d["pos_centred"][:, sel, :].astype(float)
    k, core = knot_ratio(P)
    rng = np.random.default_rng(1)
    idx = np.sort(rng.choice(P.shape[1], size=min(CLOUD_POINTS, P.shape[1]), replace=False))
    t = float(d["t_start"]) + np.asarray(d["times_rel"])
    return dict(
        label=label, M=int(d["M"]), S=float(d["S"]), centerM=centerM,
        n_particles=int(d["n_particles"]), n_inner=n_inner, knot=round(k, 3),
        best_observer_chi2=round(float(d["best_observer_chi2"]), 4),
        growth=round(float(d["growth_factor"]), 3),
        t_Gyr=r(t, 3),
        core=[bool(c) for c in core[idx]],
        # frames x points x 3, Gpc, 2 decimals
        frames=[[r(p, 2) for p in P[s][idx]] for s in range(P.shape[0])],
    ), d


def lcdm_a(t, Om=0.3, H0_per_Gyr=70.0 / 977.792):
    """Flat LCDM a(t) (shape only; renormalised by the caller). 977.792 = (km/s/Mpc)^-1 in Gyr."""
    return np.sinh(1.5 * np.sqrt(1 - Om) * H0_per_Gyr * t) ** (2 / 3)


def main():
    pan = load_pantheon()
    z, mu, sig = pan["z"], pan["mu"], pan["sigma"]

    # ---- particle clouds -------------------------------------------------------------------
    hero, dh = cloud_json(HERO, 3000, "universe-like")
    knot, _ = cloud_json(KNOT, 1, "collapsing core")
    dump("cloud_hero.json", hero)
    dump("cloud_knot.json", knot)

    # ---- Hubble diagram ----------------------------------------------------------------------
    t_obs = np.asarray(dh["obs_t"], float)
    a_best = np.asarray(dh["a_best"], float)
    t_start = float(dh["t_start"])
    dm = sim_to_distance_modulus(z, a_best, t_obs, t_start_Gyr=t_start)
    m = dm["in_range"]
    ev_model = evaluate_precomputed(z[m], mu[m], sig[m], dm["mu"])
    mu_l, mu_e = model_distance_modulus(z, "lcdm"), model_distance_modulus(z, "einstein_de_sitter")
    ev_l = evaluate_precomputed(z, mu, sig, mu_l, "lcdm")
    ev_e = evaluate_precomputed(z, mu, sig, mu_e, "eds")
    print(f"chi2/dof  model(best obs)={ev_model['chi2_dof']:.4f} (npz says {float(dh['best_observer_chi2']):.4f})"
          f"  LCDM={ev_l['chi2_dof']:.4f}  EdS={ev_e['chi2_dof']:.4f}  n_model={int(m.sum())}")

    # Offsets: put every curve on the data (marginalised offsets, as the chi2 does).
    zg = np.geomspace(0.01, 2.3, 160)
    off_l = fit_offset(mu, mu_l, sig)[0]
    off_e = fit_offset(mu, mu_e, sig)[0]
    off_m = fit_offset(mu[m], dm["mu"], sig[m])[0]
    curve_l = model_distance_modulus(zg, "lcdm") + off_l
    curve_e = model_distance_modulus(zg, "einstein_de_sitter") + off_e
    dg = sim_to_distance_modulus(zg, a_best, t_obs, t_start_Gyr=t_start)
    zgm = zg[dg["in_range"]]
    curve_m = dg["mu"] + off_m

    b = bin_for_plot(z, mu - (mu_l + off_l), sig, n_bins=18) if "n_bins" in bin_for_plot.__code__.co_varnames \
        else bin_for_plot(z, mu - (mu_l + off_l), sig)
    dump("hubble.json", dict(
        sne=dict(z=r(z, 4), mu=r(mu, 3), sigma=r(sig, 3)),
        n_sne=int(len(z)), n_model=int(m.sum()),
        grid=dict(z=r(zg, 4), lcdm=r(curve_l, 3), eds=r(curve_e, 3)),
        model=dict(z=r(zgm, 4), mu=r(curve_m, 3),
                   resid=r(curve_m - (model_distance_modulus(zgm, "lcdm") + off_l), 4)),
        resid=dict(eds=r(curve_e - curve_l, 4),
                   bins_z=r(b["z"], 4), bins_d=r(b["mu"], 4), bins_err=r(b["err"], 4)),
        chi2=dict(model=round(ev_model["chi2_dof"], 3), lcdm=round(ev_l["chi2_dof"], 3),
                  eds=round(ev_e["chi2_dof"], 3)),
        config=dict(M=int(dh["M"]), S=float(dh["S"]), centerM=3000, n=int(dh["n_particles"])),
    ))

    # ---- size of the universe vs time (a normalised to 1 at t_start) -------------------------
    tg = np.linspace(t_start, T0, 120)
    al = lcdm_a(tg) / lcdm_a(np.array([t_start]))[0]
    ae = (tg / t_start) ** (2 / 3)
    am = np.interp(tg, t_start + t_obs - t_obs[0], a_best / a_best[0])
    dump("growth.json", dict(t=r(tg, 3), lcdm=r(al), eds=r(ae), model=r(am)))

    # ---- virialized medium (node positions, Gpc) ---------------------------------------------
    md = np.load(MEDIUM)
    pos, w = md["pos"][1:], md["weights"][1:]   # drop node 0 ("us" = the progenitor/cloud)
    S = float(hero["S"])
    pos = pos * (S / nearest_neighbour_spacing(pos, "median"))
    dump("medium.json", dict(S=S, pos=[r(p, 2) for p in pos], mass=r(w / w.mean(), 3),
                             progenitor_mass=round(float(md["weights"][0] / w.mean()), 3)))

    # ---- 72-cell universe-like verdict -------------------------------------------------------
    import csv
    rows = list(csv.DictReader(open(VERDICT)))
    def num(v, nd):
        try:
            f = float(v)
        except ValueError:
            return None
        return round(f, nd) if np.isfinite(f) else None   # runaway cells have no fit

    cells = [dict(cm=int(float(x["centerM"])), seed=int(float(x["seed"])),
                  bo=num(x["bo"], 4), knot=num(x["knot"], 3),
                  ok=x["qualifies"] == "True") for x in rows]
    print(f"verdict: {sum(c['ok'] for c in cells)}/{len(cells)} universe-like")
    dump("verdict.json", dict(cells=cells))


if __name__ == "__main__":
    main()
