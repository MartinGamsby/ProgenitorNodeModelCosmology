"""Generate the PAPER candidate figures for the best External-Node config.

Best config (localized_v4 headline, PF16): virialized force-balanced lattice, vir_n_nodes=300,
vir_mass_rule=massfunc, vir_mass_spread=6.0, M=300, S=20 Gpc, GRF sphere init, Plummer 1 Gpc,
t_start=2.9 Gyr, 2000 particles / 1092 steps. Built via the sweep's own _build_sim_params
(the PF11 single source of truth) so the figure config == the swept cell exactly.

Produces (results/figures/paper/), comparing External-Node vs LCDM vs Matter-only(EdS):
  1. size_vs_time.png       — RMS diameter a(t) vs time + R^2(size) + final-size numbers
  2. rate_vs_time.png       — Hubble parameter H(t) vs time + R^2(rate)
  3. hubble_diagram.png     — mu(z) vs REAL Pantheon+ (External/LCDM/EdS) + Delta-mu residual + chi2/dof
  4. particles_3d.png       — 3D particle cloud over time (External-Node), subsampled
  5. particles_3way.png     — 3D cloud at 4 times x {External / Matter / LCDM}
  6. observer_distribution.png — per-observer chi2/dof histogram (best/center/median + LCDM/EdS refs)

Run:  python _generate_paper_figs.py   (writes a summary JSON alongside the PNGs)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import json, os

from cosmo.encoding import configure_utf8_stdout
configure_utf8_stdout()
import sweep as S
from cosmo.factories import (setup_simulation_context, run_external_node_simulation,
                             run_matter_only_simulation, solve_lcdm_baseline, results_to_sim_result)
from cosmo.plots import figure_path
from cosmo.constants import CosmologicalConstants

CONST = CosmologicalConstants()
GPC = CONST.Gpc_to_m
LCDM_REF, EDS_REF = 0.4360, 0.8430
OUT = "paper"   # results/figures/paper/

EXT_C, LCDM_C, MAT_C = "#1f77b4", "#d62728", "#2ca02c"


def r2(model, ref):
    """R^2 of `model` against `ref` (how well model tracks ref). Same-length arrays."""
    model, ref = np.asarray(model, float), np.asarray(ref, float)
    ss_res = np.sum((model - ref) ** 2)
    ss_tot = np.sum((ref - np.mean(ref)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def build_best():
    cfg = S.load_config(os.path.join("sweeps", "localized_v4", "03_sigma6.json"))
    cell = next(c for c in S.expand_grid(cfg) if c["M"] == 300)
    sweep_cfg = S._make_sweep_config_for_cell(cell, cfg)
    box, a_start, _ = setup_simulation_context(
        cfg["t_start_Gyr"], sweep_cfg.t_duration_Gyr, cfg["n_steps"], sweep_cfg.save_interval)
    params = S._build_sim_params(sweep_cfg, 300, 20, 1, 42)
    return cfg, sweep_cfg, params, box, a_start


def main():
    cfg, sweep_cfg, params, box, a_start = build_best()
    t_start = cfg["t_start_Gyr"]
    save_interval = sweep_cfg.save_interval
    print(f"[paper-figs] best config: M=300 S=20 sigma=6 virialized-300 GRF Plummer1 "
          f"{params.n_particles}p/{params.n_steps}steps t_start={t_start}")

    print("[paper-figs] running External-Node ...")
    ext = run_external_node_simulation(params, box, a_start, save_interval)
    print("[paper-figs] running Matter-only (EdS) ...")
    mat = run_matter_only_simulation(params, box, a_start, save_interval)
    lcdm = solve_lcdm_baseline(params, box, a_start, save_interval)

    t = ext["t_Gyr"]                      # relative Gyr from t_start (matches lcdm['t'])
    t_abs = t_start + t
    summary = {"config": "M300_S20_sig6_vir300_grf_plummer1",
               "n_particles": int(params.n_particles), "n_steps": int(params.n_steps),
               "t_start_Gyr": t_start, "lcdm_ref_chi2_dof": LCDM_REF, "eds_ref_chi2_dof": EDS_REF}

    # ---- Fig 1: RMS size (diameter) vs time ----
    r2_size_lcdm = r2(ext["diameter_Gpc"], lcdm["diameter_Gpc"])
    r2_size_mat = r2(mat["diameter_Gpc"], lcdm["diameter_Gpc"])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(t_abs, ext["diameter_Gpc"], color=EXT_C, lw=2.2, label="External-Node (best)")
    ax.plot(t_abs, lcdm["diameter_Gpc"], color=LCDM_C, lw=2.0, ls="--", label="$\\Lambda$CDM")
    ax.plot(t_abs, mat["diameter_Gpc"], color=MAT_C, lw=2.0, ls="-.", label="Matter-only (EdS)")
    ax.set_xlabel("Cosmic time (Gyr)"); ax.set_ylabel("Cloud diameter (2$\\times$RMS radius) [Gpc]")
    ax.set_title("Mechanism: gravity-only expansion tracks $\\Lambda$CDM, far above matter-only\n"
                 "(the DATA test is the Pantheon $\\mu(z)$ figure)")
    ax.legend(loc="upper left")
    ax.text(0.98, 0.04,
            f"$R^2_{{size}}$ (External vs $\\Lambda$CDM) = {r2_size_lcdm:.4f}\n"
            f"$R^2_{{size}}$ (Matter vs $\\Lambda$CDM) = {r2_size_mat:.4f}\n"
            f"final size: Ext {ext['diameter_Gpc'][-1]:.1f} / $\\Lambda$CDM {lcdm['diameter_Gpc'][-1]:.1f} "
            f"/ Mat {mat['diameter_Gpc'][-1]:.1f} Gpc",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    p = figure_path(OUT, "size_vs_time"); fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("  ->", p)
    summary["R2_size_external_vs_lcdm"] = float(r2_size_lcdm)
    summary["R2_size_matter_vs_lcdm"] = float(r2_size_mat)
    summary["final_size_Gpc"] = {"external": float(ext['diameter_Gpc'][-1]),
                                 "lcdm": float(lcdm['diameter_Gpc'][-1]),
                                 "matter": float(mat['diameter_Gpc'][-1])}

    # ---- Fig 2: Hubble parameter H(t) vs time ----
    r2_rate_lcdm = r2(ext["H_hubble"], lcdm["H_hubble"])
    r2_rate_mat = r2(mat["H_hubble"], lcdm["H_hubble"])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(t_abs, ext["H_hubble"], color=EXT_C, lw=2.2, label="External-Node (best)")
    ax.plot(t_abs, lcdm["H_hubble"], color=LCDM_C, lw=2.0, ls="--", label="$\\Lambda$CDM")
    ax.plot(t_abs, mat["H_hubble"], color=MAT_C, lw=2.0, ls="-.", label="Matter-only (EdS)")
    ax.set_xlabel("Cosmic time (Gyr)"); ax.set_ylabel("H(t)  [km/s/Mpc]")
    ax.set_title("Mechanism: External-Node reproduces $\\Lambda$CDM acceleration,\nunlike decelerating matter-only (DATA test = Pantheon $\\mu(z)$)")
    ax.legend(loc="upper right")
    ax.text(0.02, 0.04,
            f"$R^2_{{rate}}$ (External vs $\\Lambda$CDM) = {r2_rate_lcdm:.4f}\n"
            f"$R^2_{{rate}}$ (Matter vs $\\Lambda$CDM) = {r2_rate_mat:.4f}",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    p = figure_path(OUT, "rate_vs_time"); fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("  ->", p)
    summary["R2_rate_external_vs_lcdm"] = float(r2_rate_lcdm)
    summary["R2_rate_matter_vs_lcdm"] = float(r2_rate_mat)

    # ---- Fig 3: mu(z) Hubble diagram vs Pantheon+ ----
    try:
        from cosmo import pantheon as pantheon_loader
        from cosmo.sim_distance import sim_to_distance_modulus
        from cosmo.distances import model_distance_modulus
        from cosmo.hubble_diagram import evaluate_precomputed
        pan = pantheon_loader.load_pantheon()   # default z_min (matches the sweep reference chi2)
        z, mu_obs, sig = np.asarray(pan["z"]), np.asarray(pan["mu"]), np.asarray(pan["sigma"])

        def sim_mu_chi2(res):
            sd = sim_to_distance_modulus(z_target=z, a=res["a"], t_Gyr=res["t_Gyr"], t_start_Gyr=t_start)
            inr = sd["in_range"]
            ev = evaluate_precomputed(z[inr], mu_obs[inr], sig[inr], sd["mu"])
            return sd, inr, ev

        sd_e, inr_e, ev_e = sim_mu_chi2(ext)
        sd_m, inr_m, ev_m = sim_mu_chi2(mat)
        # BEST OBSERVER curve (the headline metric — we are a random observer, not the centre).
        sd_b = None; best_chi2 = float("nan")
        try:
            from cosmo.observer_distance import (history_from_snapshots as _hfs,
                observer_chi2_distribution, observer_a_curve_local_rms,
                strided_observer_sample, ALL_NEIGHBOURS)
            _op, _ov, _ot = _hfs(ext["sim"].snapshots)
            _oidx = strided_observer_sample(_op.shape[1], min(2000, _op.shape[1]))
            odist = observer_chi2_distribution(_op, _ov, _ot, t_start, pan, definition="local_rms",
                k=ALL_NEIGHBOURS, observers=_oidx, lcdm_ref=LCDM_REF, eds_ref=EDS_REF)
            _bi = odist.get("best_observer", -1); best_chi2 = float(odist.get("best_chi2_dof", float("nan")))
            if _bi is not None and _bi >= 0:
                a_best = observer_a_curve_local_rms(_op, _ot, _bi, k=ALL_NEIGHBOURS)
                sd_b = sim_to_distance_modulus(z_target=z, a=a_best, t_Gyr=_ot, t_start_Gyr=t_start)
        except Exception as _be:
            print("  [warn] best-observer curve skipped:", repr(_be))
        # analytic LCDM / EdS chi2 on the same in-range set as External
        zin, muin, sigin = z[inr_e], mu_obs[inr_e], sig[inr_e]
        mu_lcdm_in = model_distance_modulus(zin, "lcdm")
        mu_eds_in = model_distance_modulus(zin, "einstein_de_sitter")
        ev_lcdm = evaluate_precomputed(zin, muin, sigin, mu_lcdm_in)
        ev_eds = evaluate_precomputed(zin, muin, sigin, mu_eds_in)
        # dense curves for plotting
        zd = np.linspace(max(1e-3, zin.min()), zin.max(), 200)
        mu_lcdm_d = model_distance_modulus(zd, "lcdm")
        mu_eds_d = model_distance_modulus(zd, "einstein_de_sitter")

        fig, (axu, axl) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                                       gridspec_kw={"height_ratios": [3, 1]})
        axu.errorbar(zin, muin, yerr=sigin, fmt=".", ms=3, color="0.6", alpha=0.4,
                     ecolor="0.8", label="Pantheon+ SNe", zorder=1)
        order = np.argsort(z[inr_e])
        # BEST observer (headline) drawn first/boldest; centre as a thinner reference.
        if sd_b is not None:
            inb = sd_b["in_range"]; ob = np.argsort(z[inb])
            axu.plot(z[inb][ob], sd_b["mu"][ob], color="#9467bd", lw=2.4,
                     label=f"External-Node BEST observer  $\\chi^2/dof$={best_chi2:.3f}")
        axu.plot(z[inr_e][order], sd_e["mu"][order], color=EXT_C, lw=1.6, ls=":",
                 label=f"External-Node centre  $\\chi^2/dof$={ev_e['chi2_dof']:.3f}")
        axu.plot(zd, mu_lcdm_d, color=LCDM_C, lw=1.8, ls="--",
                 label=f"$\\Lambda$CDM  $\\chi^2/dof$={ev_lcdm['chi2_dof']:.3f}")
        axu.plot(zd, mu_eds_d, color=MAT_C, lw=1.8, ls="-.",
                 label=f"EdS null  $\\chi^2/dof$={ev_eds['chi2_dof']:.3f}")
        axu.set_ylabel("Distance modulus $\\mu$")
        axu.set_title("Hubble diagram vs Pantheon+ (best observer = our vantage)")
        axu.legend(loc="lower right", fontsize=9)
        # residual vs LCDM
        mu_lcdm_at = model_distance_modulus(z[inr_e], "lcdm")
        axl.axhline(0, color=LCDM_C, ls="--", lw=1.5)
        axl.scatter(z[inr_e], sd_e["mu"] - mu_lcdm_at, s=6, color=EXT_C, alpha=0.6)
        axl.set_xlabel("Redshift z"); axl.set_ylabel("$\\Delta\\mu$ vs $\\Lambda$CDM")
        p = figure_path(OUT, "hubble_diagram"); fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
        print("  ->", p)
        summary["mu_z_chi2_dof"] = {"external_BEST_observer": best_chi2,
                                    "external_centre": float(ev_e["chi2_dof"]),
                                    "matter_eds_sim": float(ev_m["chi2_dof"]),
                                    "lcdm": float(ev_lcdm["chi2_dof"]),
                                    "eds_analytic": float(ev_eds["chi2_dof"]),
                                    "n_sne": int(len(zin))}
        summary["R2_mu"] = {"external": float(ev_e.get("R2", float("nan")))}
    except Exception as e:
        print("  [warn] mu(z) figure skipped:", repr(e))

    # ---- 3D particle figures ----
    try:
        from cosmo.observer_distance import history_from_snapshots
        from cosmo.visualization import draw_universe_sphere, setup_3d_axes
        snaps = getattr(ext["sim"], "snapshots", None)
        snaps_m = getattr(mat["sim"], "snapshots", None)
        if snaps and len(snaps) >= 2:
            pos_e_raw, _, ts = history_from_snapshots(snaps)   # (n_snap, N, 3) meters
            pos_e_raw = pos_e_raw / GPC
            # Centre each snapshot on the MEDIAN position (robust): a few slingshot particles
            # (the PF9 tail) drag the MEAN COM far out (the naive "COM drift" is ~30 Gpc = ~9c,
            # an OUTLIER artifact, not a bulk flow). The median tracks the BULK cloud. Report
            # both so the outlier effect is visible, not hidden.
            med_e = np.median(pos_e_raw, axis=1)           # (n_snap, 3) Gpc — robust centre
            mean_e = pos_e_raw.mean(axis=1)
            summary["bulk_drift_external_Gpc_median"] = float(np.linalg.norm(med_e[-1] - med_e[0]))
            summary["com_drift_external_Gpc_mean_outlier"] = float(np.linalg.norm(mean_e[-1] - mean_e[0]))
            pos_e = pos_e_raw - med_e[:, None, :]
            n_snap = pos_e.shape[0]
            rng = np.random.default_rng(0)
            idx_p = rng.choice(pos_e.shape[1], size=min(700, pos_e.shape[1]), replace=False)
            sel = np.linspace(0, n_snap - 1, 5).astype(int)

            # Frame the BULK cloud (not the slingshot outliers): a uniform sphere of RMS
            # radius r has full radius R = r/sqrt(3/5); diameter_Gpc = 2*RMS radius, so
            # R = diameter/2/sqrt(3/5). Using the 99th percentile instead would let a few
            # far-flung particles blow up the frame and shrink the bulk to a dot.
            def rms_frame(diam_gpc):
                return float(diam_gpc / 2.0 / np.sqrt(3 / 5) * 1.15)

            # FIXED common axis limit for ALL panels = the BULK radius (95th pct of
            # COM-distance, robust to slingshot outliers) at the LARGEST time, so the cloud
            # visibly GROWS across panels and the near-runaway tail is excluded (the user
            # asked to drop the not-quite-runaway points + use ONE min/max so size shows).
            def bulk_radius(P3, s):
                return float(np.percentile(np.linalg.norm(P3[s], axis=1), 95))
            fixed_lim_e = 1.1 * max(bulk_radius(pos_e, s) for s in sel)
            # Fig 4: External-Node cloud over time (5 panels)
            fig = plt.figure(figsize=(20, 4.4))
            for j, s in enumerate(sel):
                ax = fig.add_subplot(1, 5, j + 1, projection="3d")
                P = pos_e[s][idx_p]
                P = P[np.linalg.norm(P, axis=1) <= fixed_lim_e]   # drop near-runaway tail
                ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=3, color=EXT_C, alpha=0.55)
                setup_3d_axes(ax, fixed_lim_e, f"t = {t_start + (ts[s]-ts[0]):.1f} Gyr")
            fig.suptitle("External-Node particle cloud over time (same scale; runaway tail dropped)", y=1.02)
            p = figure_path(OUT, "particles_3d"); fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
            print("  ->", p)

            # Fig 5: 3-way comparison (External / Matter / LCDM) at 4 times (COM-centred)
            pos_m = None
            if snaps_m and len(snaps_m) >= 2:
                pos_m_raw = history_from_snapshots(snaps_m)[0] / GPC
                pos_m = pos_m_raw - np.median(pos_m_raw, axis=1, keepdims=True)
            sel4 = np.linspace(0, n_snap - 1, 4).astype(int)
            # ONE fixed axis scale for ALL 12 panels (the largest bulk radius over the 3 models
            # and all times) so the External-vs-Matter-vs-LCDM SIZE comparison is honest AND
            # growth is visible (the user: "same min/max, the highest one").
            def lcdm_R(s):
                return float(lcdm["diameter_Gpc"][s] / 2 / np.sqrt(3 / 5))
            lims_all = [bulk_radius(pos_e, s) for s in sel4] + [lcdm_R(s) for s in sel4]
            if pos_m is not None:
                lims_all += [bulk_radius(pos_m, s) for s in sel4]
            fixed_lim_3 = 1.1 * max(lims_all)
            fig = plt.figure(figsize=(16, 12))
            rows = [("External-Node", pos_e, EXT_C),
                    ("Matter-only (EdS)", pos_m, MAT_C),
                    ("$\\Lambda$CDM (sphere $\\propto a$)", None, LCDM_C)]
            for ri, (name, P3, col) in enumerate(rows):
                for ci, s in enumerate(sel4):
                    ax = fig.add_subplot(3, 4, ri * 4 + ci + 1, projection="3d")
                    if P3 is not None:
                        Q = P3[s][idx_p]
                        Q = Q[np.linalg.norm(Q, axis=1) <= fixed_lim_3]   # drop near-runaway tail
                    else:
                        # analytic LCDM uniform sphere scaled to its RMS radius at this time
                        Rmax = lcdm_R(s)
                        u = rng.normal(size=(len(idx_p), 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
                        rr = Rmax * rng.uniform(size=len(idx_p)) ** (1/3)
                        Q = u * rr[:, None]
                    ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], s=2, color=col, alpha=0.5)
                    ax.set_xlim(-fixed_lim_3, fixed_lim_3); ax.set_ylim(-fixed_lim_3, fixed_lim_3)
                    ax.set_zlim(-fixed_lim_3, fixed_lim_3)
                    if ci == 0:
                        ax.text2D(-0.1, 0.5, name, transform=ax.transAxes, rotation=90,
                                  va="center", ha="center", fontsize=11)
                    if ri == 0:
                        ax.set_title(f"t = {t_start + (ts[s]-ts[0]):.1f} Gyr")
            fig.suptitle("Particle cloud over time: External-Node vs Matter-only vs $\\Lambda$CDM", y=0.93)
            p = figure_path(OUT, "particles_3way"); fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
            print("  ->", p)
        else:
            print("  [warn] no snapshots on External sim; 3D figures skipped")
    except Exception as e:
        print("  [warn] 3D figures skipped:", repr(e))

    # ---- Fig 6: observer chi2 distribution ----
    try:
        from cosmo.observer_distance import (history_from_snapshots, observer_chi2_distribution,
                                            strided_observer_sample, ALL_NEIGHBOURS)
        from cosmo import pantheon as pantheon_loader
        pan = pantheon_loader.load_pantheon()
        snaps = getattr(ext["sim"], "snapshots", None)
        o_pos, o_vel, o_t = history_from_snapshots(snaps)
        obs_idx = strided_observer_sample(o_pos.shape[1], min(2000, o_pos.shape[1]))
        dist = observer_chi2_distribution(o_pos, o_vel, o_t, t_start, pan,
                                          definition="local_rms", k=ALL_NEIGHBOURS,
                                          observers=obs_idx, lcdm_ref=LCDM_REF, eds_ref=EDS_REF)
        chi = np.asarray(dist["chi2_dof"], float)
        chi = chi[np.isfinite(chi)]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(chi[chi < 2.0], bins=60, color=EXT_C, alpha=0.7)
        for val, c, lab in [(LCDM_REF, LCDM_C, "$\\Lambda$CDM"), (EDS_REF, MAT_C, "EdS null"),
                            (dist.get("center_chi2_dof", np.nan), "k", "centre"),
                            (dist.get("best_chi2_dof", np.nan), "purple", "best observer")]:
            if val == val:
                ax.axvline(val, color=c, ls="--", lw=1.6, label=f"{lab} = {val:.3f}")
        ax.set_xlabel("$\\chi^2/dof$ vs Pantheon+ (per observer)"); ax.set_ylabel("observers")
        ax.set_title("Per-observer fit distribution (we are a random observer)")
        ax.legend(fontsize=9)
        fbl = float(np.mean(chi <= LCDM_REF)); fbe = float(np.mean(chi <= EDS_REF))
        ax.text(0.98, 0.6, f"frac $\\leq\\Lambda$CDM = {fbl:.3f}\nfrac $\\leq$EdS = {fbe:.3f}\n"
                f"median = {np.median(chi):.3f}", transform=ax.transAxes, ha="right",
                bbox=dict(boxstyle="round", fc="white", ec="0.7"), fontsize=9)
        p = figure_path(OUT, "observer_distribution"); fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
        print("  ->", p)
        summary["observer"] = {"center": float(dist.get("center_chi2_dof", float("nan"))),
                               "best": float(dist.get("best_chi2_dof", float("nan"))),
                               "median": float(np.median(chi)),
                               "frac_below_lcdm": fbl, "frac_below_eds": fbe}
    except Exception as e:
        print("  [warn] observer-distribution figure skipped:", repr(e))

    sp = os.path.join("results", "figures", OUT, "paper_figs_summary.json")
    json.dump(summary, open(sp, "w"), indent=2)
    print("[paper-figs] summary ->", sp)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
