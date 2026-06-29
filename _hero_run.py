"""Run ONE high-resolution "hero" config (e.g. 100k particles) and save everything
needed for great images + the chi2 numbers — WITHOUT the sweep's figure-rerun/CSV
overhead. Reuses the sweep machinery (_build_sim_params, the External-Node runner,
compute_pantheon_metrics) so the config == a real swept cell.

Usage:  python _hero_run.py sweeps/hero/01_M300_S20_sig6.json

Writes:
  results/hero/<tag>.npz          — downsampled MEDIAN-centred particle snapshots (Gpc) +
                                     times + diameter(t) + centre a(t) + best-observer a(t),
                                     for the image scripts (no re-run needed).
  results/hero/<tag>_result.json  — center/best-observer chi2, frac_below_lcdm/eds, growth,
                                     n_particles/n_steps, com-drift (mean vs median).
"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
from cosmo.encoding import configure_utf8_stdout
configure_utf8_stdout()
import sweep as S
from cosmo.factories import (setup_simulation_context, run_external_node_simulation,
                             results_to_sim_result)
import cosmo.parameter_sweep as PS
from cosmo.pantheon import load_pantheon
from cosmo.constants import CosmologicalConstants
from cosmo.observer_distance import observer_a_curve_local_rms, ALL_NEIGHBOURS

GPC = CosmologicalConstants().Gpc_to_m
LCDM_REF, EDS_REF = 0.4360, 0.8430
N_SNAP_KEEP = 60        # snapshots to save (image smoothness vs memory)
N_PART_KEEP = 6000      # particles per snapshot to save for the 3D scatter


def main(cfg_path):
    cfg = S.load_config(cfg_path)
    cell = S.expand_grid(cfg)[0]
    sc = S._make_sweep_config_for_cell(cell, cfg)
    M, Sv = int(cfg["M_values"][0]), int(cfg["S_values"][0])
    tag = cfg.get("tag", os.path.splitext(os.path.basename(cfg_path))[0])
    box, a0, _ = setup_simulation_context(cfg["t_start_Gyr"], sc.t_duration_Gyr,
                                          cfg["n_steps"], sc.save_interval)
    params = S._build_sim_params(sc, M, Sv, 1, cfg.get("node_mass_seeds", [42])[0])
    n_steps = params.n_steps
    save_interval = max(1, n_steps // N_SNAP_KEEP)
    t_start = cfg["t_start_Gyr"]
    print(f"[hero {tag}] M={M} S={Sv} sigma={cell.get('vir_spread')} "
          f"{params.n_particles}p / {n_steps} steps, save_interval={save_interval}", flush=True)

    t0 = time.time()
    ext = run_external_node_simulation(params, box, a0, save_interval=save_interval)
    print(f"[hero {tag}] sim done in {(time.time()-t0)/60:.1f} min", flush=True)

    # --- metrics (centre + best observer; observer_sample modest at high N) ---
    pan = load_pantheon()
    sr = results_to_sim_result(ext, params)
    osample = int(cfg.get("observer_sample", 500))
    m = PS.compute_pantheon_metrics(sr, pan, t_start, score_observers=True,
                                    observer_sample=osample, observer_k=-1,
                                    lcdm_ref=LCDM_REF, eds_ref=EDS_REF)

    # --- snapshot history -> median-centred, downsampled particle positions (Gpc) ---
    snaps = ext["sim"].snapshots
    n_snap = len(snaps)
    keep_s = np.unique(np.linspace(0, n_snap - 1, min(N_SNAP_KEEP, n_snap)).astype(int))
    N = snaps[0]["positions"].shape[0]
    rng = np.random.default_rng(0)
    keep_p = rng.choice(N, size=min(N_PART_KEEP, N), replace=False)
    pos_keep = np.stack([(snaps[s]["positions"][keep_p] / GPC) for s in keep_s])   # (ns, np, 3)
    com_mean = np.stack([snaps[s]["positions"].mean(0) / GPC for s in keep_s])
    com_med = np.stack([np.median(snaps[s]["positions"], 0) / GPC for s in keep_s])
    pos_keep_centred = pos_keep - com_med[:, None, :]
    times = np.array([snaps[s]["time_s"] for s in keep_s]) / (1e9 * 365.25 * 24 * 3600)
    times_rel = times - times[0]
    diam = np.array(ext["diameter_Gpc"])
    a_curve = np.array(ext["a"])
    t_Gyr = np.array(ext["t_Gyr"])

    # best-observer a(t) for the mu(z) image
    a_best = None
    try:
        from cosmo.observer_distance import history_from_snapshots
        op, ov, ot = history_from_snapshots(snaps)
        bi = int(m.get("best_observer", -1)) if m.get("best_observer") is not None else -1
        # compute_pantheon_metrics doesn't return the index; recompute the best observer index
        from cosmo.observer_distance import observer_chi2_distribution, strided_observer_sample
        oidx = strided_observer_sample(op.shape[1], min(osample, op.shape[1]))
        od = observer_chi2_distribution(op, ov, ot, t_start, pan, definition="local_rms",
                                        k=ALL_NEIGHBOURS, observers=oidx,
                                        lcdm_ref=LCDM_REF, eds_ref=EDS_REF)
        bi = int(od.get("best_observer", -1))
        if bi >= 0:
            a_best = observer_a_curve_local_rms(op, ot, bi, k=ALL_NEIGHBOURS)
            obs_t = ot
    except Exception as e:
        print(f"[hero {tag}] best-observer a(t) skipped: {e!r}", flush=True)
        obs_t = t_Gyr

    os.makedirs("results/hero", exist_ok=True)
    np.savez_compressed(
        f"results/hero/{tag}.npz",
        tag=tag, M=M, S=Sv, sigma=float(cell.get("vir_spread", 0.0)),
        n_particles=params.n_particles, n_steps=n_steps, t_start=t_start,
        pos_centred=pos_keep_centred.astype(np.float32), times_rel=times_rel,
        diameter_Gpc=diam, a_curve=a_curve, t_Gyr=t_Gyr,
        a_best=(a_best if a_best is not None else np.array([])),
        obs_t=(obs_t if a_best is not None else np.array([])),
        com_mean=com_mean, com_med=com_med,
    )

    drift_mean = float(np.linalg.norm(com_mean[-1] - com_mean[0]))
    drift_med = float(np.linalg.norm(com_med[-1] - com_med[0]))
    result = {
        "tag": tag, "M": M, "S": Sv, "sigma": float(cell.get("vir_spread", 0.0)),
        "n_particles": int(params.n_particles), "n_steps": int(n_steps),
        "center_chi2_dof": float(m.get("center_chi2_dof", float("nan"))),
        "best_observer_chi2": float(m.get("best_observer_chi2", float("nan"))),
        "chi2_lcdm": LCDM_REF, "chi2_eds": EDS_REF,
        "frac_below_lcdm": float(m.get("frac_below_lcdm", float("nan"))),
        "frac_below_eds": float(m.get("frac_below_eds", float("nan"))),
        "growth_factor": float(m.get("growth_factor", float("nan"))),
        "com_drift_mean_Gpc": drift_mean, "com_drift_median_Gpc": drift_med,
        "runtime_min": (time.time() - t0) / 60.0,
    }
    json.dump(result, open(f"results/hero/{tag}_result.json", "w"), indent=2)
    print(f"[hero {tag}] RESULT {json.dumps(result)}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
