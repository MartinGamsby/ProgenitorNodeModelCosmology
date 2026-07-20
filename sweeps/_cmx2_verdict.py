"""Self-contained cmx2 verdict (runs at the end of the guarded re-sweep driver, no AI needed):
builds the guarded universe-fit table straight from the cmx2 CSVs (knot_ratio is a native column
now) + recomputes the 15 Gpc nearest-HMEA gate at each cell's SELECTED S. Compares against the
unguarded campaign: how many knot-failing cells did the guard rescue, at what chi2 cost?
Writes results/figures/centerm/cmx2_verdict.{csv,txt}.
"""
import os, sys, glob, re, csv
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
import numpy as np
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "cmx2"
from cosmo.node_geometry import build_virialized_grid
from cosmo.constants import CosmologicalConstants

GPC = CosmologicalConstants.Gpc_to_m
M_VALUE = 3000

def _f(v):
    try: return float(v)
    except (TypeError, ValueError): return None

old = {}
for r in csv.DictReader(open("results/figures/centerm/universe_fit_ranking.csv")):
    old[(int(float(r["centerM"])), int(float(r["seed"])))] = r

rows = []
for fn in sorted(glob.glob(f"results/ws1_sweep_{PREFIX}_s*_cm*.csv")):
    m = re.search(PREFIX + r"_s(\d+)_cm(\d+)", fn)
    seed, cM = int(m.group(1)), int(m.group(2))
    for r in csv.DictReader(open(fn)):
        S = _f(r.get("S_gpc")); bo = _f(r.get("best_observer_chi2")) or _f(r.get("chi2_dof"))
        knot = _f(r.get("knot_ratio")); moved = str(r.get("knot_guard_moved", "")).lower() == "true"
        growth = _f(r.get("growth_factor")); anchor = str(r.get("anchor_ok")).lower() == "true"
        d_near = None
        if S:
            pos, _mass = build_virialized_grid(S * GPC, n_nodes=300, M_ext_kg=1.0,
                vir_mass_spread=0.5, vir_s_metric="median", vir_relax_mode="medium",
                center_mass_frac=cM / M_VALUE, seed=seed)
            d_near = float(np.linalg.norm(pos, axis=1).min() / GPC)
        o = old.get((cM, seed), {})
        rows.append(dict(centerM=cM, seed=seed, S=S, bo=bo, growth=growth, anchor=anchor,
                         knot=knot, moved=moved, S_from=r.get("knot_guard_S_from", ""),
                         d_near=(round(d_near, 1) if d_near else None),
                         old_S=_f(o.get("S")), old_bo=_f(o.get("bo")), old_knot=_f(o.get("knot")),
                         old_qualified=(o.get("qualifies") == "True"),
                         qualifies=(anchor and bo is not None and bo <= 0.46
                                    and (d_near or 0) > 15.0 and knot is not None and knot >= 0.9)))

rows.sort(key=lambda r: (not r["qualifies"], r["bo"] if r["bo"] is not None else 9))
with open(f"results/figures/centerm/{PREFIX}_verdict.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

q = [r for r in rows if r["qualifies"]]
rescued = [r for r in q if not r["old_qualified"]]
lost = [r for r in rows if r["old_qualified"] and not r["qualifies"]]
moved = [r for r in rows if r["moved"]]
lines = [f"=== {PREFIX} GUARDED universe-fit verdict ===", "",
         f"qualifiers: {len(q)}/{len(rows)}  (unguarded campaign: 43/71)",
         f"guard moved S in {len(moved)} cells; RESCUED {len(rescued)} previously-failing cells; "
         f"lost {len(lost)} previous qualifiers", ""]
lines.append(f"{'cM':>5} {'seed':>4} {'S':>4} {'S_old':>5} {'best_obs':>8} {'knot':>6} {'moved':>5} {'nearest Gpc':>11} {'qual':>5}")
for r in rows:
    lines.append(f"{r['centerM']:5d} {r['seed']:4d} {r['S'] or -1:4.0f} {r['old_S'] or -1:5.0f} "
                 f"{r['bo'] if r['bo'] is not None else float('nan'):8.4f} "
                 f"{r['knot'] if r['knot'] is not None else float('nan'):6.2f} "
                 f"{str(r['moved'])[:1]:>5} {r['d_near'] or -1:11.1f} {str(r['qualifies'])[:1]:>5}")
open(f"results/figures/centerm/{PREFIX}_verdict.txt", "w").write("\n".join(lines))
print("\n".join(lines[:12])); print(f"... full table in cmx2_verdict.txt ({len(rows)} rows)")
