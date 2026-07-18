"""Universe-fit ranking (user criterion, 2026-07-16): "best" = best fit to our CURRENT
UNDERSTANDING + OBSERVATIONS of the universe, NOT best Pantheon chi2 alone.

Composite criterion per cell (all must hold to qualify):
  1. PANTHEON:  best-observer chi2/dof LCDM-quality (<= 0.46 vs LCDM 0.436)
  2. EXPANSION: anchor_ok (total growth within +/-20% of LCDM's 3.30)
  3. STRUCTURE: no collapsing Gpc-scale core -> knot ratio >= 0.9 (we observe none)
  4. NEIGHBOURS: no HMEA within 15 Gpc (we would likely have detected it)
Qualifiers are then ranked by best_obs. Self-contained: measures knots from every
cmknot_/cmxsnap_/cmxsnap2_/cmxconv_ npz (observable-cloud-only recipe), joins the campaign
verdict CSV, writes results/figures/centerm/universe_fit_ranking.{csv,txt}.
"""
import os, sys, glob, re, csv
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
import numpy as np

def cloud_metrics(fn, n_inner, centerM):
    d = np.load(fn)
    n_total = n_inner + int(min((centerM - 1) * n_inner, 2 * n_inner))
    rng = np.random.default_rng(0)
    keep = rng.choice(n_total, size=min(6000, n_total), replace=False)
    P = d["pos_centred"][:, keep < n_inner, :]
    rf = np.linalg.norm(P[-1], axis=1); r90 = np.percentile(rf, 90)
    cs = rf < 0.25 * r90
    med = [float(np.median(np.linalg.norm(P[s][cs], axis=1))) for s in range(len(P))]
    return dict(core=float(np.mean(cs)), knot=med[-1] / med[0],
                bo=float(d["best_observer_chi2"]), growth=float(d["growth_factor"]))

# knots per (centerM, seed) from every snapshot family (highest-N npz wins)
PAT = re.compile(r"(?:cmknot|cmxsnap2?|cmxconv)_s(\d+)_cm(\d+)(?:_(\d+)k)?(?:_S\d+)?\.npz$")
knots = {}
for fn in glob.glob("results/hero/cm*.npz"):
    m = PAT.search(os.path.basename(fn))
    if not m: continue
    seed, cM = int(m.group(1)), int(m.group(2))
    n_inner = int(m.group(3)) * 1000 if m.group(3) else 4000
    cur = knots.get((cM, seed))
    if cur is None or n_inner > cur[0]:
        try:
            knots[(cM, seed)] = (n_inner, cloud_metrics(fn, n_inner, cM), os.path.basename(fn))
        except Exception as e:
            print(f"skip {fn}: {e!r}")

rows = list(csv.DictReader(open("results/figures/centerm/cmx_verdict.csv")))
out, qual = [], []
def _f(v):
    try: return float(v)
    except (TypeError, ValueError): return None
for r in rows:
    cM, seed = int(float(r["centerM"])), int(float(r["seed"]))
    k = knots.get((cM, seed))
    # runaway cells have empty best_obs/growth -> None -> non-qualifying (bo check below)
    rec = dict(centerM=cM, seed=seed, S=_f(r["S_cofit_gpc"]), bo=_f(r["best_obs"]),
               growth=_f(r["growth"]), anchor=r["anchor_ok"] == "True",
               d_near=float(r["d_near_gpc"]),
               knot=(round(k[1]["knot"], 3) if k else None),
               core=(round(k[1]["core"], 3) if k else None),
               knot_src=(k[2] if k else ""))
    rec["qualifies"] = (rec["anchor"] and rec["bo"] is not None and rec["bo"] <= 0.46 and rec["d_near"] > 15.0
                        and rec["knot"] is not None and rec["knot"] >= 0.9)
    out.append(rec)
    if rec["qualifies"]: qual.append(rec)

qual.sort(key=lambda r: r["bo"])
os.makedirs("results/figures/centerm", exist_ok=True)
with open("results/figures/centerm/universe_fit_ranking.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)

lines = ["=== UNIVERSE-FIT RANKING (pantheon<=0.46 + anchor + knot>=0.9 + no HMEA<15Gpc) ===", ""]
lines.append(f"{'cM':>5} {'seed':>4} {'S Gpc':>6} {'best_obs':>8} {'growth':>7} {'knot':>6} {'core':>6} {'nearest Gpc':>11}")
for r in qual:
    lines.append(f"{r['centerM']:5d} {r['seed']:4d} {r['S']:6.0f} {r['bo']:8.4f} "
                 f"{r['growth']:7.2f} {r['knot']:6.2f} {r['core']:6.3f} {r['d_near']:11.1f}")
measured = [r for r in out if r["knot"] is not None]
failed = [r for r in measured if not r["qualifies"]]
lines += ["", f"qualifiers: {len(qual)} of {len(measured)} knot-measured cells "
              f"({len(out)-len(measured)} campaign cells still knot-unmeasured)",
          "non-qualifying measured cells (reason):"]
for r in failed:
    why = []
    if not r["anchor"]: why.append("anchor")
    if r["bo"] is None: why.append("runaway/no-fit")
    elif r["bo"] > 0.46: why.append(f"fit {r['bo']:.3f}")
    if r["d_near"] <= 15: why.append("HMEA<15Gpc")
    if r["knot"] is not None and r["knot"] < 0.9: why.append(f"knot {r['knot']:.2f}")
    lines.append(f"  cm{r['centerM']}/s{r['seed']}: {', '.join(why)}")
open("results/figures/centerm/universe_fit_ranking.txt", "w").write("\n".join(lines))
print("\n".join(lines))
