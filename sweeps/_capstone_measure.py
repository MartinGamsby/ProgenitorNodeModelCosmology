"""Self-contained capstone measurement (runs at the END of results/logs/cmx_capstone.sh, no AI
needed): the S-confound probe + paper-cloud-candidate convergence verdicts, written to
results/figures/centerm/cmx_capstone_report.txt and appended to cmx_snap2_measurements.csv.

Cells (produced by the driver):
  cmxprobe_s7_cm3000_S89  (seed7 cm3000 at S=89 vs 0.90 at S=51)   } S-confound
  cmxprobe_s7_cm300_S89   (seed7 cm300  at S=89 vs 0.81 at S=51)   } probe
  cmxconv_s0_cm3000_8k    (candidate at 8000p/1500)                } convergence vs
  cmxconv_s0_cm3000_16k   (candidate at 16000p/2000)               } 1.04/0.125/0.4404/3.06 @4k
"""
import os, sys
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
import numpy as np

CELLS = [  # (tag, n_inner, prior_note)
    ("cmxprobe_s7_cm3000_S89", 4000,  "seed7 cm3000: knot was 0.90 at S=51 Gpc"),
    ("cmxprobe_s7_cm300_S89",  4000,  "seed7 cm300:  knot was 0.81 at S=51 Gpc"),
    ("cmxconv_s0_cm3000_8k",   8000,  "candidate @8k (4k was knot 1.04, core 0.125, bo 0.4404, g 3.06)"),
    ("cmxconv_s0_cm3000_16k",  16000, "candidate @16k"),
]

lines = ["=== centerM capstone verdicts (self-measured) ===", ""]
rows = []
for tag, n_inner, note in CELLS:
    fn = f"results/hero/{tag}.npz"
    if not os.path.exists(fn):
        lines.append(f"{tag}: MISSING npz ({note})"); continue
    d = np.load(fn)
    n_total = 3 * n_inner                       # outer cap 2x binds at cm>=3
    rng = np.random.default_rng(0)
    keep = rng.choice(n_total, size=min(6000, n_total), replace=False)
    inner_sel = keep < n_inner
    P = d["pos_centred"][:, inner_sel, :]
    rf = np.linalg.norm(P[-1], axis=1); r90 = np.percentile(rf, 90)
    core = float(np.mean(rf < 0.25 * r90))
    cs = rf < 0.25 * r90
    med = [float(np.median(np.linalg.norm(P[s][cs], axis=1))) for s in range(len(P))]
    knot = med[-1] / med[0]
    cls = "COLLAPSING" if knot < 0.9 else ("expanding" if knot > 1.2 else "~static")
    bo = float(d["best_observer_chi2"]); g = float(d["growth_factor"])
    lines.append(f"{tag}: best_obs={bo:.4f} growth={g:.3f} core={core:.3f} "
                 f"knot={knot:.2f} {cls}   [{note}]")
    rows.append(f"{tag},{bo:.4f},{g:.3f},{core:.3f},{knot:.3f},{cls}")

lines += ["", "Interpretation guide:",
          "PROBE: if the two S89 probes read >=0.9 (vs 0.90/0.81 at S=51), the knot tracks the",
          "  co-fit spacing/field strength, and centerM's knot effect is indirect (via S).",
          "CONVERGENCE: candidate is converged if knot/core/best_obs at 8k+16k stay near",
          "  1.04/0.125/0.4404; drifting values mean the 4k verdict was under-resolved."]
os.makedirs("results/figures/centerm", exist_ok=True)
open("results/figures/centerm/cmx_capstone_report.txt", "w").write("\n".join(lines))
with open("results/figures/centerm/cmx_snap2_measurements.csv", "a") as f:
    for r in rows: f.write(r + "\n")
print("\n".join(lines))
