"""Generate the KNOT-GUARDED campaign re-sweep (cmx2): the same 72 cells as the centerM campaign
(centerM {1..3000} x 8 medium seeds, medium/sigma0.5/grfmass/M3000, ternary S co-fit [40,160]) but
with s_knot_guard=true — the co-fit now selects S JOINTLY on fit + knot (universe-fit criterion,
PF26): if the chi2 winner's knot_ratio < 0.9 the guard probes larger S in 5 Gpc steps and accepts
the first anchor-ok knot-free candidate within a 0.03 chi2/dof budget.

Question: how many of the 28 knot-failing campaign cells does the joint selection RESCUE, and what
is the definitive (S, knot) table per cell? knot_ratio/knot_guard_moved land in the CSVs natively
(2000p in-memory knots; the PF26 margin rule still applies to boundary values).
Run: results/logs/cmx2_driver.sh (nohup, 4-way pool, ~overnight; guarded cells recompute sims
because cached metrics lack knot_ratio — the PF-OBSCACHE keyed==run rule).
"""
import json, os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "centerm_campaign")
M_VALUE = 3000
CENTERMS = [1, 2, 3, 10, 30, 100, 300, 1000, 3000]
SEEDS = [0, 1, 2, 3, 5, 7, 9, 11]

n = 0
for seed in SEEDS:
    for cM in CENTERMS:
        src = json.load(open(os.path.join(OUT, f"cmx_s{seed}_cm{cM}.json")))
        src["s_knot_guard"] = True
        tag = f"cmx2_s{seed}_cm{cM}"
        src["tag"] = tag
        src["figures_dir"] = f"results/figures/{tag}"
        src["_comment"] = (f"KNOT-GUARDED re-sweep: centerM={cM} seed={seed}, S co-fit [40,160] with "
                           f"s_knot_guard (joint fit+knot selection, PF26). Base cell: cmx_s{seed}_cm{cM}.")
        json.dump(src, open(os.path.join(OUT, f"{tag}.json"), "w"), indent=2)
        n += 1
print(f"wrote {n} guarded configs (cmx2_*)")
