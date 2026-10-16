"""Generate the NODE-COUNT probe (idea A, user's node-count lever): does raising vir_n_nodes
make the cloud rounder (fewer dominant near-nodes -> smoother tidal compression)?

Masses are PER-NODE mean-preserving (total = N * M_ext_kg), so "more nodes" has two readings,
both tested here at the rounder cell (S25 / sigma5 / seed 42 = DEFAULT, isolating node-count
from the seed):
  * PER-NODE fixed (M200 for all N): add nodes at the same per-node mass; extras extend the ball
    outward (may be near-inert per PF7). Total mass grows with N.
  * TOTAL fixed (M ~ 1/N: 200/100/50/25 at N 300/600/1200/2400): genuinely more, SMALLER nodes
    ("fewer big ones") at constant total external mass -- the user's actual hypothesis.

Single-cell snapshot configs (one npz each) at 4000p/1092. Measure core-fraction + growth +
best_obs vs node count. Time the N2400 cell FIRST (tidal cost ~ Np*Nnodes).
"""
import json, os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nodecount")
os.makedirs(OUT, exist_ok=True)
base = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rounder", "02_primary.json")))

# (n_nodes, M, series)
CELLS = [
    (300,  200, "pernode"),   # baseline (shared)
    (600,  200, "pernode"),
    (1200, 200, "pernode"),
    (2400, 200, "pernode"),
    (600,  100, "totalfix"),  # total = N*M = 60000 held fixed -> smaller per-node nodes
    (1200, 50,  "totalfix"),
    (2400, 25,  "totalfix"),
]

arms = []
for i, (nn, M, series) in enumerate(CELLS, start=1):
    cfg = dict(base)
    cfg["node_mass_seeds"] = [42]
    cfg["M_values"] = [M]
    cfg["S_values"] = [25]
    cfg["vir_mass_spreads"] = [5.0]
    cfg["vir_n_nodes"] = nn
    cfg["particle_count"] = 4000
    cfg["observer_sample"] = 4000
    tag = f"nc_{series}_N{nn}_M{M}"
    cfg["tag"] = tag
    cfg["figures_dir"] = f"results/figures/{tag}"
    cfg["_comment"] = (f"node-count probe [{series}]: vir_n_nodes={nn}, M={M}, S25/sigma5/seed42, "
                       f"4000p/1092, snapshots -> results/hero/{tag}.npz. {'total=N*M='+str(nn*M) } "
                       f"Measure core-fraction vs node count (rounder?).")
    fn = f"{i:02d}_{tag}.json"
    json.dump(cfg, open(os.path.join(OUT, fn), "w"), indent=2)
    arms.append(f"sweeps/nodecount/{fn}")
    print(f"wrote {fn}: N={nn} M={M} ({series}, total={nn*M})")

json.dump({"family": "nodecount",
           "purpose": "Idea A node-count lever: does raising vir_n_nodes round the cloud? Tests "
                      "per-node-fixed (extras go far/inert?) vs total-fixed (more, smaller nodes) "
                      "at the rounder cell S25/sigma5/seed42. Snapshots -> core-fraction vs N.",
           "arms": arms}, open(os.path.join(OUT, "_manifest.json"), "w"), indent=2)
print("arms:", *arms, sep="\n  ")
