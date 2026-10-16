"""Generate the MEDIUM-KNOT LADDER (PF23 follow-up): is the central-knot collapse a resolution
artifact of the coarse seed-42 GRF overdensity, or robust structure formation?

Three axes on TWO node-field cells — A = seed1/sigma0.5/S55 (the static-knot cell) and
B = seed0/sigma1.0/S63 (the collapsing reference), both M3000, medium geometry:
  * N-LADDER (particle_seed 42): N {8k, 16k, 32k} with steps scaled {1500, 2000, 2500}.
    If the knot dissolves with N -> resolution artifact; persists -> real collapse.
  * GRF-SEED axis (8k/1500): particle_seed {7, 123}. The knot is baked into the seed-42 ICs
    (PF23); does a different realization move/remove it?
  * MASS-RANDOMIZE axis (8k/1500): mass_randomize 0.5 (particle masses uniform in [0.5,1.5]x
    mean, now keyed==run). Expectation (honest): random masses RAISE two-body noise
    (<m^2>/<m>^2 = 13/12) so likely accelerate clumping — tested, not assumed.

All cells: save_snapshots -> results/hero/<tag>.npz (knot trace = Lagrangian core median radius).
Run:  bash driver (results/logs/medium_ladder_driver.sh), 8k cells first, 4-way parallel.
"""
import json, os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "medium_ladder")
os.makedirs(OUT, exist_ok=True)
BASE = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "medium", "medium_best.json")))

CELLS = {"A": dict(seed=1, sig=0.5, S=55), "B": dict(seed=0, sig=1.0, S=63)}
LADDER = [(8000, 1500, 25), (16000, 2000, 33), (32000, 2500, 42)]

arms = []
def emit(tag, cell, n_p, n_steps, si, particle_seed=42, mass_randomize=0.0, comment=""):
    cfg = dict(BASE)
    cfg["node_mass_seeds"] = [cell["seed"]]
    cfg["M_values"] = [3000]
    cfg["S_values"] = [cell["S"]]
    cfg["vir_mass_spreads"] = [cell["sig"]]
    cfg["particle_count"] = n_p
    cfg["n_steps"] = n_steps
    cfg["save_interval"] = si
    cfg["observer_sample"] = 2000
    if particle_seed != 42:
        cfg["particle_seed"] = particle_seed
    if mass_randomize:
        cfg["mass_randomize"] = mass_randomize
    cfg["tag"] = tag
    cfg["figures_dir"] = f"results/figures/{tag}"
    cfg["_comment"] = comment
    json.dump(cfg, open(os.path.join(OUT, f"{tag}.json"), "w"), indent=2)
    arms.append(tag)
    print("wrote", tag)

for key, cell in CELLS.items():
    who = f"node-seed{cell['seed']}/sig{cell['sig']}/S{cell['S']}"
    for n_p, n_steps, si in LADDER:
        emit(f"mlad_{key}_N{n_p//1000}k", cell, n_p, n_steps, si,
             comment=f"N-ladder {who}: {n_p}p/{n_steps}. Knot vs resolution.")
    for gs in (7, 123):
        emit(f"mlad_{key}_g{gs}", cell, 8000, 1500, 25, particle_seed=gs,
             comment=f"GRF-seed axis {who}: particle_seed={gs}, 8000p/1500. Does the knot follow the IC realization?")
    emit(f"mlad_{key}_mr0p5", cell, 8000, 1500, 25, mass_randomize=0.5,
         comment=f"mass-randomize axis {who}: mass_randomize=0.5, 8000p/1500. Do varied particle masses help or hurt the knot?")

json.dump({"family": "medium_ladder",
           "purpose": "PF23: knot origin — N-ladder(8k/16k/32k) x GRF-seed(42/7/123) x "
                      "mass_randomize(0/0.5) on the static (seed1) and collapsing (seed0) cells.",
           "arms": arms}, open(os.path.join(OUT, "_manifest.json"), "w"), indent=2)
print("arms:", len(arms))
