"""Visualize all registered HMEA node geometries as a 3D scatter grid.

Reproducible figure for inspecting the node arrangements built by
``cosmo.node_geometry.build_node_positions``.  Writes one PNG you can open.

Run:
    set PYTHONIOENCODING=utf-8
    python visualize_geometries.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cosmo.node_geometry import build_node_positions, list_geometries  # noqa: E402
from cosmo.plots import figure_path  # noqa: E402

# Order the geometries with the default (cube26) first.
GEOMS = ["cube26", "cube_dense", "shell", "shell_multi", "fcc", "bcc"]
GEOMS += [g for g in list_geometries() if g not in GEOMS]

S = 1.0  # normalized scale (positions ~ S); geometry shape is scale-invariant

ncols = 3
nrows = int(np.ceil(len(GEOMS) / ncols))
fig = plt.figure(figsize=(5 * ncols, 4.6 * nrows))

for i, geom in enumerate(GEOMS, start=1):
    pos = build_node_positions(geom, S)
    ax = fig.add_subplot(nrows, ncols, i, projection="3d")
    r = np.linalg.norm(pos, axis=1)
    p = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                   c=r, cmap="viridis", s=45, depthshade=True,
                   edgecolors="k", linewidths=0.3)
    # mark the (empty) observable centre
    ax.scatter([0], [0], [0], c="red", marker="*", s=120, label="observable centre")
    ax.set_title(f"{geom}  (N={len(pos)} nodes)", fontsize=12)
    lim = float(np.max(np.abs(pos))) * 1.05
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_xlabel("x/S"); ax.set_ylabel("y/S"); ax.set_zlabel("z/S")
    ax.tick_params(labelsize=7)

fig.suptitle("HMEA external-node geometries (red star = observable region at centre)",
             fontsize=15, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.98))

out = figure_path("ws3", "node_geometries")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")
for geom in GEOMS:
    print(f"  {geom:12s} N={len(build_node_positions(geom, S))}")
