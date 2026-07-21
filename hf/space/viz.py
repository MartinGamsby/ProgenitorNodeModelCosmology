"""Pure data-load + plot helpers for the centerM-campaign HuggingFace Space.

No gradio dependency: every function here takes a data directory and returns a
DataFrame or a matplotlib Figure, so the logic can be previewed / unit-tested
locally (``CENTERM_DATA_DIR=results/figures/centerm python hf/preview.py``)
before it is ever deployed to a Space.

Data source resolution (``resolve_data_dir``):
  1. ``CENTERM_DATA_DIR`` env var  -> use it verbatim (local preview / tests).
  2. ``HF_DATASET_REPO`` env var    -> snapshot_download that HF Dataset.
  3. repo-local ``results/figures/centerm`` (dev fallback).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

# The verdict tables we know how to render, most-preferred (most recent
# resolution / guarded) first. Whichever exists first becomes the default view;
# adding cmx3_verdict.csv to the dataset auto-upgrades the Space.
VERDICT_PREFERENCE = (
    "cmx4_verdict.csv",
    "cmx3_verdict.csv",
    "cmx2_verdict.csv",
    "universe_fit_ranking.csv",
    "cmx_verdict.csv",
)

# Common columns every verdict/ranking table shares.
CORE_COLUMNS = ["centerM", "seed", "S", "bo", "growth", "knot", "qualifies"]

LCDM_REF = 0.436  # Pantheon+ chi2/dof reference (numerics/lcdm-baseline)
KNOT_GUARD = 0.9  # composite universe-fit criterion: knot >= 0.9 (no collapse)


def resolve_data_dir() -> Path:
    """Return the directory holding the campaign CSVs + figures."""
    env_dir = os.environ.get("CENTERM_DATA_DIR")
    if env_dir:
        return Path(env_dir)

    repo = os.environ.get("HF_DATASET_REPO")
    if repo:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(repo_id=repo, repo_type="dataset"))

    # Dev fallback: walk up to the repo root and use the canonical results dir.
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "results" / "figures" / "centerm"
        if cand.is_dir():
            return cand
    raise FileNotFoundError(
        "No data source. Set CENTERM_DATA_DIR or HF_DATASET_REPO, or run inside "
        "the repo so results/figures/centerm is reachable."
    )


def available_tables(data_dir: Path) -> dict[str, Path]:
    """Map {csv name: path} for the known verdict/ranking tables that exist."""
    out: dict[str, Path] = {}
    for name in VERDICT_PREFERENCE:
        p = data_dir / name
        if p.is_file():
            out[name] = p
    return out


def default_table(data_dir: Path) -> str | None:
    """The most-preferred existing table name (the default Space view)."""
    tables = available_tables(data_dir)
    return next(iter(tables), None)


def load_table(data_dir: Path, name: str) -> pd.DataFrame:
    """Load one verdict/ranking CSV, normalized to the common columns."""
    df = pd.read_csv(data_dir / name)
    # Coerce the shared columns; tolerate tables missing an optional one.
    for col in CORE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    if "core" not in df.columns:
        df["core"] = pd.NA
    return df


def leaderboard(df: pd.DataFrame, top: int = 25, qualifying_only: bool = True) -> pd.DataFrame:
    """Tidy leaderboard: qualifying configs by best-observer chi2 (ascending)."""
    view = df.copy()
    if qualifying_only and "qualifies" in view.columns:
        view = view[view["qualifies"] == True]  # noqa: E712 (pandas mask)
    view = view.sort_values("bo", ascending=True).head(top)
    cols = ["centerM", "seed", "S", "bo", "growth", "knot", "core", "qualifies"]
    cols = [c for c in cols if c in view.columns]
    out = view[cols].rename(columns={
        "S": "S (Gpc)", "bo": "best_obs", "growth": "growth",
    }).reset_index(drop=True)
    # Round floats for display without touching the source values.
    for c in ("best_obs", "growth", "knot", "core", "S (Gpc)"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(3)
    return out


def knot_vs_centerm_fig(df: pd.DataFrame):
    """Scatter knot ratio vs centerM (log x), coloured by qualifies."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    d = df.dropna(subset=["centerM", "knot"]).copy()
    q = d["qualifies"] == True  # noqa: E712
    ax.scatter(d.loc[q, "centerM"], d.loc[q, "knot"], s=42, c="#2a9d8f",
               edgecolor="k", linewidth=0.4, label="universe-fit qualifies", zorder=3)
    ax.scatter(d.loc[~q, "centerM"], d.loc[~q, "knot"], s=36, c="#e76f51",
               edgecolor="k", linewidth=0.4, label="fails criterion", zorder=2)
    ax.axhline(KNOT_GUARD, ls="--", c="#888", lw=1,
               label=f"knot guard = {KNOT_GUARD} (no collapse)")
    ax.axhline(1.0, ls=":", c="#bbb", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("centerM  (progenitor-node mass / M_value)")
    ax.set_ylabel("observable-cloud knot ratio  (>1 = expanding)")
    ax.set_title("Central-knot vs progenitor mass")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def bo_vs_centerm_fig(df: pd.DataFrame):
    """Best-observer chi2/dof vs centerM, with the LCDM reference line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    d = df.dropna(subset=["centerM", "bo"]).copy()
    ax.scatter(d["centerM"], d["bo"], s=38, c="#264653",
               edgecolor="k", linewidth=0.3, zorder=3)
    ax.axhline(LCDM_REF, ls="--", c="#e63946", lw=1.2,
               label=f"LCDM reference = {LCDM_REF}")
    ax.set_xscale("log")
    ax.set_xlabel("centerM  (progenitor-node mass / M_value)")
    ax.set_ylabel("best-observer chi2/dof  vs Pantheon+")
    ax.set_title("Fit quality is flat in centerM (~LCDM everywhere)")
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    return fig


def list_figures(data_dir: Path) -> list[Path]:
    """Campaign PNG figures for the image gallery, sorted by name."""
    return sorted(p for p in data_dir.glob("*.png"))


def summary_text(df: pd.DataFrame, table_name: str) -> str:
    """One-line qualifying-count summary for the current table."""
    n = len(df)
    q = int((df["qualifies"] == True).sum()) if "qualifies" in df.columns else 0  # noqa: E712
    best = pd.to_numeric(df["bo"], errors="coerce").min()
    return (
        f"**{table_name}** — {q}/{n} configs qualify under the composite "
        f"universe-fit criterion (Pantheon best_obs ≤ 0.46, anchor-ok, "
        f"knot ≥ {KNOT_GUARD}, no HMEA within 15 Gpc). "
        f"Best-observer chi2/dof floor = {best:.4f} (LCDM {LCDM_REF})."
    )
