"""Local, gradio-free preview of the Space content.

Renders the leaderboard to stdout and the two campaign plots to PNGs, using the
same pure helpers the Space uses. Verifies the data/plot layer before deploying.

    CENTERM_DATA_DIR=results/figures/centerm python hf/preview.py
    # (or just `python hf/preview.py` from the repo root — dev fallback finds
    #  results/figures/centerm automatically)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root
sys.path.insert(0, str(Path(__file__).resolve().parent / "space"))

try:  # Greek/≤ glyphs print on Windows cp1252 (matches the sweep banners)
    from cosmo.encoding import configure_utf8_stdout
    configure_utf8_stdout()
except Exception:
    pass

import viz  # noqa: E402


def main() -> int:
    data_dir = viz.resolve_data_dir()
    tables = viz.available_tables(data_dir)
    if not tables:
        print(f"No verdict/ranking CSVs found in {data_dir}")
        return 1

    name = viz.default_table(data_dir)
    print(f"Data dir : {data_dir}")
    print(f"Tables   : {', '.join(tables)}")
    print(f"Default  : {name}\n")

    df = viz.load_table(data_dir, name)
    print(viz.summary_text(df, name).replace("**", ""))
    print("\nLeaderboard (top 12 qualifying by best_obs):")
    print(viz.leaderboard(df, top=12).to_string(index=False))

    out_dir = Path("results/figures/centerm/_hf_preview")
    out_dir.mkdir(parents=True, exist_ok=True)
    viz.knot_vs_centerm_fig(df).savefig(out_dir / "knot_vs_centerm.png", dpi=110)
    viz.bo_vs_centerm_fig(df).savefig(out_dir / "bo_vs_centerm.png", dpi=110)
    print(f"\nFigures  : {len(viz.list_figures(data_dir))} campaign PNGs available")
    print(f"Preview plots written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
