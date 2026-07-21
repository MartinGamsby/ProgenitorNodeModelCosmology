"""HuggingFace Space: browse the Progenitor-Node centerM campaign.

A read-only viewer over the campaign verdict tables + figures (pushed as an HF
Dataset by ``hf/push_results.py``). All data/plot logic lives in ``viz.py`` (pure,
previewable via ``hf/preview.py``); this file only wires it into Gradio.

Data source: set ``HF_DATASET_REPO`` (Space secret/variable) to the pushed
dataset repo id; falls back to ``CENTERM_DATA_DIR`` or the in-repo results dir
for local runs.
"""

import gradio as gr

import viz

DATA_DIR = viz.resolve_data_dir()
TABLES = viz.available_tables(DATA_DIR)
DEFAULT = viz.default_table(DATA_DIR)

INTRO = """
# Progenitor-Node Cosmology — centerM campaign

Can classical tidal forces from a virialized meta-structure of hyper-massive
external nodes reproduce ΛCDM expansion **without dark energy**? This viewer
shows the `centerM` campaign: for each progenitor-node mass (`centerM`) × node
realization (`seed`), the node spacing `S` is co-fit to Pantheon+ supernovae
(with a knot guard), then scored on the composite **universe-fit** criterion —
best-observer χ²/dof ≤ 0.46, anchor-ok expansion, a non-collapsing central core
(knot ≥ 0.9), and no external node within the 15 Gpc observable horizon.
"""


def _render(table_name: str):
    df = viz.load_table(DATA_DIR, table_name)
    return (
        viz.summary_text(df, table_name),
        viz.leaderboard(df, top=25),
        viz.knot_vs_centerm_fig(df),
        viz.bo_vs_centerm_fig(df),
    )


def build() -> gr.Blocks:
    with gr.Blocks(title="Progenitor-Node centerM campaign") as demo:
        gr.Markdown(INTRO)

        if not TABLES:
            gr.Markdown(
                "⚠️ No verdict tables found. Set `HF_DATASET_REPO` to the pushed "
                "dataset, or `CENTERM_DATA_DIR` locally."
            )
            return demo

        table = gr.Dropdown(
            choices=list(TABLES), value=DEFAULT, label="Verdict table",
            info="Most recent / highest-resolution sweep first.",
        )
        summary = gr.Markdown()
        with gr.Row():
            knot_plot = gr.Plot(label="Knot vs centerM")
            bo_plot = gr.Plot(label="Best-observer χ²/dof vs centerM")
        leaderboard = gr.Dataframe(
            label="Universe-fit leaderboard (qualifying configs by best_obs)",
            interactive=False, wrap=True,
        )

        figs = [str(p) for p in viz.list_figures(DATA_DIR)]
        if figs:
            gr.Gallery(value=figs, label="Campaign figures", columns=3, height="auto")

        table.change(_render, inputs=table,
                     outputs=[summary, leaderboard, knot_plot, bo_plot])
        demo.load(_render, inputs=table,
                  outputs=[summary, leaderboard, knot_plot, bo_plot])
    return demo


if __name__ == "__main__":
    build().launch()
