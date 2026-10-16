# HuggingFace scale-out (WS7) — results store + visualization Space

HF is used **only** as a durable results store + a public viewer. Compute runs on
Martin's self-hosted GPU (`force_method="cuda_direct"`), never HF Jobs. See
`lode/plans/scale-out-gpu-hf.md`.

The agent builds this tooling but **never handles your credentials** — you log in
and you trigger the publish.

## Layout
- `push_results.py` — push campaign CSVs + figures to an HF **Dataset** (dry-run
  by default; `--yes` publishes).
- `space/` — a Gradio **Space** that reads the Dataset and browses the campaign
  (universe-fit leaderboard, knot-vs-centerM, best-obs-vs-centerM, figure gallery).
  - `viz.py` — pure data/plot helpers (no gradio).
  - `app.py` — Gradio wiring.
  - `README.md` — the Space card (HF metadata header).
- `preview.py` — render the leaderboard + plots locally, **no gradio/HF needed**:
  ```bash
  python hf/preview.py
  ```

## One-time setup
```bash
pip install huggingface_hub gradio            # gradio only needed to run the Space locally
huggingface-cli login                         # your token; the agent never sees it
```

## Publish the results dataset
```bash
python hf/push_results.py --repo <you>/progenitor-node-results          # dry run: lists files
python hf/push_results.py --repo <you>/progenitor-node-results --yes    # publish
```
Re-run after each sweep (cmx3/cmx4 verdicts land as `cmx3_verdict.csv` etc. — the
Space auto-prefers the newest table).

## Deploy the Space
1. Create a new **Gradio Space** on HF.
2. Push the contents of `hf/space/` to it (`app.py`, `viz.py`, `requirements.txt`,
   `README.md`).
3. Set the Space variable `HF_DATASET_REPO=<you>/progenitor-node-results`.

Run it locally first:
```bash
cd hf/space && HF_DATASET_REPO=<you>/progenitor-node-results python app.py
```
