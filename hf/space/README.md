---
title: Progenitor-Node centerM Campaign
emoji: 🌌
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Progenitor-Node Cosmology — centerM campaign viewer

Read-only visualizer for the `centerM` parameter campaign of the
[ProgenitorNodeModelCosmology](https://github.com/) N-body toy model: does
classical tidal shear from a virialized meta-structure of hyper-massive external
nodes reproduce ΛCDM expansion without dark energy?

## Data source

The app reads campaign verdict CSVs + figures from an HF **Dataset** repo. Set
the Space variable `HF_DATASET_REPO` to that dataset's id (e.g.
`your-name/progenitor-node-results`). Locally it falls back to
`CENTERM_DATA_DIR` or the in-repo `results/figures/centerm`.

Populate/refresh the dataset from the repo root:

```bash
huggingface-cli login
python hf/push_results.py --repo your-name/progenitor-node-results --yes
```

## Files
- `app.py` — Gradio wiring only.
- `viz.py` — pure data-load + plot helpers (also drives `hf/preview.py`).
- `requirements.txt` — Space dependencies.
