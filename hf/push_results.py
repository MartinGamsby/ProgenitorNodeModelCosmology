"""Push the centerM-campaign results (CSVs + figures) to a HuggingFace Dataset.

The durable-results-store half of WS7. Tooling only: the agent never handles
credentials — YOU authenticate (``huggingface-cli login``) and YOU trigger the
publish with ``--yes``. Without ``--yes`` this is a DRY RUN that only lists what
would be uploaded, so it can never publish by accident.

    huggingface-cli login                 # once, interactive (your token)
    python hf/push_results.py --repo your-name/progenitor-node-results        # dry run
    python hf/push_results.py --repo your-name/progenitor-node-results --yes  # publish

The Space (hf/space) reads this dataset at runtime via HF_DATASET_REPO.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Only the campaign artifacts the viewer needs (verdict/ranking tables + figures).
INCLUDE_PATTERNS = ("*.csv", "*.png")


def _iter_files(data_dir: Path) -> list[Path]:
    files: list[Path] = []
    for pat in INCLUDE_PATTERNS:
        files.extend(sorted(data_dir.glob(pat)))
    return files


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True,
                    help="HF Dataset repo id, e.g. your-name/progenitor-node-results")
    ap.add_argument("--data-dir", default="results/figures/centerm",
                    help="Local dir with the campaign CSVs + PNGs (default: %(default)s)")
    ap.add_argument("--path-in-repo", default="centerm",
                    help="Destination subfolder in the dataset (default: %(default)s)")
    ap.add_argument("--yes", action="store_true",
                    help="Actually upload. Without this it is a DRY RUN.")
    args = ap.parse_args(argv)

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"error: data dir not found: {data_dir}", file=sys.stderr)
        return 2

    files = _iter_files(data_dir)
    if not files:
        print(f"error: no {'/'.join(INCLUDE_PATTERNS)} files in {data_dir}", file=sys.stderr)
        return 2

    total_mb = sum(f.stat().st_size for f in files) / 1e6
    print(f"Dataset repo : {args.repo}")
    print(f"Source dir   : {data_dir}  ->  {args.path_in_repo}/")
    print(f"Files        : {len(files)}  ({total_mb:.1f} MB)")
    for f in files:
        print(f"  {f.name}  ({f.stat().st_size/1e3:.0f} KB)")

    if not args.yes:
        print("\nDRY RUN - nothing uploaded. Re-run with --yes to publish "
              "(after `huggingface-cli login`).")
        return 0

    from huggingface_hub import HfApi, HfFolder

    if HfFolder.get_token() is None:
        print("error: not logged in. Run `huggingface-cli login` first "
              "(your token is never seen by the agent).", file=sys.stderr)
        return 3

    api = HfApi()
    api.create_repo(repo_id=args.repo, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        repo_id=args.repo,
        repo_type="dataset",
        folder_path=str(data_dir),
        path_in_repo=args.path_in_repo,
        allow_patterns=list(INCLUDE_PATTERNS),
        commit_message="Update centerM campaign results (CSVs + figures)",
    )
    print(f"\nPublished {len(files)} files to dataset {args.repo}/{args.path_in_repo}.")
    print(f"Point the Space at it:  HF_DATASET_REPO={args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
