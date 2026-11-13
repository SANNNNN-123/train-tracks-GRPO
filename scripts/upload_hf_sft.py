#!/usr/bin/env python
"""Upload the combined SFT dataset folder to the Hugging Face Hub.

Usage:
    $env:HF_TOKEN = "hf_..."      # or put HF_TOKEN=... in .env, or run `hf auth login`
    .\\.venv\\Scripts\\python.exe scripts\\upload_hf_sft.py

Options:
    --repo-id   target dataset repo (default: zuhairsan/train-tracks-sft)
    --private   create the repo as private
    --dry-run   validate the local folder and print what would be uploaded
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
FOLDER = ROOT / "hf_export" / "train-tracks-sft"
DEFAULT_REPO = "zuhairsan/train-tracks-sft"


def load_dotenv(path: Path) -> None:
    """Populate os.environ from a simple KEY=VALUE .env file, without overwriting."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    if not (FOLDER / "README.md").exists():
        raise SystemExit(f"missing {FOLDER / 'README.md'}")
    parquet = sorted((FOLDER / "data").glob("*.parquet"))
    if not parquet:
        raise SystemExit(f"no parquet files in {FOLDER / 'data'}")

    print(f"folder: {FOLDER}")
    for path in parquet:
        print(f"  {path.relative_to(FOLDER)}  {path.stat().st_size / 1e6:.1f} MB")
    print(f"  README.md  {(FOLDER / 'README.md').stat().st_size} bytes")

    if args.dry_run:
        print("dry run: nothing uploaded")
        return 0

    if not os.environ.get("HF_TOKEN"):
        print("warning: HF_TOKEN is not set; relying on a cached `hf auth login` token")

    api = HfApi()
    who = api.whoami()
    print(f"authenticated as: {who.get('name')}")

    api.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    commit = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(FOLDER),
        commit_message="Upload Train Tracks SFT: deductions and contradictions",
    )
    print(f"uploaded: https://huggingface.co/datasets/{args.repo_id}")
    print(f"commit: {commit.commit_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
