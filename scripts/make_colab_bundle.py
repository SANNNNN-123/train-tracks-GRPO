"""Bundle the environment for Colab: a small zip the notebook can unzip in one cell.

Only what training needs: `tt_env/` (verifier, sandbox, rewards, prompt, sampler) and the two data
files. The reference solver and the PNGs stay local - the reward path never touches them.

    uv run python scripts/make_colab_bundle.py
    # -> dist/train_tracks_colab.zip  (~200 KB)
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INCLUDE = [
    "tt_env/__init__.py",
    "tt_env/puzzle.py",
    "tt_env/rules.py",
    "tt_env/sandbox.py",
    "tt_env/rewards.py",
    "tt_env/prompt.py",
    "tt_env/policies.py",
    "tt_env/rl.py",
    "tt_env/report.py",
    "data/puzzles.jsonl",
    "data/splits.json",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "dist" / "train_tracks_colab.zip")
    args = ap.parse_args()

    missing = [name for name in INCLUDE if not (ROOT / name).exists()]
    if missing:
        raise SystemExit(f"missing required files: {missing}; run scripts/freeze_splits.py first")

    # A new tt_env module that is not bundled fails in Colab, at the worst possible moment. The adapter
    # is the one deliberate exception: it imports `tracks`, which only exists on this machine.
    local_modules = sorted(p.name for p in (ROOT / "tt_env").glob("*.py") if p.name != "tracks_adapter.py")
    unbundled = [name for name in local_modules if f"tt_env/{name}" not in INCLUDE]
    if unbundled:
        raise SystemExit(f"tt_env modules missing from INCLUDE: {unbundled}")

    splits = json.loads((ROOT / "data" / "splits.json").read_text(encoding="utf-8"))
    puzzles = sum(1 for line in (ROOT / "data" / "puzzles.jsonl").read_text(encoding="utf-8").splitlines() if line)
    if puzzles != 608:
        raise SystemExit(f"corpus has {puzzles} puzzles, expected 608")
    if splits.get("counts", {}).get("train") != 212:
        raise SystemExit(f"train split has {splits.get('counts', {}).get('train')} puzzles, expected 212")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in INCLUDE:
            archive.write(ROOT / name, arcname=name)

    size = args.out.stat().st_size / 1024
    print(f"bundled {len(INCLUDE)} files -> {args.out} ({size:.0f} KB)")
    print(f"corpus {puzzles} puzzles, splits {splits['counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
