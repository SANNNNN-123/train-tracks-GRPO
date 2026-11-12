"""Freeze the train / validation / test split from the HF dataset.

The HF dataset already defines the split contract we must honour (212 small train, 46 validation,
46 test_small, 304 medium test). Puzzle ids are stable, so the split is simply copied from the
`split` column of the cached decisions and joined against the corpus we can actually build.

Writes `data/splits.json`:
    {"seed", "source", "counts", "splits": {name: [puzzle_id, ...]}, "unassigned": [...]}

Usage:
    uv run python scripts/fetch_hf_decisions.py --split train --split validation --split test_small --split test_medium
    uv run python scripts/freeze_splits.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPECTED = {"train": 212, "validation": 46, "test_small": 46, "test_medium": 304}
SEED = 20261110


def read_split(path: Path) -> str | None:
    seen: str | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if seen is None:
                    seen = row["split"]
                elif row["split"] != seen:
                    raise SystemExit(f"{path}: mixed splits {seen!r} and {row['split']!r}")
    return seen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions-dir", type=Path, default=ROOT / "data" / "hf")
    ap.add_argument("--puzzles", type=Path, default=ROOT / "data" / "puzzles.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "splits.json")
    args = ap.parse_args()

    corpus = {json.loads(line)["puzzle_id"] for line in args.puzzles.read_text("utf-8").splitlines() if line}
    difficulty = {
        json.loads(line)["puzzle_id"]: json.loads(line).get("difficulty", "")
        for line in args.puzzles.read_text("utf-8").splitlines()
        if line
    }

    assignment: dict[str, str] = {}
    puzzle_counts: dict[str, set[str]] = defaultdict(set)
    for path in sorted(args.decisions_dir.glob("*_decisions.jsonl")):
        split = read_split(path)
        if split is None:
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                puzzle_id = json.loads(line)["puzzle_id"]
                previous = assignment.setdefault(puzzle_id, split)
                if previous != split:
                    raise SystemExit(f"{puzzle_id} appears in {previous} and {split}")
                puzzle_counts[split].add(puzzle_id)

    missing_splits = set(EXPECTED) - set(puzzle_counts)
    if missing_splits:
        print(f"warning: no decisions cached for {sorted(missing_splits)}; split will be incomplete")

    problems = []
    for split, expected in EXPECTED.items():
        actual = len(puzzle_counts.get(split, ()))
        flag = "ok " if actual == expected else "!! "
        print(f"{flag}{split:<12} {actual:>4} puzzles (expected {expected})")
        if actual != expected:
            problems.append(f"{split}: {actual} != {expected}")

    splits = {}
    unassigned = []
    for split in sorted(puzzle_counts):
        ids = sorted(puzzle_counts[split])
        built = [i for i in ids if i in corpus]
        absent = [i for i in ids if i not in corpus]
        splits[split] = built
        unassigned.extend(absent)
        print(f"   {split:<12} built {len(built):>4}, not in corpus {len(absent):>3} {absent[:6]}")

    if any("medium" in difficulty.get(pid, "") for pid in splits.get("train", [])):
        problems.append("medium puzzle found in train split")
    overlap = Counter(pid for ids in splits.values() for pid in ids)
    overlaps = [pid for pid, count in overlap.items() if count > 1]
    if overlaps:
        problems.append(f"{len(overlaps)} puzzles in more than one split: {overlaps[:5]}")

    payload = {
        "seed": SEED,
        "source": "zuhairsan/train-tracks-decisions",
        "expected": EXPECTED,
        "counts": {name: len(ids) for name, ids in splits.items()},
        "by_size": {
            name: dict(sorted(Counter(_size(args.puzzles, pid) for pid in ids).items())) for name, ids in splits.items()
        },
        "splits": splits,
        "not_in_corpus": sorted(unassigned),
        "problems": problems,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if problems:
        print("\nproblems:")
        for problem in problems:
            print(f"  - {problem}")
    print(f"\nsplits -> {args.out}")
    return 0 if not problems else 1


_SIZE_CACHE: dict[str, str] = {}


def _size(path: Path, puzzle_id: str) -> str:
    if not _SIZE_CACHE:
        for line in path.read_text("utf-8").splitlines():
            if line:
                row = json.loads(line)
                _SIZE_CACHE[row["puzzle_id"]] = f"{row['rows']}x{row['cols']}"
    return _SIZE_CACHE.get(puzzle_id, "?")


if __name__ == "__main__":
    raise SystemExit(main())
