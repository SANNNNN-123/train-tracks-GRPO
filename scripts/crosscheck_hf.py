"""Cross-check `data/puzzles.jsonl` against the HF decision dataset (G0 gate).

The dataset stores one row per decision the originaL authoring pipeline made, with 1-based top-down
co-ordinates: `state.target_cell` and `label`. Givens never appear as targets, so

    givens  ==  all cells  -  {target_cell}

and `label` gives the correct piece for every non-given cell. That makes two independent checks:

  * structure - dimensions, given positions, given count, decision count
  * content   - reference solution rebuilt from `givens + labels` must equal ours cell for cell

Usage:
    uv run python scripts/fetch_hf_decisions.py
    uv run python scripts/crosscheck_hf.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tt_env import Puzzle, load_jsonl, verify  # noqa: E402

LABELS = {"empty": ".."}
VALID_PIECES = {"EW", "NE", "NS", "NW", "SE", "SW"}


def load_decisions(paths: list[Path]) -> dict[str, dict]:
    """puzzle_id -> {dims, targets, labels, difficulty, date, steps, split}."""
    grouped: dict[str, dict] = defaultdict(lambda: {"targets": set(), "labels": {}, "steps": 0})
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                entry = grouped[row["puzzle_id"]]
                entry["rows"] = row["rows"]
                entry["columns"] = row["columns"]
                entry["difficulty"] = row["difficulty"]
                entry["date"] = row["source_date"]
                entry["split"] = row["split"]
                entry["steps"] += 1
                target = (int(row["row"]) - 1, int(row["column"]) - 1)
                if target in entry["targets"]:
                    entry.setdefault("repeat_targets", set()).add(target)
                entry["targets"].add(target)
                entry["labels"][target] = LABELS.get(row["label"], row["label"])
    return dict(grouped)


def rebuild(puzzle: Puzzle, labels: dict[tuple[int, int], str]) -> list[str] | None:
    cells = [[".."] * puzzle.cols for _ in range(puzzle.rows)]
    for r, c, piece in puzzle.givens:
        cells[r][c] = piece
    for (r, c), label in labels.items():
        if not (0 <= r < puzzle.rows and 0 <= c < puzzle.cols):
            return None
        if (r, c) in {(gr, gc) for gr, gc, _ in puzzle.givens}:
            if cells[r][c] != label:
                return None  # dataset disagrees with the parsed given piece
            continue
        cells[r][c] = label
    return ["".join(row) for row in cells]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions-dir", type=Path, default=ROOT / "data" / "hf")
    ap.add_argument("--puzzles", type=Path, default=ROOT / "data" / "puzzles.jsonl")
    ap.add_argument("--report", type=Path, default=ROOT / "data" / "hf_crosscheck.json")
    args = ap.parse_args()

    paths = sorted(args.decisions_dir.glob("*_decisions.jsonl"))
    if not paths:
        raise SystemExit(f"no decisions in {args.decisions_dir}; run scripts/fetch_hf_decisions.py first")

    decisions = load_decisions(paths)
    puzzles = {p.puzzle_id: p for p in load_jsonl(args.puzzles)}
    print(f"HF puzzles {len(decisions)} from {len(paths)} split file(s)   corpus puzzles {len(puzzles)}")

    checks = {"dims": 0, "givens": 0, "given_count": 0, "steps": 0, "solution": 0}
    per_split: dict[str, dict] = {}
    mismatches: list[dict] = []
    missing = sorted(set(decisions) - set(puzzles))
    extra = sorted(set(puzzles) - set(decisions))

    for puzzle_id, entry in sorted(decisions.items()):
        puzzle = puzzles.get(puzzle_id)
        if puzzle is None:
            continue
        split = entry.get("split", "?")
        bucket = per_split.setdefault(split, {"compared": 0, "ok": 0})
        bucket["compared"] += 1
        rows, cols = entry["rows"], entry["columns"]
        problems = []

        if (rows, cols) == (puzzle.rows, puzzle.cols):
            checks["dims"] += 1
        else:
            problems.append(f"dims HF {rows}x{cols} vs ours {puzzle.rows}x{puzzle.cols}")

        all_cells = {(r, c) for r in range(rows) for c in range(cols)}
        implied_givens = all_cells - entry["targets"]
        ours = {(r, c) for r, c, _ in puzzle.givens}
        if implied_givens == ours:
            checks["givens"] += 1
        else:
            problems.append(
                f"givens: missing {sorted(ours - implied_givens)[:4]} extra {sorted(implied_givens - ours)[:4]}"
            )
        if len(ours) == len(implied_givens):
            checks["given_count"] += 1
        if entry["steps"] == rows * cols - len(ours):
            checks["steps"] += 1
        else:
            problems.append(f"steps {entry['steps']} != cells {rows * cols} - givens {len(ours)}")
        if entry.get("repeat_targets"):
            problems.append(f"repeated targets {sorted(entry['repeat_targets'])[:4]}")

        rebuilt = rebuild(puzzle, entry["labels"])
        if rebuilt is None:
            problems.append("dataset label contradicts a parsed given piece")
        else:
            verdict = verify(puzzle, rebuilt)
            if verdict.ok:
                checks["solution"] += 1
            if puzzle.reference_solution and rebuilt == list(puzzle.reference_solution):
                pass
            else:
                problems.append("solution from labels differs from ours")

        if problems:
            mismatches.append({"puzzle_id": puzzle_id, "problems": problems})
        else:
            bucket["ok"] += 1

    total = len(set(decisions) & set(puzzles))
    report = {
        "hf_puzzles": len(decisions),
        "corpus_puzzles": len(puzzles),
        "compared": total,
        "missing_from_corpus": missing,
        "not_in_hf": extra[:20],
        "not_in_hf_count": len(extra),
        "checks_passed": checks,
        "agreement": {name: round(count / total, 4) if total else 0.0 for name, count in checks.items()},
        "per_split": {name: {**bucket, "agreement": round(bucket["ok"] / bucket["compared"], 4)} for name, bucket in sorted(per_split.items())},
        "mismatches": mismatches,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"compared {total}")
    for name, count in checks.items():
        print(f"  {name:<12} {count:>4}/{total}  {100 * count / total:6.2f}%")
    for name, bucket in sorted(per_split.items()):
        print(f"  split {name:<12} {bucket['ok']:>4}/{bucket['compared']:<4} fully agreeing")
    print(f"missing from corpus {len(missing)} {missing[:5]}")
    print(f"corpus puzzles with no HF decisions {len(extra)} {extra[:5]}")
    print(f"mismatches {len(mismatches)}")
    for item in mismatches[:5]:
        print(f"  - {item['puzzle_id']}: {item['problems'][:2]}")
    print(f"report -> {args.report}")
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
