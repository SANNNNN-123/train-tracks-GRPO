"""Quality report for `data/puzzles.jsonl`.

Checks that do not need the HF dataset:
  * clue integrity      - sum(row_clues) == sum(col_clues), each clue within 0..n
  * structural integrity - every grid is one path, endpoints on the boundary
  * coverage            - counts per month and per size, missing days
  * duplicates          - exact and symmetry-normalised (4 transforms; square grids also get
                          full dihedral coverage via transpose)

Usage: uv run python scripts/qa_corpus.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tt_env import Puzzle, load_jsonl, verify  # noqa: E402

FLIP = {"N": "S", "S": "N", "E": "E", "W": "W"}
MIRROR = {"N": "N", "S": "S", "E": "W", "W": "E"}
TRANSPOSE = {"N": "W", "S": "E", "E": "S", "W": "N"}


def _remap(puzzle: Puzzle, rows, cols, givens, row_clues, col_clues) -> str:
    givens_sorted = sorted((r, c, p) for r, c, p in givens)
    return json.dumps(
        {
            "r": rows,
            "c": cols,
            "rc": list(row_clues),
            "cc": list(col_clues),
            "g": [[r, c, p] for r, c, p in givens_sorted],
        },
        sort_keys=True,
    )


def symmetries(puzzle: Puzzle) -> list[str]:
    """Canonical forms under identity, 180deg, mirror-x, mirror-y (and transpose if square)."""
    R, C = puzzle.rows, puzzle.cols
    givens = [(r, c, p) for r, c, p in puzzle.givens]
    out = [
        _remap(puzzle, R, C, givens, puzzle.row_clues, puzzle.col_clues),
        _remap(
            puzzle,
            R,
            C,
            [(R - 1 - r, C - 1 - c, "".join(sorted(FLIP[d] for d in p))) for r, c, p in givens],
            tuple(reversed(puzzle.row_clues)),
            tuple(reversed(puzzle.col_clues)),
        ),
        _remap(
            puzzle,
            R,
            C,
            [(r, C - 1 - c, "".join(sorted(MIRROR[d] for d in p))) for r, c, p in givens],
            puzzle.row_clues,
            tuple(reversed(puzzle.col_clues)),
        ),
        _remap(
            puzzle,
            R,
            C,
            [(R - 1 - r, c, "".join(sorted(FLIP[d] for d in p))) for r, c, p in givens],
            tuple(reversed(puzzle.row_clues)),
            puzzle.col_clues,
        ),
    ]
    if R == C:
        out.append(
            _remap(
                puzzle,
                R,
                C,
                [(c, r, "".join(sorted(TRANSPOSE[d] for d in p))) for r, c, p in givens],
                puzzle.col_clues,
                puzzle.row_clues,
            )
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--puzzles", type=Path, default=ROOT / "data" / "puzzles.jsonl")
    ap.add_argument("--report", type=Path, default=ROOT / "data" / "qa_report.json")
    args = ap.parse_args()

    puzzles = load_jsonl(args.puzzles)
    problems: list[dict] = []

    by_size = Counter()
    by_month = Counter()
    givens_hist = Counter()
    dates_seen: dict[str, set[str]] = defaultdict(set)

    for puzzle in puzzles:
        by_size[f"{puzzle.rows}x{puzzle.cols}"] += 1
        givens_hist[len(puzzle.givens)] += 1
        match = re.match(r"(\d{2})(\d{2})(\d{4})-", puzzle.puzzle_id)
        if match:
            day, month, year = match.groups()
            month_key = f"{year}-{month}"
            by_month[month_key] += 1
            dates_seen[month_key].add(day)

        if sum(puzzle.row_clues) != sum(puzzle.col_clues):
            problems.append({"id": puzzle.puzzle_id, "issue": "clue sums differ"})
        if any(not 0 <= v <= puzzle.cols for v in puzzle.row_clues):
            problems.append({"id": puzzle.puzzle_id, "issue": "row clue out of range"})
        if any(not 0 <= v <= puzzle.rows for v in puzzle.col_clues):
            problems.append({"id": puzzle.puzzle_id, "issue": "col clue out of range"})
        for point, name in ((puzzle.start, "start"), (puzzle.end, "end")):
            r, c = point
            if not (r in (0, puzzle.rows - 1) or c in (0, puzzle.cols - 1)):
                problems.append({"id": puzzle.puzzle_id, "issue": f"{name} not on boundary"})
        verdict = verify(puzzle, list(puzzle.reference_solution or []))
        if not verdict.ok:
            problems.append({"id": puzzle.puzzle_id, "issue": f"verify: {verdict.failures[:3]}"})

    # duplicates under symmetry
    groups: dict[str, list[str]] = defaultdict(list)
    for puzzle in puzzles:
        key = min(symmetries(puzzle))
        groups[key].append(puzzle.puzzle_id)
    duplicates = [ids for ids in groups.values() if len(ids) > 1]

    month_gaps = {}
    for month_key, days in sorted(dates_seen.items()):
        year, month = (int(x) for x in month_key.split("-"))
        expected = 31 if month in (1, 3, 5, 7, 8, 10, 12) else 30 if month != 2 else 28
        missing = sorted(f"{d:02d}" for d in range(1, expected + 1) if f"{d:02d}" not in days)
        if missing:
            month_gaps[month_key] = missing

    report = {
        "puzzles": len(puzzles),
        "by_size": dict(sorted(by_size.items())),
        "by_month": dict(sorted(by_month.items())),
        "givens_per_puzzle": dict(sorted(givens_hist.items())),
        "problems": problems,
        "duplicate_groups": duplicates,
        "month_gaps": month_gaps,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"puzzles         {len(puzzles)}")
    print(f"by size         {report['by_size']}")
    print(f"givens / puzzle {report['givens_per_puzzle']}")
    print(f"problems        {len(problems)}")
    for problem in problems[:10]:
        print(f"  - {problem['id']}: {problem['issue']}")
    print(f"duplicates      {len(duplicates)} group(s)")
    for ids in duplicates[:5]:
        print(f"  - {ids}")
    print(f"month gaps      {report['month_gaps'] or 'none'}")
    print(f"report -> {args.report}")
    return 0 if not problems and not duplicates else 1


if __name__ == "__main__":
    raise SystemExit(main())
