"""Audit the reward system against plan.md §6.5 on real corpus puzzles.

Every row is *measured*, not asserted from memory: the script prints the breakdown and only then
checks the invariants that matter.

    uv run python scripts/audit_rewards.py
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402

sys.path.insert(0, str(ROOT))

from tt_env import Puzzle, grade, load_jsonl  # noqa: E402
from tt_env.policies import all_empty, givens_only, program  # noqa: E402
from tt_env.rules import EMPTY, OFFSETS  # noqa: E402

CEILING = 31.5
NON_SOLVE_CEILING = 4.0  # format 0.5 + no_cheat 1.0 + shaping 1.5, plus headroom


def random_exact_counts(puzzle: Puzzle, reference: list[str], rng: random.Random) -> list[str] | None:
    """Random occupancy with the reference row/col sums, filled with random legal pieces."""
    rows, cols = puzzle.rows, puzzle.cols
    given = {(r, c) for r, c, _ in puzzle.givens}
    occupied = {
        (r, c) for r in range(rows) for c in range(cols) if reference[r][2 * c : 2 * c + 2] != EMPTY
    }
    movable = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in given]
    if len(movable) < 4:
        return None
    for _ in range(4000):
        (r1, c1), (r2, c2) = rng.sample(movable, 2)
        if r1 == r2 or c1 == c2:
            continue
        corners = [(r1, c1), (r1, c2), (r2, c1), (r2, c2)]
        if any(cell in given for cell in corners):
            continue
        values = [cell in occupied for cell in corners]
        if values in ([True, False, False, True], [False, True, True, False]):
            for cell, value in zip(corners, values):
                if value:
                    occupied.discard(cell)
                else:
                    occupied.add(cell)
    pieces = ("NS", "EW", "NE", "NW", "SE", "SW")
    cells = [[EMPTY] * cols for _ in range(rows)]
    for r, c, piece in puzzle.givens:
        cells[r][c] = piece
    for r, c in occupied:
        if (r, c) not in given:
            cells[r][c] = rng.choice(pieces)
    return ["".join(row) for row in cells]


def two_components(puzzle: Puzzle, reference: list[str]) -> list[str] | None:
    """Reference path plus a 2x2 loop dropped into an empty 2x2 block, if one exists."""
    grid = [list(reference[r][2 * c : 2 * c + 2] for c in range(puzzle.cols)) for r in range(puzzle.rows)]
    for r in range(puzzle.rows - 1):
        for c in range(puzzle.cols - 1):
            block = [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]
            if all(grid[pr][pc] == EMPTY for pr, pc in block):
                grid[r][c], grid[r][c + 1] = "SE", "SW"
                grid[r + 1][c], grid[r + 1][c + 1] = "NE", "NW"
                return ["".join(row) for row in grid]
    return None


def row_shuffle(puzzle: Puzzle, reference: list[str], rng: random.Random) -> list[str]:
    """Keep row counts, destroy columns and reciprocity: the cheapest 'looks plausible' cheat."""
    cells = [[reference[r][2 * c : 2 * c + 2] for c in range(puzzle.cols)] for r in range(puzzle.rows)]
    given = {(r, c) for r, c, _ in puzzle.givens}
    for r in range(puzzle.rows):
        indices = [c for c in range(puzzle.cols) if (r, c) not in given]
        values = [cells[r][c] for c in indices]
        rng.shuffle(values)
        for c, value in zip(indices, values):
            cells[r][c] = value
    return ["".join(row) for row in cells]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--puzzles", type=Path, default=ROOT / "data" / "puzzles.jsonl")
    ap.add_argument("--splits", type=Path, default=ROOT / "data" / "splits.json")
    ap.add_argument("--sample", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--report", type=Path, default=ROOT / "data" / "reward_audit.json")
    args = ap.parse_args()

    puzzles = {p.puzzle_id: p for p in load_jsonl(args.puzzles)}
    if args.splits.exists():
        split = json.loads(args.splits.read_text("utf-8"))["splits"]["train"]
    else:
        split = sorted(puzzles)[: args.sample * 20]
    rng = random.Random(args.seed)
    chosen = [puzzles[pid] for pid in split if pid in puzzles][: args.sample]
    if not chosen:
        raise SystemExit("no puzzles selected; build the corpus and freeze the splits first")

    rows: list[dict] = []
    for puzzle in chosen:
        reference = list(puzzle.reference_solution)
        cases: dict[str, str] = {
            "reference": program(reference),
            "no-code": "The answer is a path from A to B.",
            "wrong-signature": "```python\ndef solve(board):\n    return []\n```",
            "import-cheat": "```python\ndef solve(puzzle):\n    import tracks\n    return []\n```",
            "dunder-cheat": "```python\ndef solve(puzzle):\n    return puzzle.__class__.__mro__\n```",
            "all-empty": program(all_empty(puzzle)),
            "givens-only": program(givens_only(puzzle)),
            "row-shuffle": program(row_shuffle(puzzle, reference, rng)),
            "exception": "```python\ndef solve(puzzle):\n    raise ValueError('nope')\n```",
            "timeout": "```python\ndef solve(puzzle):\n    while True:\n        pass\n```",
        }
        if (random_grid := random_exact_counts(puzzle, reference, rng)) is not None:
            cases["random-exact-counts"] = program(random_grid)
        if (loop_grid := two_components(puzzle, reference)) is not None:
            cases["plus-2x2-loop"] = program(loop_grid)

        for name, completion in cases.items():
            result = grade(completion, puzzle, max_steps=5_000 if name == "timeout" else None)
            rows.append(
                {
                    "puzzle_id": puzzle.puzzle_id,
                    "size": f"{puzzle.rows}x{puzzle.cols}",
                    "case": name,
                    **result.to_dict(),
                }
            )

    header = f"{'puzzle':<22}{'case':<20}{'fmt':>6}{'cheat':>7}{'solved':>8}{'shape':>7}{'total':>8}  status"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['puzzle_id']:<22}{row['case']:<20}{row['format_ok']:>6.1f}{row['no_cheat']:>7.1f}"
            f"{row['solved']:>8.1f}{row['shaping']:>7.2f}{row['total']:>8.2f}  {row['status']}"
        )

    references = [r for r in rows if r["case"] == "reference"]
    others = [r for r in rows if r["case"] != "reference"]
    problems: list[str] = []
    for row in references:
        if abs(row["total"] - CEILING) > 1e-9:
            problems.append(f"{row['puzzle_id']} reference scored {row['total']}, expected {CEILING}")
        if row["shaping"] != 0:
            problems.append(f"{row['puzzle_id']} reference collected shaping {row['shaping']}")
    for row in others:
        if row["total"] >= CEILING:
            problems.append(f"{row['puzzle_id']} {row['case']} reached the ceiling ({row['total']})")
        if row["total"] > NON_SOLVE_CEILING:
            problems.append(f"{row['puzzle_id']} {row['case']} scored {row['total']} > {NON_SOLVE_CEILING}")
        if row["solved"] == 30.0:
            problems.append(f"{row['puzzle_id']} {row['case']} was accepted as a solve")

    for case in ("import-cheat", "dunder-cheat"):
        wrong = [r for r in rows if r["case"] == case and r["no_cheat"] != -20.0]
        if wrong:
            problems.append(f"{case}: no_cheat != -20 for {len(wrong)} puzzles")
    empty_rows = [r for r in rows if r["case"] == "all-empty" and r["shaping"] != 0.0]
    if empty_rows:
        problems.append(f"all-empty earned shaping on {len(empty_rows)} puzzles")

    summary = {
        "sample": [p.puzzle_id for p in chosen],
        "seed": args.seed,
        "rows": rows,
        "reference_mean": round(sum(r["total"] for r in references) / len(references), 4),
        "max_non_solve": max((r["total"] for r in others), default=0.0),
        "problems": problems,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nreference mean {summary['reference_mean']}  max non-solve {summary['max_non_solve']}")
    if problems:
        print("PROBLEMS:")
        for problem in problems:
            print(f"  - {problem}")
    print(f"report -> {args.report}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
