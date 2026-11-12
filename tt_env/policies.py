"""Reference-free baseline policies, expressed as programs the sandbox can run.

Every baseline here is a *policy*, not a solver: it writes a grid from the payload alone and is graded
by exactly the same code path as a model completion. That property is what makes the §7.5 table
comparable with a trained model.

    empty     - no track at all
    givens    - copy the given pieces, leave the rest blank
    random    - occupy the right number of cells per row/col, fill them with random legal pieces
    reference - the `tracks.py` DFS (the ceiling)
"""

from __future__ import annotations

import random

from .puzzle import Puzzle
from .rules import EMPTY

PIECES = ("NS", "EW", "NE", "NW", "SE", "SW")


def program(grid: list[str]) -> str:
    """Wrap a concrete grid in the completion format a model is asked to produce."""
    rows = ",\n        ".join(repr(row) for row in grid)
    return f"```python\ndef solve(puzzle):\n    return [\n        {rows},\n    ]\n```"


def all_empty(puzzle: Puzzle) -> list[str]:
    return [EMPTY * puzzle.cols for _ in range(puzzle.rows)]


def givens_only(puzzle: Puzzle) -> list[str]:
    cells = [[EMPTY] * puzzle.cols for _ in range(puzzle.rows)]
    for r, c, piece in puzzle.givens:
        cells[r][c] = piece
    return ["".join(row) for row in cells]


def random_clues(puzzle: Puzzle, rng: random.Random) -> list[str]:
    """Random occupancy matching the row/col clues, random legal pieces.

    Occupancy is built row by row with the column remainders as the target, so the counts are close
    but not guaranteed exact - which is honest: a policy that hits the clues exactly and still draws
    is the `audit_rewards.py` case, and both must score far below a solve.
    """
    rows, cols = puzzle.rows, puzzle.cols
    given = {(r, c) for r, c, _ in puzzle.givens}
    remaining = list(puzzle.col_clues)
    occupied: set[tuple[int, int]] = set()
    for r in range(rows):
        need = puzzle.row_clues[r] - sum(1 for c in range(cols) if (r, c) in given)
        free = [c for c in range(cols) if (r, c) not in given]
        need = max(0, min(need, len(free)))
        # Prefer columns that still need cells in proportion to what is left.
        weights = [max(1, remaining[c]) for c in free]
        for c in rng.choices(free, weights=weights, k=need):
            occupied.add((r, c))
            remaining[c] = max(0, remaining[c] - 1)

    cells = [[EMPTY] * cols for _ in range(rows)]
    for r, c, piece in puzzle.givens:
        cells[r][c] = piece
    for r, c in occupied:
        cells[r][c] = rng.choice(PIECES)
    return ["".join(row) for row in cells]


__all__ = ["PIECES", "all_empty", "givens_only", "program", "random_clues"]
