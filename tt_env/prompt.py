"""The one prompt.

Design A asks for a *general* solver: the harness calls `solve(puzzle)` once per puzzle, so the
prompt itself never contains a puzzle. That is what makes the corpus a rollout budget rather than 608
memorisable rows, and it is why the same text is used for training and for evaluation.

`render(payload)` appends one concrete instance - useful when a human wants to see what the function
receives, and deliberately *not* used for training or for the headline eval.

Order mirrors the verifier: rules, then the input/output contract, then the constraints.
"""

from __future__ import annotations

import json

RULES = """\
TRAIN TRACKS RULES
1. The grid contains exactly one continuous track. It enters at the start cell and leaves at the end
   cell through a port on the outer border (the outward port direction is given for each endpoint).
2. Given pieces are fixed; never change them.
3. Adjacent pieces must connect reciprocally: if a cell has a port N, the cell above must have port S.
   A port that runs off the grid is only allowed at the start and end cells.
4. Each row clue is the number of cells in that row occupied by track; each column clue likewise.
5. The track may not branch, cross, touch itself, or form a separate loop. Every occupied cell lies on
   the single path from start to end: interior path cells have exactly two connections and the two
   endpoints have exactly one."""

CONTRACT = """\
INPUT
`solve` receives one dict:
  rows, cols            grid size, 6..10 each, not necessarily square
  row_clues, col_clues  clue per row (top to bottom) and per column (left to right)
  givens                list of [row, col, piece], 0-based, row 0 = top row, col 0 = leftmost
  start, end            [row, col, outward_port] of the two endpoints
A piece is two letters from N, E, S, W naming the sides it connects (letter order is irrelevant):
NS, EW, NE, NW, SE, SW.

OUTPUT
Return a list of `rows` strings. Each string has `2 * cols` characters: two characters per cell,
left to right. Use the two-letter piece, or ".." for an empty cell. Example for a 2x3 grid:
  ["EW..NS", "..NSEW"]   ->  (0,0)=EW  (0,1)=empty  (0,2)=NS  /  (1,0)=empty  (1,1)=NS  (1,2)=EW

CONSTRAINTS
- Write exactly one ```python block containing `def solve(puzzle):`.
- Return the grid; do not describe it in prose.
- Standard Python only: no imports, no file or network access, no printing.
- Do not hardcode a grid: the same function is called on many different puzzles.
- Finish within 10 seconds per puzzle, including the search and the backtracking."""

INSTRUCTIONS = """\
You are an expert at solving Train Tracks logic puzzles. Write a Python function that solves ANY
Train Tracks puzzle whose input follows the contract below. The harness calls your function once per
puzzle with that puzzle's data, so it must compute the answer from `puzzle` alone and must not assume
a particular grid. A depth-first search over ports with constraint propagation is usually enough."""

CLOSING = """\
Write the function now. Start your reply with ```python and end it with ```; put no prose before or
after the block. Keep the code compact: no comments, no docstrings, no type annotations, and aim for
under 80 lines - the reply has to fit a fixed token budget."""


def prompt_text() -> str:
    """The fixed training/eval prompt. Contains no puzzle data."""
    return f"{INSTRUCTIONS}\n\n{RULES}\n\n{CONTRACT}\n\n{CLOSING}"


def render(payload: dict | None = None) -> str:
    """The prompt, plus one concrete instance when a payload is supplied (debug / demo only)."""
    if payload is None:
        return prompt_text()
    body = json.dumps(payload, separators=(",", ":"))
    return f"{prompt_text()}\n\nSAMPLE INSTANCE (for reference only; solve puzzles you have not seen)\n{body}"


def messages(payload: dict | None = None) -> list[dict]:
    """Chat-format prompt. Text only, no images, identical for every puzzle by default."""
    return [
        {"role": "system", "content": "You write correct, self-contained Python for logic puzzles."},
        {"role": "user", "content": render(payload)},
    ]


__all__ = ["CLOSING", "CONTRACT", "INSTRUCTIONS", "RULES", "messages", "prompt_text", "render"]
