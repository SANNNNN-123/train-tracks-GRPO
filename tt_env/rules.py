"""Exact, label-free rules for Train Tracks. No solver, no stored solution.

Internal convention: 0-based, row 0 = TOP, col 0 = LEFT.
Grids are packed rows: R strings of length 2*C, two chars per cell, empty = "..".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

DIRS = ("N", "E", "S", "W")
OFFSETS = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}
EMPTY = ".."

# Only these seven values are legal; keyed by letter-set so permutations normalise.
_LEGAL = {
    frozenset("NS"): "NS",
    frozenset("EW"): "EW",
    frozenset("NE"): "NE",
    frozenset("NW"): "NW",
    frozenset("SE"): "SE",
    frozenset("SW"): "SW",
}


def canonical_piece(piece: str) -> str | None:
    """Return the canonical two-letter piece, or None if illegal."""
    letters = "".join(piece)
    if len(letters) != 2:
        return None
    if letters[0] == letters[1] or any(ch not in DIRS for ch in letters):
        return None
    return _LEGAL.get(frozenset(letters))


def dirs_of(piece: str) -> tuple[str, ...]:
    return tuple(ch for ch in piece if ch in DIRS)


def outward_port(piece: str, r: int, c: int, rows: int, cols: int) -> str | None:
    """The port of `piece` that leaves the grid, or None if it stays inside."""
    for d in dirs_of(piece):
        dr, dc = OFFSETS[d]
        if not (0 <= r + dr < rows and 0 <= c + dc < cols):
            return d
    return None


@dataclass
class Verdict:
    ok: bool
    failures: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    def __bool__(self) -> bool:  # allows `if verify(...)`
        return self.ok


def decode_grid(grid: Sequence[str], rows: int, cols: int) -> tuple[list[list[str]] | None, list[str]]:
    """Unpack packed rows into cells, or explain why the shape is wrong."""
    failures: list[str] = []
    if not isinstance(grid, (list, tuple)):
        return None, ["shape:not-a-list"]
    if len(grid) != rows:
        failures.append(f"shape:row-count {len(grid)} != {rows}")
        return None, failures
    cells: list[list[str]] = []
    for r, row in enumerate(grid):
        if not isinstance(row, str) or len(row) != 2 * cols:
            failures.append(f"shape:row-{r}-length")
            return None, failures
        out = []
        for c in range(cols):
            chunk = row[2 * c : 2 * c + 2]
            if chunk == EMPTY:
                out.append(EMPTY)
                continue
            piece = canonical_piece(chunk)
            if piece is None:
                failures.append(f"shape:illegal-piece-{chunk!r}-at-{r},{c}")
                return None, failures
            out.append(piece)
        cells.append(out)
    return cells, failures


def verify(puzzle, grid) -> Verdict:
    """Check a packed grid against the five rules. `puzzle` needs rows, cols,
    row_clues, col_clues, givens, start, end."""
    rows, cols = puzzle.rows, puzzle.cols
    cells, failures = decode_grid(grid, rows, cols)
    if cells is None:
        return Verdict(False, failures, {})

    diag: dict = {}

    # --- rule 2: givens untouched -------------------------------------------------
    for r, c, piece in puzzle.givens:
        if cells[r][c] != piece:
            failures.append(f"givens:{r},{c} expected {piece} got {cells[r][c]}")

    # --- rule 3: reciprocity, and off-grid ports only at the endpoints ------------
    start = tuple(puzzle.start)
    end = tuple(puzzle.end)
    dangling: list[str] = []
    boundary: list[tuple[int, int, str]] = []
    in_grid_degree: dict[tuple[int, int], int] = {}
    for r in range(rows):
        for c in range(cols):
            piece = cells[r][c]
            if piece == EMPTY:
                continue
            degree = 0
            for d in dirs_of(piece):
                dr, dc = OFFSETS[d]
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    boundary.append((r, c, d))
                    continue
                neighbour = cells[nr][nc]
                if neighbour == EMPTY or OPPOSITE[d] not in dirs_of(neighbour):
                    dangling.append(f"{r},{c}{d}")
                    continue
                degree += 1
            in_grid_degree[(r, c)] = degree
    diag["dangling"] = len(dangling)
    diag["boundary_ports"] = len(boundary)

    if dangling:
        failures.append(f"dangling:{len(dangling)}")
    if len(boundary) != 2:
        failures.append(f"boundary:{len(boundary)}")
    for r, c, d in boundary:
        if (r, c) not in (start, end):
            failures.append(f"boundary:off-grid port at non-endpoint {r},{c}{d}")
    for endpoint in (start, end):
        ports = [b for b in boundary if b[:2] == endpoint]
        if len(ports) != 1:
            failures.append(f"boundary:endpoint {endpoint} has {len(ports)} off-grid ports")

    # --- rule 4: clue counts ------------------------------------------------------
    counted_rows = [sum(1 for c in range(cols) if cells[r][c] != EMPTY) for r in range(rows)]
    counted_cols = [sum(1 for r in range(rows) if cells[r][c] != EMPTY) for c in range(cols)]
    row_diff = [counted_rows[r] - puzzle.row_clues[r] for r in range(rows)]
    col_diff = [counted_cols[c] - puzzle.col_clues[c] for c in range(cols)]
    diag["clue_diff_rows"] = row_diff
    diag["clue_diff_cols"] = col_diff
    if any(row_diff):
        failures.append(f"clue_row:{sum(1 for d in row_diff if d)}")
    if any(col_diff):
        failures.append(f"clue_col:{sum(1 for d in col_diff if d)}")

    # --- rule 5: one simple path --------------------------------------------------
    occupied = [(r, c) for r in range(rows) for c in range(cols) if cells[r][c] != EMPTY]
    diag["occupied"] = len(occupied)
    seen: set[tuple[int, int]] = set()
    components = 0
    for cell in occupied:
        if cell in seen:
            continue
        components += 1
        stack = [cell]
        seen.add(cell)
        while stack:
            r, c = stack.pop()
            for d in dirs_of(cells[r][c]):
                dr, dc = OFFSETS[d]
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and cells[nr][nc] != EMPTY:
                    if OPPOSITE[d] in dirs_of(cells[nr][nc]) and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        stack.append((nr, nc))
    diag["components"] = components
    if components > 1:
        failures.append(f"disconnected:{components}")

    degree_hist: dict[int, int] = {}
    for cell in occupied:
        degree_hist[in_grid_degree.get(cell, 0)] = degree_hist.get(in_grid_degree.get(cell, 0), 0) + 1
    diag["degree_histogram"] = dict(sorted(degree_hist.items()))
    for cell in occupied:
        expected = 1 if cell in (start, end) else 2
        if in_grid_degree.get(cell, 0) != expected:
            failures.append(f"degree:{cell} {in_grid_degree.get(cell, 0)} != {expected}")
    if not occupied:
        failures.append("disconnected:empty")

    return Verdict(not failures, failures, diag)
