"""Bridge between our top-down puzzle space and the reference solver in `tracks-solver/`.

`tracks.py` indexes layout rows bottom-up (row 0 = bottom), while images, the HF dataset and
everything in `tt_env` are top-down. Directions (N/E/S/W) are *not* flipped: only the row index is,
which is exactly what `track_solver.parse_puzzle_image` does. Keep all conversions here.

The solver can leave the final cell as `Track.TEMP` (it raises "Solved" before converting it), so the
path is re-walked here by following ports, treating a TEMP cell as an unknown that accepts any port.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOLVER_DIR = ROOT / "tracks-solver"

from .rules import DIRS, OFFSETS, OPPOSITE, canonical_piece, dirs_of  # noqa: E402

EMPTY = ".."
TEMP = ""  # sentinel for a cell whose piece the solver never resolved

# `tracks.py` searches depth-first in a fixed N, S, W, E order. On a few hard instances that order
# explores millions of dead ends, so the builder retries with a better ordering instead of failing
# the puzzle. Strategies are tried in order; the first is the solver's own behaviour.
STRATEGIES = (
    {"name": "plain-NSWE", "order": "NSWE", "heuristic": "plain", "move_max": 1_000_000},
    {"name": "warnsdorff-NSWE", "order": "NSWE", "heuristic": "warnsdorff", "move_max": 20_000_000},
    {"name": "warnsdorff-ESWN", "order": "ESWN", "heuristic": "warnsdorff", "move_max": 20_000_000},
    {"name": "plain-ESWN", "order": "ESWN", "heuristic": "plain", "move_max": 20_000_000},
)

_BASE_MOVES = None


def install_move_order(order: str, heuristic: str) -> None:
    """Wrap `Layout.moves` so the DFS consumes candidates in a chosen order.

    `warnsdorff` prefers the neighbour with the fewest onward moves, which prunes long dead ends;
    ties and plain mode fall back to `order` (e.g. "NSWE"). Re-installing never nests wrappers.
    """
    global _BASE_MOVES

    tracks = _tracks_module()
    if _BASE_MOVES is None:
        _BASE_MOVES = tracks.Layout.moves
    base = _BASE_MOVES
    offsets = {"N": (1, 0), "S": (-1, 0), "W": (0, -1), "E": (0, 1)}

    def moves(self, cell):
        candidates = base(self, cell)
        if heuristic == "warnsdorff":

            def onward(direction):
                dr, dc = offsets[direction]
                row, col = cell.row + dr, cell.col + dc
                if not (0 <= row < self.rows and 0 <= col < self.columns):
                    return 99
                return len(base(self, self.layout[row][col]))

            return sorted(candidates, key=lambda d: (onward(d), order.index(d)))
        return sorted(candidates, key=lambda d: order.index(d))

    tracks.Layout.moves = moves


def _tracks_module():
    if str(SOLVER_DIR) not in sys.path:
        sys.path.insert(0, str(SOLVER_DIR))
    import tracks  # noqa: PLC0415

    return tracks


def to_tracks_params(puzzle) -> dict:
    """Build the dict `tracks.parse` accepts for this puzzle."""
    pieces = []
    for r, c, piece in puzzle.givens:
        marker = []
        if (r, c) == tuple(puzzle.start):
            marker = ["s"]
        elif (r, c) == tuple(puzzle.end):
            marker = ["e"]
        pieces.append([piece, puzzle.rows - 1 - r, c, *marker])
    return {"rows": list(puzzle.row_clues), "columns": list(puzzle.col_clues), "pieces": pieces}


def neighbours(r: int, c: int, rows: int, cols: int):
    for d in DIRS:
        dr, dc = OFFSETS[d]
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield d, nr, nc


def walk_path(puzzle, cells: dict[tuple[int, int], str]) -> list[tuple[int, int]]:
    """Follow ports from `start` to `end`. `cells` maps position -> piece, with TEMP as ''."""
    rows, cols = puzzle.rows, puzzle.cols
    start, end = tuple(puzzle.start), tuple(puzzle.end)
    if start not in cells:
        raise ValueError(f"start {start} is not occupied")
    if end not in cells:
        raise ValueError(f"end {end} is not occupied")

    path = [start]
    came: str | None = None
    cur = start
    while cur != end:
        piece = cells[cur]
        if not piece:
            raise ValueError(f"cell {cur} has no resolved piece before reaching the end")
        options = []
        for d in dirs_of(piece):
            if d == came:
                continue
            dr, dc = OFFSETS[d]
            nxt = (cur[0] + dr, cur[1] + dc)
            if nxt not in cells:
                continue
            nxt_piece = cells[nxt]
            if nxt_piece == TEMP or OPPOSITE[d] in dirs_of(nxt_piece):
                options.append((d, nxt))
        if len(options) != 1:
            raise ValueError(f"path from {cur} (piece {piece}) has {len(options)} continuations: {options}")
        d, nxt = options[0]
        came = OPPOSITE[d]
        cur = nxt
        path.append(cur)
        if len(path) > rows * cols:
            raise ValueError("path exceeded the grid size")
    return path


def derive_solution(puzzle, cells: dict[tuple[int, int], str]) -> list[str]:
    """Rebuild the piece letters for every occupied cell from the solved path."""
    rows, cols = puzzle.rows, puzzle.cols
    path = walk_path(puzzle, cells)
    if set(path) != set(cells):
        off = sorted(set(cells) - set(path))[:6]
        raise ValueError(f"{len(set(cells) - set(path))} occupied cells are off the path (e.g. {off})")

    outward = _outward_ports(puzzle)
    grid = [[EMPTY] * cols for _ in range(rows)]
    for index, cell in enumerate(path):
        dirs = set()
        if index > 0:
            dirs.add(_dir_between(cell, path[index - 1]))
        if index + 1 < len(path):
            dirs.add(_dir_between(cell, path[index + 1]))
        if cell in outward:
            dirs.add(outward[cell])
        piece = canonical_piece("".join(sorted(dirs)))
        if piece is None:
            raise ValueError(f"cannot form a piece at {cell} from {sorted(dirs)}")
        grid[cell[0]][cell[1]] = piece

    return ["".join(grid[r][c] for c in range(cols)) for r in range(rows)]


def _dir_between(a, b) -> str:
    for d in DIRS:
        dr, dc = OFFSETS[d]
        if (a[0] + dr, a[1] + dc) == b:
            return d
    raise ValueError(f"{a} and {b} are not adjacent")


def _outward_ports(puzzle) -> dict[tuple[int, int], str]:
    """The single port leaving the grid at each endpoint, taken from the given piece."""
    ports: dict[tuple[int, int], str] = {}
    for endpoint in (tuple(puzzle.start), tuple(puzzle.end)):
        piece = next((p for r, c, p in puzzle.givens if (r, c) == endpoint), None)
        if piece is None:
            raise ValueError(f"endpoint {endpoint} has no given piece")
        for d in dirs_of(piece):
            dr, dc = OFFSETS[d]
            nr, nc = endpoint[0] + dr, endpoint[1] + dc
            if not (0 <= nr < puzzle.rows and 0 <= nc < puzzle.cols):
                ports[endpoint] = d
        if endpoint not in ports:
            raise ValueError(f"endpoint {endpoint} has no outward port in {piece}")
    return ports


def solve_with_tracks(payload: dict, strategy: dict | None = None) -> dict:
    """Solve one puzzle with the reference DFS. Runs in a child process.

    `strategy` selects the move ordering and budget (see `STRATEGIES`). Returns
    {"status": "solved"|"failed"|"timeout"|"error", "grid": [...], "seconds": float, "moves": int}.
    """
    from .puzzle import Puzzle  # local import keeps the child process light

    tracks = _tracks_module()
    strategy = strategy or STRATEGIES[0]
    install_move_order(strategy["order"], strategy["heuristic"])
    puzzle = Puzzle.from_dict(payload)
    started = time.perf_counter()
    try:
        board = tracks.parse(to_tracks_params(puzzle), graphical=False)
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "detail": f"parse: {exc}", "seconds": time.perf_counter() - started}
    board.move_max = strategy["move_max"]

    message = "failed"
    try:
        board.solve()
    except ValueError as exc:
        message = str(exc)
    except RecursionError:
        return {"status": "failed", "detail": "recursion", "seconds": time.perf_counter() - started}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "detail": f"solve: {exc}", "seconds": time.perf_counter() - started}

    cells: dict[tuple[int, int], str] = {}
    for r in range(puzzle.rows):
        for c in range(puzzle.cols):
            track = board.layout[r][c].track
            if track is None:
                continue
            name = getattr(track, "name", "")
            cells[(puzzle.rows - 1 - r, c)] = TEMP if name == "TEMP" else name

    elapsed = time.perf_counter() - started
    try:
        grid = derive_solution(puzzle, cells)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "detail": f"derive: {exc} (solver said {message})",
            "seconds": elapsed,
            "moves": board.move_count,
        }

    return {
        "status": "solved" if message == "Solved" else "failed",
        "detail": message,
        "grid": grid,
        "seconds": elapsed,
        "moves": board.move_count,
        "strategy": strategy["name"],
    }


__all__ = [
    "EMPTY",
    "TEMP",
    "derive_solution",
    "solve_with_tracks",
    "to_tracks_params",
    "walk_path",
]
