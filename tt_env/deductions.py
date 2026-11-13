"""Combine the sound one-step rules and run them together to a fixpoint.

Not a solver: this only records facts that are forced by the current board.

Cell state is one of:
  ``None``            unknown
  ``"XX"``            known empty
  ``frozenset()``     known track, orientation unknown
  ``frozenset({...})`` known track with one or two known ports
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .rules import EMPTY, OFFSETS, OPPOSITE, canonical_piece, dirs_of

DIR_NAME = {"N": "north", "E": "east", "S": "south", "W": "west"}
PORT_ORDER = "NESW"
UNKNOWN = None
EMPTY_CELL = "XX"


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


@dataclass
class Step:
    rule: str
    explanation: str
    cells: list[tuple[int, int, str]] = field(default_factory=list)


def cell_token(state) -> str:
    """Render one cell state as the board token used by the SFT data."""
    if state is None:
        return "??"
    if state == EMPTY_CELL:
        return "XX"
    if not state:
        return "?"
    if len(state) == 1:
        return next(iter(state)) + "?"
    return canonical_piece("".join(sorted(state, key=PORT_ORDER.index)))


def initial_state(puzzle) -> list[list]:
    state = [[UNKNOWN for _ in range(puzzle.cols)] for _ in range(puzzle.rows)]
    for r, c, piece in puzzle.givens:
        state[r][c] = frozenset(dirs_of(piece))
    return state


def _reference_cells(puzzle) -> list[list[str]]:
    return [[puzzle.reference_solution[r][2 * c : 2 * c + 2] for c in range(puzzle.cols)]
            for r in range(puzzle.rows)]


def _clue_steps(puzzle, state, ref) -> list[Step]:
    steps = []
    for axis, count, clue_name in (("row", puzzle.rows, "row_clues"),
                                   ("column", puzzle.cols, "col_clues")):
        clues = getattr(puzzle, clue_name)
        for i in range(count):
            indices = ([(i, j) for j in range(puzzle.cols)] if axis == "row"
                       else [(j, i) for j in range(puzzle.rows)])
            clue = clues[i]
            occupied = sum(isinstance(state[r][c], frozenset) for r, c in indices)
            unknown = [(r, c) for r, c in indices if state[r][c] is UNKNOWN]
            remaining = clue - occupied
            if remaining < 0:
                raise ValueError(f"clue contradicted: {puzzle.puzzle_id} {axis} {i + 1}")
            if not unknown or remaining not in (0, len(unknown)):
                continue
            changes = []
            for r, c in unknown:
                if remaining == 0:
                    if ref[r][c] != EMPTY:
                        raise ValueError(f"reference contradicts empty deduction: {puzzle.puzzle_id} r{r}c{c}")
                    state[r][c] = EMPTY_CELL
                    changes.append((r, c, "X"))
                else:
                    if ref[r][c] == EMPTY:
                        raise ValueError(f"reference contradicts track deduction: {puzzle.puzzle_id} r{r}c{c}")
                    state[r][c] = frozenset()
                    changes.append((r, c, "?"))
            if remaining == 0:
                explanation = (f"{axis.title()} {i + 1} clue {clue} is met "
                               f"({occupied} occupied) → {len(unknown)} remaining cells marked empty.")
                rule = "clue_satisfied"
            else:
                explanation = (f"{axis.title()} {i + 1} clue {clue} needs {remaining} more track(s) "
                               f"and has exactly {len(unknown)} unknown cells → all marked as tracks "
                               f"(orientation still open).")
                rule = "all_remaining_cells_required"
            steps.append(Step(rule, explanation, changes))
    return steps


def _port_steps(puzzle, state, ref) -> list[Step]:
    steps = []
    for r in range(puzzle.rows):
        for c in range(puzzle.cols):
            piece = state[r][c]
            if not isinstance(piece, frozenset) or not piece:
                continue
            ports = sorted(piece, key=PORT_ORDER.index)
            forced = []
            for port in ports:
                dr, dc = OFFSETS[port]
                nr, nc = r + dr, c + dc
                if not (0 <= nr < puzzle.rows and 0 <= nc < puzzle.cols):
                    continue  # off-grid port at the start/end endpoint
                need = OPPOSITE[port]
                neighbour = state[nr][nc]
                if neighbour == EMPTY_CELL:
                    raise ValueError(f"port hits an empty cell: {puzzle.puzzle_id} r{r}c{c}{port}")
                if isinstance(neighbour, frozenset) and need in neighbour:
                    continue  # already reciprocal
                if ref[nr][nc] == EMPTY or need not in dirs_of(ref[nr][nc]):
                    raise ValueError(f"reference contradicts port deduction: {puzzle.puzzle_id} r{nr}c{nc} needs {need}")
                state[nr][nc] = (neighbour or frozenset()) | {need}
                forced.append((nr, nc, need))
            if not forced:
                continue
            source = cell_token(piece)
            targets = []
            for nr, nc, need in forced:
                after = cell_token(state[nr][nc])
                suffix = f" (now {after})" if isinstance(state[nr][nc], frozenset) and len(state[nr][nc]) == 2 else ""
                targets.append(f"r{nr + 1}c{nc + 1} must carry {_article(DIR_NAME[need])} {DIR_NAME[need]} port{suffix}")
            explanation = (f"r{r + 1}c{c + 1} ({source}) points "
                           f"{' and '.join(DIR_NAME[p] for p in ports)}: " + "; ".join(targets) + ".")
            steps.append(Step("port_neighbour", explanation,
                              [(nr, nc, cell_token(state[nr][nc])) for nr, nc, _ in forced]))
    return steps


def combined_deductions(puzzle):
    """Apply clause + port rules until nothing changes. Returns (state, steps)."""
    if not puzzle.reference_solution:
        raise ValueError(f"no reference solution: {puzzle.puzzle_id}")
    ref = _reference_cells(puzzle)
    state = initial_state(puzzle)
    steps: list[Step] = []
    changed = True
    while changed:
        changed = False
        for batch in (_clue_steps(puzzle, state, ref), _port_steps(puzzle, state, ref)):
            if batch:
                steps.extend(batch)
                changed = True
    return state, steps
