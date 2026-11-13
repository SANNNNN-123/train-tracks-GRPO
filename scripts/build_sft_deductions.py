"""Build conservative one-step SFT examples from forced Train Tracks deductions."""

from __future__ import annotations

import json
from pathlib import Path

from tt_env.puzzle import load_jsonl
from tt_env.rules import EMPTY, OFFSETS, OPPOSITE, dirs_of, verify
from tt_env.rl import load_split

DIR_NAME = {"N": "north", "E": "east", "S": "south", "W": "west"}


def packed(cells: list[list[str]]) -> list[str]:
    return ["".join(row) for row in cells]


def initial_cells(puzzle) -> list[list[str]]:
    cells = [["??" for _ in range(puzzle.cols)] for _ in range(puzzle.rows)]
    for r, c, piece in puzzle.givens:
        cells[r][c] = piece
    return cells


def board_text(puzzle, cells: list[list[str]]) -> str:
    lines = ["    " + " ".join(map(str, puzzle.col_clues))]
    for r, row in enumerate(cells):
        lines.append(f"{puzzle.row_clues[r]:>2}  " + " ".join(row))
    return "\n".join(lines)


def port_neighbour_examples(puzzle, target):
    """Rule 2: a known port forces the adjacent cell to carry the reciprocal port.

    Only the forced port is written (e.g. "N?"); the cell's other port is not
    determined by this rule alone. Applies to the given pieces only, so every
    example starts from the same givens-only board.

    Example (01012026-small): the given at r1c3 is "SE", so r2c3 must carry a
    north port and r1c4 must carry a west port.
    """
    givens_board = initial_cells(puzzle)
    out = []
    for r, c, piece in puzzle.givens:
        forced = []
        for port in dirs_of(piece):
            dr, dc = OFFSETS[port]
            nr, nc = r + dr, c + dc
            if not (0 <= nr < puzzle.rows and 0 <= nc < puzzle.cols):
                continue  # off-grid port at the start/end endpoint
            need = OPPOSITE[port]
            current = givens_board[nr][nc]
            if current != "??":
                if not dirs_of(current) or need not in dirs_of(current):
                    raise ValueError(
                        f"givens contradict port reciprocity: {puzzle.puzzle_id} r{r}c{c} {port} -> r{nr}c{nc} {current}")
                continue  # neighbour is already a given; nothing new to deduce
            if target[nr][nc] == EMPTY or need not in dirs_of(target[nr][nc]):
                raise ValueError(
                    f"reference contradicts port reciprocity: {puzzle.puzzle_id} r{nr}c{nc} needs {need}")
            forced.append((nr, nc, need, port))
        if not forced:
            continue
        after = [row.copy() for row in givens_board]
        action_cells = []
        reasons = []
        for nr, nc, need, port in forced:
            after[nr][nc] = f"{need}?"
            action_cells.append([nr, nc, f"{need}?"])
            reasons.append(
                f"The given at r{r + 1}c{c + 1} has a {DIR_NAME[port]} port, "
                f"so r{nr + 1}c{nc + 1} must be a track with a {DIR_NAME[need]} port.")
        out.append({
            "puzzle_id": puzzle.puzzle_id,
            "rule": "port_neighbour",
            "input": {"puzzle": puzzle.payload(), "board": board_text(puzzle, givens_board)},
            "action": {"kind": "occupied", "cells": action_cells, "explanation": " ".join(reasons)},
            "target_board": board_text(puzzle, after),
        })
    return out


def examples(puzzle):
    if not puzzle.reference_solution:
        return []
    target = [[puzzle.reference_solution[r][2*c:2*c+2] for c in range(puzzle.cols)]
              for r in range(puzzle.rows)]
    if not verify(puzzle, puzzle.reference_solution).ok:
        raise ValueError(f"invalid reference solution: {puzzle.puzzle_id}")
    cells = initial_cells(puzzle)
    out = []

    # Each rule action is independently forced by a clue and the current known cells.
    for axis, count, clue_name in (("row", puzzle.rows, "row_clues"),
                                   ("column", puzzle.cols, "col_clues")):
        for i in range(count):
            indices = [(i, j) for j in range(puzzle.cols)] if axis == "row" else [(j, i) for j in range(puzzle.rows)]
            clue = getattr(puzzle, clue_name)[i]
            occupied = sum(cells[r][c] not in (EMPTY, "??", "XX") for r, c in indices)
            unknown = [(r, c) for r, c in indices if cells[r][c] == "??"]
            remaining = clue - occupied
            if not unknown or remaining not in (0, len(unknown)):
                continue
            value = "empty" if remaining == 0 else "occupied"
            before = [row.copy() for row in cells]
            action_cells = []
            for r, c in unknown:
                if remaining == 0:
                    cells[r][c] = "XX"
                    action_cells.append([r, c, "X"])
                else:
                    if target[r][c] == EMPTY:
                        raise ValueError(f"reference contradicts forced clue deduction: {puzzle.puzzle_id} {axis} {i}")
                    # Occupancy is forced, but the orientation is not known yet.
                    cells[r][c] = "?"
                    action_cells.append([r, c, "?"])
            if not action_cells:
                continue
            explanation = f"The {axis} clue is {clue}; {occupied} occupied and {len(unknown)} unknown."
            if remaining == 0:
                explanation += " The line is full, so every remaining cell is empty."
            else:
                explanation += " Every remaining cell must be a track, but its orientation is still open."
            out.append({
                "puzzle_id": puzzle.puzzle_id,
                "rule": "clue_satisfied" if remaining == 0 else "all_remaining_cells_required",
                "input": {"puzzle": puzzle.payload(), "board": board_text(puzzle, before)},
                "action": {"kind": value, "cells": action_cells, "explanation": explanation},
                "target_board": board_text(puzzle, cells),
            })

    out.extend(port_neighbour_examples(puzzle, target))
    return out


def main():
    puzzle_path = Path("data/puzzles.jsonl")
    split_path = Path("data/splits.json")
    train = load_split(puzzle_path, split_path, "train")
    records = [record for puzzle in train for record in examples(puzzle)]
    dest = Path("data/sft_deductions_train.jsonl")
    dest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    print(f"wrote {len(records)} examples from {len(train)} training puzzles to {dest}")


if __name__ == "__main__":
    main()
