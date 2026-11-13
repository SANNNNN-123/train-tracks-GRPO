"""Build contradiction-detection SFT examples from verified Train Tracks solutions.

Each example shows a completed board with exactly one broken local rule:

  clue_mismatch      one row/column clue disagrees with the number of tracks
  port_reciprocity   one piece was rotated, so a port is not answered

The board is otherwise the verified reference solution, so the intended rule is
the only local rule that fails. Connectivity/degree failures are consequences of
the broken port and are not counted as a second rule.
"""

from __future__ import annotations

import json
import zlib
from collections import Counter
from pathlib import Path

from tt_env.rl import load_split
from tt_env.rules import EMPTY, OFFSETS, OPPOSITE, dirs_of, verify

DIR_NAME = {"N": "north", "E": "east", "S": "south", "W": "west"}
# Every 90-degree rotation of each legal piece (the original excluded).
ROTATIONS = {
    "NS": ("EW",),
    "EW": ("NS",),
    "NE": ("SE", "SW", "NW"),
    "SE": ("SW", "NW", "NE"),
    "SW": ("NW", "NE", "SE"),
    "NW": ("NE", "SE", "SW"),
}


def stable_index(seed: str, size: int) -> int:
    """Deterministic choice so reruns produce the same dataset."""
    return zlib.crc32(seed.encode("utf-8")) % size


def reference_cells(puzzle) -> list[list[str]]:
    return [[puzzle.reference_solution[r][2 * c : 2 * c + 2] for c in range(puzzle.cols)]
            for r in range(puzzle.rows)]


def board_text(puzzle, cells, row_clues=None, col_clues=None) -> str:
    rows = row_clues if row_clues is not None else puzzle.row_clues
    cols = col_clues if col_clues is not None else puzzle.col_clues
    lines = ["    " + " ".join(map(str, cols))]
    for r, row in enumerate(cells):
        lines.append(f"{rows[r]:>2}  " + " ".join(row))
    return "\n".join(lines)


def payload(puzzle, row_clues=None, col_clues=None) -> dict:
    data = puzzle.payload()
    if row_clues is not None:
        data["row_clues"] = list(row_clues)
    if col_clues is not None:
        data["col_clues"] = list(col_clues)
    return data


def dangling_edges(puzzle, cells):
    """Every on-grid port that the neighbour does not answer."""
    out = []
    for r in range(puzzle.rows):
        for c in range(puzzle.cols):
            piece = cells[r][c]
            if piece == EMPTY:
                continue
            for direction in dirs_of(piece):
                dr, dc = OFFSETS[direction]
                nr, nc = r + dr, c + dc
                if not (0 <= nr < puzzle.rows and 0 <= nc < puzzle.cols):
                    continue
                neighbour = cells[nr][nc]
                if neighbour == EMPTY or OPPOSITE[direction] not in dirs_of(neighbour):
                    out.append((r, c, direction, nr, nc, neighbour))
    return out


def rule_breakdown(puzzle, cells):
    """Classify failures, ignoring connectivity/degree consequences."""
    verdict = verify(puzzle, ["".join(row) for row in cells])
    groups = {"givens": [], "clue": [], "boundary": [], "port": [], "other": []}
    for failure in verdict.failures:
        if failure.startswith("givens:"):
            groups["givens"].append(failure)
        elif failure.startswith("clue_"):
            groups["clue"].append(failure)
        elif failure.startswith("boundary:"):
            groups["boundary"].append(failure)
        elif failure.startswith("dangling:"):
            groups["port"].append(failure)
        else:  # degree / disconnected are consequences of a broken port
            groups["other"].append(failure)
    return groups


def clue_mismatch_example(puzzle, cells):
    """Flip one clue so it disagrees with the verified board."""
    counted_rows = [sum(1 for c in range(puzzle.cols) if cells[r][c] != EMPTY) for r in range(puzzle.rows)]
    counted_cols = [sum(1 for r in range(puzzle.rows) if cells[r][c] != EMPTY) for c in range(puzzle.cols)]

    candidates = []
    for axis, clues, counted, length in (("row", puzzle.row_clues, counted_rows, puzzle.cols),
                                         ("column", puzzle.col_clues, counted_cols, puzzle.rows)):
        for index, (clue, actual) in enumerate(zip(clues, counted)):
            if clue < length:
                candidates.append((axis, index, clue, actual, clue + 1))
            elif clue > 0:
                candidates.append((axis, index, clue, actual, clue - 1))
    if not candidates:
        return None

    # Alternate the axis across puzzles, then spread the offending line out.
    axis = "row" if stable_index(puzzle.puzzle_id, 2) == 0 else "column"
    on_axis = [item for item in candidates if item[0] == axis] or candidates
    axis, index, clue, actual, mutated = on_axis[stable_index(puzzle.puzzle_id + ":line", len(on_axis))]
    if actual == mutated:
        return None

    row_clues = list(puzzle.row_clues)
    col_clues = list(puzzle.col_clues)
    if axis == "row":
        row_clues[index] = mutated
    else:
        col_clues[index] = mutated

    word = "row" if axis == "row" else "column"
    explanation = (f"{word.title()} {index + 1} clue says {mutated}, but the board contains "
                   f"{actual} track(s) in that {word}.")
    record_puzzle = payload(puzzle, row_clues=row_clues, col_clues=col_clues)
    cells_evidence = [[index, c, cells[index][c]] for c in range(puzzle.cols)] if axis == "row" else \
                     [[r, index, cells[r][index]] for r in range(puzzle.rows)]
    return {
        "puzzle_id": puzzle.puzzle_id,
        "rule": "clue_mismatch",
        "input": {"puzzle": record_puzzle,
                  "board": board_text(puzzle, cells, row_clues=row_clues, col_clues=col_clues)},
        "action": {
            "kind": "violation",
            "cells": cells_evidence,
            "evidence": {"type": "clue", "axis": axis, "index": index,
                         "clue": mutated, "actual": actual},
            "explanation": explanation,
        },
        "target": f"clue_mismatch: {explanation}",
    }


def port_mismatch_examples(puzzle, cells):
    """Rotate one non-given interior piece and keep examples with only port failures."""
    givens = {(r, c) for r, c, _ in puzzle.givens}
    empty_neighbour = []
    track_neighbour = []
    for r in range(1, puzzle.rows - 1):
        for c in range(1, puzzle.cols - 1):
            if (r, c) in givens or cells[r][c] == EMPTY:
                continue
            original = cells[r][c]
            for piece in ROTATIONS[original]:
                mutated = [row.copy() for row in cells]
                mutated[r][c] = piece
                groups = rule_breakdown(puzzle, mutated)
                if groups["givens"] or groups["clue"] or groups["boundary"]:
                    continue
                dangling = dangling_edges(puzzle, mutated)
                if not dangling:
                    continue
                touched_empty = any(neighbour == EMPTY for _, _, _, _, _, neighbour in dangling)
                item = (r, c, original, piece, mutated, dangling)
                (empty_neighbour if touched_empty else track_neighbour).append(item)

    picks = []
    used_cells: set[tuple[int, int]] = set()
    for name, pool in (("empty", empty_neighbour), ("track", track_neighbour)):
        if not pool:
            continue
        offset = stable_index(puzzle.puzzle_id + ":" + name, len(pool))
        ordered = pool[offset:] + pool[:offset]
        pick = next((item for item in ordered if (item[0], item[1]) not in used_cells), None)
        if pick is not None:
            picks.append(pick)
            used_cells.add((pick[0], pick[1]))

    out = []
    for r, c, original, piece, mutated, dangling in picks:
        reasons = []
        cells_evidence = [[r, c, piece]]
        for sr, sc, direction, nr, nc, neighbour in dangling:
            if neighbour == EMPTY:
                reasons.append(f"r{sr + 1}c{sc + 1} points {DIR_NAME[direction]} into empty r{nr + 1}c{nc + 1}")
            else:
                reasons.append(f"r{sr + 1}c{sc + 1} points {DIR_NAME[direction]} at r{nr + 1}c{nc + 1} "
                               f"({neighbour}), which has no {DIR_NAME[OPPOSITE[direction]]} port")
            if [nr, nc, neighbour] not in cells_evidence:
                cells_evidence.append([nr, nc, neighbour])
        explanation = (f"r{r + 1}c{c + 1} is {piece} (was {original}), leaving "
                       f"{len(dangling)} unanswered port(s): " + "; ".join(reasons) + ".")
        out.append({
            "puzzle_id": puzzle.puzzle_id,
            "rule": "port_reciprocity",
            "input": {"puzzle": puzzle.payload(),
                      "board": board_text(puzzle, mutated)},
            "action": {
                "kind": "violation",
                "cells": cells_evidence,
                "evidence": {"type": "port", "cell": [r, c], "was": original, "now": piece,
                             "edges": [[sr, sc, direction] for sr, sc, direction, _, _, _ in dangling]},
                "explanation": explanation,
            },
            "target": f"port_reciprocity: {explanation}",
        })
    return out


def examples(puzzle):
    if not puzzle.reference_solution:
        return []
    if not verify(puzzle, puzzle.reference_solution).ok:
        raise ValueError(f"invalid reference solution: {puzzle.puzzle_id}")
    cells = reference_cells(puzzle)
    out = []
    clue = clue_mismatch_example(puzzle, cells)
    if clue:
        out.append(clue)
    out.extend(port_mismatch_examples(puzzle, cells))
    return out


def main():
    puzzle_path = Path("data/puzzles.jsonl")
    split_path = Path("data/splits.json")
    train = load_split(puzzle_path, split_path, "train")
    records = [record for puzzle in train for record in examples(puzzle)]
    dest = Path("data/sft_contradictions_train.jsonl")
    dest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    counts = Counter(record["rule"] for record in records)
    print(f"wrote {len(records)} contradiction examples from {len(train)} training puzzles to {dest}")
    print(f"  {dict(counts)}")


if __name__ == "__main__":
    main()
