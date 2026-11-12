"""Verifier tests: hand-built fixture, targeted mutants, and the 608-puzzle corpus gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tt_env import Puzzle, canonical_piece, verify

ROOT = Path(__file__).resolve().parent.parent

# A hand-built 4x4 puzzle. Path: (1,0) -> (1,1) -> (0,1) -> (0,2) -> (1,2) -> (1,3)
# with the track entering from the west and leaving to the east.
FIXTURE = Puzzle(
    puzzle_id="fixture-4x4",
    rows=4,
    cols=4,
    row_clues=(2, 4, 0, 0),
    col_clues=(1, 2, 2, 1),
    givens=((1, 0, "EW"), (1, 3, "EW")),
    start=(1, 0),
    end=(1, 3),
)

SOLVED = ["..SESW..", "EWNWNEEW", "........", "........"]


def cell(grid, r, c):
    return grid[r][2 * c : 2 * c + 2]


def with_cell(grid, r, c, piece):
    row = grid[r]
    return [*grid[:r], row[: 2 * c] + piece + row[2 * c + 2 :], *grid[r + 1 :]]


def test_fixture_is_accepted():
    verdict = verify(FIXTURE, SOLVED)
    assert verdict.ok, verdict.failures
    assert verdict.diagnostics["components"] == 1
    assert verdict.diagnostics["occupied"] == 6
    assert verdict.diagnostics["degree_histogram"] == {1: 2, 2: 4}


def test_canonical_pieces():
    assert canonical_piece("ES") == "SE"
    assert canonical_piece("WN") == "NW"
    assert canonical_piece("NS") == "NS"
    assert canonical_piece("NN") is None
    assert canonical_piece("NX") is None
    assert canonical_piece("N") is None


@pytest.mark.parametrize(
    ("grid", "expected"),
    [
        (SOLVED[:3], "shape"),
        (["..SESW.", *SOLVED[1:]], "shape"),
        (with_cell(SOLVED, 0, 1, "XX"), "shape"),
        (with_cell(SOLVED, 0, 1, ".."), "dangling"),          # hole in the path
        (with_cell(SOLVED, 1, 1, "NS"), "dangling"),          # rotated corner
        (with_cell(SOLVED, 1, 0, "NS"), "givens"),            # modified given
        (with_cell(SOLVED, 0, 0, "NW"), "boundary"),          # port off the edge
        (["........"] * 4, "clue_row"),                       # everything empty
        (["..SESW..", "EWNWNEEW", "SESW....", "NW......"], "disconnected"),  # extra 2x2 loop
    ],
)
def test_mutants_are_rejected(grid, expected):
    verdict = verify(FIXTURE, grid)
    assert not verdict.ok
    assert any(f.startswith(expected) for f in verdict.failures), verdict.failures


def test_second_valid_path_is_accepted():
    """The verifier grades rules, not a stored answer: any valid path must pass."""
    alternate = Puzzle(
        puzzle_id="fixture-alt",
        rows=4,
        cols=3,
        row_clues=(0, 3, 0, 0),
        col_clues=(1, 1, 1),
        givens=((1, 0, "EW"), (1, 2, "EW")),
        start=(1, 0),
        end=(1, 2),
    )
    grid = ["......", "EWEWEW", "......", "......"]
    assert verify(alternate, grid).ok


def test_clue_sums_must_match():
    """A puzzle whose clues disagree cannot be satisfied; the verifier must notice."""
    broken = Puzzle(
        puzzle_id="fixture-broken",
        rows=4,
        cols=4,
        row_clues=(2, 4, 0, 0),
        col_clues=(1, 2, 2, 5),
        givens=FIXTURE.givens,
        start=FIXTURE.start,
        end=FIXTURE.end,
    )
    verdict = verify(broken, SOLVED)
    assert not verdict.ok
    assert any(f.startswith("clue_col") for f in verdict.failures)


CORPUS = ROOT / "data" / "puzzles.jsonl"


@pytest.mark.skipif(not CORPUS.exists(), reason="run scripts/build_puzzles.py first")
def test_reference_solutions_verify():
    puzzles = [Puzzle.from_dict(json.loads(line)) for line in CORPUS.read_text("utf-8").splitlines() if line]
    assert len(puzzles) == 608, f"corpus should hold every puzzle, found {len(puzzles)}"
    bad = []
    for puzzle in puzzles:
        verdict = verify(puzzle, list(puzzle.reference_solution))
        if not verdict.ok:
            bad.append((puzzle.puzzle_id, verdict.failures[:3]))
    assert not bad, f"{len(bad)} reference solutions failed: {bad[:5]}"
