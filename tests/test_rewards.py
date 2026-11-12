"""Reward tests: the plan's §6.5 table, with the ceiling pinned at exactly 31.5."""

from __future__ import annotations

import pytest

from tt_env import Puzzle, format_ok, grade, shaping
from tt_env.rewards import MAX_SHAPING

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

CEILING = 31.5


def program(grid: list[str]) -> str:
    rows = ",\n        ".join(repr(row) for row in grid)
    return f"```python\ndef solve(puzzle):\n    return [\n        {rows},\n    ]\n```"


def cell(grid, r, c):
    return grid[r][2 * c : 2 * c + 2]


def with_cell(grid, r, c, piece):
    row = grid[r]
    return [*grid[:r], row[: 2 * c] + piece + row[2 * c + 2 :], *grid[r + 1 :]]


def test_reference_solution_scores_the_ceiling():
    result = grade(program(SOLVED), FIXTURE)
    assert result.status == "solved", result.detail
    assert result.solved == 30.0
    assert result.total == pytest.approx(CEILING)
    assert result.shaping == 0.0, "a solve must never also collect shaping"


def test_no_code_block():
    result = grade("I solved it! The answer is a path.", FIXTURE)
    assert result.status == "no-code"
    assert result.format_ok == -0.5
    assert result.no_cheat == 1.0
    assert result.total == pytest.approx(0.5)
    assert "found 0" in result.detail, result.detail


def test_two_code_blocks_are_rejected_but_explained():
    completion = "```python\ndef solve(puzzle):\n    return []\n```\nand a test:\n```python\nassert True\n```"
    result = grade(completion, FIXTURE)
    assert result.status == "no-code"
    assert "found 2" in result.detail, result.detail


def test_chat_and_multimodal_completions_are_unwrapped():
    """TRL hands rewards messages, sometimes with content as parts; an empty parse looks like no-code."""
    code = "def solve(puzzle):\n    return ['....'] * puzzle['rows']\n"
    shapes = [
        [{"role": "assistant", "content": "```python\n" + code + "```"}],
        [{"role": "assistant", "content": [{"type": "text", "text": "```python\n" + code + "```"}]}],
        {"role": "assistant", "content": "```python\n" + code + "```"},
    ]
    for completion in shapes:
        assert format_ok(completion) == 0.5
        result = grade(completion, FIXTURE)
        assert result.status != "no-code", result.detail
        assert result.format_ok == 0.5


def test_syntax_error_costs_format_only():
    result = grade("```python\ndef solve(puzzle)\n    return []\n```", FIXTURE)
    assert result.format_ok == -0.5
    assert result.total == pytest.approx(0.5)


def test_wrong_signature_costs_format_only():
    result = grade("```python\ndef solve(board):\n    return []\n```", FIXTURE)
    assert result.format_ok == -0.5


def test_import_is_a_cheat():
    completion = "```python\ndef solve(puzzle):\n    import tracks\n    return []\n```"
    result = grade(completion, FIXTURE)
    assert result.status == "cheat"
    assert result.no_cheat == -20.0
    assert result.format_ok == 0.5, "it is still a usable-looking program"
    assert result.total == pytest.approx(-19.5)


def test_exception_is_a_small_penalty():
    completion = "```python\ndef solve(puzzle):\n    return 1 / 0\n```"
    result = grade(completion, FIXTURE)
    assert result.status == "error"
    assert result.solved == -0.5
    assert result.total == pytest.approx(1.0)


def test_timeout_is_a_small_penalty():
    completion = "```python\ndef solve(puzzle):\n    while True:\n        pass\n```"
    result = grade(completion, FIXTURE, max_steps=5_000)
    assert result.status == "timeout"
    assert result.solved == -0.5


def test_malformed_return_is_not_a_penalty():
    completion = "```python\ndef solve(puzzle):\n    return 'nope'\n```"
    result = grade(completion, FIXTURE)
    assert result.status == "error"
    assert result.solved == 0.0


def test_all_empty_grid_earns_no_shaping():
    empty = ["." * 8] * 4
    assert shaping(FIXTURE, empty) == 0.0
    result = grade(program(empty), FIXTURE)
    assert result.status == "not-solved"
    assert result.shaping == 0.0
    assert result.total == pytest.approx(1.5)  # format 0.5 + no_cheat 1.0


def test_givens_only_grid_earns_nothing():
    """Copying the givens must not pay: it was the first run's degenerate optimum."""
    givens_only = ["." * 8, "EW....EW", "." * 8, "." * 8]
    assert shaping(FIXTURE, givens_only) == 0.0
    result = grade(program(givens_only), FIXTURE)
    assert result.solved == 0.0
    assert result.shaping == 0.0
    assert result.total == pytest.approx(1.5)  # format 0.5 + no_cheat 1.0


def test_one_extra_cell_beyond_the_givens_earns_shaping():
    one_more = ["........", "EWNS..EW", "........", "........"]  # givens + one NS at (1,1)
    value = shaping(FIXTURE, one_more)
    assert 0 < value <= MAX_SHAPING


def test_mutating_a_given_zeroes_shaping():
    mutated = with_cell(SOLVED, 1, 0, "NS")
    assert shaping(FIXTURE, mutated) == 0.0


def test_two_components_earn_shaping_but_never_a_solve():
    """A 2x2 loop plus the real path: it is never +30, whatever the shaping does."""
    grid = list(SOLVED)
    grid[2] = "SESW...."
    grid[3] = "NW......"
    result = grade(program(grid), FIXTURE)
    assert result.solved == 0.0
    assert result.status == "not-solved"
    assert 0 <= result.shaping <= 1.5


def test_shaping_is_capped():
    assert shaping(FIXTURE, SOLVED) <= 1.5


def test_reference_grid_round_trip_through_payload():
    """The payload must carry everything a solver needs: dims, clues, givens and endpoint ports."""
    payload = FIXTURE.payload()
    assert payload["start"] == [1, 0, "W"]
    assert payload["end"] == [1, 3, "E"]
    assert payload["givens"] == [[1, 0, "EW"], [1, 3, "EW"]]
