"""Report helpers: the grid renderer and the honest comparison metric."""

from __future__ import annotations

from tt_env import Puzzle
from tt_env.report import (
    compare,
    givens_cells,
    grid_cells,
    line_chart_svg,
    render_grid_svg,
    status_strip_svg,
)
from tt_env.rules import EMPTY

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


def test_grid_cells_packs_two_characters_per_cell():
    cells = grid_cells(["EW..NS", "..NSEW"], 2, 3)
    assert cells == [["EW", "..", "NS"], ["..", "NS", "EW"]]


def test_grid_cells_survives_a_short_grid():
    cells = grid_cells(["EW"], 2, 2)
    assert cells == [["EW", ".."], ["..", ".."]]


def test_givens_cells_is_only_the_givens():
    cells = givens_cells(FIXTURE)
    occupied = [(r, c) for r in range(4) for c in range(4) if cells[r][c] != EMPTY]
    assert occupied == [(1, 0), (1, 3)]


def test_compare_reports_accuracy_and_iou():
    actual = grid_cells(SOLVED, 4, 4)
    givens_only = givens_cells(FIXTURE)
    stats = compare(givens_only, actual)
    assert stats["occupied"] == 2
    assert stats["actual_occupied"] == 6
    assert stats["occupied_iou"] == 2 / 6
    assert stats["cell_accuracy"] == (16 - 4) / 16  # 4 cells of the path are missing

    perfect = compare(actual, actual)
    assert perfect["cell_accuracy"] == 1.0 and perfect["occupied_iou"] == 1.0


def test_render_grid_svg_contains_clues_and_pieces():
    svg = render_grid_svg(FIXTURE, givens_cells(FIXTURE), title="fixture")
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    assert "<title>fixture</title>" in svg
    assert ">1<" in svg and ">4<" in svg          # row clue 1, column clue 4
    assert svg.count('stroke="#1d4ed8"') == 4    # two EW givens, two ports each
    assert ">A<" in svg and ">B<" in svg


def test_render_grid_svg_marks_highlights():
    highlight = {(0, 0): "#fecaca"}
    svg = render_grid_svg(FIXTURE, givens_cells(FIXTURE), highlight=highlight)
    assert 'fill="#fecaca"' in svg


def test_line_chart_and_strip_render():
    chart = line_chart_svg([{"name": "total", "colour": "#000", "values": [0.5, 1.0, 0.75]}])
    assert "<polyline" in chart and "total" in chart
    strip = status_strip_svg(
        [
            {"step": 1, "puzzle_id": "a", "statuses": ["solved", "no-code"]},
            {"step": 2, "puzzle_id": "b", "statuses": ["error", "no-code"]},
        ]
    )
    assert strip.count("<rect") >= 2 + 8  # two steps + legend swatches
