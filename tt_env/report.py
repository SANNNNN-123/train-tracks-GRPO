"""Render Train Tracks grids and simple charts as inline SVG (no plotting dependency).

Used by `scripts/make_run_report.py`. Everything is returned as an SVG string so the report is a
single self-contained HTML file that opens offline.
"""

from __future__ import annotations

from .puzzle import Puzzle
from .rules import EMPTY, OFFSETS, dirs_of, outward_port

GIVEN = "#1d4ed8"
MODEL = "#ea580c"
SOLUTION = "#15803d"
MISMATCH = "#fecaca"
AGREE = "#dcfce7"
GRID_LINE = "#cbd5e1"
TEXT = "#334155"

STATUS_COLOURS = {
    "solved": "#15803d",
    "not-solved": "#f59e0b",
    "no-code": "#64748b",
    "invalid": "#7c3aed",
    "syntax": "#4b5563",
    "error": "#dc2626",
    "timeout": "#b91c1c",
    "cheat": "#111827",
}


def grid_cells(grid, rows: int, cols: int) -> list[list[str]]:
    """Packed rows -> cells; short rows are padded so a malformed grid still renders."""
    cells = []
    for r in range(rows):
        row = grid[r] if r < len(grid) else ""
        cells.append([(row[2 * c : 2 * c + 2] or EMPTY)[:2].ljust(2, ".") for c in range(cols)])
    return cells


def givens_cells(puzzle: Puzzle) -> list[list[str]]:
    cells = [[EMPTY] * puzzle.cols for _ in range(puzzle.rows)]
    for r, c, piece in puzzle.givens:
        cells[r][c] = piece
    return cells


def occupied(cells: list[list[str]]) -> set[tuple[int, int]]:
    return {(r, c) for r, row in enumerate(cells) for c, piece in enumerate(row) if piece != EMPTY}


def compare(predicted: list[list[str]], actual: list[list[str]]) -> dict:
    """Cell accuracy (empties count) and occupancy IoU - the second is the honest one."""
    rows, cols = len(actual), len(actual[0])
    matches = sum(1 for r in range(rows) for c in range(cols) if predicted[r][c] == actual[r][c])
    pred_set, act_set = occupied(predicted), occupied(actual)
    union = pred_set | act_set
    return {
        "cell_accuracy": matches / (rows * cols),
        "occupied_iou": (len(pred_set & act_set) / len(union)) if union else 1.0,
        "occupied": len(pred_set),
        "actual_occupied": len(act_set),
        "matches": matches,
        "cells": rows * cols,
    }


def _piece_paths(r: int, c: int, piece: str, x0: float, y0: float, cell: float) -> list[tuple[float, float, float, float]]:
    cx, cy = x0 + (c + 0.5) * cell, y0 + (r + 0.5) * cell
    segments = []
    for d in dirs_of(piece):
        dr, dc = OFFSETS[d]
        segments.append((cx, cy, cx + dc * cell / 2, cy + dr * cell / 2))
    return segments


def render_grid_svg(
    puzzle: Puzzle,
    cells: list[list[str]] | None = None,
    *,
    cell: float = 30,
    base_colour: str = MODEL,
    highlight: dict[tuple[int, int], str] | None = None,
    title: str = "",
    show_clues: bool = True,
    row_clues_right: bool = False,
    bold_clues: bool = False,
    row_labels: bool = False,
    col_labels: bool = False,
) -> str:
    """Draw one grid: clues, given pieces in blue, other pieces in `base_colour`.

    `row_labels` / `col_labels` add non-bold r1..rN / c1..cN indices on the left
    and bottom. They live in extra viewBox padding, so grid coordinates do not
    move.
    """
    cells = cells or givens_cells(puzzle)
    highlight = highlight or {}
    left = 8 if row_clues_right or not show_clues else 30
    top = 24 if show_clues else 8
    right = 30 if row_clues_right and show_clues else 8
    width = left + puzzle.cols * cell + right
    height = top + puzzle.rows * cell + 8
    pad_left = 26 if row_labels else 0
    pad_bottom = 18 if col_labels else 0
    view_width = width + pad_left
    view_height = height + pad_bottom
    parts = [f'<svg viewBox="{-pad_left} 0 {view_width:.0f} {view_height:.0f}" width="{view_width:.0f}" height="{view_height:.0f}" role="img">']
    if title:
        parts.append(f"<title>{title}</title>")

    # Cell backgrounds (diff highlights) and the grid lines.
    for r in range(puzzle.rows):
        for c in range(puzzle.cols):
            if (r, c) in highlight:
                parts.append(
                    f'<rect x="{left + c * cell:.1f}" y="{top + r * cell:.1f}" width="{cell:.1f}" height="{cell:.1f}" fill="{highlight[(r, c)]}"/>'
                )
    for r in range(puzzle.rows + 1):
        y = top + r * cell
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + puzzle.cols * cell:.1f}" y2="{y:.1f}" stroke="{GRID_LINE}" stroke-width="1"/>')
    for c in range(puzzle.cols + 1):
        x = left + c * cell
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + puzzle.rows * cell:.1f}" stroke="{GRID_LINE}" stroke-width="1"/>')

    if show_clues:
        weight = ' font-weight="700"' if bold_clues else ""
        for c, clue in enumerate(puzzle.col_clues):
            x = left + (c + 0.5) * cell
            parts.append(f'<text x="{x:.1f}" y="{top - 7}" text-anchor="middle" font-size="11"{weight} fill="{TEXT}">{clue}</text>')
        for r, clue in enumerate(puzzle.row_clues):
            y = top + (r + 0.5) * cell + 4
            if row_clues_right:
                parts.append(f'<text x="{left + puzzle.cols * cell + 6}" y="{y:.1f}" text-anchor="start" font-size="11"{weight} fill="{TEXT}">{clue}</text>')
            else:
                parts.append(f'<text x="{left - 6}" y="{y:.1f}" text-anchor="end" font-size="11"{weight} fill="{TEXT}">{clue}</text>')

    if row_labels:
        for r in range(puzzle.rows):
            y = top + (r + 0.5) * cell + 4
            parts.append(f'<text x="{left - 6:.1f}" y="{y:.1f}" text-anchor="end" font-size="11" fill="{TEXT}">r{r + 1}</text>')
    if col_labels:
        bottom = top + puzzle.rows * cell
        for c in range(puzzle.cols):
            x = left + (c + 0.5) * cell
            parts.append(f'<text x="{x:.1f}" y="{bottom + 14:.1f}" text-anchor="middle" font-size="11" fill="{TEXT}">c{c + 1}</text>')

    givens = {(r, c): piece for r, c, piece in puzzle.givens}
    for r in range(puzzle.rows):
        for c in range(puzzle.cols):
            piece = cells[r][c]
            if piece == EMPTY or piece is None:
                continue
            colour = GIVEN if (r, c) in givens else base_colour
            for x1, y1, x2, y2 in _piece_paths(r, c, piece, left, top, cell):
                parts.append(
                    f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{colour}" stroke-width="{3 if (r, c) in givens else 2.5}" stroke-linecap="round"/>'
                )

    # Endpoints: A/B plus the outward port arrow.
    for label, (r, c) in (("A", puzzle.start), ("B", puzzle.end)):
        cx, cy = left + (c + 0.5) * cell, top + (r + 0.5) * cell
        parts.append(f'<circle cx="{cx - cell / 2 + 5:.1f}" cy="{cy - cell / 2 + 5:.1f}" r="7" fill="#fff" stroke="{TEXT}" stroke-width="1"/>')
        parts.append(f'<text x="{cx - cell / 2 + 5:.1f}" y="{cy - cell / 2 + 8:.1f}" text-anchor="middle" font-size="9" fill="{TEXT}">{label}</text>')
        piece = givens.get((r, c))
        port = outward_port(piece, r, c, puzzle.rows, puzzle.cols) if piece else None
        if port:
            dr, dc = OFFSETS[port]
            parts.append(
                f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{cx + dc * cell * 0.72:.1f}" y2="{cy + dr * cell * 0.72:.1f}" stroke="{TEXT}" stroke-width="1.2" stroke-dasharray="3 3"/>'
            )
    parts.append("</svg>")
    return "".join(parts)


def line_chart_svg(
    series: list[dict],
    *,
    width: int = 760,
    height: int = 240,
    x_label: str = "step",
    y_label: str = "",
    y_min: float | None = None,
    y_max: float | None = None,
) -> str:
    """Minimal multi-series line chart. `series = [{"name", "colour", "values"}]`."""
    pad_l, pad_r, pad_t, pad_b = 56, 16, 18, 34
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    all_values = [v for s in series for v in s["values"] if v is not None]
    if not all_values:
        return "<svg></svg>"
    lo = y_min if y_min is not None else min(all_values)
    hi = y_max if y_max is not None else max(all_values)
    if hi - lo < 1e-9:
        hi = lo + 1
    n = max(len(s["values"]) for s in series)

    def xy(i: int, v: float) -> tuple[float, float]:
        x = pad_l + (i / max(1, n - 1)) * plot_w
        y = pad_t + (1 - (v - lo) / (hi - lo)) * plot_h
        return x, y

    parts = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img">']
    for k in range(5):
        value = lo + (hi - lo) * k / 4
        y = pad_t + (1 - k / 4) * plot_h
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" y2="{y:.1f}" stroke="#e2e8f0"/>')
        parts.append(f'<text x="{pad_l - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="10" fill="{TEXT}">{value:.2f}</text>')
    for k in range(5):
        step = 1 + round((n - 1) * k / 4)
        x = pad_l + (k / 4) * plot_w
        parts.append(f'<text x="{x:.1f}" y="{height - 12}" text-anchor="middle" font-size="10" fill="{TEXT}">{step}</text>')
    for s in series:
        points = " ".join(
            f"{xy(i, v)[0]:.1f},{xy(i, v)[1]:.1f}" for i, v in enumerate(s["values"]) if v is not None
        )
        parts.append(f'<polyline points="{points}" fill="none" stroke="{s["colour"]}" stroke-width="1.8"/>')
    parts.append(f'<text x="{pad_l + plot_w / 2}" y="{height - 1}" text-anchor="middle" font-size="10" fill="{TEXT}">{x_label}</text>')
    legend_x = pad_l
    for s in series:
        parts.append(f'<line x1="{legend_x}" y1="8" x2="{legend_x + 18}" y2="8" stroke="{s["colour"]}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x + 23}" y="12" font-size="10" fill="{TEXT}">{s["name"]}</text>')
        legend_x += 34 + 6.2 * len(s["name"])
    parts.append("</svg>")
    return "".join(parts)


def status_strip_svg(records: list[dict], *, width: int = 760, height: int = 46) -> str:
    """One thin column per step, coloured by the worse of its two statuses."""
    order = ["cheat", "solved", "not-solved", "error", "timeout", "no-code", "invalid", "syntax"]
    slot = width / max(1, len(records))
    parts = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img">']
    for i, record in enumerate(records):
        statuses = record["statuses"]
        worst = min(statuses, key=lambda s: order.index(s) if s in order else len(order))
        colour = STATUS_COLOURS.get(worst, "#94a3b8")
        x = i * slot
        parts.append(
            f'<rect x="{x:.2f}" y="18" width="{max(1.0, slot + 0.2):.2f}" height="20" fill="{colour}">'
            f"<title>step {record['step']} {record['puzzle_id']}: {', '.join(statuses)}</title></rect>"
        )
    legend_x = 0
    for name, colour in STATUS_COLOURS.items():
        parts.append(f'<rect x="{legend_x}" y="2" width="9" height="9" fill="{colour}"/>')
        parts.append(f'<text x="{legend_x + 12}" y="10" font-size="9" fill="{TEXT}">{name}</text>')
        legend_x += 30 + 5.5 * len(name)
    parts.append("</svg>")
    return "".join(parts)


__all__ = [
    "AGREE",
    "GIVEN",
    "MISMATCH",
    "MODEL",
    "SOLUTION",
    "STATUS_COLOURS",
    "compare",
    "givens_cells",
    "grid_cells",
    "line_chart_svg",
    "occupied",
    "render_grid_svg",
    "status_strip_svg",
]
