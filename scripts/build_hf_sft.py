#!/usr/bin/env python
"""Build the combined Hugging Face SFT dataset: deductions + contradictions.

Output layout (ready to upload as a dataset repo):

    hf_export/train-tracks-sft/
        README.md
        data/train.parquet
        data/validation.parquet
        data/test_small.parquet
        data/test_medium.parquet

One row per example. `images` holds one rendered PNG of the input board, so the
image never contains the answer. Coordinates in the exported rows are 1-based,
matching the existing train-tracks-decisions dataset.
"""

from __future__ import annotations

import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset, Features, Image as HFImage, List, Value
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_sft_contradictions import examples as contradiction_examples  # noqa: E402
from build_sft_deductions import examples as deduction_examples  # noqa: E402
from tt_env.puzzle import load_jsonl  # noqa: E402
from tt_env.rl import load_split  # noqa: E402
from tt_env.rules import OFFSETS  # noqa: E402

SPLIT_NAMES = ("train", "validation", "test_small", "test_medium")
DEST = ROOT / "hf_export" / "train-tracks-sft"
IMAGE_SIZE = 512
COORDINATES = ("Rows top to bottom; columns left to right; both start at 1. "
               "Track shapes name connected sides N/E/S/W.")

CRITERIA = {
    "clue_satisfied": "A row/column clue is already met, so every remaining unknown cell in that line is empty.",
    "all_remaining_cells_required": (
        "A row/column clue needs as many tracks as there are unknown cells, so all of them are tracks "
        "(the orientation may stay open)."
    ),
    "port_neighbour": "A known port forces the adjacent cell to carry the reciprocal port.",
    "clue_mismatch": "A row/column clue disagrees with the number of tracks in that line.",
    "port_reciprocity": "A port points at a neighbour that does not answer it, or at an empty cell.",
}

STATE_TASK = {
    "deduction": (
        "Current board. ?? = unknown, XX = confirmed empty, ? = track with unknown orientation, "
        "N? = track with a known north port, NE/NS/... = complete piece. Apply the single forced deduction."
    ),
    "contradiction": (
        "Completed board. Exactly one local rule is broken: a clue count or port reciprocity. "
        "Find the offending cell or clue."
    ),
}

INSTRUCTIONS = {
    "deduction": "Apply the single forced deduction. Return the updated board in the same format.",
    "contradiction": "Exactly one local rule is broken. Name the rule and point at the offending cell or clue.",
}

FEATURES = Features({
    "id": Value("string"),
    "puzzle_id": Value("string"),
    "task": Value("string"),
    "step_index": Value("int64"),
    "images": List(HFImage()),
    "state": {
        "rows": Value("int64"),
        "columns": Value("int64"),
        "coordinates": Value("string"),
        "task": Value("string"),
        "board": Value("string"),
        "row_clues": List(Value("int64")),
        "col_clues": List(Value("int64")),
        "givens": List({
            "row": Value("int64"),
            "column": Value("int64"),
            "value": Value("string"),
        }),
        "start": {"row": Value("int64"), "column": Value("int64"), "port": Value("string")},
        "end": {"row": Value("int64"), "column": Value("int64"), "port": Value("string")},
    },
    "question": {
        "type": Value("string"),
        "instructions": Value("string"),
        "criteria": {name: Value("string") for name in CRITERIA},
    },
    "label": Value("string"),
    "answer": Value("string"),
    "action": {
        "kind": Value("string"),
        "cells": List({
            "row": Value("int64"),
            "column": Value("int64"),
            "value": Value("string"),
        }),
        "evidence": Value("string"),
        "explanation": Value("string"),
    },
    "difficulty": Value("string"),
    "rows": Value("int64"),
    "columns": Value("int64"),
    "source_date": Value("string"),
    "split": Value("string"),
})


# ---------------------------------------------------------------------------
# Board rendering
# ---------------------------------------------------------------------------

def _font(size: int, bold: bool = False):
    names = (("segoeuib.ttf", "arialbd.ttf", "calibrib.ttf") if bold
             else ("segoeui.ttf", "arial.ttf", "calibri.ttf"))
    for name in names:
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                pass
    return ImageFont.load_default()


def _text_center(draw, xy, text: str, font, fill: str) -> None:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    draw.text((xy[0] - (right - left) / 2 - left, xy[1] - (bottom - top) / 2 - top),
              text, font=font, fill=fill)


def _text_right(draw, xy, text: str, font, fill: str) -> None:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    draw.text((xy[0] - (right - left) - left, xy[1] - (bottom - top) / 2 - top),
              text, font=font, fill=fill)


def _text_left(draw, xy, text: str, font, fill: str) -> None:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    draw.text((xy[0] - left, xy[1] - (bottom - top) / 2 - top), text, font=font, fill=fill)


@dataclass(frozen=True)
class BoardSpec:
    rows: int
    cols: int
    row_clues: tuple[int, ...]
    col_clues: tuple[int, ...]
    givens: tuple[tuple[int, int, str], ...]
    start: tuple
    end: tuple


def render_board(spec: BoardSpec, cells: list[list[str]], size: int = IMAGE_SIZE) -> PILImage.Image:
    """Draw one board: blue givens, black other tracks, red X, orange ?, clue numbers."""
    left, right, top, bottom = 30, 42, 42, 30
    avail_w, avail_h = size - left - right, size - top - bottom
    cell = min(avail_w / spec.cols, avail_h / spec.rows)
    x0 = left + (avail_w - spec.cols * cell) / 2
    y0 = top + (avail_h - spec.rows * cell) / 2

    image = PILImage.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)

    grid = "#cbd5e1"
    for r in range(spec.rows + 1):
        y = y0 + r * cell
        draw.line([(x0, y), (x0 + spec.cols * cell, y)], fill=grid, width=1)
    for c in range(spec.cols + 1):
        x = x0 + c * cell
        draw.line([(x, y0), (x, y0 + spec.rows * cell)], fill=grid, width=1)

    clue_font = _font(max(10, int(cell * 0.28)), bold=True)
    label_font = _font(max(9, int(cell * 0.22)), bold=False)
    for c, clue in enumerate(spec.col_clues):
        _text_center(draw, (x0 + (c + 0.5) * cell, y0 - 7), str(clue), clue_font, "#17202a")
    for r, clue in enumerate(spec.row_clues):
        _text_left(draw, (x0 + spec.cols * cell + 8, y0 + (r + 0.5) * cell),
                   str(clue), clue_font, "#17202a")
    for r in range(spec.rows):
        _text_right(draw, (x0 - 8, y0 + (r + 0.5) * cell), f"r{r + 1}", label_font, "#52616b")
    for c in range(spec.cols):
        _text_center(draw, (x0 + (c + 0.5) * cell, y0 + spec.rows * cell + 14),
                     f"c{c + 1}", label_font, "#52616b")

    given_map = {(r, c): piece for r, c, piece in spec.givens}
    for r in range(spec.rows):
        for c in range(spec.cols):
            token = cells[r][c]
            if token in ("??", "..", "", None):
                continue
            cx, cy = x0 + (c + 0.5) * cell, y0 + (r + 0.5) * cell
            if token == "XX":
                inset = cell * 0.27
                width = max(2, int(cell * 0.08))
                draw.line([(cx - inset, cy - inset), (cx + inset, cy + inset)], fill="#dc2626", width=width)
                draw.line([(cx - inset, cy + inset), (cx + inset, cy - inset)], fill="#dc2626", width=width)
                continue
            if token == "?":
                _text_center(draw, (cx, cy), "?", _font(max(12, int(cell * 0.4)), bold=True), "#ea580c")
                continue

            known = [ch for ch in token if ch in "NESW"]
            colour = "#1d4ed8" if (r, c) in given_map else "#334155"
            width = max(2, int(cell * 0.07))
            for direction in known:
                dr, dc = OFFSETS[direction]
                draw.line([(cx, cy), (cx + dc * cell / 2, cy + dr * cell / 2)], fill=colour, width=width)
            if "?" in token:
                _text_center(draw, (cx + cell * 0.22, cy - cell * 0.22), "?",
                             _font(max(10, int(cell * 0.3)), bold=True), "#ea580c")

    for label, cellpos in (("A", spec.start), ("B", spec.end)):
        r, c = cellpos[0], cellpos[1]
        cx, cy = x0 + (c + 0.5) * cell, y0 + (r + 0.5) * cell
        radius = max(6, int(cell * 0.16))
        bx, by = x0 + c * cell + radius + 2, y0 + r * cell + radius + 2
        draw.ellipse([(bx - radius, by - radius), (bx + radius, by + radius)],
                     fill="white", outline="#334155", width=1)
        _text_center(draw, (bx, by), label, _font(max(8, int(cell * 0.2)), bold=True), "#334155")
        if len(cellpos) > 2 and cellpos[2]:
            dr, dc = OFFSETS[cellpos[2]]
            draw.line([(cx, cy), (cx + dc * cell * 0.72, cy + dr * cell * 0.72)],
                      fill="#334155", width=1)

    return image


def evidence_1based(evidence: dict) -> dict:
    """Rewrite the source-native 0-based evidence coordinates to 1-based."""
    out = dict(evidence)
    if out.get("type") == "clue":
        out["index"] = out["index"] + 1
    elif out.get("type") == "port":
        out["cell"] = [out["cell"][0] + 1, out["cell"][1] + 1]
        out["edges"] = [[r + 1, c + 1, d] for r, c, d in out["edges"]]
    return out


def parse_board(text: str, rows: int, cols: int) -> list[list[str]]:
    lines = text.splitlines()[1:]
    if len(lines) != rows:
        raise ValueError(f"board has {len(lines)} rows, expected {rows}")
    cells = []
    for line in lines:
        tokens = line.split()[1:]
        if len(tokens) < cols:
            raise ValueError(f"short board row: {line!r}")
        cells.append(tokens[-cols:])
    return cells


# ---------------------------------------------------------------------------
# Row construction
# ---------------------------------------------------------------------------

def make_row(puzzle, record: dict, task: str, step_index: int, split: str) -> dict:
    payload = record["input"]["puzzle"]
    rows, cols = payload["rows"], payload["cols"]
    cells = parse_board(record["input"]["board"], rows, cols)
    spec = BoardSpec(
        rows=rows,
        cols=cols,
        row_clues=tuple(payload["row_clues"]),
        col_clues=tuple(payload["col_clues"]),
        givens=tuple((r, c, p) for r, c, p in payload["givens"]),
        start=tuple(payload["start"]),
        end=tuple(payload["end"]),
    )
    action = record["action"]
    return {
        "id": f"{puzzle.puzzle_id}-{task[:3]}-{step_index:04d}",
        "puzzle_id": puzzle.puzzle_id,
        "task": task,
        "step_index": step_index,
        "images": [render_board(spec, cells)],
        "state": {
            "rows": rows,
            "columns": cols,
            "coordinates": COORDINATES,
            "task": STATE_TASK[task],
            "board": record["input"]["board"],
            "row_clues": list(payload["row_clues"]),
            "col_clues": list(payload["col_clues"]),
            "givens": [{"row": r + 1, "column": c + 1, "value": p}
                       for r, c, p in payload["givens"]],
            "start": {"row": payload["start"][0] + 1, "column": payload["start"][1] + 1,
                      "port": payload["start"][2]},
            "end": {"row": payload["end"][0] + 1, "column": payload["end"][1] + 1,
                    "port": payload["end"][2]},
        },
        "question": {
            "type": task,
            "instructions": INSTRUCTIONS[task],
            "criteria": dict(CRITERIA),
        },
        "label": record["rule"],
        "answer": record["target_board"] if task == "deduction" else record["target"],
        "action": {
            "kind": action["kind"],
            "cells": [{"row": r + 1, "column": c + 1,
                       "value": "XX" if v == "X" else v} for r, c, v in action["cells"]],
            "evidence": (json.dumps(evidence_1based(action["evidence"]), ensure_ascii=False)
                         if task == "contradiction" else ""),
            "explanation": action["explanation"],
        },
        "difficulty": puzzle.difficulty or ("medium" if puzzle.puzzle_id.endswith("-medium") else "small"),
        "rows": rows,
        "columns": cols,
        "source_date": puzzle.date,
        "split": split,
    }


def iter_split(split: str):
    puzzles = load_split(ROOT / "data" / "puzzles.jsonl", ROOT / "data" / "splits.json", split)
    for puzzle in puzzles:
        for index, record in enumerate(deduction_examples(puzzle)):
            yield make_row(puzzle, record, "deduction", index, split)
        for index, record in enumerate(contradiction_examples(puzzle)):
            yield make_row(puzzle, record, "contradiction", index, split)


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------

README = """---
language:
- en
tags:
- train-tracks
- supervised-fine-tuning
- reasoning
- multimodal
- deduction
- contradiction-detection
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.parquet
  - split: validation
    path: data/validation.parquet
  - split: test_small
    path: data/test_small.parquet
  - split: test_medium
    path: data/test_medium.parquet
---
# Train Tracks SFT: deductions and contradictions

{total:,} supervised examples from {puzzles} Train Tracks puzzles, covering two tasks on the same boards:

- **deduction** — apply one forced deduction to a partial board
- **contradiction** — find the single local rule broken by a completed board

Each row contains one rendered board image, the board as text, a nested `state`, a nested `question`, the rule `label`, and the expected `answer`.

| Split | Puzzles | Deductions | Contradictions | Total |
|---|---:|---:|---:|---:|
| train | 212 small | {train_ded:,} | {train_con:,} | {train_total:,} |
| validation | 46 small | {val_ded:,} | {val_con:,} | {val_total:,} |
| test_small | 46 small | {test_ded:,} | {test_con:,} | {test_total:,} |
| test_medium | 304 medium | {med_ded:,} | {med_con:,} | {med_total:,} |
| **total** | **608** | **{ded_total:,}** | **{con_total:,}** | **{total:,}** |

## Tasks

**Deduction** (`question.type = "deduction"`). The board is partial: `??` unknown, `XX` confirmed empty, `?` a track whose orientation is open, `N?`/`E?`/`S?`/`W?` a track with one known port, and two-letter tokens complete pieces. Apply the single deduction forced by the row/column clues or by a known port. `label` is `clue_satisfied`, `all_remaining_cells_required`, or `port_neighbour`; `answer` is the full updated board.

**Contradiction** (`question.type = "contradiction"`). The board is a completed reference solution with exactly one local rule broken. `label` is `clue_mismatch` (one clue disagrees with the number of tracks in its line) or `port_reciprocity` (one rotated piece leaves a port unanswered). `answer` names the rule and the offender. `action.cells` lists the offending cell(s), and `action.evidence` is a JSON string with the exact clue/edge evidence.

## Usage

```python
from datasets import load_dataset
data = load_dataset("zuhairsan/train-tracks-sft")
row = data["train"][0]
image = row["images"][0]
state = row["state"]
question = row["question"]
label = row["label"]
answer = row["answer"]
```

PNG bytes are embedded in Parquet. `images` is a list with one Hugging Face Image. `state` and `question` are nested structures. Coordinates in `state.givens`, `state.start`, `state.end`, `action.cells`, and the JSON in `action.evidence` are **one-based**, matching `train-tracks-decisions`. `action.cells[].value` is the board token written to that cell (`XX`, `?`, `N?`, `NE`, ...).

## Schema

| Field | Meaning |
|---|---|
| `id` | stable row id, e.g. `01022026-small-ded-0000` |
| `puzzle_id` | source puzzle id |
| `task` | `deduction` or `contradiction` |
| `step_index` | index within the puzzle/task record list |
| `images` | one rendered PNG of the input board |
| `state` | rows, columns, coordinates, board text, clues, givens, start/end |
| `question` | task type, instructions, and rule descriptions |
| `label` | the rule that applies or is violated |
| `answer` | updated board (deduction) or violation description (contradiction) |
| `action` | kind, changed/offending cells, evidence JSON, explanation |
| `difficulty` | source category (`small`/`medium`) |
| `rows`, `columns` | grid size |
| `source_date` | ISO date of the source puzzle |
| `split` | split name |

## Rendering

The image is the **input only**, so it never contains the answer. Given pieces are blue, other tracks are black, `XX` is a red X, `?` is an orange question mark, and partial tokens show their known port plus an orange `?`. Row clues are on the right; `r1…` labels are on the left and `c1…` labels are on the bottom. Clue numbers are bold.

## Provenance and audit

Source puzzles: https://puzzlemadness.co.uk/traintracks/ from January through October 2026. All reference solutions were independently checked; a second-solution MILP audit found each unique relative to the extracted constraints. Frozen puzzle-level splits use seed 20261110. The test splits contain labels for scoring: never use them for training or prompt selection.

This card does not assert ownership or grant a new license over the source puzzles/artwork.
"""


def main() -> int:
    (DEST / "data").mkdir(parents=True, exist_ok=True)
    counts: dict[str, dict[str, int]] = {}
    for split in SPLIT_NAMES:
        dataset = Dataset.from_generator(
            lambda split=split: iter_split(split),
            features=FEATURES,
        )
        path = DEST / "data" / f"{split}.parquet"
        dataset.to_parquet(str(path))
        per_task: dict[str, int] = {}
        for task in ("deduction", "contradiction"):
            per_task[task] = sum(1 for value in dataset["task"] if value == task)
        counts[split] = per_task
        print(f"{split}: {len(dataset):,} rows -> {path} ({path.stat().st_size / 1e6:.1f} MB)")

    ded_total = sum(v["deduction"] for v in counts.values())
    con_total = sum(v["contradiction"] for v in counts.values())
    total = ded_total + con_total
    readme = README.format(
        total=total,
        puzzles=608,
        train_ded=counts["train"]["deduction"], train_con=counts["train"]["contradiction"],
        train_total=sum(counts["train"].values()),
        val_ded=counts["validation"]["deduction"], val_con=counts["validation"]["contradiction"],
        val_total=sum(counts["validation"].values()),
        test_ded=counts["test_small"]["deduction"], test_con=counts["test_small"]["contradiction"],
        test_total=sum(counts["test_small"].values()),
        med_ded=counts["test_medium"]["deduction"], med_con=counts["test_medium"]["contradiction"],
        med_total=sum(counts["test_medium"].values()),
        ded_total=ded_total, con_total=con_total,
    )
    (DEST / "README.md").write_text(readme, encoding="utf-8")
    print(f"wrote {DEST / 'README.md'}")
    print(f"total: {total:,} rows ({ded_total:,} deductions + {con_total:,} contradictions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
