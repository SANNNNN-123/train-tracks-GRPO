#!/usr/bin/env python
"""Build a local, Hugging-Face-style HTML preview of the combined SFT dataset.

Output:

    outputs/hf_preview/
        index.html
        data/meta.js
        data/card.js
        data/files.js
        data/train.js, validation.js, test_small.js, test_medium.js
        images/<split>/00000.png ...

Open `index.html` directly in a browser. Data and images are loaded from
relative paths, so no web server is needed. Each split's rows are loaded on
demand with a plain <script> tag (which works from file://, unlike fetch()).

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\make_hf_preview.py
"""

from __future__ import annotations

import html
import json
import re
import shutil
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "hf_export" / "train-tracks-sft"
DEST = ROOT / "outputs" / "hf_preview"
DATASET = "zuhairsan/train-tracks-sft"
SPLITS = ("train", "validation", "test_small", "test_medium")
SPLIT_PUZZLES = {"train": 212, "validation": 46, "test_small": 46, "test_medium": 304}
TAGS = [
    "train-tracks",
    "supervised-fine-tuning",
    "reasoning",
    "multimodal",
    "deduction",
    "contradiction-detection",
]


# ---------------------------------------------------------------------------
# Minimal Markdown -> HTML for the dataset card
# ---------------------------------------------------------------------------

def _inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        r'<a href="\2" target="_blank" rel="noreferrer">\1</a>',
        text,
    )
    return text


def _table(lines: list[str]) -> str:
    rows = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines]
    if len(rows) >= 2 and all(re.match(r"^:?-{2,}:?$", cell) for cell in rows[1]):
        head = "".join(f"<th>{_inline(cell)}</th>" for cell in rows[0])
        body = "".join(
            "<tr>" + "".join(f"<td>{_inline(cell)}</td>" for cell in row) + "</tr>"
            for row in rows[2:]
        )
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    body = "".join(
        "<tr>" + "".join(f"<td>{_inline(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<table><tbody>{body}</tbody></table>"


def md_to_html(md: str) -> str:
    lines = md.splitlines()
    if lines and lines[0].strip() == "---":
        for index in range(1, len(lines)):
            if lines[index].strip() == "---":
                lines = lines[index + 1:]
                break

    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]

        if line.startswith("```"):
            i += 1
            code: list[str] = []
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1
            out.append(f"<pre><code>{html.escape(chr(10).join(code))}</code></pre>")
            continue

        if not line.strip():
            i += 1
            continue

        match = re.match(r"^(#{1,6})\s+(.*)$", line)
        if match:
            level = len(match.group(1))
            out.append(f"<h{level}>{_inline(match.group(2))}</h{level}>")
            i += 1
            continue

        if line.lstrip().startswith("|"):
            table: list[str] = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                table.append(lines[i])
                i += 1
            out.append(_table(table))
            continue

        if re.match(r"^\s*[-*]\s+", line):
            items: list[str] = []
            while i < len(lines) and re.match(r"^\s*[-*]\s+", lines[i]):
                items.append(re.sub(r"^\s*[-*]\s+", "", lines[i]))
                i += 1
            out.append("<ul>" + "".join(f"<li>{_inline(item)}</li>" for item in items) + "</ul>")
            continue

        paragraph: list[str] = []
        while (
            i < len(lines)
            and lines[i].strip()
            and not lines[i].startswith("```")
            and not re.match(r"^(#{1,6})\s+", lines[i])
            and not lines[i].lstrip().startswith("|")
            and not re.match(r"^\s*[-*]\s+", lines[i])
        ):
            paragraph.append(lines[i].strip())
            i += 1
        out.append("<p>" + _inline(" ".join(paragraph)) + "</p>")

    return "\n".join(out)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "data").mkdir(exist_ok=True)
    if (DEST / "images").exists():
        shutil.rmtree(DEST / "images")
    (DEST / "images").mkdir()

    counts: dict[str, dict[str, int]] = {}
    labels: dict[str, list[str]] = {}
    state_task: dict[str, str] = {}
    instructions: dict[str, str] = {}
    criteria: dict[str, str] = {}

    for split in SPLITS:
        source = SRC / "data" / f"{split}.parquet"
        rows = pq.ParquetFile(source).read().to_pylist()
        image_dir = DEST / "images" / split
        image_dir.mkdir(parents=True, exist_ok=True)

        out_rows = []
        for index, row in enumerate(rows):
            image = row["images"][0]["bytes"]
            image_rel = f"images/{split}/{index:05d}.png"
            (DEST / image_rel).write_bytes(image)

            state = row["state"]
            question = row["question"]
            task = row["task"]
            state_task.setdefault(task, state["task"])
            instructions.setdefault(task, question["instructions"])
            criteria.update(question["criteria"])

            out_rows.append({
                "index": index + 1,
                "id": row["id"],
                "puzzle_id": row["puzzle_id"],
                "task": task,
                "step_index": row["step_index"],
                "image": image_rel,
                "state": {
                    "rows": state["rows"],
                    "columns": state["columns"],
                    "board": state["board"],
                    "row_clues": state["row_clues"],
                    "col_clues": state["col_clues"],
                    "givens": state["givens"],
                    "start": state["start"],
                    "end": state["end"],
                },
                "label": row["label"],
                "answer": row["answer"],
                "action": {
                    "kind": row["action"]["kind"],
                    "cells": row["action"]["cells"],
                    "evidence": row["action"]["evidence"],
                    "explanation": row["action"]["explanation"],
                },
                "difficulty": row["difficulty"],
                "rows": row["rows"],
                "columns": row["columns"],
                "source_date": row["source_date"],
                "split": row["split"],
            })

        counts[split] = {
            "deduction": sum(1 for row in out_rows if row["task"] == "deduction"),
            "contradiction": sum(1 for row in out_rows if row["task"] == "contradiction"),
            "total": len(out_rows),
        }
        labels[split] = sorted({row["label"] for row in out_rows})

        payload = {"rows": out_rows, "counts": counts[split], "labels": labels[split]}
        data_js = (
            "window.HF_PREVIEW_DATA = window.HF_PREVIEW_DATA || {};\n"
            f'window.HF_PREVIEW_DATA["{split}"] = '
            f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))};\n"
        )
        (DEST / "data" / f"{split}.js").write_text(data_js, encoding="utf-8")
        print(f"{split:12s} {len(out_rows):5d} rows, {len(labels[split])} labels")

    total = sum(value["total"] for value in counts.values())
    meta = {
        "dataset": DATASET,
        "title": "Train Tracks SFT: deductions and contradictions",
        "description": (
            f"{total:,} supervised examples from 608 Train Tracks puzzles, covering "
            "deduction and contradiction tasks on the same boards."
        ),
        "tags": TAGS,
        "splits": list(SPLITS),
        "splitPuzzles": SPLIT_PUZZLES,
        "counts": counts,
        "labels": labels,
        "total": total,
        "puzzles": 608,
        "coordinates": (
            "Rows top to bottom; columns left to right; both start at 1. "
            "Track shapes name connected sides N/E/S/W."
        ),
        "stateTask": state_task,
        "instructions": instructions,
        "criteria": criteria,
    }
    (DEST / "data" / "meta.js").write_text(
        "window.HF_PREVIEW_META = " + json.dumps(meta, ensure_ascii=False, indent=1) + ";\n",
        encoding="utf-8",
    )

    card = md_to_html((SRC / "README.md").read_text(encoding="utf-8"))
    (DEST / "data" / "card.js").write_text(
        "window.HF_PREVIEW_CARD = " + json.dumps(card, ensure_ascii=False) + ";\n",
        encoding="utf-8",
    )

    files = []
    for path in sorted((SRC / "data").glob("*.parquet")):
        files.append({
            "path": f"data/{path.name}",
            "size": path.stat().st_size,
            "url": f"https://huggingface.co/datasets/{DATASET}/resolve/main/data/{path.name}",
        })
    files.append({
        "path": "README.md",
        "size": (SRC / "README.md").stat().st_size,
        "url": f"https://huggingface.co/datasets/{DATASET}/resolve/main/README.md",
    })
    (DEST / "data" / "files.js").write_text(
        "window.HF_PREVIEW_FILES = " + json.dumps(files, ensure_ascii=False, indent=1) + ";\n",
        encoding="utf-8",
    )

    print(f"total {total:,} rows")
    print(f"open {DEST / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
