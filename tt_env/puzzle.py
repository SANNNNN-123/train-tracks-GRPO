"""Puzzle container shared by the env, verifier and reward functions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .rules import EMPTY, dirs_of, outward_port


@dataclass(frozen=True)
class Puzzle:
    puzzle_id: str
    rows: int
    cols: int
    row_clues: tuple[int, ...]
    col_clues: tuple[int, ...]
    givens: tuple[tuple[int, int, str], ...]
    start: tuple[int, int]
    end: tuple[int, int]
    difficulty: str = ""
    date: str = ""
    source: str = ""
    reference_solution: tuple[str, ...] | None = None
    meta: dict = field(default_factory=dict)

    def payload(self) -> dict:
        """The dict handed to a candidate `solve(puzzle)` implementation."""
        return {
            "rows": self.rows,
            "cols": self.cols,
            "row_clues": list(self.row_clues),
            "col_clues": list(self.col_clues),
            "givens": [[r, c, p] for r, c, p in self.givens],
            "start": [*self.start, self._port(self.start)],
            "end": [*self.end, self._port(self.end)],
        }

    def _port(self, cell) -> str:
        piece = next((p for r, c, p in self.givens if (r, c) == tuple(cell)), None)
        if piece is None:
            return ""
        return outward_port(piece, cell[0], cell[1], self.rows, self.cols) or ""

    def to_dict(self) -> dict:
        data = {
            "puzzle_id": self.puzzle_id,
            "difficulty": self.difficulty,
            "date": self.date,
            "source": self.source,
            "rows": self.rows,
            "cols": self.cols,
            "row_clues": list(self.row_clues),
            "col_clues": list(self.col_clues),
            "givens": [[r, c, p] for r, c, p in self.givens],
            "start": list(self.start),
            "end": list(self.end),
            "reference_solution": list(self.reference_solution) if self.reference_solution else None,
        }
        if self.meta:
            data["meta"] = self.meta
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Puzzle:
        return cls(
            puzzle_id=data["puzzle_id"],
            rows=int(data["rows"]),
            cols=int(data["cols"]),
            row_clues=tuple(int(v) for v in data["row_clues"]),
            col_clues=tuple(int(v) for v in data["col_clues"]),
            givens=tuple((int(r), int(c), str(p)) for r, c, p in data["givens"]),
            start=tuple(data["start"]),
            end=tuple(data["end"]),
            difficulty=data.get("difficulty", ""),
            date=data.get("date", ""),
            source=data.get("source", ""),
            reference_solution=tuple(data["reference_solution"]) if data.get("reference_solution") else None,
            meta=data.get("meta", {}),
        )

    def ascii(self, grid=None) -> str:
        """Readable dump of the puzzle, optionally with a candidate grid."""
        cells = None
        if grid is not None:
            cells = [[grid[r][2 * c : 2 * c + 2] for c in range(self.cols)] for r in range(self.rows)]
        else:
            cells = [[EMPTY] * self.cols for _ in range(self.rows)]
            for r, c, piece in self.givens:
                cells[r][c] = piece
        width = max(2, max(len(x) for row in cells for x in row))
        head = "    " + "".join(f"{self.col_clues[c]:>{width + 1}}" for c in range(self.cols))
        lines = [head]
        for r in range(self.rows):
            row = "".join(f"{cells[r][c]:>{width + 1}}" for c in range(self.cols))
            lines.append(f"{self.row_clues[r]:>3} {row}")
        return "\n".join(lines)


def load_jsonl(path: str | Path) -> list[Puzzle]:
    path = Path(path)
    return [Puzzle.from_dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]


def save_jsonl(path: str | Path, puzzles) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for puzzle in puzzles:
            handle.write(json.dumps(puzzle.to_dict(), ensure_ascii=False) + "\n")
