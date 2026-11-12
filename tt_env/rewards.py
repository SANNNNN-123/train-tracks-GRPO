"""Reward functions for code-as-action Train Tracks.

Mirrors the sudoku notebook's `function_works` / `no_cheating` / `strategy_succeeds`, plus the bounded
shaping term from `plan.md` §6.

| component | value | meaning |
|---|---|---|
| `format_ok` | +0.5 / -0.5 | exactly one fenced block, parses, `def solve(puzzle)`, compiles |
| `no_cheat` | +1.0 / -20.0 | static AST allow-list passes / any escape attempt |
| `solved` | +30.0 | verifier accepts the returned grid |
| | -0.5 | the program raised or timed out |
| | 0 | ran, but the return value is not a grid, or never ran (statically rejected) |
| shaping | 0 … 1.5 | only when the grid parses and the givens are intact, and never on a solve |

A solve therefore scores exactly 31.5, which the audit script asserts is the maximum.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .rules import EMPTY, OFFSETS, OPPOSITE, decode_grid, dirs_of, verify
from .sandbox import (
    MAX_STEPS,
    check_source,
    compile_solution,
    escape_violations,
    extract_code,
    fence_count,
    run_solution,
)

FORMAT_GOOD = 0.5
FORMAT_BAD = -0.5
CHEAT_CLEAN = 1.0
CHEAT_BAD = -20.0
SOLVED = 30.0
RUN_FAILED = -0.5
MAX_SHAPING = 1.5


@dataclass
class Grade:
    format_ok: float = FORMAT_BAD
    no_cheat: float = CHEAT_CLEAN
    solved: float = 0.0
    shaping: float = 0.0
    status: str = "no-code"
    detail: str = ""
    grid: list[str] | None = None
    violations: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    @property
    def total(self) -> float:
        return self.format_ok + self.no_cheat + self.solved + self.shaping

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "format_ok": self.format_ok,
            "no_cheat": self.no_cheat,
            "solved": self.solved,
            "shaping": round(self.shaping, 4),
            "total": round(self.total, 4),
            "detail": self.detail,
            "violations": self.violations,
            "diagnostics": self.diagnostics,
            # The returned grid, when the program produced one, so an eval report can show the
            # prediction next to the reference instead of guessing it from diagnostics.
            "grid": self.grid,
        }


def _signature_ok(source: str) -> bool:
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    solves = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "solve"]
    if len(solves) != 1:
        return False
    args = solves[0].args
    positional = [*args.posonlyargs, *args.args]
    return len(positional) == 1 and positional[0].arg == "puzzle" and not (args.vararg or args.kwarg)


def format_ok(completion) -> float:
    """Is it a usable program? Deliberately small: GRPO subtracts the group mean anyway."""
    source = extract_code(completion_text(completion))
    if source is None:
        return FORMAT_BAD
    if not _signature_ok(source):
        return FORMAT_BAD
    try:
        compile(source, "<candidate>", "exec")
    except SyntaxError:
        return FORMAT_BAD
    return FORMAT_GOOD


def no_cheat(completion) -> float:
    """Sandbox integrity, scored statically."""
    source = extract_code(completion_text(completion))
    if source is None:
        return CHEAT_CLEAN  # nothing to execute, nothing to escape from
    return CHEAT_BAD if escape_violations(source) else CHEAT_CLEAN


def shaping(puzzle, grid) -> float:
    """Bounded partial credit for a parseable grid with the givens intact.

    "Copy the givens and stop" earns **nothing**: with the givens fixed, an occupied count equal to
    the number of givens means no new track was placed. Leaving that case paid (0.25-0.49 in the
    audit, 0.31-0.45 in the first Kaggle run) and the policy collapsed onto it, because any attempt
    to extend the path risks mutating a given (0) or timing out (-0.5). Every real attempt adds at
    least one cell and still scores.
    """
    cells, failures = decode_grid(grid, puzzle.rows, puzzle.cols)
    if cells is None:
        return 0.0
    for r, c, piece in puzzle.givens:
        if cells[r][c] != piece:
            return 0.0
    if sum(1 for row in cells for piece in row if piece != EMPTY) <= len(puzzle.givens):
        return 0.0

    counted_rows = [sum(1 for c in range(puzzle.cols) if cells[r][c] != EMPTY) for r in range(puzzle.rows)]
    counted_cols = [sum(1 for r in range(puzzle.rows) if cells[r][c] != EMPTY) for c in range(puzzle.cols)]
    errors = [
        abs(counted_rows[r] - puzzle.row_clues[r]) / max(puzzle.row_clues[r], 1) for r in range(puzzle.rows)
    ] + [abs(counted_cols[c] - puzzle.col_clues[c]) / max(puzzle.col_clues[c], 1) for c in range(puzzle.cols)]
    clue_fit = 1.0 - sum(errors) / len(errors)

    start, end = tuple(puzzle.start), tuple(puzzle.end)
    satisfied = total_ports = 0
    for r in range(puzzle.rows):
        for c in range(puzzle.cols):
            if cells[r][c] == EMPTY:
                continue
            for d in dirs_of(cells[r][c]):
                total_ports += 1
                dr, dc = OFFSETS[d]
                nr, nc = r + dr, c + dc
                if not (0 <= nr < puzzle.rows and 0 <= nc < puzzle.cols):
                    satisfied += 1 if (r, c) in (start, end) else 0
                elif cells[nr][nc] != EMPTY and OPPOSITE[d] in dirs_of(cells[nr][nc]):
                    satisfied += 1
    port_fit = satisfied / total_ports if total_ports else 0.0

    value = MAX_SHAPING * (0.5 * clue_fit + 0.5 * port_fit)
    return max(0.0, min(MAX_SHAPING, value))


def grade(completion, puzzle, execute: bool = True, max_steps: int | None = None) -> Grade:
    """Full breakdown for one completion against one puzzle."""
    result = Grade()
    text = completion_text(completion)
    source = extract_code(text)
    result.format_ok = format_ok(text)
    if source is None:
        # Say *why*: "no code" from an empty string is a different bug from two blocks.
        result.status = "no-code"
        result.detail = (
            f"expected exactly one fenced block, found {fence_count(text)} in {len(text)} chars"
        )
        return result

    violations = check_source(source)
    escapes = escape_violations(source)
    result.violations = violations
    result.no_cheat = CHEAT_BAD if escapes else CHEAT_CLEAN
    if escapes:
        result.status = "cheat"
        result.detail = "; ".join(escapes)
        return result
    if violations:
        result.status = "invalid"
        result.detail = "; ".join(violations)
        return result

    try:
        compile_solution(source)
    except SyntaxError as exc:
        result.status = "syntax"
        result.detail = exc.msg
        return result

    if not execute:
        result.status = "static"
        return result

    run = run_solution(source, puzzle.payload(), max_steps=max_steps or MAX_STEPS)
    result.status = run.status
    result.detail = run.detail
    if run.status == "timeout":
        result.solved = RUN_FAILED
        return result
    if run.status == "error":
        result.solved = 0.0 if run.detail.startswith("return-not-grid") else RUN_FAILED
        return result
    if run.status != "ok" or run.grid is None:
        return result

    result.grid = run.grid
    verdict = verify(puzzle, run.grid)
    result.diagnostics = verdict.diagnostics
    if verdict.ok:
        result.solved = SOLVED
        result.status = "solved"
        return result

    result.status = "not-solved"
    result.detail = "; ".join(verdict.failures[:4])
    result.shaping = shaping(puzzle, run.grid)
    return result


# --- TRL-compatible wrappers (each returns a single float) -------------------------------


def _content_text(content) -> str:
    """Text of one message's `content`, which may be a string or multimodal parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return _content_text(content.get("text", ""))
    if isinstance(content, (list, tuple)):
        chunks = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict):
                # {"type": "text", "text": "..."} is the Vision/TRL structured form.
                chunks.append(str(part.get("text", "")))
        return "".join(chunks)
    return "" if content is None else str(content)


def completion_text(completion) -> str:
    """Normalise a TRL completion to text.

    TRL hands reward functions a plain string, a chat-style list of messages
    (`[{"role": "assistant", "content": "..."}]`), or - with a multimodal processor - the same
    messages with `content` as a list of parts. Getting this wrong yields an empty string, which
    looks exactly like a model that never writes code, so all three shapes are accepted.
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return _content_text(completion.get("content", ""))
    if isinstance(completion, (list, tuple)):
        parts = [
            _content_text(item.get("content", "")) if isinstance(item, dict) else str(item)
            for item in completion
        ]
        return "\n".join(part for part in parts if part)
    return "" if completion is None else str(completion)


def reward_format_ok(completions, **kwargs) -> list[float]:
    return [format_ok(completion_text(text)) for text in completions]


def reward_no_cheat(completions, **kwargs) -> list[float]:
    return [no_cheat(completion_text(text)) for text in completions]


__all__ = [
    "CHEAT_BAD",
    "CHEAT_CLEAN",
    "FORMAT_BAD",
    "FORMAT_GOOD",
    "Grade",
    "MAX_SHAPING",
    "RUN_FAILED",
    "SOLVED",
    "completion_text",
    "format_ok",
    "grade",
    "no_cheat",
    "shaping",
]
