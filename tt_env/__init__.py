"""Train Tracks: exact rule verifier, puzzle model, sandbox and rewards."""

from .puzzle import Puzzle, load_jsonl, save_jsonl
from .rewards import Grade, format_ok, grade, no_cheat, shaping
from .rules import Verdict, canonical_piece, decode_grid, dirs_of, verify
from .sandbox import RunResult, check_source, extract_code, run_solution

__all__ = [
    "Grade",
    "Puzzle",
    "RunResult",
    "Verdict",
    "canonical_piece",
    "check_source",
    "decode_grid",
    "dirs_of",
    "extract_code",
    "format_ok",
    "grade",
    "load_jsonl",
    "no_cheat",
    "run_solution",
    "save_jsonl",
    "shaping",
    "verify",
]
