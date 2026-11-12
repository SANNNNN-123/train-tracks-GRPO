"""Can the eval prediction be reconstructed from its diagnostics?

Every eval record reports occupied == len(givens). If the diagnostics from verifying a givens-only
grid match the recorded diagnostics exactly, the prediction was that grid, bit for bit.
"""

import json
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\ZUHAIR\Desktop\2026\Train_Track_GRPO")
sys.path.insert(0, str(ROOT))
from tt_env import load_jsonl  # noqa: E402
from tt_env.rules import EMPTY, verify  # noqa: E402

corpus = {p.puzzle_id: p for p in load_jsonl(ROOT / "data" / "puzzles.jsonl")}
rows = [json.loads(line) for line in (ROOT / "outputs" / "test_small_partial.txt").read_text(encoding="utf-8").splitlines() if line.strip()]

for record in rows:
    puzzle = corpus[record["puzzle_id"]]
    cells = [[EMPTY] * puzzle.cols for _ in range(puzzle.rows)]
    for r, c, piece in puzzle.givens:
        cells[r][c] = piece
    grid = ["".join(row) for row in cells]
    verdict = verify(puzzle, grid)
    same_diag = verdict.diagnostics == record["diagnostics"]
    same_detail = "; ".join(verdict.failures[:4]) == record["detail"]
    print(
        f"step {record['sample']} {record['puzzle_id']:<16} "
        f"diag match {same_diag}  detail match {same_detail}"
    )
    if not same_diag:
        print("   ours:", json.dumps(verdict.diagnostics, sort_keys=True))
        print("   eval:", json.dumps(record["diagnostics"], sort_keys=True))
