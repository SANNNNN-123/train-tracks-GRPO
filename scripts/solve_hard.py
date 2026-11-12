"""Probe which move ordering the reference DFS needs on the puzzles that exhaust it.

`tracks-solver/tracks.py` is a plain depth-first search with a hard-coded N, S, W, E order, so a few
instances explore millions of dead ends. The production path handles that with
`tt_env.tracks_adapter.STRATEGIES`; this script reports the same comparison per puzzle, with each
attempt in its own process and a hard wall-clock limit, because a bad ordering can run for hours.

Usage:
    uv run python scripts/solve_hard.py                        # the known failures
    uv run python scripts/solve_hard.py --id 20082026-medium --timeout 30
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tracks-solver"))

from tt_env.puzzle import Puzzle  # noqa: E402
from tt_env.rules import canonical_piece, verify  # noqa: E402
from tt_env.tracks_adapter import STRATEGIES, solve_with_tracks  # noqa: E402

KNOWN_FAILURES = ["09022026-medium", "22062026-medium", "20082026-medium"]


def attempt(payload: dict, strategy: dict) -> dict:
    """Run one strategy in this process and check the grid it produces."""
    outcome = solve_with_tracks(payload, strategy)
    if outcome.get("grid") is None:
        return {
            "status": outcome["status"],
            "seconds": outcome.get("seconds", 0.0),
            "moves": outcome.get("moves", -1),
            "detail": outcome.get("detail", ""),
        }
    verdict = verify(Puzzle.from_dict(payload), outcome["grid"])
    return {
        "status": "solved" if verdict.ok else "verify",
        "seconds": outcome.get("seconds", 0.0),
        "moves": outcome.get("moves", -1),
        "detail": "; ".join(verdict.failures[:2]),
        "grid": outcome["grid"],
    }


def _worker(payload: dict, strategy: dict, sink) -> None:
    try:
        sink.put(attempt(payload, strategy))
    except Exception as exc:  # noqa: BLE001
        sink.put({"status": "error", "detail": f"worker: {exc}", "seconds": 0.0, "moves": -1})


def attempt_isolated(payload: dict, strategy: dict, timeout: float) -> dict:
    ctx = mp.get_context("spawn")
    sink = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(payload, strategy, sink), daemon=True)
    started = time.perf_counter()
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        return {
            "status": "timeout",
            "seconds": time.perf_counter() - started,
            "moves": -1,
            "detail": f">{timeout:.0f}s",
        }
    try:
        return sink.get_nowait()
    except queue.Empty:
        return {"status": "error", "detail": "no result", "seconds": time.perf_counter() - started, "moves": -1}


def load_puzzle(puzzle_id: str) -> Puzzle:
    import track_solver  # needs cv2

    matches = list((ROOT / "assets").rglob(f"{puzzle_id}.png"))
    if not matches:
        raise SystemExit(f"no image for {puzzle_id}")
    data, _ = track_solver.parse_puzzle_image(matches[0])
    rows, cols = len(data["rows"]), len(data["columns"])
    givens, start, end = [], None, None
    for piece, solver_row, col, *marker in data["pieces"]:
        r = rows - 1 - int(solver_row)  # tracks.py is bottom-up
        c = int(col)
        givens.append((r, c, canonical_piece(piece)))
        if marker and marker[0] == "s":
            start = (r, c)
        elif marker and marker[0] == "e":
            end = (r, c)
    return Puzzle(
        puzzle_id=puzzle_id,
        rows=rows,
        cols=cols,
        row_clues=tuple(int(v) for v in data["rows"]),
        col_clues=tuple(int(v) for v in data["columns"]),
        givens=tuple(sorted(givens)),
        start=start,
        end=end,
        source=str(matches[0].relative_to(ROOT)),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", action="append", default=None)
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--report", type=Path, default=ROOT / "data" / "hard_report.json")
    args = ap.parse_args()

    ids = args.id or KNOWN_FAILURES
    solved: dict[str, dict] = {}
    results: dict[str, list] = {}

    for puzzle_id in ids:
        puzzle = load_puzzle(puzzle_id)
        print(f"\n{puzzle_id}  {puzzle.rows}x{puzzle.cols} givens={len(puzzle.givens)}", flush=True)
        results[puzzle_id] = []
        for strategy in STRATEGIES:
            outcome = attempt_isolated(puzzle.to_dict(), strategy, args.timeout)
            print(
                f"  {strategy['name']:<18} {outcome['status']:<9} {outcome['seconds']:7.2f}s "
                f"{outcome['moves']:>11,} moves  {outcome.get('detail', '')[:60]}",
                flush=True,
            )
            record = {
                "strategy": strategy["name"],
                "status": outcome["status"],
                "seconds": round(outcome["seconds"], 3),
                "moves": outcome["moves"],
                "detail": outcome.get("detail", "")[:200],
            }
            results[puzzle_id].append(record)
            if outcome["status"] == "solved":
                solved[puzzle_id] = {**record, "grid": outcome.get("grid")}
                break

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"solved": solved, "results": results}, indent=2), encoding="utf-8")
    print(f"\nsolved {len(solved)}/{len(ids)}: { {k: v['strategy'] for k, v in solved.items()} }")
    print(f"report -> {args.report}")
    return 0 if len(solved) == len(ids) else 1


if __name__ == "__main__":
    raise SystemExit(main())
