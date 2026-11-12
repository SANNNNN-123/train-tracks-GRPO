"""Build `data/puzzles.jsonl` from the 608 puzzle PNGs.

For each image: parse clues + givens (track_solver.py), solve with the reference DFS in an isolated
process (hard timeout), re-derive the piece letters, then certify the result with the exact verifier.
Nothing is hand-patched: a puzzle either passes every gate or is reported as a failure.

Usage:
    uv run python scripts/build_puzzles.py             # all 608
    uv run python scripts/build_puzzles.py --limit 8    # smoke test
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import sys
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tt_env.puzzle import Puzzle, load_jsonl, save_jsonl  # noqa: E402
from tt_env.rules import canonical_piece, verify  # noqa: E402
from tt_env.tracks_adapter import STRATEGIES, solve_with_tracks  # noqa: E402

ASSETS = ROOT / "assets"
OUT = ROOT / "data" / "puzzles.jsonl"
REPORT = ROOT / "data" / "build_report.json"
SKIP_DIRS = {"traintracks-rules"}


def _worker(payload: dict, strategy: dict, sink) -> None:
    try:
        sink.put(solve_with_tracks(payload, strategy))
    except Exception as exc:  # noqa: BLE001
        sink.put({"status": "error", "detail": f"worker: {exc}", "seconds": 0.0})


def solve_isolated(puzzle: Puzzle, timeout: float, strategy: dict) -> dict:
    ctx = mp.get_context("spawn")
    sink = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(puzzle.to_dict(), strategy, sink), daemon=True)
    started = time.perf_counter()
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        return {
            "status": "timeout",
            "detail": f">{timeout:.0f}s ({strategy['name']})",
            "seconds": time.perf_counter() - started,
        }
    try:
        return sink.get_nowait()
    except queue.Empty:
        return {"status": "error", "detail": "no result", "seconds": time.perf_counter() - started}


def solve_with_fallbacks(puzzle: Puzzle, timeout: float) -> dict:
    """Try each ordering in turn; the first solved result wins."""
    attempts = []
    for strategy in STRATEGIES:
        outcome = solve_isolated(puzzle, timeout, strategy)
        attempts.append(
            {
                "strategy": strategy["name"],
                "status": outcome["status"],
                "seconds": round(outcome.get("seconds", 0.0), 3),
                "moves": outcome.get("moves"),
            }
        )
        if outcome["status"] == "solved":
            outcome["attempts"] = attempts
            return outcome
    last = dict(attempts[-1])
    return {"status": last["status"], "detail": last["status"], "attempts": attempts, "seconds": 0.0}


def parse_filename(stem: str) -> tuple[str, str]:
    """`01012026-small` -> (ISO date, difficulty)."""
    name, _, difficulty = stem.rpartition("-")
    if len(name) == 8 and name.isdigit():
        day, month, year = name[:2], name[2:4], name[4:]
        try:
            return date(int(year), int(month), int(day)).isoformat(), difficulty
        except ValueError:
            pass
    return "", difficulty


def puzzle_from_parser(data: dict, stem: str, source: str) -> Puzzle:
    rows, cols = len(data["rows"]), len(data["columns"])
    givens = []
    start = end = None
    for piece in data["pieces"]:
        dirs, solver_row, col = piece[0], int(piece[1]), int(piece[2])
        marker = piece[3] if len(piece) > 3 else ""
        canonical = canonical_piece(dirs)
        if canonical is None:
            raise ValueError(f"illegal piece {dirs!r} at r{solver_row} c{col}")
        r = rows - 1 - solver_row
        givens.append((r, col, canonical))
        if marker == "s":
            start = (r, col)
        elif marker == "e":
            end = (r, col)
    if start is None or end is None:
        raise ValueError("parser did not mark both endpoints")
    iso, difficulty = parse_filename(stem)
    return Puzzle(
        puzzle_id=stem,
        rows=rows,
        cols=cols,
        row_clues=tuple(int(v) for v in data["rows"]),
        col_clues=tuple(int(v) for v in data["columns"]),
        givens=tuple(sorted(givens)),
        start=start,
        end=end,
        difficulty=difficulty,
        date=iso,
        source=source,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", type=Path, default=ASSETS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--report", type=Path, default=REPORT)
    ap.add_argument("--timeout", type=float, default=120.0, help="per strategy attempt, in seconds")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", action="append", default=None, help="rebuild just these puzzle ids")
    ap.add_argument("--skip-existing", action="store_true", help="keep puzzles already in --out and add new ones")
    args = ap.parse_args()

    import track_solver  # needs cv2; imported here so the child processes never load it

    existing: list[Puzzle] = []
    if args.out.exists() and (args.skip_existing or args.only):
        existing = load_jsonl(args.out)
        print(f"loaded {len(existing)} existing puzzles from {args.out}")

    images = sorted(
        p
        for p in args.assets.rglob("*.png")
        if not any(part in SKIP_DIRS for part in p.parts)
    )
    images_found = len(images)
    if args.only:
        wanted = set(args.only)
        images = [p for p in images if p.stem in wanted]
        missing = wanted - {p.stem for p in images}
        if missing:
            print(f"warning: no image for {sorted(missing)}")
    if args.skip_existing and existing:
        done = {p.puzzle_id for p in existing}
        images = [p for p in images if p.stem not in done]
        print(f"resuming: {len(images)} images left")
    if args.limit:
        images = images[: args.limit]

    puzzles: list[Puzzle] = []
    failures: list[dict] = []
    parse_seconds: list[float] = []

    for index, path in enumerate(images, 1):
        stem = path.stem
        started = time.perf_counter()
        try:
            data, _ = track_solver.parse_puzzle_image(path)
            puzzle = puzzle_from_parser(data, stem, str(path.relative_to(ROOT)))
        except Exception as exc:  # noqa: BLE001
            failures.append({"puzzle_id": stem, "stage": "parse", "detail": str(exc)})
            print(f"[{index:3}/{len(images)}] {stem:<22} PARSE FAIL  {exc}")
            continue
        parse_seconds.append(time.perf_counter() - started)

        result = solve_with_fallbacks(puzzle, args.timeout)
        if result["status"] != "solved":
            failures.append(
                {
                    "puzzle_id": stem,
                    "stage": "solve",
                    "detail": f"{result['status']}: {result.get('detail', '')}",
                    "attempts": result.get("attempts"),
                }
            )
            print(f"[{index:3}/{len(images)}] {stem:<22} SOLVE {result['status']:<8} {result.get('detail', '')}")
            continue

        verdict = verify(puzzle, result["grid"])
        if not verdict.ok:
            failures.append(
                {"puzzle_id": stem, "stage": "verify", "detail": "; ".join(verdict.failures[:4]), "diag": verdict.diagnostics}
            )
            print(f"[{index:3}/{len(images)}] {stem:<22} VERIFY FAIL {verdict.failures[:3]}")
            continue

        puzzles.append(
            Puzzle(
                **{
                    **puzzle.__dict__,
                    "reference_solution": tuple(result["grid"]),
                    "meta": {
                        "givens": len(puzzle.givens),
                        "occupied": verdict.diagnostics.get("occupied"),
                        "solve_seconds": round(result.get("seconds", 0.0), 3),
                        "strategy": result.get("strategy", "plain-NSWE"),
                        "attempts": result.get("attempts"),
                    },
                }
            )
        )
        print(
            f"[{index:3}/{len(images)}] {stem:<22} ok  {puzzle.rows}x{puzzle.cols} "
            f"givens={len(puzzle.givens)} {result.get('seconds', 0):5.2f}s"
        )

    if existing:
        replaced = {p.puzzle_id for p in puzzles}
        puzzles = [p for p in existing if p.puzzle_id not in replaced] + puzzles
        puzzles.sort(key=lambda p: p.puzzle_id)
    by_size = Counter(f"{p.rows}x{p.cols}" for p in puzzles)
    strategy_use = Counter(p.meta.get("strategy", "plain-NSWE") for p in puzzles)
    all_solve_seconds = [p.meta.get("solve_seconds", 0.0) for p in puzzles]
    save_jsonl(args.out, puzzles)

    def pct(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 3)

    report = {
        "images_found": images_found,
        "images_this_run": len(images),
        "built": len(puzzles),
        "failed_this_run": len(failures),
        "by_size": dict(sorted(by_size.items())),
        "strategies": dict(sorted(strategy_use.items())),
        "parse_seconds_this_run": {"p50": pct(parse_seconds, 0.5), "p90": pct(parse_seconds, 0.9)},
        "solve_seconds": {
            "p50": pct(all_solve_seconds, 0.5),
            "p90": pct(all_solve_seconds, 0.9),
            "max": pct(all_solve_seconds, 1.0),
        },
        "failures": failures,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== summary ===")
    print(f"corpus {len(puzzles)} / {images_found} images   failures this run {len(failures)}")
    print(f"by size: {dict(sorted(by_size.items()))}")
    print(f"strategies: {dict(sorted(strategy_use.items()))}")
    print(f"solve p50/p90/max: {report['solve_seconds']}")
    if failures:
        stages = Counter(f["stage"] for f in failures)
        print(f"failure stages: {dict(stages)}")
        print(f"details in {args.report}")
    return 0 if puzzles else 1


if __name__ == "__main__":
    raise SystemExit(main())
