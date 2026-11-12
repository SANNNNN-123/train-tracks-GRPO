"""One command for the G0 data gate.

Aggregates the artefacts the other scripts produce (nothing is recomputed twice) and re-verifies every
reference solution, because that is the one check cheap enough to repeat.

    uv run python scripts/verify_corpus.py

Exit code 0 means every gate below passed.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tt_env import load_jsonl, verify  # noqa: E402

EXPECTED_PUZZLES = 608
EXPECTED_SPLITS = {"train": 212, "validation": 46, "test_small": 46, "test_medium": 304}


def main() -> int:
    data = ROOT / "data"
    puzzles = load_jsonl(data / "puzzles.jsonl")
    gates: list[tuple[str, bool, str]] = []

    gates.append(("corpus size", len(puzzles) == EXPECTED_PUZZLES, f"{len(puzzles)}/{EXPECTED_PUZZLES}"))

    broken = [p.puzzle_id for p in puzzles if not verify(p, list(p.reference_solution or [])).ok]
    gates.append(("reference solutions verify", not broken, f"{len(puzzles) - len(broken)}/{len(puzzles)}"))

    qa = _load(data / "qa_report.json")
    if qa is None:
        gates.append(("qa report", False, "missing; run scripts/qa_corpus.py"))
    else:
        gates.append(("qa problems", not qa["problems"], f"{len(qa['problems'])}"))
        gates.append(("duplicates", not qa["duplicate_groups"], f"{len(qa['duplicate_groups'])} group(s)"))
        gates.append(("month gaps", not qa["month_gaps"], f"{len(qa['month_gaps'])} month(s)"))

    cross = _load(data / "hf_crosscheck.json")
    if cross is None:
        gates.append(("HF cross-check", False, "missing; run scripts/crosscheck_hf.py"))
    else:
        agreement = cross["agreement"]
        worst = min(agreement.values()) if agreement else 0.0
        gates.append(
            (
                "HF agreement",
                worst == 1.0 and cross["compared"] == EXPECTED_PUZZLES and not cross["mismatches"],
                f"{cross['compared']} compared, worst {worst:.1%}, {len(cross['mismatches'])} mismatch(es)",
            )
        )

    splits = _load(data / "splits.json")
    if splits is None:
        gates.append(("splits", False, "missing; run scripts/freeze_splits.py"))
    else:
        counts = splits["counts"]
        gates.append(("split counts", counts == EXPECTED_SPLITS, json.dumps(counts)))
        gates.append(("split problems", not splits["problems"], f"{len(splits['problems'])}"))

    build = _load(data / "build_report.json")
    if build is None:
        gates.append(("build report", False, "missing; run scripts/build_puzzles.py"))
    else:
        sizes = Counter(p.rows for p in puzzles if p.difficulty == "small")
        medium = sum(1 for p in puzzles if p.difficulty == "medium")
        gates.append(("size mix", bool(sizes) and medium > 0, f"{dict(sorted(sizes.items()))} small, {medium} medium"))

    width = max(len(name) for name, _, _ in gates)
    print("G0 data gate")
    print("-" * (width + 60))
    for name, ok, detail in gates:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    failed = [name for name, ok, _ in gates if not ok]
    print("-" * (width + 60))
    print("G0 green" if not failed else f"G0 blocked by: {', '.join(failed)}")
    return 0 if not failed else 1


def _load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text("utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
