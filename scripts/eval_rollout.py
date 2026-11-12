"""T2 evaluation harness: independent rollouts, graded by the same verifier as training.

The baselines in `plan.md` §7.5 are policies, so they run through the identical path as a model
completion: produce text -> extract code -> sandbox -> verify -> reward. That keeps the comparison
honest and lets the harness be tested without a GPU.

Backends
    --backend empty      never places track
    --backend givens     copies the givens, nothing else
    --backend random     right number of cells per clue, random legal pieces
    --backend reference  the `tracks.py` DFS (ceiling; slow on a few mediums)
    --backend hf         a local causal LM via transformers (needs torch; GPU host only)

Usage
    uv run python scripts/eval_rollout.py --backend empty --split test_small
    uv run python scripts/eval_rollout.py --backend random --split train --limit 20
    uv run python scripts/eval_rollout.py --backend hf --model unsloth/gemma-4-E2B-it \
        --adapter outputs/grpo/checkpoint-200 --split test_small --k 4 --temperature 1.0

Outputs
    outputs/eval/<backend>-<split>-<utcstamp>.jsonl    one record per sample
    outputs/eval/<backend>-<split>-<utcstamp>.json      summary, metrics from §7.6

Nothing here reads `reference_solution`; the verifier is the only judge.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tt_env import Puzzle, grade, load_jsonl  # noqa: E402
from tt_env import policies  # noqa: E402

Z = 1.959963984540054  # 95%


def _display(path: Path) -> str:
    """Repo-relative when possible, absolute otherwise (the harness may write to a temp dir)."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def wilson(successes: int, total: int) -> tuple[float, float]:
    """Wilson score interval; behaves sanely at 0/n, unlike the normal approximation."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1 + Z * Z / total
    centre = (p + Z * Z / (2 * total)) / denominator
    margin = Z * math.sqrt(p * (1 - p) / total + Z * Z / (4 * total * total)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def pass_at_k(solved_per_puzzle: list[list[bool]], k: int) -> float:
    """Unbiased pass@k estimator (Chen et al.): no re-ranking, just combinatorics."""
    if not solved_per_puzzle or k <= 0:
        return 0.0
    values = []
    for solved in solved_per_puzzle:
        n, c = len(solved), sum(solved)
        if n - c < k:
            values.append(1.0)
        else:
            values.append(1.0 - math.comb(n - c, k) / math.comb(n, k))
    return sum(values) / len(values)


# --- backends ------------------------------------------------------------------------------


def make_backend(name: str, args, rng: random.Random):
    """Return `call(puzzle, sample_index) -> (completion, extra_seconds)`."""

    if name in {"empty", "givens", "random"}:
        builder = {
            "empty": policies.all_empty,
            "givens": policies.givens_only,
            "random": lambda puzzle: policies.random_clues(puzzle, rng),
        }[name]

        def stub(puzzle: Puzzle, sample_index: int):
            return policies.program(builder(puzzle)), 0.0

        return stub

    if name == "reference":
        from tt_env.tracks_adapter import STRATEGIES, solve_with_tracks

        strategy = next(s for s in STRATEGIES if s["name"] == "warnsdorff-NSWE")

        def reference(puzzle: Puzzle, sample_index: int):
            outcome = solve_with_tracks(puzzle.to_dict(), strategy)
            if outcome.get("grid") is None:
                # A non-solution is still a data point: it scores as an error/not-solved.
                return "```python\ndef solve(puzzle):\n    raise ValueError('reference failed')\n```", outcome.get("seconds", 0.0)
            return policies.program(outcome["grid"]), outcome.get("seconds", 0.0)

        return reference

    if name == "hf":
        return make_hf_backend(args)

    raise SystemExit(f"unknown backend {name!r}")


def make_hf_backend(args):
    """Load a causal LM once and sample `k` completions per prompt at temperature 1.0."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # noqa: BLE001 - documented as GPU-host-only
        raise SystemExit(
            f"the hf backend needs torch + transformers ({exc}); run it on the GPU host"
        ) from exc

    from tt_env.prompt import messages

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if args.fp16 else "auto",
        device_map="auto",
        trust_remote_code=True,
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    def hf(puzzle: Puzzle, sample_index: int):
        # Design A: one fixed prompt. The puzzle reaches the model only through the function
        # argument, so every puzzle in a split gets byte-identical prompt text.
        text = tokenizer.apply_chat_template(
            messages(), tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text=text, return_tensors="pt", add_special_tokens=False).to(model.device)
        started = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1] :]
        return tokenizer.decode(generated, skip_special_tokens=True), time.perf_counter() - started

    return hf


# --- main ----------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="empty", choices=["empty", "givens", "random", "reference", "hf"])
    ap.add_argument("--split", default="test_small")
    ap.add_argument("--puzzles", type=Path, default=ROOT / "data" / "puzzles.jsonl")
    ap.add_argument("--splits", type=Path, default=ROOT / "data" / "splits.json")
    ap.add_argument("--k", type=int, default=1, help="samples per puzzle (pass@k uses this)")
    ap.add_argument("--limit", type=int, default=0, help="first N puzzles of the split")
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--max-steps", type=int, default=None, help="sandbox step budget override")
    ap.add_argument("--model", default="unsloth/gemma-4-E2B-it")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--step", type=int, default=0, help="training step this checkpoint came from")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "eval")
    args = ap.parse_args()

    puzzles = {p.puzzle_id: p for p in load_jsonl(args.puzzles)}
    if args.splits.exists():
        declared = json.loads(args.splits.read_text("utf-8"))["splits"]
    else:
        raise SystemExit("missing data/splits.json; run scripts/freeze_splits.py first")
    if args.split not in declared:
        raise SystemExit(f"no split {args.split!r}; have {sorted(declared)}")

    ids = declared[args.split]
    if args.limit:
        ids = ids[: args.limit]
    selected = [puzzles[pid] for pid in ids if pid in puzzles]
    missing = [pid for pid in ids if pid not in puzzles]
    if not selected:
        raise SystemExit(f"split {args.split!r} has no puzzles in the corpus")

    rng = random.Random(args.seed)
    backend = make_backend(args.backend, args, rng)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.backend}-{args.split}-{stamp}"
    samples_path = args.out_dir / f"{stem}.jsonl"
    summary_path = args.out_dir / f"{stem}.json"

    records: list[dict] = []
    per_puzzle: dict[str, list[bool]] = defaultdict(list)
    wall: list[float] = []

    print(f"backend {args.backend}   split {args.split}   puzzles {len(selected)}   k {args.k}")
    if missing:
        print(f"warning: {len(missing)} split ids are not in the corpus, e.g. {missing[:3]}")

    with samples_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, puzzle in enumerate(selected, 1):
            for sample_index in range(args.k):
                completion, elapsed = backend(puzzle, sample_index)
                result = grade(completion, puzzle, max_steps=args.max_steps)
                wall.append(elapsed)
                record = {
                    "puzzle_id": puzzle.puzzle_id,
                    "rows": puzzle.rows,
                    "cols": puzzle.cols,
                    "difficulty": puzzle.difficulty or "?",
                    "sample_index": sample_index,
                    "backend": args.backend,
                    "model": args.model if args.backend == "hf" else args.backend,
                    "adapter": args.adapter,
                    "step": args.step,
                    "backend_seconds": round(elapsed, 4),
                    **result.to_dict(),
                }
                handle.write(json.dumps(record) + "\n")
                records.append(record)
                per_puzzle[puzzle.puzzle_id].append(result.status == "solved")
            if index % 25 == 0 or index == len(selected):
                solved = sum(sum(v) for v in per_puzzle.values())
                print(f"  {index}/{len(selected)} puzzles   {solved} solved samples", flush=True)

    total = len(records)
    solved_count = sum(1 for r in records if r["status"] == "solved")
    valid = sum(
        1 for r in records if r["format_ok"] == 0.5 and r["status"] not in {"cheat", "invalid", "syntax"}
    )
    cheat = sum(1 for r in records if r["status"] == "cheat")
    timeout = sum(1 for r in records if r["status"] == "timeout")
    shaped = [r["shaping"] for r in records if r["status"] != "solved"]

    def breakdown(key: str) -> dict:
        buckets: dict[str, list[dict]] = defaultdict(list)
        for record in records:
            buckets[str(record[key])].append(record)
        out = {}
        for name, rows in sorted(buckets.items()):
            ok = sum(1 for r in rows if r["status"] == "solved")
            out[name] = {
                "puzzles": len({r["puzzle_id"] for r in rows}),
                "samples": len(rows),
                "solve_rate": round(ok / len(rows), 4),
            }
        return out

    used = sorted({r["puzzle_id"] for r in records})
    low, high = wilson(solved_count, total)
    summary = {
        "backend": args.backend,
        "model": args.model if args.backend == "hf" else args.backend,
        "adapter": args.adapter,
        "step": args.step,
        "split": args.split,
        "seed": args.seed,
        "k": args.k,
        "temperature": args.temperature if args.backend == "hf" else 0.0,
        "puzzles": len(used),
        "samples": total,
        "solve_rate": round(solved_count / total, 4) if total else 0.0,
        "solve_rate_ci95": [round(low, 4), round(high, 4)],
        f"pass@{args.k}": round(pass_at_k(list(per_puzzle.values()), args.k), 4),
        "puzzles_solved": sum(1 for v in per_puzzle.values() if any(v)),
        "valid_program_rate": round(valid / total, 4) if total else 0.0,
        "cheat_rate": round(cheat / total, 4) if total else 0.0,
        "timeout_rate": round(timeout / total, 4) if total else 0.0,
        "mean_shaping": round(statistics.fmean(shaped), 4) if shaped else 0.0,
        "median_solve_seconds": (
            round(statistics.median(wall), 4) if wall else 0.0
        ),
        "by_size": breakdown("rows"),
        "by_difficulty": breakdown("difficulty"),
        "missing_from_corpus": missing,
        "samples_file": _display(samples_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("")
    print(f"  puzzles            {summary['puzzles']}   samples {summary['samples']}")
    print(f"  solve rate         {summary['solve_rate']:.1%}  (95% CI {low:.1%}-{high:.1%})")
    print(f"  pass@{args.k}            {summary[f'pass@{args.k}']:.1%}   puzzles solved {summary['puzzles_solved']}")
    print(f"  valid programs     {summary['valid_program_rate']:.1%}")
    print(f"  cheat / timeout    {summary['cheat_rate']:.1%} / {summary['timeout_rate']:.1%}")
    print(f"  mean shaping       {summary['mean_shaping']:.3f}")
    print(f"  median seconds     {summary['median_solve_seconds']:.3f}")
    for name, bucket in summary["by_size"].items():
        print(f"    rows {name:<3} {bucket['puzzles']:>4} puzzles  solve {bucket['solve_rate']:.1%}")
    print(f"\nsamples -> {samples_path}")
    print(f"summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
