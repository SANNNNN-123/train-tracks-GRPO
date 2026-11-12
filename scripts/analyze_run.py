"""Summarise the Kaggle run: training log, reward log and partial eval."""

import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(r"C:\Users\ZUHAIR\Desktop\2026\Train_Track_GRPO")
sys.path.insert(0, str(ROOT))
from tt_env import load_jsonl  # noqa: E402

# --- training log -----------------------------------------------------------
log = (ROOT / "outputs" / "train-track-grpo.log").read_text(encoding="utf-8", errors="replace")
lines = log.splitlines()
print(f"train log: {len(lines)} lines, {len(log)} chars")
print("first 3 non-debug lines:")
for line in lines:
    if "Debugger" not in line and line.strip():
        print("   ", line[:160])
        if len([1 for x in lines if "Debugger" not in x and x.strip()][:3]) == 0:
            break

reward_rows = []
for line in lines:
    m = re.search(r"'reward':\s*'([-\d.]+)'", line)
    if m:
        reward_rows.append(float(m.group(1)))
print("trainer reward rows found:", len(reward_rows))
if reward_rows:
    print("  first 5", reward_rows[:5], "last 5", reward_rows[-5:])

for marker in ("---- first", "Traceback", "Error", "OutOfMemory", "CUDA"):
    hits = [line for line in lines if marker in line]
    print(f"  {marker!r}: {len(hits)}")
    for line in hits[:5]:
        print("     ", line[:160])

# --- reward log -------------------------------------------------------------
rows = [json.loads(line) for line in (ROOT / "outputs" / "train_rewards_20261112T092308Z.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
print(f"\nreward log: {len(rows)} steps")
statuses = Counter(s for r in rows for s in r["statuses"])
print("  statuses:", dict(statuses.most_common()))
solved_steps = [r["step"] for r in rows if "solved" in r["statuses"]]
print("  steps with a solve:", solved_steps)
valid = sum(1 for r in rows if all(s not in {"no-code", "invalid", "cheat", "syntax"} for s in r["statuses"]))
print(f"  valid-program steps: {valid}/{len(rows)} = {valid/len(rows):.1%}")
totals = [t for r in rows for t in r["totals"]]
shaped = [s for r in rows for s in r["shaping"]]
def blocks(values, n=5):
    size = max(1, len(values) // n)
    return [round(statistics.fmean(values[i:i+size]), 3) for i in range(0, len(values), size)][:n]
print("  mean total by fifth:", blocks(totals))
print("  mean shaping by fifth:", blocks(shaped))
print("  max total:", max(totals), " max shaping:", max(shaped))
nonzero_shaping = sum(1 for s in shaped if s > 0)
print(f"  samples with shaping > 0: {nonzero_shaping}/{len(shaped)}")

# --- eval -------------------------------------------------------------------
eval_rows = [json.loads(line) for line in (ROOT / "outputs" / "test_small_partial.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
print(f"\neval: {len(eval_rows)} records, {len({r['puzzle_id'] for r in eval_rows})} puzzles")
print("  statuses:", dict(Counter(r["status"] for r in eval_rows)))
print("  solved:", sum(1 for r in eval_rows if r["status"] == "solved"))
for pid in sorted({r["puzzle_id"] for r in eval_rows}):
    group = [r for r in eval_rows if r["puzzle_id"] == pid]
    first = group[0]
    print(f"  {pid:<18} samples {len(group)} shaping {first['shaping']} occupied {first['diagnostics'].get('occupied')} "
          f"components {first['diagnostics'].get('components')} unique_totals {sorted({r['total'] for r in group})}")

corpus = {p.puzzle_id: p for p in load_jsonl(ROOT / "data" / "puzzles.jsonl")}
print("\ncorpus check (givens count vs eval 'occupied'):")
for pid in sorted({r["puzzle_id"] for r in eval_rows}):
    puzzle = corpus[pid]
    first = next(r for r in eval_rows if r["puzzle_id"] == pid)
    occ = first["diagnostics"].get("occupied")
    print(f"  {pid:<18} {puzzle.rows}x{puzzle.cols} givens {len(puzzle.givens):>2} occupied {occ:>2} "
          f"{'== givens-only' if occ == len(puzzle.givens) else 'differs'}")
