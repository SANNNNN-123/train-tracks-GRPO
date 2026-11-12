"""Build a self-contained HTML report for one training run (training curves + predicted vs actual).

It reads the artefacts a run leaves behind:

    outputs/train_rewards_<utc>.jsonl|.txt   one JSON record per step (statuses, totals, shaping)
    outputs/test_small_partial.txt           eval records (one per puzzle x sample)
    outputs/train-track-grpo.log             notebook/trainer stdout (optional: wall clock, notes)

and writes `outputs/eval_report.html`, which needs no network and no plotting library.

The eval cell historically stored only diagnostics, not the returned grid. When a record has no
`grid`, the report reconstructs it *only* if the recorded diagnostics are identical to the ones the
verifier produces for the givens-only grid (the degenerate policy this run converged to), and labels
it as reconstructed. Future runs store the grid, so the panels are exact.

Usage:
    uv run python scripts/make_run_report.py
    uv run python scripts/make_run_report.py --eval path.jsonl --rewards path.jsonl --log path.log
"""

from __future__ import annotations

import argparse
import html
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tt_env import load_jsonl, verify  # noqa: E402
from tt_env.report import (  # noqa: E402
    AGREE,
    MISMATCH,
    SOLUTION,
    STATUS_COLOURS,
    compare,
    givens_cells,
    grid_cells,
    line_chart_svg,
    render_grid_svg,
    status_strip_svg,
)
from tt_env.rules import EMPTY  # noqa: E402

VALID_INVALID = {"no-code", "invalid", "cheat", "syntax"}


def normalise(value):
    """JSON round-trip with stringified dict keys: diagnostics are compared across formats."""
    if isinstance(value, dict):
        return {str(k): normalise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalise(v) for v in value]
    return value


def infer_prediction(puzzle, record) -> tuple[list[list[str]] | None, str]:
    """The eval record's grid if present; else the givens-only grid when diagnostics prove it."""
    if record.get("grid"):
        return grid_cells(record["grid"], puzzle.rows, puzzle.cols), "stored by the eval cell"
    base = givens_cells(puzzle)
    verdict = verify(puzzle, ["".join(row) for row in base])
    if normalise(verdict.diagnostics) == normalise(record.get("diagnostics", {})):
        return base, "reconstructed: diagnostics identical to the givens-only grid"
    return None, "not stored in the eval output, and not a givens-only grid"


def signature(record) -> str:
    return json.dumps(normalise(record.get("diagnostics", {})), sort_keys=True)


def moving_average(values: list[float], window: int = 10) -> list[float | None]:
    out: list[float | None] = []
    for i in range(len(values)):
        chunk = values[max(0, i - window + 1) : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def wall_clock(log_text: str) -> tuple[str, float]:
    stamps = [float(m) for m in re.findall(r"^(\d+(?:\.\d+)?)s ", log_text, flags=re.MULTILINE)]
    if not stamps:
        return "unknown", 0.0
    seconds = max(stamps)
    hours, rest = divmod(int(seconds), 3600)
    minutes = rest // 60
    return f"{hours}h {minutes:02d}m", seconds


def build_report(eval_rows, reward_rows, corpus, log_text: str) -> str:
    puzzles = {r["puzzle_id"] for r in eval_rows}
    groups: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for record in eval_rows:
        groups[record["puzzle_id"]][signature(record)].append(record)

    # --- training summary ---------------------------------------------------
    totals_by_step = [statistics.fmean(r["totals"]) for r in reward_rows if r["totals"]]
    shaped_by_step = [statistics.fmean(r["shaping"]) for r in reward_rows if r["shaping"]]
    status_counts = Counter(s for r in reward_rows for s in r["statuses"])
    solved_steps = [r["step"] for r in reward_rows if "solved" in r["statuses"]]
    valid_steps = sum(1 for r in reward_rows if all(s not in VALID_INVALID for s in r["statuses"]))
    last_total = statistics.fmean(totals_by_step[-20:]) if totals_by_step else 0.0
    first_total = statistics.fmean(totals_by_step[:20]) if totals_by_step else 0.0
    last_shape = statistics.fmean(shaped_by_step[-20:]) if shaped_by_step else 0.0
    wall, seconds = wall_clock(log_text)
    fps32 = log_text.count("float32")

    # --- eval summary -------------------------------------------------------
    solved_samples = sum(1 for r in eval_rows if r["status"] == "solved")
    solved_puzzles = sum(1 for pid in puzzles if any(r["status"] == "solved" for r in groups[pid][next(iter(groups[pid]))]))
    sample_shaping = [r["shaping"] for r in eval_rows]

    # How many predictions are the degenerate "copy the givens" grid? That is the headline finding.
    givens_only = 0
    for record in eval_rows:
        predicted, how = infer_prediction(corpus[record["puzzle_id"]], record)
        if predicted is not None and how.startswith("reconstructed"):
            givens_only += 1
    outliers = [
        r["puzzle_id"]
        for r in eval_rows
        if infer_prediction(corpus[r["puzzle_id"]], r)[1].startswith("not stored")
    ]

    chart = line_chart_svg(
        [
            {"name": "mean total (10-step avg)", "colour": "#0f172a", "values": moving_average(totals_by_step)},
            {"name": "solved+shaped (10-step avg)", "colour": "#ea580c", "values": moving_average(shaped_by_step)},
        ],
        y_min=0.0,
    )
    strip = status_strip_svg(reward_rows)

    # --- panels -------------------------------------------------------------
    panels = []
    for pid in sorted(puzzles):
        puzzle = corpus[pid]
        sigs = groups[pid]
        first_record = next(iter(sigs.values()))[0]
        total_values = sorted({round(r["total"], 3) for r in eval_rows if r["puzzle_id"] == pid})
        cards = []
        pred_panels = []
        for sig, records in sigs.items():
            predicted, how = infer_prediction(puzzle, records[0])
            samples = ", ".join(str(r["sample"]) for r in records)
            if predicted is None:
                pred_panels.append(
                    f'<div class="gridbox"><h4>Predicted (samples {samples})</h4>'
                    f'<p class="warn">grid {html.escape(how)}</p></div>'
                )
                continue
            actual = grid_cells(puzzle.reference_solution, puzzle.rows, puzzle.cols)
            stats = compare(predicted, actual)
            highlight = {}
            for r in range(puzzle.rows):
                for c in range(puzzle.cols):
                    if predicted[r][c] == actual[r][c]:
                        if predicted[r][c] != EMPTY:
                            highlight[(r, c)] = AGREE
                    else:
                        highlight[(r, c)] = MISMATCH
            pred_panels.append(
                f'<div class="gridbox"><h4>Predicted (samples {samples})</h4>'
                + render_grid_svg(puzzle, predicted, highlight=highlight, title=f"predicted {pid}")
                + f'<p class="note">{html.escape(how)}<br>'
                f'occupied {stats["occupied"]} of {stats["actual_occupied"]} · '
                f'cell accuracy {stats["cell_accuracy"]:.1%} · occupancy IoU {stats["occupied_iou"]:.1%}'
                f"</p></div>"
            )
        actual = grid_cells(puzzle.reference_solution, puzzle.rows, puzzle.cols)
        cards.append(
            f'<div class="gridbox"><h4>Puzzle</h4>'
            + render_grid_svg(puzzle, givens_cells(puzzle), title=f"puzzle {pid}")
            + f'<p class="note">{puzzle.rows}x{puzzle.cols} · {len(puzzle.givens)} givens</p></div>'
        )
        cards.extend(pred_panels)
        cards.append(
            f'<div class="gridbox"><h4>Actual (reference)</h4>'
            + render_grid_svg(puzzle, actual, base_colour=SOLUTION, title=f"actual {pid}")
            + f'<p class="note">verified solution · {sum(1 for row in actual for x in row if x != EMPTY)} occupied</p></div>'
        )
        panels.append(
            f"""<section class="panel">
  <div class="panelhead">
    <h3>{html.escape(pid)}</h3>
    <span class="badge" style="border-color:{STATUS_COLOURS.get(first_record['status'], '#94a3b8')}">
      {html.escape(first_record['status'])}</span>
    <span class="meta">total {', '.join(str(v) for v in total_values)} ·
      shaping {', '.join(sorted({str(round(r['shaping'], 3)) for r in eval_rows if r['puzzle_id'] == pid}))} ·
      {html.escape(first_record['detail'][:90])}</span>
  </div>
  <div class="grids">{''.join(cards)}</div>
</section>"""
        )

    status_rows = "".join(
        f'<tr><td><span class="dot" style="background:{STATUS_COLOURS.get(name, "#94a3b8")}"></span>{name}</td>'
        f"<td>{count}</td><td>{count / max(1, len(reward_rows) * 2):.1%}</td></tr>"
        for name, count in status_counts.most_common()
    )

    legend = (
        '<p class="legend">'
        f'<span class="sw" style="background:#1d4ed8"></span>given piece '
        f'<span class="sw" style="background:#ea580c"></span>model piece '
        f'<span class="sw" style="background:#15803d"></span>reference piece '
        f'<span class="sw" style="background:{MISMATCH}"></span>cell differs from reference '
        f'<span class="sw" style="background:{AGREE}"></span>matches reference</p>'
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Train Tracks GRPO - run report</title>
<style>
:root {{ color-scheme: light; }}
body {{ font: 14px/1.5 -apple-system, Segoe UI, Roboto, sans-serif; color:#0f172a; margin:0; background:#f8fafc; }}
main {{ max-width: 1240px; margin: 0 auto; padding: 28px; }}
h1 {{ font-size: 22px; margin: 0 0 4px; }}
h2 {{ font-size: 17px; margin: 34px 0 10px; }}
h3 {{ font-size: 15px; margin: 0; }}
h4 {{ font-size: 12px; text-transform: uppercase; letter-spacing: .04em; color:#475569; margin: 0 0 6px; }}
.sub {{ color:#64748b; margin-bottom: 18px; }}
.cards {{ display:flex; flex-wrap:wrap; gap:12px; }}
.card {{ background:#fff; border:1px solid #e2e8f0; border-radius:10px; padding:12px 16px; min-width:150px; }}
.card .v {{ font-size:22px; font-weight:600; }}
.card .k {{ color:#64748b; font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
.note {{ color:#64748b; font-size:12px; margin:6px 0 0; }}
.warn {{ color:#b91c1c; font-size:12px; }}
.panel {{ background:#fff; border:1px solid #e2e8f0; border-radius:12px; padding:14px 16px 6px; margin:12px 0; }}
.panelhead {{ display:flex; gap:12px; align-items:baseline; flex-wrap:wrap; border-bottom:1px solid #f1f5f9; padding-bottom:8px; }}
.badge {{ border:2px solid #94a3b8; border-radius:999px; padding:1px 9px; font-size:12px; }}
.meta {{ color:#64748b; font-size:12px; }}
.grids {{ display:flex; gap:22px; flex-wrap:wrap; padding:12px 0 6px; }}
.gridbox svg {{ background:#fff; border-radius:8px; }}
table {{ border-collapse: collapse; background:#fff; border:1px solid #e2e8f0; border-radius:10px; }}
th, td {{ padding:6px 12px; border-bottom:1px solid #f1f5f9; text-align:left; font-size:13px; }}
th {{ color:#475569; font-weight:600; }}
.dot {{ display:inline-block; width:9px; height:9px; border-radius:2px; margin-right:6px; }}
.legend {{ color:#475569; font-size:12px; }}
.sw {{ display:inline-block; width:11px; height:11px; border-radius:2px; margin:0 4px 0 14px; vertical-align:-1px; }}
ul {{ margin:6px 0 0 18px; }} li {{ margin:3px 0; }}
code {{ background:#f1f5f9; padding:1px 4px; border-radius:4px; }}
</style></head><body><main>

<h1>Train Tracks &times; Gemma 4 E2B &mdash; GRPO run report</h1>
<div class="sub">{len(reward_rows)} steps &middot; wall clock {wall} &middot; {len(eval_rows)} eval samples on
{len(puzzles)} puzzles &middot; adapter <code>gemma_4_tracks_lora</code></div>

<div class="cards">
  <div class="card"><div class="k">steps with a solve</div><div class="v">{len(solved_steps)} / {len(reward_rows)}</div></div>
  <div class="card"><div class="k">valid-program steps</div><div class="v">{valid_steps}/{len(reward_rows)}</div><div class="k">{valid_steps / max(1, len(reward_rows)):.1%}</div></div>
  <div class="card"><div class="k">mean total (first 20 &rarr; last 20)</div><div class="v">{first_total:.2f} &rarr; {last_total:.2f}</div></div>
  <div class="card"><div class="k">mean solved+shaped (last 20)</div><div class="v">{last_shape:.2f}</div></div>
  <div class="card"><div class="k">eval solve rate</div><div class="v">{solved_samples / max(1, len(eval_rows)):.0%}</div><div class="k">{solved_puzzles}/{len(puzzles)} puzzles</div></div>
  <div class="card"><div class="k">mean eval shaping</div><div class="v">{statistics.fmean(sample_shaping):.2f}</div></div>
</div>

<h2>Training signal</h2>
{chart}
<p class="note">Mean per step over the two generations. "total" includes format and no_cheat, so it
starts at ~0.5 even for a program that returns an empty grid. {fps32} log mentions of float32: Unsloth
could not use fp16 for this checkpoint, so the T4 trained in fp32 (slow).</p>
{strip}
<table><tr><th>status</th><th>samples</th><th>share</th></tr>{status_rows}</table>

<h2>Evaluation: predicted vs actual</h2>
<p class="note">Each row is one puzzle. Predicted panels are grouped by identical diagnostics, so a
deterministic policy shows one panel. Givens are blue; model pieces orange; the reference is green;
cells that differ are tinted red.</p>
{legend}
{''.join(panels)}

<h2>Reading this run</h2>
<ul>
<li><b>{len(solved_steps)} solves in {len(reward_rows)} steps.</b> The model produced valid, parseable
programs ({valid_steps / max(1, len(reward_rows)):.0%} of steps) but never a correct grid.</li>
<li><b>The policy converged on "copy the givens".</b> {givens_only} of {len(eval_rows)} eval predictions
have exactly as many occupied cells as the puzzle has givens
({', '.join(f"{html.escape(pid)}: {next(iter(groups[pid].values()))[0]['diagnostics'].get('occupied')}" for pid in sorted(puzzles))}), with
{statistics.fmean(sample_shaping):.2f} mean shaping. That is the degenerate policy the reward audit
predicted at 1.75&ndash;1.99 total.
{f'The other sample(s) came from {sorted(set(outliers))} and are not reconstructible (one cell, shaping 0).' if outliers else ''}</li>
<li><b>Why it is a local optimum.</b> Shaping pays for clue fit and reciprocal ports, and the givens
already supply some of both; any attempt that moves a given scores 0, and a slow search costs
&minus;0.5 on timeout. Doing nothing is the highest expected value once solving is out of reach.</li>
<li><b>Cost of a step was high</b> ({seconds / max(1, len(reward_rows)):.0f}s/step): fp32 instead of fp16 on
the T4, and up to 2048 completion tokens per generation. Steps 1&ndash;200 took {wall}.</li>
<li><b>Eval coverage is 4 puzzles &times; 4 samples</b> (LIMIT=4), so pass@4 = 0/4 is not the full
46-puzzle gate.</li>
</ul>

<h2>What to change next</h2>
<ul>
<li><b>Do not pay shaping for the givens-only grid.</b> Zero shaping when
<code>occupied &le; len(givens)</code>: every added cell still scores, but "do nothing" earns nothing.
This is the one degenerate case the cap did not cover. <em>Implemented after this run</em>: the audit
now scores givens-only at <b>1.50</b> (was 1.75&ndash;1.99), and a single extra cell still earns.</li>
<li><b>Store the grid in eval records</b> (now in <code>Grade.to_dict()</code>), so the next report is
exact rather than reconstructed.</li>
<li><b>Consider a small penalty for timeouts only after a real attempt</b>, or raise the step budget:
{status_counts.get('timeout', 0)} timeouts and {status_counts.get('error', 0)} runtime errors per
{len(reward_rows) * 2} samples make "attempt nothing" comparatively safe.</li>
<li><b>Run the full eval</b> (<code>LIMIT = None</code>) before judging G3; 4 puzzles is a smoke test.</li>
<li><b>bf16 hardware would roughly halve the step time</b> (L4/A100). On T4, expect fp32 and ~2&ndash;3
min/step at 2048 completion tokens.</li>
</ul>

<p class="note">Generated by <code>scripts/make_run_report.py</code> from
<code>outputs/train_rewards_*.jsonl</code>, <code>outputs/test_small_partial.txt</code> and
<code>outputs/train-track-grpo.log</code>.</p>
</main></body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--puzzles", type=Path, default=ROOT / "data" / "puzzles.jsonl")
    ap.add_argument("--eval", dest="eval_path", type=Path, default=ROOT / "outputs" / "test_small_partial.txt")
    ap.add_argument("--rewards", type=Path, default=None)
    ap.add_argument("--log", type=Path, default=ROOT / "outputs" / "train-track-grpo.log")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "eval_report.html")
    args = ap.parse_args()

    if args.rewards is None:
        candidates = sorted((ROOT / "outputs").glob("train_rewards_*.jsonl")) + sorted(
            (ROOT / "outputs").glob("train_rewards_*.txt")
        )
        if not candidates:
            raise SystemExit("no outputs/train_rewards_*.jsonl|.txt found")
        args.rewards = candidates[-1]

    corpus = {p.puzzle_id: p for p in load_jsonl(args.puzzles)}
    eval_rows = [json.loads(line) for line in args.eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    reward_rows = [json.loads(line) for line in args.rewards.read_text(encoding="utf-8").splitlines() if line.strip()]
    log_text = args.log.read_text(encoding="utf-8", errors="replace") if args.log.exists() else ""

    unknown = sorted({r["puzzle_id"] for r in eval_rows} - set(corpus))
    if unknown:
        raise SystemExit(f"eval puzzles not in the corpus: {unknown}")

    report = build_report(eval_rows, reward_rows, corpus, log_text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"eval {len(eval_rows)} samples / {len({r['puzzle_id'] for r in eval_rows})} puzzles")
    print(f"rewards {len(reward_rows)} steps from {args.rewards.name}")
    print(f"report -> {args.out}  ({args.out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
