"""Create a manually reviewable HTML page for contradiction-detection examples.

One card per record in data/sft_contradictions_train.jsonl:

  * the invalid board with the offender highlighted in red
  * the verified reference solution for the same puzzle
  * the violated rule, the evidence, and the expected answer

Writes outputs/sft_contradictions_review.html.
"""

from __future__ import annotations

import html
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from tt_env.puzzle import load_jsonl
from tt_env.report import render_grid_svg
from tt_env.rl import load_split
from tt_env.rules import EMPTY, OFFSETS, OPPOSITE, verify

DATA = "data/sft_contradictions_train.jsonl"
PUZZLES = "data/puzzles.jsonl"
SPLITS = "data/splits.json"
SPLIT_NAMES = ("train", "validation", "test_small", "test_medium")
RULE_ORDER = ("clue_mismatch", "port_reciprocity")
DIR_NAME = {"N": "north", "E": "east", "S": "south", "W": "west"}

records = [json.loads(line) for line in
           Path(DATA).read_text(encoding="utf-8").splitlines() if line]
puzzles = {puzzle.puzzle_id: puzzle for puzzle in load_jsonl(PUZZLES)}

split_of: dict[str, str] = {}
for name in SPLIT_NAMES:
    for puzzle in load_split(PUZZLES, SPLITS, name):
        split_of[puzzle.puzzle_id] = name


def reference_cells(puzzle) -> list[list[str]]:
    return [[puzzle.reference_solution[r][2 * c : 2 * c + 2] for c in range(puzzle.cols)]
            for r in range(puzzle.rows)]


def cells_from_board(record, puzzle) -> list[list[str]]:
    lines = record["input"]["board"].splitlines()[1:]
    if len(lines) != puzzle.rows:
        raise ValueError(f"{record['puzzle_id']}: board has {len(lines)} rows, expected {puzzle.rows}")
    cells = []
    for line in lines:
        tokens = line.split()
        if len(tokens) < puzzle.cols:
            raise ValueError(f"{record['puzzle_id']}: short board row {line!r}")
        cells.append(tokens[-puzzle.cols:])
    return cells


def invalid_puzzle(record, puzzle):
    """The puzzle as shown in the record: a clue_mismatch rewrites one clue."""
    if record["rule"] != "clue_mismatch":
        return puzzle
    evidence = record["action"]["evidence"]
    row_clues = list(puzzle.row_clues)
    col_clues = list(puzzle.col_clues)
    if evidence["axis"] == "row":
        row_clues[evidence["index"]] = evidence["clue"]
    else:
        col_clues[evidence["index"]] = evidence["clue"]
    return replace(puzzle, row_clues=tuple(row_clues), col_clues=tuple(col_clues))


def classify(puzzle, cells) -> dict[str, list[str]]:
    verdict = verify(puzzle, ["".join(row) for row in cells])
    groups: dict[str, list[str]] = {"givens": [], "clue": [], "boundary": [], "port": [], "other": []}
    for failure in verdict.failures:
        if failure.startswith("givens:"):
            groups["givens"].append(failure)
        elif failure.startswith("clue_"):
            groups["clue"].append(failure)
        elif failure.startswith("boundary:"):
            groups["boundary"].append(failure)
        elif failure.startswith("dangling:"):
            groups["port"].append(failure)
        else:
            groups["other"].append(failure)
    return groups


def board_svg(puzzle, cells, highlights, *, clue=None, clue_colours=("#fecaca", "#dc2626"),
              edges=()) -> str:
    """Render one board, then stamp clue/edge overlays on top."""
    cell_size = 420 / max(puzzle.rows, puzzle.cols)
    svg = render_grid_svg(puzzle, cells, cell=cell_size, highlight=highlights,
                          row_clues_right=True, bold_clues=True,
                          row_labels=True, col_labels=True)
    left, top = 8, 24
    back, front = [], []

    if clue is not None:
        axis, index = clue
        fill, stroke = clue_colours
        if axis == "row":
            x = left + puzzle.cols * cell_size + 3
            y = top + index * cell_size + cell_size / 2 - 9
            width, height = cell_size * 0.45, 18
        else:
            x = left + (index + 0.5) * cell_size - cell_size * 0.3
            y = top - 20
            width, height = cell_size * 0.6, 16
        back.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
                    f'rx="3" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')

    for sr, sc, direction in edges:
        dr, dc = OFFSETS[direction]
        x = left + (sc + 0.5) * cell_size + dc * cell_size / 2
        y = top + (sr + 0.5) * cell_size + dr * cell_size / 2
        radius = cell_size * 0.13
        front.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" '
                     f'fill="#fff" stroke="#dc2626" stroke-width="2.2"/>')
        front.append(f'<text x="{x:.1f}" y="{y + radius * 0.55:.1f}" text-anchor="middle" '
                     f'font-size="{radius * 1.3:.1f}" font-weight="700" fill="#dc2626">✕</text>')

    if back:
        svg = svg.replace(">", ">" + "".join(back), 1)
    if front:
        svg = svg.replace("</svg>", "".join(front) + "</svg>")
    return svg


def edge_rows(record, cells) -> tuple[str, int, int]:
    """Table of unanswered ports, plus (empty-neighbour, track-neighbour) counts."""
    evidence = record["action"]["evidence"]
    source = tuple(evidence["cell"])
    puzzle = puzzles[record["puzzle_id"]]
    rows = []
    empty_hits = track_hits = 0
    for sr, sc, direction in evidence["edges"]:
        nr, nc = sr + OFFSETS[direction][0], sc + OFFSETS[direction][1]
        neighbour = cells[nr][nc] if 0 <= nr < puzzle.rows and 0 <= nc < puzzle.cols else EMPTY
        if neighbour == EMPTY:
            empty_hits += 1
            status = "neighbour is empty"
        else:
            track_hits += 1
            status = f"no {DIR_NAME[OPPOSITE[direction]]} port"
        role = "mutated piece" if (sr, sc) == source else "neighbour"
        rows.append(f'<tr><td>r{sr + 1}c{sc + 1}</td><td>{role}</td>'
                    f'<td>{DIR_NAME[direction]}</td><td>r{nr + 1}c{nc + 1}</td>'
                    f'<td>{html.escape(neighbour)}</td><td>{html.escape(status)}</td></tr>')
    return "".join(rows), empty_hits, track_hits


cards = []
summary = Counter()
bad_records = []
edge_histogram = Counter()
clue_axis = Counter()

for index, record in enumerate(records):
    puzzle = puzzles[record["puzzle_id"]]
    rule = record["rule"]
    split = split_of.get(record["puzzle_id"], "?")
    cells = cells_from_board(record, puzzle)
    shown_puzzle = invalid_puzzle(record, puzzle)
    groups = classify(shown_puzzle, cells)

    if rule == "clue_mismatch":
        valid = (len(groups["clue"]) == 1 and not groups["givens"]
                 and not groups["boundary"] and not groups["port"])
        clue_axis[record["action"]["evidence"]["axis"]] += 1
        edge_histogram[0] += 1
    else:
        valid = (len(groups["port"]) == 1 and not groups["givens"]
                 and not groups["clue"] and not groups["boundary"])
        edge_histogram[len(record["action"]["evidence"]["edges"])] += 1
    summary[rule] += 1
    if not valid:
        bad_records.append((record["puzzle_id"], rule, {k: len(v) for k, v in groups.items()}))

    reference = reference_cells(puzzle)
    evidence = record["action"]["evidence"]

    if rule == "clue_mismatch":
        axis, line_index = evidence["axis"], evidence["index"]
        line_cells = ([(line_index, c) for c in range(puzzle.cols)] if axis == "row"
                      else [(r, line_index) for r in range(puzzle.rows)])
        invalid_svg = board_svg(
            shown_puzzle, cells,
            {cell: "#fee2e2" for cell in line_cells},
            clue=(axis, line_index),
            clue_colours=("#fecaca", "#dc2626"),
        )
        reference_svg = board_svg(
            puzzle, reference,
            {cell: "#dcfce7" for cell in line_cells},
            clue=(axis, line_index),
            clue_colours=("#dcfce7", "#15803d"),
        )
        word = "row" if axis == "row" else "column"
        invalid_caption = (f"{word.title()} {line_index + 1} clue says "
                           f"{evidence['clue']} (board has {evidence['actual']})")
        reference_caption = (f"{word.title()} {line_index + 1} clue correctly says "
                             f"{evidence['actual']}")
        detail_rows = (
            f'<tr><th>Violated rule</th><td>clue_mismatch</td></tr>'
            f'<tr><th>Offending line</th><td>{word.title()} {line_index + 1}</td></tr>'
            f'<tr><th>Clue shown</th><td>{evidence["clue"]}</td></tr>'
            f'<tr><th>Tracks on the board</th><td>{evidence["actual"]}</td></tr>'
            f'<tr><th>Expected answer</th><td><code>{html.escape(record["target"])}</code></td></tr>'
        )
        extra = ""
    else:
        cell = tuple(evidence["cell"])
        r, c = cell
        dangling_cells = set()
        for sr, sc, direction in evidence["edges"]:
            dangling_cells.add((sr, sc))
            nr, nc = sr + OFFSETS[direction][0], sc + OFFSETS[direction][1]
            if 0 <= nr < puzzle.rows and 0 <= nc < puzzle.cols:
                dangling_cells.add((nr, nc))
        invalid_svg = board_svg(
            puzzle, cells,
            {cell: "#fecaca", **{other: "#ffedd5" for other in dangling_cells if other != cell}},
            edges=evidence["edges"],
        )
        reference_svg = board_svg(puzzle, reference, {cell: "#dcfce7"})
        invalid_caption = (f"r{r + 1}c{c + 1} rotated {evidence['was']} → {evidence['now']}; "
                           f"{len(evidence['edges'])} unanswered port(s)")
        reference_caption = f"r{r + 1}c{c + 1} correctly carries {evidence['was']}"
        rows, empty_hits, track_hits = edge_rows(record, cells)
        detail_rows = (
            f'<tr><th>Violated rule</th><td>port_reciprocity</td></tr>'
            f'<tr><th>Offending cell</th><td>r{r + 1}c{c + 1}: '
            f'{html.escape(evidence["was"])} → {html.escape(evidence["now"])}</td></tr>'
            f'<tr><th>Unanswered ports</th><td>{len(evidence["edges"])} '
            f'({empty_hits} into empty cells, {track_hits} at tracks without the reciprocal port)</td></tr>'
            f'<tr><th>Expected answer</th><td><code>{html.escape(record["target"])}</code></td></tr>'
        )
        extra = (f'<h3>Unanswered ports</h3><table class="edges"><thead><tr><th>Source</th>'
                 f'<th>Role</th><th>Direction</th><th>Neighbour</th><th>Piece</th>'
                 f'<th>Why it fails</th></tr></thead><tbody>{rows}</tbody></table>')

    badge = "ok" if valid else "bad"
    badge_text = "exactly one local rule broken" if valid else "CHECK: extra rule failure"
    edge_count = 0 if rule == "clue_mismatch" else len(evidence["edges"])
    cards.append(f'''<article class="card" data-puzzle="{html.escape(record["puzzle_id"])}" data-rule="{html.escape(rule)}" data-split="{html.escape(split)}" data-edges="{edge_count}" data-valid="{1 if valid else 0}">
<h2>{html.escape(record["puzzle_id"])} <span class="split">{html.escape(split)}</span> <span class="rule {html.escape(rule)}">{html.escape(rule)}</span> <span class="badge {badge}">{badge_text}</span></h2>
<p class="clues">Rows: {html.escape(str(list(shown_puzzle.row_clues)))} · Columns: {html.escape(str(list(shown_puzzle.col_clues)))}</p>
<div class="boards">
<section><h3>Invalid board</h3><div class="visual">{invalid_svg}</div><p class="caption bad">{html.escape(invalid_caption)}</p></section>
<section><h3>Reference solution</h3><div class="visual">{reference_svg}</div><p class="caption good">{html.escape(reference_caption)}</p></section>
</div>
<h3>Why it is invalid</h3><p>{html.escape(record["action"]["explanation"])}</p>
<table class="details">{detail_rows}</table>{extra}</article>''')

rule_counts = {rule: summary[rule] for rule in RULE_ORDER}
rule_options = "".join(
    f'<option value="{rule}">{rule} ({rule_counts[rule]})</option>'
    for rule in RULE_ORDER if rule_counts[rule]
)
split_options = "".join(
    f'<option value="{name}">{name} ({sum(1 for record in records if split_of.get(record["puzzle_id"]) == name)})</option>'
    for name in SPLIT_NAMES
    if any(split_of.get(record["puzzle_id"]) == name for record in records)
)

valid_count = len(records) - len(bad_records)
edge_summary = " · ".join(f"{count} unanswered ports: {total}"
                          for count, total in sorted(edge_histogram.items()) if count)
axis_summary = " · ".join(f"{name}: {total}" for name, total in sorted(clue_axis.items()))

page = '''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Train Tracks contradiction review</title><style>
body{font:16px system-ui;margin:2rem auto;padding:0 1rem;max-width:1180px;background:#f4f6f8;color:#17202a}
h1{margin-bottom:.3rem}h2{margin:.2rem 0 .4rem}h3{margin:.9rem 0 .4rem;font-size:15px}
.toolbar{position:sticky;top:0;background:#f4f6f8;padding:.8rem 0;display:flex;gap:1rem;align-items:center;flex-wrap:wrap;z-index:2;border-bottom:1px solid #d7dde3}
.card{background:white;border:1px solid #ccd3da;border-radius:10px;padding:1rem;margin:1rem 0}
.boards{display:flex;gap:2rem;flex-wrap:wrap}.boards section{flex:1;min-width:340px}
.visual{width:min(100%,460px);height:460px;display:flex;align-items:center;justify-content:center}
.visual svg{max-width:100%;max-height:100%}
.clues{color:#52616b;margin:.2rem 0 .6rem}
.caption{font-size:13px;margin:.3rem 0 0;font-weight:600}.caption.bad{color:#b91c1c}.caption.good{color:#15803d}
.split{font-size:12px;font-weight:600;color:#475569;background:#eef1f4;border-radius:10px;padding:2px 8px;vertical-align:middle}
.rule{font-size:12px;font-weight:700;color:#fff;border-radius:10px;padding:2px 9px;vertical-align:middle}
.rule.clue_mismatch{background:#b91c1c}.rule.port_reciprocity{background:#7c3aed}
.badge{display:inline-block;font-size:12px;font-weight:700;border-radius:10px;padding:2px 9px;vertical-align:middle}
.badge.ok{background:#dcfce7;color:#166534}.badge.bad{background:#fee2e2;color:#b91c1c}
table{border-collapse:collapse;width:100%;margin-top:.6rem;font-size:14px}
th,td{border:1px solid #d7dde3;padding:.35rem .5rem;text-align:left}th{background:#f1f5f9}
table.details{max-width:960px}table.details th{width:190px}
table.edges th{background:#faf5ff}
.summary{border-left:6px solid #7c3aed;background:#faf5ff;padding:1rem 1.2rem;border-radius:8px;margin:1rem 0}
.summary b{color:#6d28d9}.warn{border-left:6px solid #b91c1c;background:#fef2f2;padding:.8rem 1.2rem;border-radius:8px;margin:1rem 0}
.legend{color:#52616b;font-size:14px}.legend code{background:#eef1f4;padding:1px 4px;border-radius:3px}
button,select{font:inherit;padding:.4rem}h1 span{font-size:15px;color:#52616b;font-weight:400}
table.counts{max-width:820px}
</style>
<h1>Train Tracks contradiction review <span>· invalid boards from verified solutions</span></h1>
<p>Each card shows a completed board that breaks <b>exactly one local rule</b>, the verified reference solution for the same puzzle, and the evidence for the violation. Port breaks also list every unanswered port; connectivity and degree failures are accepted consequences of the broken port.</p>
<div class="summary"><b>Dataset:</b> ''' + f"{len(records):,} records from {len({r['puzzle_id'] for r in records})} training puzzles · " + " · ".join(f"{rule}: {rule_counts[rule]}" for rule in RULE_ORDER) + '''. <b>Validation:</b> ''' + f"{valid_count:,} / {len(records):,} break exactly one local rule." + '''</div>
''' + ("" if not bad_records else '<div class="warn"><b>Validation warnings:</b> ' + html.escape(str(bad_records[:5])) + '</div>') + '''
<h2>Counts</h2><table class="counts"><tbody>
<tr><th>Records</th><td>''' + f"{len(records):,}" + '''</td></tr>
<tr><th>clue_mismatch</th><td>''' + f"{rule_counts['clue_mismatch']:,}" + '''</td></tr>
<tr><th>port_reciprocity</th><td>''' + f"{rule_counts['port_reciprocity']:,}" + '''</td></tr>
<tr><th>Port breaks by unanswered ports</th><td>''' + html.escape(edge_summary or "—") + '''</td></tr>
<tr><th>Clue breaks by axis</th><td>''' + html.escape(axis_summary or "—") + '''</td></tr>
<tr><th>Records with an extra local-rule failure</th><td>''' + f"{len(bad_records)}" + '''</td></tr>
</tbody></table>
<p class="legend">Legend: red cell = mutated piece · orange cells = cells touching an unanswered port · red ✕ = the exact edge that does not match · red clue box = the wrong clue · green clue box = the correct clue. Given pieces are blue, other track pieces orange.</p>
<div class="toolbar"><label>Rule <select id="rule"><option value="">All rules</option>''' + rule_options + '''</select></label>
<label>Puzzle <select id="puzzle"><option value="">All puzzles</option></select></label>
<label>Split <select id="split"><option value="">All splits</option>''' + split_options + '''</select></label>
<label>Port break <select id="edges"><option value="">Any size</option><option value="2">2 unanswered ports</option><option value="4">4 unanswered ports</option></select></label>
<span id="count"></span></div>
''' + "\n".join(cards) + '''
<script>
const cards=[...document.querySelectorAll('.card')],ruleSelect=document.querySelector('#rule'),puzzleSelect=document.querySelector('#puzzle'),splitSelect=document.querySelector('#split'),edgeSelect=document.querySelector('#edges'),count=document.querySelector('#count');
[...new Set(cards.map(c=>c.dataset.puzzle))].sort().forEach(id=>{const n=cards.filter(c=>c.dataset.puzzle===id).length,o=document.createElement('option');o.value=id;o.textContent=id+' ('+n+')';puzzleSelect.append(o)});
function show(){let n=0;cards.forEach(c=>{c.hidden=!!(ruleSelect.value&&c.dataset.rule!==ruleSelect.value)||!!(puzzleSelect.value&&c.dataset.puzzle!==puzzleSelect.value)||!!(splitSelect.value&&c.dataset.split!==splitSelect.value)||!!(edgeSelect.value&&c.dataset.edges!==edgeSelect.value);if(!c.hidden)n++});count.textContent=n+' cards'}
ruleSelect.addEventListener('change',show);puzzleSelect.addEventListener('change',show);splitSelect.addEventListener('change',show);edgeSelect.addEventListener('change',show);show();
</script>'''

dest = Path("outputs/sft_contradictions_review.html")
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(page, encoding="utf-8")
print(f"wrote {len(cards)} cards to {dest.resolve()}")
print(f"  {dict(rule_counts)}")
print(f"  valid: {valid_count}/{len(records)}; warnings: {len(bad_records)}")
print(f"  port breaks by edges: {dict(sorted(edge_histogram.items()))}")
print(f"  clue axis: {dict(clue_axis)}")
