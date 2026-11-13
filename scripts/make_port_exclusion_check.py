"""Check the proposed 'corner port exclusion marks the neighbour X' rule.

Theory: for a given corner piece, every side without a port means the
immediately adjacent cell must be empty (X).

Ground truth: the verified reference solutions in data/puzzles.jsonl (built
from the images with track_solver.py + the reference DFS solver).

One card per puzzle: reference board next to the theory board, plus every
corner slot with its verdict. Writes outputs/port_exclusion_check.html.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from tt_env import load_jsonl
from tt_env.report import render_grid_svg
from tt_env.rl import load_split
from tt_env.rules import EMPTY, OFFSETS, OPPOSITE, dirs_of

CORPUS = "data/puzzles.jsonl"
SPLITS = "data/splits.json"
SPLIT_NAMES = ("train", "validation", "test_small", "test_medium")
BENDS = frozenset({"NE", "NW", "SE", "SW"})
DIR_NAME = {"N": "north", "E": "east", "S": "south", "W": "west"}

# Cards the user named during the discussion, shown first.
NAMED = ["01022026-small", "02022026-small", "11012026-small", "20012026-small",
         "31072026-small"]

puzzles = load_jsonl(CORPUS)
split_of: dict[str, str] = {}
for name in SPLIT_NAMES:
    for puzzle in load_split(CORPUS, SPLITS, name):
        split_of[puzzle.puzzle_id] = name


def reference_cells(puzzle) -> list[list[str]]:
    return [[puzzle.reference_solution[r][2 * c : 2 * c + 2] for c in range(puzzle.cols)]
            for r in range(puzzle.rows)]


def empty_cells(puzzle) -> list[list[str]]:
    cells = [[EMPTY] * puzzle.cols for _ in range(puzzle.rows)]
    for r, c, piece in puzzle.givens:
        cells[r][c] = piece
    return cells


def scan(puzzle):
    """All corner slots of this puzzle, plus reciprocal-port sanity checks."""
    sol = reference_cells(puzzle)
    ends = {tuple(puzzle.start), tuple(puzzle.end)}
    slots = []
    reciprocal_checked = 0
    reciprocal_bad = []
    for r, c, piece in puzzle.givens:
        ports = set(dirs_of(piece))
        for direction in "NESW":
            nr, nc = r + OFFSETS[direction][0], c + OFFSETS[direction][1]
            if not (0 <= nr < puzzle.rows and 0 <= nc < puzzle.cols):
                continue
            truth = sol[nr][nc]
            if direction in ports:
                reciprocal_checked += 1
                if truth == EMPTY or OPPOSITE[direction] not in dirs_of(truth):
                    reciprocal_bad.append((r, c, piece, direction, nr, nc, truth))
            elif piece in BENDS:
                slots.append({
                    "source": (r, c), "piece": piece, "ports": ports,
                    "missing": direction, "target": (nr, nc), "truth": truth,
                    "correct": truth == EMPTY,
                    "endpoint": (r, c) in ends,
                })
    return slots, reciprocal_checked, reciprocal_bad


scans = {p.puzzle_id: scan(p) for p in puzzles}

# ---------------------------------------------------------------------------
# Summary numbers
# ---------------------------------------------------------------------------

def summary(predicate, scope: str):
    slots = wrong = 0
    puzzles_wrong = set()
    for puzzle in puzzles:
        if scope != "all" and split_of.get(puzzle.puzzle_id) != scope:
            continue
        for slot in scans[puzzle.puzzle_id][0]:
            if not predicate(slot):
                continue
            slots += 1
            if not slot["correct"]:
                wrong += 1
                puzzles_wrong.add(puzzle.puzzle_id)
    return slots, wrong, puzzles_wrong


def pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f}%" if whole else "—"


def row(label: str, slots: int, wrong: int, puzzles_wrong) -> str:
    colour = "#15803d" if wrong == 0 else "#b91c1c"
    return (f"<tr><td>{html.escape(label)}</td><td>{slots}</td><td>{slots - wrong}</td>"
            f"<td style=\"color:{colour};font-weight:700\">{wrong}</td>"
            f"<td>{pct(wrong, slots)}</td><td>{len(puzzles_wrong)}</td></tr>")


summary_rows = [
    row("All puzzles · corner givens", *summary(lambda s: True, "all")),
    row("Train · corner givens", *summary(lambda s: True, "train")),
    row("Validation · corner givens", *summary(lambda s: True, "validation")),
    row("Test small · corner givens", *summary(lambda s: True, "test_small")),
    row("Test medium · corner givens", *summary(lambda s: True, "test_medium")),
    row("All puzzles · endpoint corner givens", *summary(lambda s: s["endpoint"], "all")),
    row("Train · endpoint corner givens", *summary(lambda s: s["endpoint"], "train")),
]

all_slots, all_wrong, all_puzzles_wrong = summary(lambda s: True, "all")
train_slots, train_wrong, train_puzzles_wrong = summary(lambda s: True, "train")
reciprocal_checked = sum(v[1] for v in scans.values())
reciprocal_bad = [(pid, v[2]) for pid, v in scans.items() if v[2]]

# Same rule widened to straight givens, for the note in the verdict box.
widened_slots = widened_wrong = 0
for puzzle in puzzles:
    sol = reference_cells(puzzle)
    for r, c, piece in puzzle.givens:
        ports = set(dirs_of(piece))
        for direction in "NESW":
            if direction in ports:
                continue
            nr, nc = r + OFFSETS[direction][0], c + OFFSETS[direction][1]
            if not (0 <= nr < puzzle.rows and 0 <= nc < puzzle.cols):
                continue
            widened_slots += 1
            widened_wrong += sol[nr][nc] != EMPTY

# ---------------------------------------------------------------------------
# SVG helpers
# ---------------------------------------------------------------------------

def board_svg(puzzle, cells, highlights, marks=None):
    cell_size = 420 / max(puzzle.rows, puzzle.cols)
    svg = render_grid_svg(puzzle, cells, cell=cell_size, highlight=highlights,
                          row_clues_right=True, bold_clues=True,
                          row_labels=True, col_labels=True)
    glyphs = []
    for r, c, value, correct in marks or []:
        x = 8 + (c + .5) * cell_size
        y = 24 + (r + .5) * cell_size
        colour = "#7c3aed" if correct else "#dc2626"
        glyphs.append(f'<text x="{x:.1f}" y="{y + 6:.1f}" text-anchor="middle" '
                      f'font-size="19" font-weight="700" fill="{colour}">X</text>')
    return svg.replace("</svg>", "".join(glyphs) + "</svg>")


def slot_row(slot) -> str:
    (r, c), (nr, nc) = slot["source"], slot["target"]
    ports = ", ".join(DIR_NAME[d] for d in "NESW" if d in slot["ports"])
    verdict = ('<span class="ok">theory holds</span>' if slot["correct"]
               else '<span class="bad">contradicted by solution</span>')
    reference = "empty" if slot["truth"] == EMPTY else f'track {slot["truth"]}'
    return (f'<tr class="{"good" if slot["correct"] else "fail"}">'
            f'<td>r{r + 1}c{c + 1} = {html.escape(slot["piece"])}</td>'
            f'<td>{html.escape(ports)}</td><td>{DIR_NAME[slot["missing"]]}</td>'
            f'<td>r{nr + 1}c{nc + 1}</td><td>X</td>'
            f'<td>{html.escape(reference)}</td><td>{verdict}</td></tr>')


# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------

cards = []
order = []
for puzzle in puzzles:
    slots = scans[puzzle.puzzle_id][0]
    if not slots:
        continue
    wrong = sum(1 for s in slots if not s["correct"])
    named_rank = NAMED.index(puzzle.puzzle_id) if puzzle.puzzle_id in NAMED else 999
    order.append((named_rank, -wrong, puzzle.puzzle_id, puzzle, slots, wrong))
order.sort(key=lambda item: item[:3])

for _, _, puzzle_id, puzzle, slots, wrong in order:
    ref = reference_cells(puzzle)
    theory = empty_cells(puzzle)
    ref_highlights, theory_highlights, marks = {}, {}, []
    for slot in slots:
        colour = "#dcfce7" if slot["correct"] else "#fee2e2"
        ref_highlights[slot["target"]] = colour
        theory_highlights[slot["target"]] = "#ede9fe" if slot["correct"] else "#fee2e2"
        marks.append((*slot["target"], "X", slot["correct"]))
    ref_svg = board_svg(puzzle, ref, ref_highlights)
    theory_svg = board_svg(puzzle, theory, theory_highlights, marks)
    badge = ('<span class="badge ok">all corner X\'s correct here</span>' if not wrong
             else f'<span class="badge bad">{wrong} of {len(slots)} corner X\'s wrong</span>')
    split = split_of.get(puzzle_id, "?")
    rows = "".join(slot_row(s) for s in slots)
    cards.append(f'''<article class="card" data-puzzle="{html.escape(puzzle_id)}" data-split="{html.escape(split)}" data-wrong="{wrong}">
<h2>{html.escape(puzzle_id)} <span class="split">{html.escape(split)}</span> {badge}</h2>
<p class="clues">Rows: {html.escape(str(list(puzzle.row_clues)))} · Columns: {html.escape(str(list(puzzle.col_clues)))}</p>
<div class="boards">
<section><h3>Reference solution</h3><div class="visual">{ref_svg}</div>
<p class="note">Green = the theory's target cell is empty. Red = it is a track, so the X is wrong.</p></section>
<section><h3>Port-exclusion theory</h3><div class="visual">{theory_svg}</div>
<p class="note">Purple X = matches the solution. Red X = contradicts the solution.</p></section>
</div>
<table class="slots"><thead><tr><th>Corner given</th><th>Ports</th><th>Missing side</th><th>Target</th><th>Theory</th><th>Reference</th><th>Verdict</th></tr></thead>
<tbody>{rows}</tbody></table></article>''')

split_options = "".join(
    f'<option value="{name}">{name}</option>' for name in SPLIT_NAMES
)

page = '''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Port exclusion check · corner givens vs solutions</title><style>
body{font:16px system-ui;margin:2rem auto;padding:0 1rem;max-width:1180px;background:#f4f6f8;color:#17202a}
h1{margin-bottom:.2rem}h2{margin:.2rem 0 .4rem}h3{margin:.2rem 0 .5rem;font-size:15px}
.verdict{border-left:6px solid #b91c1c;background:#fef2f2;padding:1rem 1.2rem;border-radius:8px;margin:1rem 0}
.verdict b{color:#b91c1c}
.toolbar{position:sticky;top:0;background:#f4f6f8;padding:.8rem 0;display:flex;gap:1rem;align-items:center;flex-wrap:wrap;z-index:2;border-bottom:1px solid #d7dde3}
.card{background:white;border:1px solid #ccd3da;border-radius:10px;padding:1rem;margin:1rem 0}
.boards{display:flex;gap:2rem;flex-wrap:wrap}.boards section{flex:1;min-width:340px}
.visual{width:min(100%,460px);height:460px;display:flex;align-items:center;justify-content:center}
.visual svg{max-width:100%;max-height:100%}
.clues{color:#52616b;margin:.2rem 0 .6rem}.note{color:#52616b;font-size:13px;margin:.3rem 0 0}
.split{font-size:12px;font-weight:600;color:#475569;background:#eef1f4;border-radius:10px;padding:2px 8px;vertical-align:middle}
.badge{display:inline-block;font-size:12px;font-weight:700;border-radius:10px;padding:2px 9px;vertical-align:middle}
.badge.ok{background:#dcfce7;color:#166534}.badge.bad{background:#fee2e2;color:#b91c1c}
table{border-collapse:collapse;width:100%;margin-top:.8rem;font-size:14px}
th,td{border:1px solid #d7dde3;padding:.35rem .5rem;text-align:left}th{background:#f1f5f9}
tr.fail td{background:#fff1f2}tr.good td{background:#f8fafc}
.ok{color:#15803d;font-weight:600}.bad{color:#b91c1c;font-weight:600}
table.summary{max-width:900px}table.summary td,table.summary th{text-align:left}
.legend{color:#52616b;font-size:14px}.legend code{background:#eef1f4;padding:1px 4px;border-radius:3px}
button,select{font:inherit;padding:.4rem}h1 span{font-size:15px;color:#52616b;font-weight:400}
</style>
<h1>Port exclusion check <span>· corner givens vs verified solutions</span></h1>
<p>Proposed rule: if a given corner piece has no port on a side, the immediately adjacent cell must be empty (X).</p>
<div class="verdict"><b>Theory fails on the corpus:</b> ''' + f"{all_wrong:,} of {all_slots:,} corner-given missing sides ({pct(all_wrong, all_slots)}) are real tracks in the reference solution, spread over {len(all_puzzles_wrong)} of {len(puzzles)} puzzles. On the train split it is {train_wrong} of {train_slots} ({pct(train_wrong, train_slots)}), across {len(train_puzzles_wrong)} of {sum(1 for p in puzzles if split_of.get(p.puzzle_id) == 'train')} puzzles. Widened to straight givens the rule fails {widened_wrong:,} of {widened_slots:,} ({pct(widened_wrong, widened_slots)})." + '''</div>
<h2>Reciprocal-port sanity check</h2>
<p>Every port of a given piece was checked against the cell it points at in the reference solution: ''' + f"{reciprocal_checked:,} port links, {len(reciprocal_bad)} failures." + (" All reciprocal ports agree with the solutions." if not reciprocal_bad else " Mismatches: " + html.escape(str(reciprocal_bad[:5]))) + '''</p>
<h2>Summary</h2><table class="summary"><thead><tr><th>Scope</th><th>Corner slots</th><th>X correct</th><th>X wrong</th><th>Wrong rate</th><th>Puzzles with a wrong X</th></tr></thead><tbody>''' + "".join(summary_rows) + '''</tbody></table>
<p class="legend">Legend: <code>X</code> = the theory marks the neighbour empty. Purple X = the reference cell is empty, so the theory holds. Red X = the reference cell is a track, so the theory is contradicted. Reference boards come from <code>data/puzzles.jsonl</code> (built from the images with <code>track_solver.py</code> and the reference DFS solver).</p>
<div class="toolbar"><label>Puzzle <select id="puzzle"><option value="">All puzzles</option></select></label>
<label>Split <select id="split"><option value="">All splits</option>''' + split_options + '''</select></label>
<label>Verdict <select id="verdict"><option value="">All</option><option value="fail">Only puzzles with a wrong X</option><option value="ok">Only fully correct puzzles</option></select></label>
<span id="count"></span></div>
''' + "\n".join(cards) + '''
<script>
const cards=[...document.querySelectorAll('.card')],puzzleSelect=document.querySelector('#puzzle'),splitSelect=document.querySelector('#split'),verdictSelect=document.querySelector('#verdict'),count=document.querySelector('#count');
[...new Set(cards.map(c=>c.dataset.puzzle))].sort().forEach(id=>{const n=cards.filter(c=>c.dataset.puzzle===id).length,o=document.createElement('option');o.value=id;o.textContent=id+' ('+n+')';puzzleSelect.append(o)});
function show(){let n=0;cards.forEach(c=>{const wrong=+c.dataset.wrong;c.hidden=!!(puzzleSelect.value&&c.dataset.puzzle!==puzzleSelect.value)||!!(splitSelect.value&&c.dataset.split!==splitSelect.value)||(verdictSelect.value==='fail'&&wrong===0)||(verdictSelect.value==='ok'&&wrong>0);if(!c.hidden)n++});count.textContent=n+' puzzles'}
puzzleSelect.addEventListener('change',show);splitSelect.addEventListener('change',show);verdictSelect.addEventListener('change',show);show();
</script>'''

dest = Path("outputs/port_exclusion_check.html")
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(page, encoding="utf-8")
print(f"wrote {len(cards)} puzzle cards to {dest.resolve()}")
print(f"corner slots: {all_slots}, X correct: {all_slots - all_wrong}, X wrong: {all_wrong}")
print(f"train corner slots: {train_slots}, X wrong: {train_wrong}")
print(f"reciprocal port links: {reciprocal_checked}, failures: {len(reciprocal_bad)}")
