"""Create a manually reviewable HTML page for small-puzzle deduction examples."""
import html
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()))
from tt_env.deductions import cell_token, combined_deductions
from tt_env.puzzle import load_jsonl
from tt_env.report import render_grid_svg


def board_svg(puzzle, tokens, highlights, marks):
    """Render tokens as SVG and stamp X / ? glyphs for the requested cells."""
    cell_size = 420 / max(puzzle.rows, puzzle.cols)
    cells = [[".." if x in ("??", "XX") else x for x in row] for row in tokens]
    svg = render_grid_svg(puzzle, cells, cell=cell_size, highlight=highlights,
                          title="After deduction", row_clues_right=True,
                          bold_clues=True, row_labels=True, col_labels=True)
    glyphs = []
    for r, c, value in marks:
        x = 8 + (c + .5) * cell_size
        y = 24 + (r + .5) * cell_size
        if value in ("X", "XX"):
            glyphs.append(f'<text x="{x:.1f}" y="{y + 5:.1f}" text-anchor="middle" font-size="18" font-weight="700" fill="#dc2626">X</text>')
        elif value == "?":
            glyphs.append(f'<text x="{x:.1f}" y="{y + 7:.1f}" text-anchor="middle" font-size="20" font-weight="700" fill="#ea580c">?</text>')
    return svg.replace("</svg>", "".join(glyphs) + "</svg>")


records = [json.loads(line) for line in
           Path("data/sft_deductions_train.jsonl").read_text(encoding="utf-8").splitlines() if line]
puzzles = {p.puzzle_id: p for p in load_jsonl("data/puzzles.jsonl")}
small = [item for item in records if item["puzzle_id"].endswith("-small")]

cards = []
for item in small:
    puzzle = puzzles[item["puzzle_id"]]
    after_text = item["target_board"].splitlines()
    after = [[token for token in line.split()[1:]] for line in after_text[1:]]
    highlights = {(r, c): "#dcfce7" for r, c, value in item["action"]["cells"] if value != "X"}
    image_src = "../" + puzzle.source.replace("\\", "/")
    after_svg = board_svg(puzzle, after, highlights, item["action"]["cells"])
    title = f'{item["puzzle_id"]} · {item["rule"]}'
    cards.append(f'''<article class="card" data-puzzle="{html.escape(item["puzzle_id"])}" data-rule="{html.escape(item["rule"])}">
<h2>{html.escape(title)}</h2><p class="clues">Rows: {html.escape(str(list(puzzle.row_clues)))} · Columns: {html.escape(str(list(puzzle.col_clues)))}</p>
<div class="boards"><section><h3>Before · original puzzle</h3><div class="visual"><img class="source-image" src="{html.escape(image_src, quote=True)}" alt="Original {html.escape(item['puzzle_id'])} puzzle"></div></section>
<section><h3>After</h3><div class="visual">{after_svg}</div></section></div>
<p><b>Rule:</b> {html.escape(item["action"]["explanation"])}</p>
<p><b>Action:</b> {html.escape(json.dumps(item["action"]["cells"]))}</p></article>''')

# One card per puzzle: run the three rules together to a fixpoint.
combined_cards = []
for puzzle_id in dict.fromkeys(item["puzzle_id"] for item in small):
    puzzle = puzzles[puzzle_id]
    state, steps = combined_deductions(puzzle)
    tokens = [[cell_token(state[r][c]) for c in range(puzzle.cols)] for r in range(puzzle.rows)]
    givens = {(r, c) for r, c, _ in puzzle.givens}
    highlights = {(r, c): "#dcfce7" for r in range(puzzle.rows) for c in range(puzzle.cols)
                  if tokens[r][c] not in ("??", "XX") and (r, c) not in givens}
    marks = [(r, c, tokens[r][c]) for r in range(puzzle.rows) for c in range(puzzle.cols)
             if tokens[r][c] in ("XX", "?")]
    image_src = "../" + puzzle.source.replace("\\", "/")
    after_svg = board_svg(puzzle, tokens, highlights, marks)
    known = sum(1 for row in tokens for token in row if token not in ("??", "XX"))
    empty = sum(1 for row in tokens for token in row if token == "XX")
    unknown = sum(1 for row in tokens for token in row if token == "??")
    steps_html = "".join(
        f'<li><span class="tag {html.escape(step.rule, quote=True)}">{html.escape(step.rule)}</span> '
        f'{html.escape(step.explanation)}</li>'
        for step in steps
    )
    combined_cards.append(f'''<article class="card combined" data-puzzle="{html.escape(puzzle_id)}" data-rule="combined">
<h2>{html.escape(puzzle_id)} · all rules combined</h2>
<p class="clues">Rows: {html.escape(str(list(puzzle.row_clues)))} · Columns: {html.escape(str(list(puzzle.col_clues)))}</p>
<p class="summary">{len(steps)} steps · {known} track cells known · {empty} empty · {unknown} still unknown</p>
<div class="boards"><section><h3>Before · original puzzle</h3><div class="visual"><img class="source-image" src="{html.escape(image_src, quote=True)}" alt="Original {html.escape(puzzle_id)} puzzle"></div></section>
<section><h3>After · all rules applied</h3><div class="visual">{after_svg}</div></section></div>
<h3>Reasoning</h3><ol class="steps">{steps_html}</ol></article>''')

RULE_ORDER = ["clue_satisfied", "all_remaining_cells_required", "port_neighbour"]
rule_counts = {rule: sum(1 for item in small if item["rule"] == rule) for rule in RULE_ORDER}
rule_options = f'<option value="combined">combined · all rules ({len(combined_cards)})</option>' + "".join(
    f'<option value="{html.escape(rule, quote=True)}">{html.escape(rule)} ({rule_counts[rule]})</option>'
    for rule in RULE_ORDER if rule_counts[rule]
)
page = '''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Train Tracks deduction review · small</title><style>
body{font:16px system-ui;margin:2rem auto;padding:0 1rem;max-width:1100px;background:#f4f6f8;color:#17202a}h1{margin-bottom:.3rem}.toolbar{position:sticky;top:0;background:#f4f6f8;padding:1rem 0;display:flex;gap:1rem;align-items:center;flex-wrap:wrap}.card{background:white;border:1px solid #ccd3da;border-radius:10px;padding:1rem;margin:1rem 0}.card.combined{border:2px solid #1d4ed8}.boards{display:flex;gap:2rem;flex-wrap:wrap}.boards section{flex:1;min-width:320px}.visual{width:min(100%,500px);height:500px;display:flex;align-items:center;justify-content:center}.source-image{display:block;max-width:500px;max-height:500px;width:auto;height:auto}.visual svg{max-width:100%;max-height:100%}pre{font:16px/1.5 ui-monospace,monospace;background:#f7f8fa;padding:1rem;border-radius:6px}.clues{color:#52616b}.summary{color:#1d4ed8;font-size:14px;font-weight:600}.legend{color:#52616b;font-size:14px}.legend code{background:#eef1f4;padding:1px 4px;border-radius:3px}.steps{margin:.4rem 0 0 1.2rem;padding:0}.steps li{margin:.3rem 0}.tag{display:inline-block;font-size:11px;font-weight:700;padding:1px 7px;border-radius:10px;margin-right:.45rem;color:#fff}.tag.clue_satisfied{background:#dc2626}.tag.all_remaining_cells_required{background:#ea580c}.tag.port_neighbour{background:#1d4ed8}button,select{font:inherit;padding:.4rem}h2.section{margin-top:2rem}
</style>
<h1>Train Tracks deduction review</h1><p>Small training puzzles only. Compare each before/after board and action. <a href="#combined">Jump to the combined view</a>.</p>
<p class="legend">Legend: <code>??</code> unknown · <code>XX</code> empty · <code>?</code> track, orientation unknown · <code>N?</code> track with a north port, other port unknown · <code>NE</code> complete piece · red <b>X</b> = marked empty.</p>
<div class="toolbar"><label>Puzzle <select id="puzzle"><option value="">All small puzzles</option></select></label><label>Rule <select id="rule"><option value="">All rules</option>''' + rule_options + '''</select></label><span id="count"></span></div>
<section class="group" id="combined-section">
<h2 class="section" id="combined">All three rules combined</h2>
<p>One card per puzzle: the original board, the board after every rule has been applied to a fixpoint, and the ordered reasoning steps.</p>
''' + "\n".join(combined_cards) + '''
</section>
<section class="group" id="examples-section">
<h2 class="section" id="examples">Per-rule examples</h2>
<p>Each card isolates one forced deduction, in the same order the dataset presents them.</p>
''' + "\n".join(cards) + '''
</section>
<script>
const cards=[...document.querySelectorAll('.card')],puzzleSelect=document.querySelector('#puzzle'),ruleSelect=document.querySelector('#rule'),count=document.querySelector('#count');
[...new Set(cards.map(c=>c.dataset.puzzle))].sort().forEach(id=>{let n=cards.filter(c=>c.dataset.puzzle===id).length,o=document.createElement('option');o.value=id;o.textContent=id+' ('+n+')';puzzleSelect.append(o)});
function show(){let n=0;cards.forEach(c=>{c.hidden=!!(puzzleSelect.value&&c.dataset.puzzle!==puzzleSelect.value)||!!(ruleSelect.value&&c.dataset.rule!==ruleSelect.value);if(!c.hidden)n++});count.textContent=n+' cards';document.querySelectorAll('section.group').forEach(g=>{g.hidden=!g.querySelector('.card:not([hidden])')})}
puzzleSelect.addEventListener('change',show);ruleSelect.addEventListener('change',show);show();</script>'''
dest = Path("outputs/sft_deductions_small_review.html")
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(page, encoding="utf-8")
print(f"wrote {len(cards)} examples + {len(combined_cards)} combined cards to {dest.resolve()}")
