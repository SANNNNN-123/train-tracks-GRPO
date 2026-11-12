# How one GRPO training step works

Companion to [`plan.md`](plan.md) · Design A (code-as-action) · T4 profile

This document explains the mechanics end-to-end. It is deliberately concrete: every number below is a real
value from the reward tables in `plan.md` §6, not a placeholder.

---

## 0. The one-sentence version

The model writes a Python solver, we **run** it on a puzzle, **grade the output with exact rules**, and
update the model so the code shapes that produced better outputs become more likely — the model never sees
a solution and is never told which puzzle it will be graded on.

---

## 1. The cast

| Piece | What it is | Where |
|---|---|---|
| **Prompt** | One fixed instruction: "write `def solve(puzzle)` …" | `tt_env/prompt.py` |
| **Puzzle bank** | 608 parsed puzzles, train/val/test split | `data/puzzles.jsonl`, `data/splits.json` |
| **Sandbox** | AST allow-list, no imports/IO, 10 s timeout | `tt_env/sandbox.py` |
| **Verifier** | 5 exact rules → `Verdict(ok, failures, diagnostics)` | `tt_env/rules.py` |
| **Rewards** | `format_ok`, `no_cheat`, `solved` | `tt_env/rewards.py` |
| **Trainer** | `GRPOTrainer` (TRL) + LoRA on Gemma 4 E2B | `notebooks/train_track_grpo.ipynb` |

Nothing here is supervised: there is no `label` column, no `answer` field, no distillation.

---

## 2. The puzzle the model sees at runtime — and the code it never sees

The model generates **code**. The grid is produced by *executing* that code. Gradients flow through the
code tokens only.

```python
# prompt (fixed, no puzzle inside)
# "Write solve(puzzle). Return the grid as a list of strings, two chars per cell
#  ('NS', 'EW', 'NE', 'NW', 'SE', 'SW', '..'). No imports. Helpers must be nested."

# model's completion
def solve(puzzle):
    R, C = puzzle["rows"], puzzle["cols"]
    ...
    return [ "EW..NS..", ... ]   # 2*C chars per row
```

At execution time the sandbox passes a plain dict — so the same code has to work on any puzzle:

```python
{
  "rows": 8, "cols": 8,
  "row_clues": [5,4,2,5,5,6,2,1],
  "col_clues": [5,3,4,4,6,1,6,1],
  "givens": [[0,2,"NE"], [5,7,"EW"], [7,3,"NS"]],
  "start": [5,7,"E"], "end": [7,3,"S"],
}
```

Consequence: the model cannot "see the answer" and cannot see *which* puzzle it will be tested on. It can
only get better at *algorithms*. That is the entire point of the design — it is why one prompt suffices and
why improvement transfers to unseen puzzles.

---

## 3. One step, in order

### Stage 1 — sample a group

Ask for `N` completions of the same prompt at `temperature = 1.0`.

- T4 profile: **N = 2** (VRAM).
- ≥24 GB: N = 8 (better advantage estimates).

All completions are different attempts at the same question.

### Stage 2 — draw ONE puzzle for the group

```python
puzzle_id = train_ids[rng.randrange(len(train_ids))]   # once, outside the completion loop
```

Every attempt in this group is graded on the **same** puzzle. This is non-negotiable: GRPO ranks attempts
*against each other*, so they must face an identical test. (This is the trick the sudoku notebook uses with
`seed = np.random.randint(10000)` placed outside its loop.)

### Stage 3 — extract code

Take exactly one ``` block; require `def solve(puzzle):`. Anything else → no code.

### Stage 4 — sandbox execute

Compile under an allow-list (no `import`, no `open`, no `__builtins__` tricks), run with a 10 s limit.
Cheating earns −20 and is logged for audit.

### Stage 5 — grade the returned grid

Verifier, in this order:

1. **shape** — `R` strings of `2C` chars, one two-char piece per cell (`NS EW NE NW SE SW`), `..` = empty
2. **givens** — the given pieces appear exactly as supplied, unmodified
3. **dangling/boundary** — every port has a reciprocal neighbour port; off-grid ports only at A and B
4. **clue_row / clue_col** — occupied cells per row/column equal the clues
5. **disconnected / cycle / branch** — occupied cells form exactly one simple path from A to B

Rules 3–5 together are complete: a grid that passes cannot branch, loop, cross, or stop short. **No answer
key is consulted.** If the model finds a different valid path from the reference one, that is a legitimate
solve and gets full marks.

### Stage 6 — score each attempt

| Attempt | Output | format | no_cheat | solve / shaping | total |
|---|---|---|---|---|---|
| A | verifier passes | +0.5 | +1 | **+30** | **31.5** |
| B | right cell counts, random pieces, some dangling ports | +0.5 | +1 | +0.91 | 2.41 |
| C | givens copied, rest empty | +0.5 | +1 | +0.25 | 1.75 |
| D | syntax error | −0.5 | +1 | 0 | 0.5 |
| E | `import tracks` | +0.5 | **−20** | 0 | **−19.5** |

(The bands are the measured ones from `data/reward_audit.json`, not illustrative numbers.)

- `shaped = 1.5 × (0.5·clue_fit + 0.5·port_fit)`, only when the grid is **parseable** and the **givens are intact**.
- Shaping is capped at 1.5 — 5% of a solve — so "match the cell counts" can never compete with "solve it".
- The constant +1 from `no_cheat` appears on every row, including D. Harmless: GRPO subtracts the group
  mean, so constants cancel. Only *differences within the group* produce gradient.

### Stage 7 — GRPO update

```
advantage_i = (reward_i − mean(reward_group)) / std(reward_group)
loss        = − Σ_i advantage_i · Σ_t log π(code_i,t | prompt, code_i,<t)
```

With the table above: mean ≈ 3.5 → A is strongly positive, B/C/D all negative (and E catastrophic).
The update **increases** the probability of the token sequences that produced A's code and **decreases**
D's and E's. No target sequence is ever supplied.

### Stage 8 — repeat

New step → new puzzle → new completions from the updated weights → new relative ranking. Over hundreds of
steps the code shapes that actually solve puzzles (constraint propagation, pruning, backtracking) become
more likely, because completions containing them keep scoring above their group's average.

---

## 4. Why one prompt instead of 608 prompts

With a per-puzzle prompt ("solve this grid"), the model could memorise grid→answer pairs, and each prompt
would be a separate generalisation problem. With a code-as-action prompt:

- the dataset is a **rollout budget**, not a training set (sudoku used 1000 copies of one prompt);
- the model must express a *general algorithm*, so training on 6×6–8×8 transfers to 10×10;
- there is nothing to memorise except algorithms;
- the prompt never changes, so "prompt overfitting" is structurally impossible. The remaining risk is
  **puzzle overfitting** (weights encoding the 212 training puzzles), which is why `validation` (46) exists.

---

## 5. Reward design rationale (the short version)

| Term | Value | Why this value |
|---|---|---|
| `format_ok` | +0.5 / −0.5 | Small on purpose. Big negatives make the policy collapse into "output nothing" (a constant, safe score). Only variance trains. |
| `no_cheat` | +1 / −20 | Cheating must dominate any partial credit, or it becomes the cheapest path to reward. |
| `solved` | +30 | Big enough that a single success outranks any amount of shaping. Sparse by design — shaping is what carries the early signal. |
| shaping | 0 … 1.5 | Gives a gradient on "almost" attempts without ever being a destination. Gated: unparseable grid, modified givens, or no track beyond the givens → 0. |

Shaping exists because on 8×8 a 2B model will score ~0 for many steps; with zero variance inside a group,
GRPO has **nothing to learn from**. The cap stops the obvious farms: matching the counts alone tops out at
1.5 (5% of a solve), and mutating a given scores 0.

---

## 6. Three ways this stalls (and what to do)

1. **All attempts fail identically** → std ≈ 0 → no gradient.
   *Fix:* increase shaping weight; train on 6×6–7×7 first; raise N so at least one attempt differs.
2. **A cheap partial is reachable** → the policy parks there. Run 1 did exactly this: copying the
   givens paid ~0.4 shaping, so after 200 steps the model returned givens-only on 15 of 16 eval samples.
   *Fix:* make the cheap partial score 0 (done for givens-only) rather than lowering the solve reward.
3. **All attempts succeed** → advantage ≈ 0 again — success is no longer informative.
   *Fix:* mix in larger puzzles (`test_medium` is held out, so add 9×9/10×10 *train* puzzles, or generate
   new ones from the reference solver).
4. **Train reward climbs, eval flat** → memorisation or hacking.
   *Fix:* hold out `validation` (46) and `test_small` (46); audit every completion scoring ≥ 20.

This is why `plan.md` has gates (G2/G3) and a fallback ladder rather than a single "train it and see".

---

## 7. FAQ / misconceptions

**"Isn't this just SFT with extra steps?"**
No. SFT needs target sequences and minimises their negative log-likelihood. Here there is no target: the
model's own attempts define a ranking, and the loss is a *relative* advantage weighted by the probability of
the tokens it actually produced.

**"The verifier is basically the answer key, right?"**
It is equivalent in power but different in kind: it encodes *rules*, not a stored solution. That is why a
second, different valid path earns full reward — and why synthetic puzzles without a unique solution are
safe to train on.

**"So it gets a reward for a wrong grid?"**
It gets a *small* partial reward if its output is structurally close (complete grid, correct counts, mostly
reciprocal ports). Maximum partial reward is 1.5 against 30 for a solve. B and C above are that case.

**"Does it learn from the grid it returns?"**
No. The returned grid is not part of the generated text, so no gradient flows through it. The code is the
only trainable surface. This is also why a beautifully commented broken function scores like D.

**"Why 10 seconds?"**
Same as the sudoku notebook. Real solvers backtrack; an artificial cap keeps rollouts affordable and gives
a clean `timeout` signal instead of hanging the trainer.

**"What exactly is the 'accuracy' we report?"**
Solve rate on held-out puzzles — the fraction of *whole puzzles* whose final grid passes the verifier —
reported as pass@1 / pass@4 with Wilson intervals, broken down by grid size. Teacher-forced decision accuracy
(from the HF dataset) is a secondary diagnostic only, because always answering `empty` scores 57% there.

---

## 8. Where this differs from the sudoku notebook

| Sudoku | Here |
|---|---|
| One fixed prompt, no puzzle inside | Same: `solve(puzzle)` receives the puzzle at run time, so the prompt is identical for all 608 puzzles |
| One puzzle per group, generated with a seed | Same trick, sampled from the 212-puzzle train split (seeded, ids logged) |
| `strategy(board, initial)` returns one move, env applies it, +0.2 per valid move | `solve(puzzle)` returns the **whole grid**; reward is exact, not per-move |
| Solved = board equals stored solution | Solved = passes 5 rule checks; no stored solution needed |
| Difficulty = cells removed | Difficulty = grid size + number of givens (6×6…10×10, rectangular too) |
| `function_works` / `no_cheating` / `strategy_succeeds` | `format_ok` / `no_cheat` / `solved` (+ shaping) |
| Small batch is optional | `per_device_train_batch_size = 1` is **required**: one reward call must hold one prompt's group |

The one structural addition is that the training reward must also be a *scheduler*: it samples the puzzle,
so it is the only part of the pipeline that touches `data/splits.json` during training.

If we later fall back to design B (per-cell decisions on the HF schema), Stages 1–2 and 6–8 stay the same;
only Stage 5 changes from "verify a whole grid" to "apply one action and check legality", with the same
verifier running at the end of the episode.

---

## 9. Where to look, and how to run it

| You want | Open |
|---|---|
| the rules (the only judge) | `tt_env/rules.py` |
| the sandbox and its limits | `tt_env/sandbox.py` |
| the reward numbers | `tt_env/rewards.py`, measured in `data/reward_audit.json` |
| the fixed prompt | `tt_env/prompt.py` |
| one training step, as code | `tt_env/rl.py`, `notebooks/train_track_grpo.ipynb` |
| the plan and the gates | `plan.md` |

```bash
uv sync
uv run python scripts/verify_corpus.py     # G0: 608/608 parsed, solved, verified, HF-agreeing
uv run python scripts/audit_rewards.py     # G1: reference 31.5, worst non-solve 2.9
uv run pytest -q                           # 81 tests
uv run python scripts/eval_rollout.py --backend empty --split test_small   # baseline, no GPU
uv run python scripts/make_colab_bundle.py # dist/train_tracks_colab.zip for Colab
```

Then upload `dist/train_tracks_colab.zip` and `notebooks/train_track_grpo.ipynb` to a free T4 Colab and
*Run all*. The notebook re-asserts G1 on that host before training, logs every step's puzzle and rewards
to `outputs/train_rewards_<utc>.jsonl`, and ends with a pass@4 evaluation on `test_small`.
