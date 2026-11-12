"""Generate `notebooks/train_track_grpo.ipynb` from plain cell sources.

Keeping the notebook as generated output means it can be reviewed as text, diffed, and regenerated
after a contract change. Run this after editing any cell:

    uv run python scripts/make_notebook.py

The notebook is adapted from Unsloth's Gemma 4 RL Sudoku notebook (LGPL-3.0); the installation and
model-loading cells are kept as close to it as possible, because that combination is the one that is
known to fit a free T4.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "train_track_grpo.ipynb"

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        """# Train Tracks x Gemma 4 (E2B) - GRPO
Run this on a **free Tesla T4** (Colab) or **Kaggle T4 x2** instance with *Run all*.

Adapted from Unsloth's [Gemma 4 RL Sudoku notebook](https://github.com/unslothai/notebooks)
(**LGPL-3.0**); the environment, prompt and rewards here are our own (`tt_env/`, see `plan.md`).

| | Sudoku notebook | This notebook |
|---|---|---|
| Action | one `(row, col, digit)` per step | **one Python function that returns a whole grid** |
| Prompt | fixed text, puzzle generated inside the reward | fixed text, puzzle injected as a function argument |
| Reward | valid moves (0.2 each) | `format_ok` + `no_cheat` + `solved` (+1.5 shaped) |
| Ceiling | 30 | **31.5**, asserted below before training |

Because the prompt contains no puzzle, the model must learn a *general* solver. The corpus is a
rollout budget, not 608 training rows.""",
    ),
    (
        "markdown",
        """## Installation
Runs on Colab or Kaggle. Enable the accelerator first (Colab: *Runtime -> Change runtime type -> T4*;
Kaggle: *Settings -> Accelerator -> GPU T4 x2*), and on Kaggle enable *Settings -> Internet* too - the
cell below installs from PyPI and GitHub.

If this cell has to **upgrade** torch (Kaggle images sometimes ship one older than 2.8), restart the
session before continuing: the running kernel keeps the old `torch` module, and the model load would hit
the new CUDA libraries with the old one.""",
    ),
    (
        "code",
        r"""%%capture
import os, importlib.util
!pip install --upgrade -qqq uv

# Gemma 4 needs torch >= 2.8. Colab ships none; Kaggle ships one that may be older, so check the
# version rather than just its presence, or the model load fails after a long install.
_have_torch = importlib.util.find_spec("torch") is not None
_old_torch = False
if _have_torch:
    import torch as _t
    _old_torch = tuple(int(x) for x in _t.__version__.split("+")[0].split(".")[:2]) < (2, 8)

if not _have_torch or _old_torch or "COLAB_" in "".join(os.environ.keys()):
    try: import numpy, PIL; _numpy = f"numpy=={numpy.__version__}"; _pil = f"pillow=={PIL.__version__}"
    except: _numpy = "numpy"; _pil = "pillow"
    # Gemma 4 requires transformers >= 5.5.0 - do NOT pin to 4.x here
    !uv pip install -qqq \
        "torch>=2.8.0" "triton>=3.4.0" {_numpy} {_pil} torchvision bitsandbytes \
        "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \
        "unsloth[base] @ git+https://github.com/unslothai/unsloth" \
        git+https://github.com/triton-lang/triton.git@0add68262ab0a2e33b84524346cb27cbb2787356#subdirectory=python/triton_kernels
elif importlib.util.find_spec("unsloth") is None:
    !uv pip install -qqq unsloth
# Gemma 4 requires transformers >= 5.5.0; 5.16+ needs tokenizers >= 0.23.1, past the pin below
!uv pip install --upgrade --no-deps "transformers>=5.5.0,<5.16" "tokenizers>=0.22.0,<=0.23.0" "trl>=0.28.0" unsloth unsloth_zoo
# transformers 5.x needs huggingface_hub >= 1.5.0, and the --no-deps above cannot pull it
!uv pip install -qqq "huggingface_hub>=1.5.0,<2.0"
!pip install --no-deps --upgrade timm # For Gemma 4 vision/audio""",
    ),
    ("markdown", "## Load the model"),
    (
        "code",
        r"""from unsloth import FastVisionModel
import torch, transformers

print("torch", torch.__version__, "| transformers", transformers.__version__)
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU - enable the accelerator")

max_seq_length = 4096  # measured prompt ~0.7k tokens; leaves ~3k for the function
lora_rank = 32

model, tokenizer = FastVisionModel.from_pretrained(
    model_name = "unsloth/gemma-4-E2B-it",
    max_seq_length = max_seq_length,
    load_in_4bit = False,   # LoRA 16bit; set True if a T4 runs out of memory
    fast_inference = False,
)""",
    ),
    (
        "code",
        r"""model = FastVisionModel.get_peft_model(
    model,
    r = lora_rank,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = lora_rank * 2,
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)""",
    ),
    (
        "markdown",
        """## The environment
Upload `dist/train_tracks_colab.zip` (built locally with
`uv run python scripts/make_colab_bundle.py`) to the notebook's file browser, then run this cell.
The zip holds `tt_env/` and the frozen data; the reference solver and the PNGs are not needed here.

**Colab or Kaggle?** Both work, with the same T4 profile - Kaggle's T4 has 16 GB and only one device is
used. On Kaggle: enable *Settings -> Internet* before running (the install cell and the model download
need it), and remember sessions stop after **12 h** with a **30 h/week** GPU quota.

On Kaggle the bundle can arrive either way:

- a **zip** uploaded to the notebook (`/kaggle/working`) or added as Data (`/kaggle/input/**`), or
- an **extracted dataset**, e.g. `/kaggle/input/datasets/<user>/train-tracks-colab/` containing
  `tt_env/` and `data/` - this cell finds it and copies it.

Either way it ends up in the working directory, so every later path (`data/puzzles.jsonl`,
`outputs/...`) is unchanged.""",
    ),
    (
        "code",
        r"""from pathlib import Path
import shutil, sys, zipfile

# Colab: a zip in cwd. Kaggle: a zip anywhere under /kaggle/input, or an extracted bundle folder.
CANDIDATES = [Path("train_tracks_colab.zip"), Path("/kaggle/working/train_tracks_colab.zip")]
if Path("/kaggle/input").exists():
    CANDIDATES += sorted(Path("/kaggle/input").glob("**/train_tracks_colab.zip"))
ZIP = next((path for path in CANDIDATES if path.exists()), None)

def find_folder():
    # An extracted bundle: a directory that holds tt_env/ and data/puzzles.jsonl.
    if not Path("/kaggle/input").exists():
        return None
    for marker in Path("/kaggle/input").glob("**/tt_env"):
        candidate = marker.parent
        if (candidate / "data" / "puzzles.jsonl").exists():
            return candidate
    return None

if ZIP is not None:
    print("bundle: zip", ZIP)
    for name in ("tt_env", "data"):
        if (Path.cwd() / name).is_dir():
            shutil.rmtree(Path.cwd() / name)
    with zipfile.ZipFile(ZIP) as archive:
        archive.extractall(".")
else:
    FOLDER = find_folder()
    assert FOLDER is not None, (
        "no bundle: upload dist/train_tracks_colab.zip, or add a dataset containing tt_env/ and data/"
    )
    print("bundle: folder", FOLDER)
    for name in ("tt_env", "data"):
        if (Path.cwd() / name).is_dir():
            shutil.rmtree(Path.cwd() / name)   # mount is read-only: copy, never write into it
        shutil.copytree(FOLDER / name, Path.cwd() / name)
sys.path.insert(0, str(Path.cwd()))
# Replacing the bundle in a running session still needs a restart (Colab: Runtime -> Restart session;
# Kaggle: Run -> Restart): Python keeps already-imported modules in memory.

from tt_env import grade, load_jsonl
from tt_env.policies import program
from tt_env.prompt import messages, prompt_text
from tt_env.rewards import reward_format_ok, reward_no_cheat
from tt_env.rl import RewardLog, TrainSampler, load_split, solved_rewards

corpus = load_jsonl("data/puzzles.jsonl")
train_puzzles = load_split("data/puzzles.jsonl", "data/splits.json", "train")
eval_puzzles = load_split("data/puzzles.jsonl", "data/splits.json", "test_small")
print(f"corpus {len(corpus)}   train {len(train_puzzles)}   test_small {len(eval_puzzles)}")
assert len(corpus) == 608 and len(train_puzzles) == 212 and len(eval_puzzles) == 46""",
    ),
    (
        "markdown",
        """### G1 on this host
The same reward code, checked here before spending GPU time: the frozen reference solution must score
**exactly 31.5** and a cheat must score **-19.5**. If either assertion fails, nothing below is worth
running.""",
    ),
    (
        "code",
        r"""sample = train_puzzles[0]
reference = grade(program(list(sample.reference_solution)), sample)
cheat = grade("```python\ndef solve(puzzle):\n    import tracks\n    return []\n```", sample)

print(sample.puzzle_id, "|", reference.status, reference.total, "|", cheat.status, cheat.total)
assert reference.status == "solved" and reference.total == 31.5, reference.to_dict()
assert cheat.status == "cheat" and cheat.total == -19.5, cheat.to_dict()
print("reward path OK")""",
    ),
    (
        "markdown",
        """## The prompt
One fixed prompt, no puzzle data: the harness passes the puzzle to `solve(puzzle)` at run time. Same
text for training and for evaluation.""",
    ),
    (
        "code",
        r"""PROMPT = prompt_text()
print(PROMPT)
print("-" * 70)

# Count TOKENS, not characters, and beware the batch dimension: this processor returns
# [[id, ...]] or a (1, N) tensor, so len() gives 1. Subtracting a wrong count from max_seq_length
# starves max_completion_length and truncates every answer before its closing fence.
_chat = tokenizer.apply_chat_template(messages(), tokenize = False, add_generation_prompt = True)
_encoded = tokenizer(images = None, text = _chat, add_special_tokens = False, return_tensors = None)
_ids = _encoded["input_ids"]
if hasattr(_ids, "shape"):                      # torch tensor: (1, N)
    n_prompt = int(_ids.shape[-1])
elif _ids and isinstance(_ids[0], (list, tuple)):   # nested list: [[id, ...]]
    n_prompt = len(_ids[0])
else:                                           # flat list of ids
    n_prompt = len(_ids)
print(f"prompt chars {len(PROMPT)}   prompt tokens {n_prompt}")
assert 200 < n_prompt < max_seq_length // 2, (
    f"prompt is {n_prompt} of {max_seq_length} tokens; shorten the prompt or raise max_seq_length"
)""",
    ),
    (
        "markdown",
        """## Reward functions
Three functions, mirroring the sudoku notebook's `function_works` / `no_cheating` / `strategy_succeeds`.

`reward_solved` samples **one** puzzle per call and grades every completion of the group on it - that
is what makes the GRPO advantages meaningful. It also logs the puzzle id, so a run can be replayed.
Keep `per_device_train_batch_size = 1` so one call holds exactly `num_generations` completions of the
same prompt.""",
    ),
    (
        "code",
        r"""from datetime import datetime, timezone
from tt_env.rewards import completion_text

sampler = TrainSampler(train_puzzles, seed = 20261110)
_run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
reward_log = RewardLog(Path(f"outputs/train_rewards_{_run}.jsonl"))
print("reward log:", reward_log.path)
_shown = {"no-code": 0, "invalid": 0}

def reward_solved(completions, **kwargs):
    puzzle = sampler.next()
    rewards, grades = solved_rewards(completions, puzzle)
    reward_log.log(puzzle, grades, rewards)
    statuses = [g.status for g in grades]
    if reward_log.step % 5 == 0 or "solved" in statuses:
        print(f"step {reward_log.step:>4}  {puzzle.puzzle_id:<16} {statuses}  "
              f"total={[round(g.total, 2) for g in grades]}  solved+shaped={[round(r, 2) for r in rewards]}")
    # If the model never writes a fence, nothing can be learned. Show the raw text for the first
    # few such samples: 0 fences (prose or truncation) and 2+ fences (a rejected answer) need
    # different fixes.
    for completion, status in zip(completions, statuses):
        if status in _shown and _shown[status] < 2:
            _shown[status] += 1
            raw = completion_text(completion)
            print(f"---- first {status} completion ({len(raw)} chars, {raw.count('```') // 2} fence pairs) ----")
            print(raw[:800])
            print("----")
    return rewards

REWARD_FUNCS = [reward_format_ok, reward_no_cheat, reward_solved]

# Smoke test: one reference completion and one empty completion, graded as a group of two.
demo = train_puzzles[0]
demo_rewards, demo_grades = solved_rewards(
    [program(list(demo.reference_solution)), program([".." * demo.cols] * demo.rows)], demo
)
print("demo rewards", demo_rewards, [g.status for g in demo_grades])
assert demo_rewards[0] == 30.0 and demo_rewards[1] == 0.0""",
    ),
    (
        "markdown",
        """## Dataset
The same prompt a thousand times. `answer` is unused; GRPO needs only the prompt. The puzzle is drawn
by the reward, so the number of rows is a rollout budget, not a dataset size.""",
    ),
    (
        "code",
        r"""from datasets import Dataset

dataset = Dataset.from_list([{"prompt": messages(), "answer": 0}] * 1000)
print(dataset)
print(dataset[0]["prompt"][-1]["content"][-300:])""",
    ),
    (
        "markdown",
        """## Train
T4 notes: fp16 only (no bf16), `num_generations = 2`, gradient checkpointing on. If you hit an OOM
error, set `load_in_4bit = True` above and/or drop `gradient_accumulation_steps` to 1.""",
    ),
    (
        "code",
        r"""from trl import GRPOConfig, GRPOTrainer
import inspect

# Prompt is ~0.7k tokens, so ~3.3k would be available. A solver function needs ~1k; 2048 bounds the
# worst-case generation time per step (a T4 costs real minutes on a rambling completion).
max_completion_length = min(2048, max_seq_length - (n_prompt + 1))

# T4 has no bf16 and Unsloth already loads the model in fp16; ask TRL for fp16 autocast only when the
# installed version still exposes the flag (transformers 5.x renamed some of these).
amp = {"fp16": True} if "fp16" in inspect.signature(GRPOConfig).parameters else {}

training_args = GRPOConfig(
    temperature = 1.0,
    learning_rate = 5e-5,
    weight_decay = 0.001,
    warmup_ratio = 0.1,
    lr_scheduler_type = "linear",
    optim = "adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 1,   # one prompt per call -> one puzzle per group
    gradient_accumulation_steps = 2,
    num_generations = 2,               # T4 budget; 4 if memory allows
    max_completion_length = max_completion_length,
    max_steps = 200,                   # G3 is measured at step 200
    save_steps = 50,
    report_to = "none",
    output_dir = "outputs/grpo",
    epsilon = 0.2,
    epsilon_high = 0.28,               # one sided
    delta = 1.5,
    loss_type = "bnpo",
    mask_truncated_completions = True,
    **amp,
)
print(f"max_completion_length {max_completion_length}   amp {amp or 'model default'}")""",
    ),
    (
        "markdown",
        """### Diagnostic: does the model even produce code?

Run this once before spending 200 steps. It answers three questions: is the current `tt_env` imported,
how many tokens the prompt costs, and does the reply contain one **complete** fenced block.

The usual failure is a block that opens and never closes because the answer hit the token cap — that
scores `no-code`. Look at the `marker(s)` count (odd = truncated) and the `tail:` line, not just the
head.""",
    ),
    (
        "code",
        r"""import torch
from tt_env.rewards import completion_text

# Proves the tt_env that is imported is the current one: multimodal content must unwrap to text.
# If this assertion fails, the old zip is still loaded - re-extract it and restart the runtime.
probe = [{"role": "assistant", "content": [{"type": "text", "text": "hello"}]}]
print("tt_env probe:", repr(completion_text(probe)), "(expect 'hello')")
assert completion_text(probe) == "hello", "stale tt_env: re-extract the zip, then Runtime -> Restart session"

def sample_completion(max_new_tokens = None):
    text = tokenizer.apply_chat_template(messages(), tokenize = False, add_generation_prompt = True)
    # images=None + text= as keywords: this is a Gemma 4 processor, and a positional string is
    # interpreted as an image (the reference sudoku notebook does the same).
    inputs = tokenizer(images = None, text = text, return_tensors = "pt", add_special_tokens = False).to("cuda")
    out = model.generate(
        **inputs,
        max_new_tokens = max_new_tokens or training_args.max_completion_length,  # what training uses
        do_sample = True, temperature = 1.0, top_p = 0.95,
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    answer = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens = True)
    return answer, int(inputs["input_ids"].shape[1])

print("max_completion_length:", training_args.max_completion_length)
for i in range(2):
    raw, prompt_tokens = sample_completion()
    marks = raw.count("```")   # triple-backtick markers, not pairs: 1 means opened but never closed
    print(f"--- sample {i}: prompt {prompt_tokens} tok, answer {len(raw)} chars, {marks} marker(s)")
    print("head:", repr(raw[:400]))
    print("tail:", repr(raw[-200:]))
    if not raw:
        print("VERDICT: empty completion - generation/template problem, do not start training")
    elif marks == 0:
        print("VERDICT: no code block at all - the model is not following the output contract")
    elif marks % 2:
        print(f"VERDICT: block opened but never closed - truncated at {training_args.max_completion_length}"
              " tokens; ask for shorter code or raise max_completion_length")
    elif marks // 2 > 1:
        print("VERDICT: several blocks - extract_code rejects those; send this output to the maintainer")
    else:
        print("VERDICT: one complete fenced block - the reward path can grade this")""",
    ),
    (
        "code",
        r"""trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = REWARD_FUNCS,
    args = training_args,
    train_dataset = dataset,
)
trainer.train()""",
    ),
    (
        "markdown",
        """### What did it learn?
The reward log has one line per step: puzzle id, per-sample statuses, the `solved+shaped` term, and the
`total` (which adds `format_ok` + `no_cheat`, so even an all-empty sample already scores 0.5).

`valid_program_rate` (G2) is the fraction of steps whose completions parsed and ran (a timeout or a
runtime error is still a *valid program*; `no-code`, `invalid` and `cheat` are not). The mean total
should climb from ~0.5 towards 31.5.""",
    ),
    (
        "code",
        r"""import json, statistics
from pathlib import Path

logs = sorted(Path("outputs").glob("train_rewards_*.jsonl"))
assert logs, "no reward log yet - run the training cell first"
print("reading", logs[-1])
rows = [json.loads(line) for line in logs[-1].read_text().splitlines() if line]
valid = sum(1 for r in rows if all(s not in {"no-code", "invalid", "cheat", "syntax"} for s in r["statuses"]))
solved_steps = sum(1 for r in rows if "solved" in r["statuses"])
totals = [t for r in rows for t in r["totals"]]      # 2 samples per step
shaped = [s for r in rows for s in r["rewards"]]
print(f"steps {len(rows)}   valid-program steps {valid / len(rows):.1%}   solved steps {solved_steps}")
print(f"mean total         first 20 steps {statistics.fmean(totals[:40]):.3f}"
      f"   last 20 {statistics.fmean(totals[-40:]):.3f}")
print(f"mean solved+shaped first 20 steps {statistics.fmean(shaped[:40]):.3f}"
      f"   last 20 {statistics.fmean(shaped[-40:]):.3f}")
for r in rows[:3] + rows[-3:]:
    print(r["step"], r["puzzle_id"], r["statuses"], r["totals"])""",
    ),
    ("markdown", "## Save the adapter"),
    (
        "code",
        r"""from pathlib import Path
import statistics

adapter_dir = Path("gemma_4_tracks_lora").resolve()
model.save_pretrained(str(adapter_dir))
tokenizer.save_pretrained(str(adapter_dir))
print("adapter ->", adapter_dir)   # Colab: /content/...   Kaggle: /kaggle/working/... (downloadable)

from safetensors import safe_open

# Check the language layers, not the first keys in the file: those are the audio/vision towers,
# which were never trained (their lora_B stays zero and that is expected).
with safe_open(str(adapter_dir / "adapter_model.safetensors"), framework = "pt") as f:
    keys = [k for k in f.keys() if "audio_tower" not in k and "vision_tower" not in k]
    b_keys = [k for k in keys if k.endswith("lora_B.weight")]
    means = [float((f.get_tensor(k) != 0).float().mean()) for k in b_keys]
    print(f"language-layer lora_B tensors: {len(b_keys)}   mean non-zero {statistics.fmean(means):.2f}")
    for key, mean in list(zip(b_keys, means))[:4]:
        print(" ", key, f"non-zero {mean:.2f}")""",
    ),
    (
        "markdown",
        """## Evaluate (T2: independent rollouts)

Same fixed prompt, k samples per puzzle, every completion graded by the verifier. `LIMIT = 4` first to
check the loop; set `LIMIT = None` for the full 46-puzzle `test_small` set (184 generations). The
generation cap is the same `max_completion_length` the trainer used, so evaluation cannot fail a sample
that training would have allowed.

This is the number that goes in the gate table, not the training reward.""",
    ),
    (
        "code",
        r"""import json
from pathlib import Path

LIMIT = 4        # None -> all 46 test_small puzzles
K = 4            # pass@4
puzzles = eval_puzzles[:LIMIT] if LIMIT else eval_puzzles

records = []
for puzzle in puzzles:
    text = tokenizer.apply_chat_template(messages(), tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(images = None, text = text, return_tensors = "pt", add_special_tokens = False).to("cuda")
    for sample_index in range(K):
        out = model.generate(
            **inputs, max_new_tokens = max_completion_length, do_sample = True, temperature = 1.0,
            top_p = 0.95,
            pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens = True)
        result = grade(completion, puzzle)
        records.append({"puzzle_id": puzzle.puzzle_id, "sample": sample_index, **result.to_dict()})
    print(puzzle.puzzle_id, [r["status"] for r in records[-K:]], flush = True)

per_puzzle = {}
for record in records:
    per_puzzle.setdefault(record["puzzle_id"], []).append(record["status"] == "solved")
solve_rate = sum(1 for r in records if r["status"] == "solved") / len(records)
pass_at_k = sum(1 for flags in per_puzzle.values() if any(flags)) / len(per_puzzle)
print(f"puzzles {len(per_puzzle)}  samples {len(records)}  solve rate {solve_rate:.1%}  pass@{K} {pass_at_k:.1%}")

Path("outputs/eval").mkdir(parents = True, exist_ok = True)
name = "outputs/eval/test_small_after_training.jsonl" if LIMIT is None else "outputs/eval/test_small_partial.jsonl"
Path(name).write_text("\n".join(json.dumps(r) for r in records))""",
    ),
    (
        "markdown",
        """## Export

Merge to 16-bit for vLLM/llama.cpp, or keep the LoRA adapter. Nothing below runs unless enabled.""",
    ),
    (
        "code",
        r"""# Merge to 16bit
if False: model.save_pretrained_merged("gemma_4_tracks_16bit", tokenizer, save_method = "merged_16bit")

# Merge to 4bit
if False: model.save_pretrained_merged("gemma_4_tracks_4bit", tokenizer, save_method = "merged_4bit")

# GGUF for llama.cpp
if False: model.save_pretrained_gguf("gemma_4_tracks", tokenizer, quantization_method = "q4_k_m")""",
    ),
]


def build_notebook(cells: list[tuple[str, str]]) -> dict:
    out = []
    for cell_type, source in cells:
        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        out.append(cell)
    return {
        "cells": out,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    notebook = build_notebook(CELLS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    code_cells = sum(1 for kind, _ in CELLS if kind == "code")
    print(f"{len(CELLS)} cells ({code_cells} code) -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
