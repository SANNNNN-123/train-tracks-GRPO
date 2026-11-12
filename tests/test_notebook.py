"""Notebook structure tests: the `.ipynb` is generated, so it must stay valid and consistent.

The install cell is real notebook syntax (cell magic + shell lines) and cannot be compiled as Python;
every other code cell must. That catches the common failure of editing a cell into something only
Jupyter can parse.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "train_track_grpo.ipynb"

pytestmark = pytest.mark.skipif(not NOTEBOOK.exists(), reason="run scripts/make_notebook.py first")


def load() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def cell_source(cell: dict) -> str:
    return "".join(cell["source"])


def test_notebook_is_well_formed():
    data = load()
    assert data["nbformat"] == 4
    assert len(data["cells"]) >= 20
    for cell in data["cells"]:
        assert cell["cell_type"] in {"markdown", "code"}
        assert cell["source"], "empty cell"
        assert isinstance(cell["source"], list)
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [], "a generated notebook must not keep outputs"


def test_code_cells_compile():
    for index, cell in enumerate(load()["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = cell_source(cell)
        if "%%capture" in source:
            continue  # cell magic + shell lines; only Jupyter can parse this one
        compile(source, f"<cell {index}>", "exec")


def test_notebook_wires_the_environment_in():
    text = "\n".join(cell_source(cell) for cell in load()["cells"])
    assert "from tt_env" in text
    assert "reward_format_ok" in text and "reward_no_cheat" in text and "reward_solved" in text
    assert "num_generations = 2" in text, "the T4 budget in plan.md §8.1"
    assert "per_device_train_batch_size = 1" in text, "one puzzle per group"
    assert "train_tracks_colab.zip" in text
    assert "31.5" in text, "the G1 assertion must run on the training host too"


def test_notebook_matches_its_generator():
    """A stale notebook looks fine in Colab until it silently does the wrong thing; compare bytes."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("make_notebook", ROOT / "scripts" / "make_notebook.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = json.dumps(module.build_notebook(module.CELLS), indent=1, ensure_ascii=False) + "\n"
    assert NOTEBOOK.read_text(encoding="utf-8") == expected, (
        "notebook drifted from scripts/make_notebook.py; re-run that script"
    )


def test_notebook_has_the_pre_training_diagnostic():
    text = "\n".join(cell_source(cell) for cell in load()["cells"])
    assert "VERDICT" in text, "the diagnostic cell must run before training"
    assert "tt_env probe" in text, "it must prove the extracted tt_env is the current one"


def test_notebook_never_calls_the_processor_positionally():
    """A Gemma 4 processor treats a positional string as an image; always pass text=/images=."""
    text = "\n".join(cell_source(cell) for cell in load()["cells"])
    assert "tokenizer(text," not in text, "positional tokenizer call: pass images=None, text=text"
    assert text.count("images = None, text = text") >= 2, "prompt cell + diagnostic + eval cell"


def test_completion_budget_is_tokens_not_characters():
    text = "\n".join(cell_source(cell) for cell in load()["cells"])
    assert '_encoded["input_ids"]' in text, "n_prompt must be counted in tokens, not characters"
    assert "shape[-1]" in text, "a processor returns a batch; len() would count 1"
    assert "max_completion_length = min(2048," in text


def test_notebook_finds_an_extracted_kaggle_dataset():
    """Kaggle mounts a dataset as folders, not a zip: /kaggle/input/.../{tt_env,data}."""
    text = "\n".join(cell_source(cell) for cell in load()["cells"])
    assert "find_folder" in text
    assert "**/tt_env" in text
    assert "copytree" in text
    assert "/kaggle/input" in text


def test_prompt_is_not_baked_into_the_dataset_cell():
    """Design A: the dataset rows carry the fixed prompt, never a concrete puzzle."""
    for cell in load()["cells"]:
        source = cell_source(cell)
        if "Dataset.from_list" in source:
            assert "messages()" in source
            assert "payload" not in source
            break
    else:
        pytest.fail("no dataset cell found")
