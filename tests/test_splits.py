"""Split-contract tests. Skipped until `scripts/freeze_splits.py` has run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tt_env import load_jsonl

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "data" / "splits.json"
CORPUS = ROOT / "data" / "puzzles.jsonl"
EXPECTED = {"train": 212, "validation": 46, "test_small": 46, "test_medium": 304}

pytestmark = pytest.mark.skipif(not SPLITS.exists() or not CORPUS.exists(), reason="run the data pipeline first")


def payload():
    return json.loads(SPLITS.read_text("utf-8"))


def test_counts_match_the_hf_contract():
    data = payload()
    assert data["counts"] == EXPECTED
    for name, expected in EXPECTED.items():
        assert len(data["splits"][name]) == expected, name


def test_every_split_id_exists_once():
    data = payload()
    corpus = {p.puzzle_id for p in load_jsonl(CORPUS)}
    seen: dict[str, str] = {}
    for name, ids in data["splits"].items():
        for puzzle_id in ids:
            assert puzzle_id in corpus, f"{puzzle_id} in {name} but not in the corpus"
            assert puzzle_id not in seen, f"{puzzle_id} in {name} and {seen[puzzle_id]}"
            seen[puzzle_id] = name


def test_medium_never_trains():
    data = payload()
    difficulties = {p.puzzle_id: p.difficulty for p in load_jsonl(CORPUS)}
    assert not [pid for pid in data["splits"]["train"] if difficulties.get(pid) == "medium"]
    assert all(difficulties.get(pid) == "medium" for pid in data["splits"]["test_medium"])


def test_split_ids_are_sorted():
    data = payload()
    for name, ids in data["splits"].items():
        assert ids == sorted(ids), name
