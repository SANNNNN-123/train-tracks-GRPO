"""Training plumbing: sampler determinism, group semantics, and TRL completion shapes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tt_env import load_jsonl
from tt_env.policies import all_empty, program
from tt_env.rewards import completion_text
from tt_env.rl import RewardLog, TrainSampler, load_split, solved_rewards

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "puzzles.jsonl"
SPLITS = ROOT / "data" / "splits.json"

pytestmark = pytest.mark.skipif(not CORPUS.exists(), reason="run scripts/build_puzzles.py first")


@pytest.fixture(scope="module")
def corpus():
    return {p.puzzle_id: p for p in load_jsonl(CORPUS)}


@pytest.fixture()
def fake_splits(tmp_path, corpus):
    ids = sorted(corpus)[:6]
    path = tmp_path / "splits.json"
    path.write_text(json.dumps({"splits": {"train": ids, "test_small": ids[:2]}}), encoding="utf-8")
    return path


def test_load_split_honours_the_declared_ids(fake_splits, corpus):
    puzzles = load_split(CORPUS, fake_splits, "train")
    assert [p.puzzle_id for p in puzzles] == sorted(corpus)[:6]


def test_load_split_rejects_an_unknown_split(fake_splits):
    with pytest.raises(KeyError):
        load_split(CORPUS, fake_splits, "test_medium")


def test_load_split_reports_missing_puzzles(fake_splits, corpus):
    data = json.loads(fake_splits.read_text("utf-8"))
    data["splits"]["train"].append("99999999-small")
    fake_splits.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="missing from the corpus"):
        load_split(CORPUS, fake_splits, "train")


def test_sampler_is_deterministic_and_records_history(corpus):
    puzzles = [corpus[pid] for pid in sorted(corpus)[:8]]
    first = TrainSampler(puzzles, seed=7)
    second = TrainSampler(puzzles, seed=7)
    draws = [first.next().puzzle_id for _ in range(20)]
    assert draws == [second.next().puzzle_id for _ in range(20)]
    assert first.history == draws


def test_sampler_needs_a_puzzle():
    with pytest.raises(ValueError):
        TrainSampler([], seed=1)


def test_completion_text_accepts_both_trl_shapes():
    assert completion_text("plain") == "plain"
    assert completion_text([{"role": "assistant", "content": "hi"}]) == "hi"
    assert completion_text([{"role": "assistant", "content": "a"}, {"role": "user", "content": "b"}]) == "a\nb"


def test_group_rewards_grade_every_completion_on_one_puzzle(corpus):
    puzzle = corpus[sorted(corpus)[0]]
    reference = puzzle.reference_solution
    assert reference is not None
    completions = [
        [{"role": "assistant", "content": program(list(reference))}],
        [{"role": "assistant", "content": program(all_empty(puzzle))}],
        [{"role": "assistant", "content": program(all_empty(puzzle))}],
    ]
    rewards, grades = solved_rewards(completions, puzzle)
    assert rewards[0] == pytest.approx(30.0)
    assert rewards[1:] == [0.0, 0.0]
    assert [g.status for g in grades] == ["solved", "not-solved", "not-solved"]
    assert all(g.grid is not None or g.status != "solved" for g in grades)


def test_group_rewards_zero_variance_for_two_identical_failures(corpus):
    """The failure mode that makes GRPO learn nothing: identical scores in the group."""
    puzzle = corpus[sorted(corpus)[0]]
    completions = [program(all_empty(puzzle))] * 2
    rewards, _ = solved_rewards(completions, puzzle)
    assert len(set(rewards)) == 1


def test_reward_log_appends_one_record_per_step(tmp_path, corpus):
    puzzle = corpus[sorted(corpus)[0]]
    log = RewardLog(tmp_path / "nested" / "train.jsonl")
    rewards, grades = solved_rewards([program(all_empty(puzzle))], puzzle)
    log.log(puzzle, grades, rewards)
    log.log(puzzle, grades, rewards)
    lines = (tmp_path / "nested" / "train.jsonl").read_text("utf-8").splitlines()
    assert [json.loads(line)["step"] for line in lines] == [1, 2]
    assert json.loads(lines[0])["puzzle_id"] == puzzle.puzzle_id


@pytest.mark.skipif(not SPLITS.exists(), reason="run scripts/freeze_splits.py first")
def test_real_train_split_never_contains_medium():
    puzzles = load_split(CORPUS, SPLITS, "train")
    assert len(puzzles) == 212
    assert {p.difficulty for p in puzzles} == {"small"}
