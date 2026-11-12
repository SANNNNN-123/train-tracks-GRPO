"""Training-side helpers: the puzzle sampler and the `reward_solved` group function.

The sudoku notebook samples one puzzle per reward call and grades every completion in that call on
it. We keep that trick, because GRPO subtracts the *group* mean: all generations of one prompt must
face the same puzzle or the advantages are meaningless. Two consequences, both deliberate:

  * the dataset is a single prompt repeated, so the corpus is a rollout budget, not 608 rows;
  * `per_device_train_batch_size` must stay 1, so one call holds exactly `num_generations`
    completions of one and the same prompt.

`TrainSampler` is seeded and records the puzzle id it returned, so any training run can be replayed.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from .puzzle import Puzzle, load_jsonl
from .rewards import MAX_STEPS, Grade, completion_text, grade


def load_split(puzzles_path: str | Path, splits_path: str | Path, split: str = "train") -> list[Puzzle]:
    """Load exactly the puzzles of one split, in sorted-id order."""
    puzzles = {p.puzzle_id: p for p in load_jsonl(puzzles_path)}
    declared = json.loads(Path(splits_path).read_text(encoding="utf-8"))["splits"]
    if split not in declared:
        raise KeyError(f"split {split!r} not in {splits_path}; have {sorted(declared)}")
    missing = [pid for pid in declared[split] if pid not in puzzles]
    if missing:
        raise ValueError(f"{len(missing)} {split} puzzle(s) missing from the corpus, e.g. {missing[:3]}")
    return [puzzles[pid] for pid in declared[split]]


@dataclass
class TrainSampler:
    """Deterministic puzzle sampler; one call per training step."""

    puzzles: list[Puzzle]
    seed: int = 20261110
    _rng: random.Random = field(init=False, repr=False)
    _recent: list[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.puzzles:
            raise ValueError("TrainSampler needs at least one puzzle")
        self._rng = random.Random(self.seed)

    def next(self) -> Puzzle:
        puzzle = self._rng.choice(self.puzzles)
        self._recent.append(puzzle.puzzle_id)
        return puzzle

    @property
    def history(self) -> list[str]:
        return list(self._recent)


def solved_rewards(
    completions,
    puzzle: Puzzle,
    max_steps: int = MAX_STEPS,
) -> tuple[list[float], list[Grade]]:
    """The `reward_solved` body: grade every completion of one group on the same puzzle.

    Returns the per-completion reward (`solved + shaping`) and the full grades for logging.
    """
    rewards: list[float] = []
    grades: list[Grade] = []
    for completion in completions:
        result = grade(completion_text(completion), puzzle, max_steps=max_steps)
        grades.append(result)
        rewards.append(result.solved + result.shaping)
    return rewards, grades


@dataclass
class RewardLog:
    """Append-only JSONL record of what training actually rewarded.

    The trainer only prints means; this keeps the per-step puzzle id and status counts so a bad run
    can be diagnosed afterwards instead of re-run. Same fields as `outputs/eval`, minus the grids.
    """

    path: Path
    step: int = 0

    def log(self, puzzle: Puzzle, grades: list[Grade], rewards: list[float]) -> None:
        self.step += 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "step": self.step,
            "puzzle_id": puzzle.puzzle_id,
            "size": f"{puzzle.rows}x{puzzle.cols}",
            "rewards": [round(value, 4) for value in rewards],
            "mean_reward": round(sum(rewards) / len(rewards), 4) if rewards else 0.0,
            "statuses": [g.status for g in grades],
            "shaping": [round(g.shaping, 4) for g in grades],
            "totals": [round(g.total, 4) for g in grades],
        }
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record) + "\n")


__all__ = ["RewardLog", "TrainSampler", "load_split", "solved_rewards"]
