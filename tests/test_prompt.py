"""Prompt tests: the contract is the most likely silent failure, so check it explicitly."""

from __future__ import annotations

from tt_env.prompt import CONTRACT, messages, prompt_text, render
from tt_env.rules import decode_grid

PAYLOAD = {
    "rows": 3,
    "cols": 4,
    "row_clues": [2, 3, 1],
    "col_clues": [1, 2, 1, 2],
    "givens": [[1, 0, "EW"], [2, 3, "NS"]],
    "start": [1, 0, "W"],
    "end": [2, 3, "S"],
}


def test_training_prompt_is_puzzle_free():
    """The harness injects the puzzle; a prompt that changes per puzzle is not design A."""
    text = prompt_text()
    assert "row_clues" in text, "the contract still names the fields"
    assert '"givens":' not in text and "[[1,0" not in text
    assert "row_clues\":[" not in text.replace("\n", "")


def test_training_prompt_is_stable():
    assert prompt_text() == prompt_text()
    assert render() == prompt_text()
    assert messages()[1]["content"] == prompt_text()


def test_render_with_a_payload_shows_the_instance():
    text = render(PAYLOAD)
    assert '"rows":3' in text and '"cols":4' in text
    assert '"givens":[[1,0,"EW"],[2,3,"NS"]]' in text
    assert '"start":[1,0,"W"]' in text
    assert text.startswith(prompt_text())


def test_prompt_fits_the_t4_budget():
    """~1000 tokens of instructions, not four thousand; measured in characters."""
    text = prompt_text()
    assert len(text) < 6_000, f"prompt is {len(text)} chars before the model writes anything"


def test_prompt_mentions_the_one_code_block_and_the_size_range():
    text = prompt_text()
    assert "```python" in text
    assert "6..10" in text or "6x6" in text
    assert "10 seconds" in text


def test_prompt_asks_for_a_compact_reply():
    """A verbose 2B answer blows the completion budget and is then truncated mid-block."""
    text = prompt_text()
    assert "compact" in text
    assert "no comments" in text
    assert "under 80 lines" in text


def test_messages_are_text_only():
    for chat in (messages(), messages(PAYLOAD)):
        assert [m["role"] for m in chat] == ["system", "user"]
        assert all(isinstance(m["content"], str) for m in chat)
        assert not any("image" in m for m in chat)


def test_documented_example_is_valid():
    example = ["EW..NS", "..NSEW"]
    cells, failures = decode_grid(example, 2, 3)
    assert not failures
    assert cells[0] == ["EW", "..", "NS"]
    assert cells[1] == ["..", "NS", "EW"]


def test_contract_mentions_the_packed_encoding():
    assert "2 * cols" in CONTRACT
    assert ".." in CONTRACT
