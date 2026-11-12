"""Sandbox tests: the eight escape vectors from plan.md §6.2, plus timeouts and malformed returns."""

from __future__ import annotations

import pytest

from tt_env.sandbox import check_source, extract_code, run_solution

PAYLOAD = {"rows": 2, "cols": 2, "row_clues": [2, 2], "col_clues": [2, 2], "givens": [], "start": [0, 0, "W"], "end": [1, 1, "E"]}


def fence(code: str) -> str:
    return f"```python\n{code}\n```"


GOOD = "def solve(puzzle):\n    return ['..' * puzzle['cols'] for _ in range(puzzle['rows'])]"


def test_good_program_runs():
    run = run_solution(GOOD, PAYLOAD)
    assert run.status == "ok", run.detail
    assert run.grid == ["....", "...."]
    assert run.steps > 0


@pytest.mark.parametrize(
    ("code", "code_prefix"),
    [
        ("def solve(puzzle):\n    import tracks\n    return []", "import"),
        ("def solve(puzzle):\n    return __import__('os').listdir('.')", "name"),
        ("def solve(puzzle):\n    return open('assets/a.png').read()", "name"),
        ("def solve(puzzle):\n    return str(__builtins__)", "name"),
        ("def solve(puzzle):\n    return puzzle.__class__.__mro__", "dunder"),
        ("def solve(puzzle):\n    puzzle.clear = None\n    return []", "monkeypatch"),
        ("def solve(puzzle):\n    global cache\n    cache = 1\n    return []", "node"),
        ("def solve(puzzle):\n    return '{0.__class__}'.format(puzzle)", "attr"),
        ("def solve(puzzle):\n    return eval('1')", "name"),
        ("def solve(puzzle):\n    return getattr(puzzle, 'keys')()", "name"),
    ],
)
def test_escape_vectors_are_rejected(code, code_prefix):
    violations = check_source(code)
    assert any(v.startswith(code_prefix) for v in violations), violations


def test_while_true_times_out_instead_of_hanging():
    code = "def solve(puzzle):\n    while True:\n        pass\n"
    run = run_solution(code, PAYLOAD, max_steps=5_000)
    assert run.status == "timeout"


def test_timeout_cannot_be_swallowed_by_except_exception():
    code = "def solve(puzzle):\n    try:\n        while True:\n            pass\n    except Exception:\n        return ['....', '....']\n"
    run = run_solution(code, PAYLOAD, max_steps=5_000)
    assert run.status == "timeout"


def test_recursion_is_an_error_not_a_crash():
    code = "def solve(puzzle):\n    return solve(puzzle)\n"
    run = run_solution(code, PAYLOAD)
    assert run.status == "error"
    assert "RecursionError" in run.detail


@pytest.mark.parametrize(
    ("result_expr", "detail_prefix"),
    [
        ("'nope'", "return-not-grid"),
        ("[[1, 2], [3, 4]]", "return-not-grid"),
        ("None", "return-not-grid"),
    ],
)
def test_malformed_returns(result_expr, detail_prefix):
    code = f"def solve(puzzle):\n    return {result_expr}\n"
    run = run_solution(code, PAYLOAD)
    assert run.status == "error"
    assert run.detail.startswith(detail_prefix)


def test_exceptions_are_reported_as_errors():
    code = "def solve(puzzle):\n    return 1 / 0\n"
    run = run_solution(code, PAYLOAD)
    assert run.status == "error"
    assert "ZeroDivisionError" in run.detail


@pytest.mark.parametrize(
    "completion",
    [
        "no code here",
        fence(GOOD) + fence(GOOD),
        "```python\ndef solve(puzzle):\n    return []\n",
        fence("def solver(puzzle):\n    return []"),
        fence("def solve():\n    return []"),
        fence("def solve(puzzle, extra=1):\n    return []"),
    ],
)
def test_extract_code_rejects_bad_shapes(completion):
    source = extract_code(completion)
    if source is not None:
        assert check_source(source), "source should be rejected"


def test_extract_code_accepts_a_single_block():
    assert extract_code(fence(GOOD)).strip() == GOOD
    assert extract_code("before\n```\n" + GOOD + "\n```\nafter").strip() == GOOD


# The constructs a model actually needs for a backtracking solver. If the allow-list rejects this,
# it rejects every plausible answer, and the run trains on the wrong lesson.
DFS_SKELETON = '''def solve(puzzle):
    rows = puzzle["rows"]
    cols = puzzle["cols"]
    empty = ".."
    grid = [[empty for _ in range(cols)] for _ in range(rows)]
    offsets = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}

    def inside(r, c):
        return 0 <= r < rows and 0 <= c < cols

    def walk(r, c, depth, seen):
        if depth > 4 or (r, c) in seen:
            return False
        seen.add((r, c))
        for name, (dr, dc) in offsets.items():
            nr, nc = r + dr, c + dc
            if inside(nr, nc) and walk(nr, nc, depth + 1, seen):
                return True
        return False

    try:
        walk(0, 0, 0, set())
    except Exception:
        pass

    counts = [sum(1 for piece in row if piece != empty) for row in grid]
    assert len(counts) == rows or True
    return ["".join(row) for row in grid]
'''


def test_realistic_dfs_skeleton_is_allowed():
    assert check_source(DFS_SKELETON) == []
    run = run_solution(DFS_SKELETON, PAYLOAD)
    assert run.status == "ok", run.detail
    assert run.grid == ["....", "...."]


def test_deep_recursion_returns_an_error_not_a_crash():
    code = "def solve(puzzle):\n    def down(n):\n        return down(n + 1)\n    down(0)\n"
    run = run_solution(code, PAYLOAD)
    assert run.status == "error"
    assert "RecursionError" in run.detail
