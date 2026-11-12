"""Sandbox for model-written `solve(puzzle)` programs.

Two independent layers:

1. **Static checks** (`check_source`) - an AST allow-list. Imports, dunder access, global/nonlocal
   writes, monkeypatching (`x.attr = ...`) and the usual escape hatches are rejected before anything
   runs. This is what `no_cheat` scores.
2. **Runtime budget** (`run_solution`) - a `sys.settrace` step counter that raises `SandboxTimeout`
   after `max_steps` line events (a `SandboxTimeout` is a `BaseException`, so `except Exception`
   inside a candidate cannot swallow it), plus a `SIGALRM` wall clock on POSIX.

Execution happens in this process with a restricted `__builtins__`. Without imports, dunder access,
`getattr` or `open`, a candidate cannot reach the filesystem or network; it also cannot see any
reference solution - the payload it receives is the puzzle only.

Known limit (documented, not hidden): a single C-level allocation such as `"x" * 10**9` is not
interrupted. Models this size do not write that, and the reward is bounded anyway.
"""

from __future__ import annotations

import ast
import json
import re
import signal
import sys
import time
from dataclasses import dataclass
from typing import Any

RUN_TIMEOUT_S = 10.0
MAX_STEPS = 2_000_000

FENCE = re.compile(r"```([A-Za-z0-9_+-]*)\s*\n(.*?)```", re.DOTALL)

# Names a candidate never needs, and every one of them is an escape route.
FORBIDDEN_NAMES = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "open",
        "input",
        "breakpoint",
        "exit",
        "quit",
        "help",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
        "object",
        "type",
        "super",
        "memoryview",
        "classmethod",
        "staticmethod",
        "property",
        "builtins",
        "__builtins__",
    }
)

FORBIDDEN_ATTRIBUTES = frozenset({"format", "format_map"})

_OPERATOR_NODES = {
    cls.__name__
    for cls in (
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.LShift,
        ast.RShift,
        ast.BitOr,
        ast.BitXor,
        ast.BitAnd,
        ast.MatMult,
        ast.UAdd,
        ast.USub,
        ast.Not,
        ast.Invert,
        ast.And,
        ast.Or,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.Is,
        ast.IsNot,
        ast.In,
        ast.NotIn,
    )
}

ALLOWED_NODES = frozenset(
    {
        "Module",
        "FunctionDef",
        "arguments",
        "arg",
        "Return",
        "Assign",
        "AugAssign",
        "AnnAssign",
        "For",
        "While",
        "If",
        "Break",
        "Continue",
        "Pass",
        "Expr",
        "List",
        "Tuple",
        "Dict",
        "Set",
        "Subscript",
        "Slice",
        "BinOp",
        "UnaryOp",
        "BoolOp",
        "Compare",
        "Constant",
        "Name",
        "Load",
        "Store",
        "Del",
        "Call",
        "keyword",
        "Lambda",
        "IfExp",
        "ListComp",
        "DictComp",
        "SetComp",
        "GeneratorExp",
        "comprehension",
        "JoinedStr",
        "FormattedValue",
        "Attribute",
        "Starred",
        "Try",
        "ExceptHandler",
        "Raise",
        "Assert",
        "ClassDef",
        "With",
        "withitem",
        "Yield",
    }
) | _OPERATOR_NODES

SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "Exception": Exception,
    "ValueError": ValueError,
    "IndexError": IndexError,
    "KeyError": KeyError,
    "TypeError": TypeError,
    "ZeroDivisionError": ZeroDivisionError,
    "StopIteration": StopIteration,
}


class SandboxTimeout(BaseException):
    """Raised by the step counter / wall clock. BaseException so candidates cannot catch it."""


@dataclass
class RunResult:
    status: str  # ok | invalid | error | timeout
    grid: list[str] | None = None
    detail: str = ""
    seconds: float = 0.0
    steps: int = 0

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "grid": self.grid,
            "detail": self.detail,
            "seconds": round(self.seconds, 4),
            "steps": self.steps,
        }


def extract_code(completion: str) -> str | None:
    """Return the single fenced code block, or None unless there is exactly one."""
    if not isinstance(completion, str):
        return None
    blocks = FENCE.findall(completion)
    if len(blocks) != 1:
        return None
    return blocks[0][1]


def fence_count(completion: str) -> int:
    """Number of fenced blocks; `extract_code` accepts exactly one."""
    if not isinstance(completion, str):
        return 0
    return len(FENCE.findall(completion))


def _parse(source: str) -> tuple[ast.Module | None, list[str]]:
    if not isinstance(source, str) or not source.strip():
        return None, ["empty"]
    try:
        return ast.parse(source), []
    except SyntaxError as exc:
        return None, [f"syntax:{exc.msg}"]


def _shape_violations(tree: ast.Module) -> list[str]:
    """Problems that make the program unusable but are not escape attempts."""
    violations: list[str] = []
    solves = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "solve"]
    if not solves:
        violations.append("no-solve")
    elif len(solves) > 1:
        violations.append("solve-defined-twice")
    else:
        args = solves[0].args
        positional = [*args.posonlyargs, *args.args]
        if len(positional) != 1 or args.vararg or args.kwarg or args.kwonlyargs:
            violations.append(f"signature:{len(positional)}-positional")
        elif positional[0].arg != "puzzle":
            violations.append(f"signature:first-arg-{positional[0].arg}")

    for node in ast.walk(tree):
        kind = type(node).__name__
        if kind not in ALLOWED_NODES:
            violations.append(f"node:{kind}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, bytes):
            violations.append("bytes-literal")
    return violations


def _escape_violations(tree: ast.Module) -> list[str]:
    """The §6.2 cheat vectors: imports, dunder/attribute escapes, forbidden builtins, global writes."""
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            violations.append("import")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            violations.append(f"node:{type(node).__name__}")
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES or node.id.startswith("__"):
                violations.append(f"name:{node.id}")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                violations.append(f"dunder:{node.attr}")
            elif node.attr in FORBIDDEN_ATTRIBUTES:
                violations.append(f"attr:{node.attr}")
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                violations.append(f"monkeypatch:{node.attr}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, bytes):
            violations.append("bytes-literal")
    return violations


def _unique(codes: list[str]) -> list[str]:
    """One code per category keeps reward audits readable."""
    seen: set[str] = set()
    unique = []
    for item in codes:
        key = item.split(":", 1)[0]
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def escape_violations(source: str) -> list[str]:
    """Only the cheat-vector subset; syntax and signature problems are not cheating."""
    tree, problems = _parse(source)
    if tree is None:
        return []
    return _unique(_escape_violations(tree))


def check_source(source: str) -> list[str]:
    """Static allow-list. Returns violation codes; an empty list means the source may run."""
    tree, problems = _parse(source)
    if tree is None:
        return problems
    return _unique([*_shape_violations(tree), *_escape_violations(tree)])


def compile_solution(source: str) -> Any:
    """Compile and exec in a restricted namespace; returns the `solve` function."""
    compiled = compile(source, "<candidate>", "exec")
    namespace: dict[str, Any] = {"__builtins__": dict(SAFE_BUILTINS), "__name__": "candidate"}
    exec(compiled, namespace)  # noqa: S102 - guarded by check_source
    return namespace["solve"]


def run_solution(
    source: str,
    payload: dict,
    timeout: float = RUN_TIMEOUT_S,
    max_steps: int = MAX_STEPS,
) -> RunResult:
    """Check, compile and run `solve(payload)` under a step and wall-clock budget."""
    violations = check_source(source)
    if violations:
        return RunResult("invalid", detail="; ".join(violations))

    started = time.perf_counter()
    counter = {"steps": 0}
    previous_trace = sys.gettrace()
    alarm_armed = False
    previous_handler = None

    def tracer(frame, event, arg):
        if event == "line":
            counter["steps"] += 1
            if counter["steps"] > max_steps:
                raise SandboxTimeout(f"step budget exhausted ({max_steps})")
        return tracer

    def on_alarm(signum, frame):
        raise SandboxTimeout(f"wall clock exceeded ({timeout:.0f}s)")

    if hasattr(signal, "setitimer"):
        previous_handler = signal.signal(signal.SIGALRM, on_alarm)
        signal.setitimer(signal.ITIMER_REAL, max(0.05, timeout))
        alarm_armed = True

    try:
        try:
            solve = compile_solution(source)
        except (SyntaxError, ValueError) as exc:
            detail = getattr(exc, "msg", str(exc))
            return RunResult("invalid", detail=f"syntax:{detail}", seconds=time.perf_counter() - started)
        sys.settrace(tracer)
        result = solve(json.loads(json.dumps(payload)))
    except SandboxTimeout as exc:
        return RunResult("timeout", detail=str(exc), seconds=time.perf_counter() - started, steps=counter["steps"])
    except BaseException as exc:  # noqa: BLE001 - candidates may raise anything
        return RunResult(
            "error",
            detail=f"{type(exc).__name__}: {exc}"[:200],
            seconds=time.perf_counter() - started,
            steps=counter["steps"],
        )
    finally:
        sys.settrace(previous_trace)
        if alarm_armed:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)

    elapsed = time.perf_counter() - started
    grid = _coerce_grid(result)
    if grid is None:
        return RunResult(
            "error",
            detail=f"return-not-grid:{type(result).__name__}",
            seconds=elapsed,
            steps=counter["steps"],
        )
    return RunResult("ok", grid=grid, seconds=elapsed, steps=counter["steps"])


def _coerce_grid(result: Any) -> list[str] | None:
    """Accept a list/tuple of strings; anything else is a malformed return, not a crash."""
    if not isinstance(result, (list, tuple)):
        return None
    if not all(isinstance(row, str) for row in result):
        return None
    return list(result)


__all__ = [
    "MAX_STEPS",
    "RUN_TIMEOUT_S",
    "RunResult",
    "SandboxTimeout",
    "check_source",
    "compile_solution",
    "escape_violations",
    "extract_code",
    "fence_count",
    "run_solution",
]
