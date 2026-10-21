import importlib
from pathlib import Path
from types import FunctionType, ModuleType

import pytest

from bench.mine.bindings import HELPER_DEPTH, MODULES, bind_candidate
from bench.mine.measurement import measure

from .helpers import collect_sources


def test_globals_resolve_helpers_nested_code_and_last_constants_without_execution(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "executed"
    trigger = f"open({str(marker)!r}, 'w')"
    candidates, _ = collect_sources(
        tmp_path,
        [
            (
                "notebook",
                f"{trigger}\n"
                f"unsafe = {trigger}\n"
                "offset = 1\n"
                f"@{trigger}\n"
                f"def helper(value: {trigger} = {trigger}) -> {trigger}:\n"
                f"    {trigger}\n"
                "    return other(value) + offset\n"
                "def other(value): return value + offset\n"
                "def candidate(value):\n"
                "    def nested(item): return helper(item) + unsafe\n"
                "    return nested(value)\n"
                "frame.map(candidate)",
            ),
            ("notebook", "offset = 7"),
            ("other", "offset = 99"),
        ],
    )
    assert len(candidates) == 1
    function = bind_candidate(candidates[0])
    namespace = function.__globals__
    assert namespace["offset"] == 7
    assert "unsafe" not in namespace
    assert set(candidates[0].bindings) == {"helper", "other", "offset"}
    helper = namespace["helper"]
    assert isinstance(helper, FunctionType)
    assert helper.__globals__ is namespace
    assert helper.__defaults__ is None
    assert helper.__annotations__ == {}
    measure(candidates[0])
    assert not marker.exists()


def test_globals_use_definition_then_assignment_then_import(tmp_path: Path) -> None:
    candidates, _ = collect_sources(
        tmp_path,
        [
            (
                "notebook",
                "def helper(value): return value\n"
                "helper = 4\nimport math as helper\n"
                "offset = 3\nimport math as offset\n"
                "frame.map(lambda value: helper(value) + offset)",
            )
        ],
    )
    namespace = bind_candidate(candidates[0]).__globals__
    assert isinstance(namespace["helper"], FunctionType)
    assert namespace["offset"] == 3


def test_assigned_lambdas_and_annotated_constants_do_not_execute_expressions(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "executed"
    trigger = f"open({str(marker)!r}, 'w')"
    candidates, _ = collect_sources(
        tmp_path,
        [
            (
                "notebook",
                f"offset: {trigger} = 3\n"
                "unsafe = 2\n"
                f"unsafe = {trigger}\n"
                f"function = lambda value={trigger}: value + offset + unsafe\n"
                "frame.map(function)",
            )
        ],
    )
    function = bind_candidate(candidates[0])
    assert function.__globals__["offset"] == 3
    assert "unsafe" not in function.__globals__
    assert measure(candidates[0]).category == "unknown global"
    assert not marker.exists()


@pytest.mark.parametrize(
    "statement",
    ["import math", "import math as calculations", "from math import sqrt as root"],
)
def test_allowed_imports_compile(tmp_path: Path, statement: str) -> None:
    expression = {
        "import math": "math.sqrt(value)",
        "import math as calculations": "calculations.sqrt(value)",
    }.get(statement, "root(value)")
    candidates, _ = collect_sources(
        tmp_path, [("notebook", f"{statement}\nframe.map(lambda value: {expression})")]
    )
    assert measure(candidates[0]).ok


@pytest.mark.parametrize("module_name", sorted(MODULES))
def test_imports_use_real_allowlisted_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module_name: str
) -> None:
    imported: list[str] = []

    def import_module(name: str) -> ModuleType:
        imported.append(name)
        return ModuleType(name)

    monkeypatch.setattr(importlib, "import_module", import_module)
    candidates, _ = collect_sources(
        tmp_path,
        [("notebook", f"import {module_name} as alias\nframe.map(lambda value: alias)")],
    )
    bind_candidate(candidates[0])
    assert imported == [module_name]


@pytest.mark.parametrize(
    "statement",
    [
        "import os as forbidden",
        "from os import system as forbidden",
        "import numpy.random as forbidden",
        "from .math import sqrt as forbidden",
        "from math import *",
    ],
)
def test_disallowed_imports_stay_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, statement: str
) -> None:
    def import_module(name: str) -> ModuleType:
        pytest.fail(f"unexpected import {name}")

    monkeypatch.setattr(importlib, "import_module", import_module)
    candidates, _ = collect_sources(
        tmp_path, [("notebook", f"{statement}\nframe.map(lambda value: forbidden(value))")]
    )
    function = bind_candidate(candidates[0])
    assert "forbidden" not in function.__globals__


def test_globals_bound_helper_depth_and_cycles(tmp_path: Path) -> None:
    sources = [f"def helper{index}(value): return helper{index + 1}(value)" for index in range(20)]
    sources.extend(
        [
            "frame.map(lambda value: helper0(value))",
            "def recursive(value): return recursive(value)",
            "frame.map(recursive)",
        ]
    )
    candidates, _ = collect_sources(tmp_path, [("notebook", "\n".join(sources))])
    assert len(candidates[0].bindings) == HELPER_DEPTH + 1
    assert "helper9" not in bind_candidate(candidates[0]).__globals__
    assert list(candidates[1].bindings) == ["recursive"]


def test_unsafe_helper_is_excluded(tmp_path: Path) -> None:
    candidates, counts = collect_sources(
        tmp_path,
        [
            (
                "notebook",
                "def helper[Item: print('unsafe')](value): return value\n"
                "frame.map(lambda value: helper(value))",
            )
        ],
    )
    assert not candidates
    assert counts.skipped_definitions.total() == 1
