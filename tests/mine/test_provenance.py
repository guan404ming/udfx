import importlib
from pathlib import Path
from types import ModuleType

import pytest

from bench.mine.measurement import measure

from .helpers import collect_sources


@pytest.mark.parametrize(
    ("statement", "external"),
    [
        ("import tensorflow as target", True),
        ("from sklearn.preprocessing import normalize as target", True),
        ("import numpy.random as target", True),
        ("from .math import sqrt as target", True),
        ("", True),
        ("target = load_data()", False),
        ("target: object = load_data()", False),
        ("target, other = load_data()", False),
        ("if ready:\n    target = load_data()", False),
        ("class target: pass", False),
        ("async def target(value): return value", False),
        ("if ready:\n    import re as target", False),
        ("from re import missing as target", False),
    ],
)
def test_unknown_global_classification_uses_notebook_origins(
    tmp_path: Path, statement: str, external: bool
) -> None:
    candidates, _ = collect_sources(
        tmp_path, [("notebook", f"{statement}\nframe.map(lambda value: target(value))")]
    )
    result = measure(candidates[0])
    assert result.category == "unknown global"
    assert result.third_party_names == (["target"] if external else [])


def test_helper_failure_classifies_import_alias_without_importing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_import = importlib.import_module

    def import_module(name: str) -> ModuleType:
        assert name != "tensorflow"
        return original_import(name)

    monkeypatch.setattr(importlib, "import_module", import_module)
    candidates, _ = collect_sources(
        tmp_path,
        [
            (
                "notebook",
                "import tensorflow as model\n"
                "def helper(value): return model.convert(value)\n"
                "frame.map(lambda value: helper(value))",
            )
        ],
    )
    result = measure(candidates[0])
    assert result.reason == "unknown global model"
    assert result.third_party_names == ["model"]


def test_deduplication_keeps_name_origins_separate(tmp_path: Path) -> None:
    call = "frame.map(lambda value: target(value))"
    candidates, _ = collect_sources(
        tmp_path,
        [
            ("undefined", call),
            ("defined", f"target = load_data()\n{call}"),
            ("external", f"import tensorflow as target\n{call}"),
            ("allowed", f"from re import missing as target\n{call}"),
            ("duplicate", f"unused = 1\n{call}"),
        ],
    )
    results = [measure(candidate) for candidate in candidates]
    assert len(results) == 4
    assert [result.third_party_names for result in results] == [["target"], [], ["target"], []]
    assert [result.occurrences for result in results] == [2, 1, 1, 1]


def test_other_failures_with_external_dependencies_stay_in_scope(tmp_path: Path) -> None:
    candidates, _ = collect_sources(
        tmp_path,
        [("notebook", "import tensorflow as target\nframe.map(lambda value: sum(value) + target)")],
    )
    result = measure(candidates[0])
    assert result.category == "no op sum"
    assert result.third_party_names == []
