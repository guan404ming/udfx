import ast
import importlib
from pathlib import Path
from types import ModuleType

import pytest

from bench.mine.measurement import failure_category, measure
from bench.mine.models import Candidate

from .helpers import collect_sources


def test_missing_module_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def import_module(name: str) -> ModuleType:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(importlib, "import_module", import_module)
    candidates, _ = collect_sources(
        tmp_path,
        [("notebook", "import pandas as pd\nframe.map(lambda value: pd.isna(value))")],
    )
    result = measure(candidates[0])
    assert result.category == "missing module pandas"


@pytest.mark.parametrize(
    ("source", "schema", "category"),
    [
        ("lambda value: value.lower()", {"value": "str?"}, None),
        ("lambda value: value / 2", {"value": "float?"}, None),
        ("lambda value: 'word'[value]", {"value": "int?"}, None),
        ("lambda value: value.casefold()", None, "no op .casefold"),
        ("lambda value: sum(value)", None, "unknown callable"),
        ("lambda value: missing(value)", None, "unknown global"),
    ],
)
def test_measure_tries_schemas_and_retains_first_failure(
    source: str, schema: dict[str, str] | None, category: str | None
) -> None:
    node = ast.parse(source, mode="eval").body
    assert isinstance(node, ast.Lambda)
    result = measure(Candidate("notebook", 1, node))
    assert result.schema == schema
    assert result.ok == (schema is not None)
    assert result.category == category
    if category == "no op .casefold":
        assert result.reason == "no op .casefold(str)"


@pytest.mark.parametrize(
    ("reason", "category"),
    [
        ("unknown callable sum", "unknown callable"),
        ("unknown global numpy", "unknown global"),
        ("no op .casefold(str) at helper.<locals>.nested+32", "no op .casefold"),
        ("loops and comprehensions at helper+16", "loops and comprehensions"),
        ("object at 0x123ABC at helper+4", "object at <address>"),
    ],
)
def test_failure_categories(reason: str, category: str) -> None:
    assert failure_category(reason) == category
