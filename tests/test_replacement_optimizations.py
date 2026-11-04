from collections.abc import Callable
from copy import copy
from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.backends.engines.datafusion import DataFusionBackend
from udfx.errors import Unsupported
from udfx.frontend import trace
from udfx.frontend.core.translation import combined_replacements
from udfx.ir import Call, Col, lit
from udfx.ops import apply
from udfx.testing import check_function
from udfx.types import STR


@pytest.mark.parametrize(
    "function",
    [
        lambda value: value.replace("a", "b").replace("b", "c"),
        lambda value: value.replace("a", "b").replace("a", "c"),
        lambda value: value.replace("a", "").replace("b", "a"),
        lambda value: value.replace("😀", "é").replace("é", "\u0301"),
        lambda value: value.replace("a", "b").replace("b", "c").replace("c", ""),
        lambda value: value.translate({97: "b", 98: "a"}).replace("a", "c"),
    ],
)
def test_character_replacements_compose_in_order(
    function: Callable[[str], str], monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = DataFusionBackend()
    schema = {"value": STR.or_none()}
    program = trace(function, schema)
    assert isinstance(program.value, Call)
    assert program.value.op.name == ".replace"
    rewritten = combined_replacements(program.value)
    assert isinstance(rewritten, Call)
    assert rewritten.op.name == ".translate"
    assert isinstance(rewritten.arguments[0], Col)
    baseline = DataFusionBackend()
    monkeypatch.setattr(baseline, "rules", copy(baseline.rules))
    rewrite = baseline.rules.rewrite_call
    monkeypatch.setattr(
        baseline.rules,
        "rewrite_call",
        lambda node: None if node.op.name == ".replace" else rewrite(node),
    )
    before = baseline.lower_program(program)
    after = backend.lower_program(program)
    values = ("", "abca", "aabb", "😀é\u0301ab", "a\x00b")
    rows = [(value,) for value in (*values, None)]
    exact = backend.evaluate(
        [backend.predicate(before.exact), backend.predicate(after.exact)], schema, rows
    )
    assert exact[0] == exact[1]
    expected = [function(value) for value in values]
    assert backend.evaluate([before.expr, after.expr], schema, rows[:-1]) == [expected, expected]
    assert check_function(function, schema, backend, rows) == []


@pytest.mark.parametrize(
    "function",
    [
        lambda value: value.replace("a", "bb").replace("b", "c"),
        lambda value: value.replace("ab", "b").replace("b", "c"),
        lambda value: value.replace("", "b").replace("b", "c"),
        lambda value: value.replace("a", "b").replace("b", "cc"),
        lambda value: value.replace("a", "b").replace("", "c"),
        lambda value: value.replace("a", "b"),
    ],
)
def test_non_character_replacements_keep_their_operations(
    function: Callable[[str], str],
) -> None:
    program = trace(function, {"value": STR})
    assert isinstance(program.value, Call)
    assert combined_replacements(program.value) is program.value


def test_dynamic_replacement_keeps_its_operation() -> None:
    def function(value: str, old: str) -> str:
        return value.replace("a", "b").replace(old, "c")

    program = trace(function, {"value": STR, "old": STR})
    assert isinstance(program.value, Call)
    assert combined_replacements(program.value) is program.value


@pytest.mark.parametrize("backend", backend_params())
def test_replacement_fusion_is_an_engine_choice(backend: Backend[Any]) -> None:
    program = trace(lambda value: value.replace("a", "b").replace("b", "c"), {"value": STR})
    assert isinstance(program.value, Call)
    assert program.value.op.name == ".replace"
    rewritten = backend.rules.rewrite_call(program.value)
    if backend.name == "datafusion":
        assert isinstance(rewritten, Call)
        assert rewritten.op.name == ".translate"
    else:
        assert rewritten is None


def test_counted_replacement_remains_unsupported() -> None:
    with pytest.raises(Unsupported, match=r"no op \.replace"):
        trace(lambda value: value.replace("a", "b", 1).replace("b", "c"), {"value": STR})


def test_replacement_fusion_preserves_the_shared_source() -> None:
    source = apply(".strip", Col("value", STR))
    first = apply(".replace", source, lit("a"), lit("b"))
    second = apply(".replace", first, lit("b"), lit("c"))
    third = apply(".replace", second, lit("c"), lit("d"))
    rewritten = combined_replacements(third)
    assert isinstance(rewritten, Call)
    assert rewritten.op.name == ".translate"
    assert rewritten.arguments[0] is source
