from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.frontend import trace
from udfx.ir import Col, lit
from udfx.ops import OpKey, apply
from udfx.testing import check_function
from udfx.types import STR, Kind, list_of


@pytest.mark.parametrize("backend", backend_params())
def test_literal_membership_uses_native_set(backend: Backend[Any]) -> None:
    def selected(words: list[str | None]) -> list[str | None]:
        return [word for word in words if word in ("a", "b", None)]

    schema = {"words": list_of(STR.or_none())}
    lowered = backend.lower_program(trace(selected, schema))
    expression = str(lowered.expr).lower()
    for search in ("list_position", "array_position", "array_contains"):
        assert search not in expression
    rows = [(["a", "z", None, "b"],), ([],), ([None],)]
    assert check_function(selected, schema, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
def test_initial_lowercase_membership_skips_string_transforms(
    backend: Backend[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def selected(words: list[str]) -> list[str]:
        excluded = ("the", "and", "i\u0307", "k", "\u03c3", "É", "a.b", "the\n")
        return [word for word in words if word and word[0].lower() + word[1:] not in excluded]

    original = backend.rules.get

    def rule(key: OpKey) -> Any:
        assert key != OpKey(".lower", (Kind.STR,))
        return original(key)

    monkeypatch.setattr(backend.rules, "get", rule)
    rows = [
        ([],),
        (["", "the", "The", "THE", "And", "İ", "i\u0307", "\u212a", "Σ", "\u03c3", "ς", "É"],),
        (["A.b", "Axb", "The\n", "the\r", "中文", "😀", "\n"],),
    ]
    assert check_function(selected, {"words": list_of(STR)}, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("negative", [False, True])
def test_comparison_chain_uses_native_membership(
    backend: Backend[Any], negative: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    schema = {"value": STR.or_none()}
    value = Col("value", schema["value"])
    literals = ("", "a.b", "é", "line\n", "a.b")
    comparisons = [apply("__eq__", lit(item), value) for item in literals]
    if negative:
        comparisons = [apply("not", item) for item in comparisons]
    expression = comparisons[0]
    for condition in comparisons[1:]:
        expression = apply("and" if negative else "or", expression, condition)
    original = backend.rules.get

    def rule(key: OpKey) -> Any:
        assert key != OpKey("__eq__", (Kind.STR, Kind.STR))
        return original(key)

    monkeypatch.setattr(backend.rules, "get", rule)
    lowered = backend.lower(expression)
    assert lowered.exact is None
    values = [None, "", "a.b", "axb", "é", "line", "line\n", "中文"]
    expected = [(item in literals) != negative for item in values]
    assert backend.evaluate([lowered.expr], schema, [(item,) for item in values]) == [expected]


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("nullable", [False, True])
def test_literal_membership_omits_unneeded_null_checks(
    backend: Backend[Any], nullable: bool
) -> None:
    if backend.name not in {"datafusion", "spark"}:
        pytest.skip("Literal membership null handling")

    def selected(value: str | None) -> bool:
        return value is not None and value in ("a", "é", None)

    schema = {"value": STR.or_none() if nullable else STR}
    expression = backend.build(trace(selected, schema))
    assert "coalesce(" not in str(expression).lower()
    values = ["a", "é", "", "other", "😀"] + ([None] if nullable else [])
    assert backend.evaluate([expression], schema, [(value,) for value in values]) == [
        [selected(value) for value in values]
    ]
