from dataclasses import replace
from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.frontend.core.comprehensions import compose
from udfx.frontend.core.splitting import joined_tokens
from udfx.ir import FALSE, TRUE, Call, Col, Comprehension, Element, Node, lit
from udfx.ops import apply
from udfx.testing import check_function
from udfx.types import STR, list_of


def joined_maps(separator: str = " ") -> Comprehension:
    source = Col("items", list_of(STR))
    first = Element("first", STR)
    token = Element("token", STR)
    selected = Comprehension(
        apply(".split", first, lit(separator)),
        token,
        token,
        apply("__gt__", apply("len", token), lit(2)),
        FALSE,
    )
    middle = Comprehension(source, first, apply(".join", lit(separator), selected), TRUE, FALSE)
    second = Element("second", STR)
    value = apply(".join", lit(separator), apply("sorted", apply(".split", second, lit(separator))))
    return Comprehension(middle, second, value, TRUE, FALSE)


def rewrite(node: Call) -> Node | None:
    return joined_tokens(node) if node.op.name == ".join" else None


def test_composition_removes_joined_token_roundtrip() -> None:
    original = joined_maps()
    result = compose(original, lambda node: False, rewrite)
    assert isinstance(original.source, Comprehension)
    assert isinstance(original.source.value, Call)
    assert result.source == original.source.source
    assert result.element == original.source.element
    assert result.value == apply(
        ".join", lit(" "), apply("sorted", original.source.value.arguments[1])
    )


@pytest.mark.parametrize("separator", ["", "aba"])
def test_joined_tokens_keep_ambiguous_separators(separator: str) -> None:
    original = joined_maps(separator)
    assert compose(original, lambda node: False, rewrite) == original


@pytest.mark.parametrize("boundary", ["fault", "condition", "rewritten", "unproved"])
def test_composition_keeps_required_boundaries(boundary: str) -> None:
    original = joined_maps()
    assert isinstance(original.source, Comprehension)
    if boundary == "fault":
        original = replace(original, source=replace(original.source, fault=TRUE))
    if boundary == "condition":
        original = replace(original, condition=FALSE)
    if boundary == "unproved":
        source = replace(
            original.source,
            value=apply(".join", lit(" "), Col("unknown", list_of(STR))),
        )
        original = replace(original, source=source)
    selected = (lambda node: node) if boundary == "rewritten" else rewrite
    assert compose(original, lambda node: False, selected) == original


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("separator", [" ", "|", "🦊"])
def test_joined_token_results_match_python(backend: Backend[Any], separator: str) -> None:
    def ordered(items: list[str]) -> list[str]:
        joined = [
            separator.join(word for word in text.split(separator) if len(word) > 2)
            for text in items
        ]
        return [separator.join(sorted(text.split(separator))) for text in joined]

    rows = [
        ([],),
        (["", "a", "bb", "ccc"],),
        ([separator.join(["zzz", "a", "", "aaa", "zzz"])],),
        ([separator.join(["ΣΣΣ", "१२३", "\x00ab", 'a"b', "a\\b"])],),
        (None,),
    ]
    assert check_function(ordered, {"items": list_of(STR).or_none()}, backend, rows) == []
