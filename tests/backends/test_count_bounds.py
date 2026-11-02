from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.frontend import trace
from udfx.ir import Col, Element, Node, lit
from udfx.ops import apply
from udfx.semantics.bindings import bound_sources
from udfx.semantics.facts import integer_fits
from udfx.semantics.ranges import integer_bounds
from udfx.testing import check_function
from udfx.types import INT64_MAX, STR


@pytest.mark.parametrize(
    ("text", "pattern", "expected"),
    [
        (Col("text", STR), lit("aaaa"), (0, INT64_MAX // 4)),
        (lit("aaaaaaa"), lit("aa"), (0, 3)),
        (apply("slice", Col("text", STR), lit(None), lit(9)), lit("😀a"), (0, 4)),
        (apply("slice", Col("text", STR), lit(-7), lit(None)), lit("e\u0301"), (0, 3)),
        (Col("text", STR), lit(""), (1, INT64_MAX + 1)),
        (apply("slice", Col("text", STR), lit(None), lit(9)), lit(""), (1, 10)),
        (Col("text", STR), Col("pattern", STR), (0, INT64_MAX + 1)),
        (
            apply("slice", Col("text", STR), lit(None), lit(9)),
            Col("pattern", STR),
            (0, 10),
        ),
    ],
)
def test_count_bounds_use_nonoverlapping_pattern_length(
    text: Node, pattern: Node, expected: tuple[int, int]
) -> None:
    assert integer_bounds(apply(".count", text, pattern), INT64_MAX) == expected


def test_count_bounds_resolve_bound_text_and_pattern() -> None:
    text, pattern = Element("text", STR), Element("pattern", STR)
    sliced = apply("slice", Col("source", STR), lit(None), lit(9))
    with bound_sources({text: sliced, pattern: lit("aa")}):
        assert integer_bounds(apply(".count", text, pattern), INT64_MAX) == (0, 4)


@pytest.mark.parametrize("pattern", [lit(""), Col("pattern", STR)])
def test_unbounded_count_sum_keeps_overflow_requirement(pattern: Node) -> None:
    count = apply(".count", Col("text", STR), pattern)
    total = apply("__add__", count, count)
    assert integer_bounds(total, INT64_MAX) is None
    assert not integer_fits(total, size_limit=INT64_MAX)


@pytest.mark.parametrize("backend", backend_params())
def test_summed_counts_remove_overflow_without_losing_faults(backend: Backend[Any]) -> None:
    def counted(text: str) -> int:
        return (
            text.count("aaaa")
            + text.count("ββββ")
            + text.count("😀😀😀😀")
            + text.count("e\u0301e\u0301")
        )

    program = trace(counted, {"text": STR})
    assert integer_bounds(program.value, INT64_MAX) == (0, 4 * (INT64_MAX // 4))
    lowered = backend.lower(program.value)
    assert lowered.exact is None
    assert lowered.overflow is None
    rows = [("",), ("a" * 13,), ("β" * 9 + "😀" * 8 + "e\u0301" * 7,), (None,)]
    assert check_function(counted, {"text": STR.or_none()}, backend, rows) == []
