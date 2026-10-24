from __future__ import annotations

from collections.abc import Callable, Iterable

import pytest

from udfx.explain import render
from udfx.frontend import trace
from udfx.ir import FALSE, Call, Comprehension, Guarded
from udfx.types import INT, STR, list_of


@pytest.mark.parametrize(
    "function, expected",
    [
        (lambda value: value.lower().casefold(), ".casefold(col value)"),
        (lambda value: value.casefold().casefold(), ".casefold(col value)"),
        (
            lambda value: value.title().swapcase().capitalize().casefold(),
            ".casefold(.capitalize(.swapcase(.title(col value))))",
        ),
        (lambda value: value.upper().lower(), "guard(.lower(col value), .isascii(col value))"),
        (
            lambda value: value.lower().capitalize().casefold(),
            ".casefold(.capitalize(.lower(col value)))",
        ),
    ],
)
def test_case_chain_keeps_backend_choices(function: Callable[[str], str], expected: str) -> None:
    program = trace(function, {"value": STR})
    assert render(program.value) == expected


def test_partitioned_sum_is_guarded_by_length() -> None:
    def total(values: list[int]) -> int:
        return values[0] + sum(values[1:-1]) + values[-1]

    program = trace(total, {"values": list_of(INT).or_none()})
    assert isinstance(program.value, Guarded)


@pytest.mark.parametrize("separator", ["_", ".", "\\", "\n", "]"])
def test_joined_split_prefix_becomes_one_match(separator: str) -> None:
    program = trace(lambda text: separator.join(text.split(separator)[:3]), {"text": STR})
    assert "re.match(" in render(program.value)
    assert ".split(" not in render(program.value)


@pytest.mark.parametrize("reduction", [any, all])
def test_literal_generator_reduction_short_circuits(
    reduction: Callable[[Iterable[bool]], bool],
) -> None:
    program = trace(lambda text: reduction(word[0] in text for word in ("a", "")), {"text": STR})
    assert not isinstance(program.value, Comprehension)
    assert program.fault != FALSE


def test_list_reduction_keeps_the_comprehension() -> None:
    program = trace(lambda text: any([word[0] in text for word in ("a", "")]), {"text": STR})
    assert isinstance(program.value, Call)
    assert isinstance(program.value.arguments[0], Comprehension)
