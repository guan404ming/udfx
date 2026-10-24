from collections.abc import Callable

import pytest

from udfx.errors import Unsupported
from udfx.frontend import trace
from udfx.ir import FALSE, Col
from udfx.ops import apply
from udfx.types import STR, list_of


@pytest.mark.parametrize(
    ("function", "operation"),
    [
        (lambda left, right: len(set(left) & set(right)), "intersection"),
        (lambda left, right: len(set(left) | set(right)), "union"),
        (lambda left, right: len(set(left) - set(right)), "difference"),
        (lambda left, right: len(set(left) ^ set(right)), "symmetric_difference"),
        (lambda left, right: len(set(left).intersection(set(right))), "intersection"),
        (lambda left, right: len(set(left).union(set(right))), "union"),
        (lambda left, right: len(set(left).difference(set(right))), "difference"),
        (
            lambda left, right: len(set(left).symmetric_difference(set(right))),
            "symmetric_difference",
        ),
    ],
)
def test_set_algebra_traces(function: Callable[..., object], operation: str) -> None:
    schema = {"left": list_of(STR), "right": list_of(STR)}
    program = trace(function, schema)
    operands = [apply("list", Col(name, type)) for name, type in schema.items()]
    assert program.value == apply(f"set_{operation}_size", *operands)
    assert program.fault == FALSE


@pytest.mark.parametrize(
    ("function", "operation"),
    [
        (lambda left, right: set(left).issubset(set(right)), "issubset"),
        (lambda left, right: set(left).issuperset(set(right)), "issuperset"),
        (lambda left, right: set(left).isdisjoint(set(right)), "isdisjoint"),
    ],
)
def test_set_predicates_trace(function: Callable[..., object], operation: str) -> None:
    schema = {"left": STR, "right": STR}
    program = trace(function, schema)
    operands = [apply("list", Col(name, type)) for name, type in schema.items()]
    assert program.value == apply(f"set_{operation}", *operands)
    assert program.fault == FALSE


@pytest.mark.parametrize(
    "function",
    [
        lambda left, right: set(left) & set(right),
        lambda left, right: set(left).union(set(right)),
        lambda left, right: set(left) & right,
        lambda left, right: left | set(right),
    ],
)
def test_set_results_and_mixed_operators_stay_unsupported(function: Callable[..., object]) -> None:
    with pytest.raises(Unsupported):
        trace(function, {"left": STR, "right": STR})


def test_set_mutation_stays_unsupported() -> None:
    def mutate(text: str) -> int:
        values = set(text)
        alias = values
        values &= set("a")
        return len(alias)

    with pytest.raises(Unsupported, match="in-place set mutation"):
        trace(mutate, {"text": STR})


def test_sorted_set_difference_combines_union_and_intersection() -> None:
    def difference(left: list[str], right: list[str]) -> list[str]:
        return sorted((set(left) | set(right)) - (set(right) & set(left)))

    schema = {"left": list_of(STR), "right": list_of(STR)}
    program = trace(difference, schema)
    operands = [apply("list", Col(name, type)) for name, type in schema.items()]
    assert program.value == apply("set_symmetric_difference", *operands)
