import math
import random
import re
from collections.abc import Callable
from datetime import datetime

import pytest

from udfx.backends.polars import PolarsBackend
from udfx.frontend import trace
from udfx.frontend.core.branches import factor_branches
from udfx.graphs import unique_size
from udfx.ir import FALSE, Call, Col, If, Lit, children, lit
from udfx.ops import apply
from udfx.testing import check_function
from udfx.types import BOOL, DATETIME, STR, Type

POLARS = PolarsBackend()


@pytest.mark.parametrize(
    "function",
    [
        lambda text, condition: text.upper() + "/" if condition else text.upper() + "!",
        lambda text, condition: "/" + text.upper() if condition else "!" + text.upper(),
        lambda text, condition: text.upper() + "/" if condition else text.upper(),
        lambda text, condition: "/" + text.upper() if condition else text.upper(),
        lambda text, condition: text.upper() if condition else text.upper() + "/",
        lambda text, condition: text.upper() if condition else "/" + text.upper(),
    ],
)
def test_concatenation_factors_the_shared_operand(function: Callable[[str, bool], str]) -> None:
    schema = {"text": STR, "condition": BOOL}
    program = trace(function, schema)
    assert isinstance(program.value, Call)
    assert program.value.op.name == "__add__"
    assert sum(isinstance(argument, If) for argument in program.value.arguments) == 1
    assert sum(isinstance(argument, Call) for argument in program.value.arguments) == 1
    rows = [("", False), ("ß😀", True), ("x", False), ("a", True)]
    assert check_function(function, schema, POLARS, rows) == []


@pytest.mark.parametrize(
    "function",
    [
        lambda text, condition: (
            text.upper() + "/" + text.strip() if condition else text.upper() + "!" + text.lower()
        ),
        lambda text, condition: (
            text.strip() + ("/" + text.upper())
            if condition
            else text.lower() + ("!" + text.upper())
        ),
        lambda text, condition: (
            text.upper() + "/" + text.strip() + ":end"
            if condition
            else text.upper() + "!" + text.lower() + ":end"
        ),
    ],
)
def test_concat_parts_share_edges_without_repeating_conditions(
    function: Callable[[str, bool], str],
) -> None:
    schema = {"text": STR, "condition": BOOL}
    program = trace(function, schema)
    nodes = []
    pending = [program.value]
    while pending:
        node = pending.pop()
        nodes.append(node)
        pending.extend(children(node))
    assert sum(isinstance(node, Call) and node.op.name == ".upper" for node in nodes) == 1
    assert sum(isinstance(node, Col) and node.name == "condition" for node in nodes) == 1
    rows = [("", False), (" ß😀 ", True), ("A b", False), ("a", True)]
    assert check_function(function, schema, POLARS, rows) == []


@pytest.mark.parametrize("depth", [12, 24])
def test_factoring_preserves_shared_concatenation_graphs(depth: int) -> None:
    def repeated(text: str, condition: bool) -> str:
        for _ in range(depth):
            text = text + text
        return text + "a" if condition else text + "b"

    schema = {"text": STR, "condition": BOOL}
    source = Col("text", STR)
    for _ in range(depth):
        source = apply("__add__", source, source)
    node = If(
        Col("condition", BOOL),
        apply("__add__", source, lit("a")),
        apply("__add__", source, lit("b")),
        STR,
    )
    (factored,) = factor_branches((node,))
    assert unique_size((factored,)) <= depth + 10
    assert isinstance(factored, Call)
    current = factored.arguments[0]
    for _ in range(depth):
        assert isinstance(current, Call)
        assert current.op.name == "__add__"
        assert current.arguments[0] is current.arguments[1]
        current = current.arguments[0]
    assert current == Col("text", STR)
    if depth == 12:
        rows = [("", True), ("", False), ("x", True)]
        assert check_function(repeated, schema, POLARS, rows) == []


def test_flattened_concatenation_keeps_balanced_depth() -> None:
    leaves = [Col(f"part{index}", STR) for index in range(1024)]
    level = leaves
    while len(level) > 1:
        level = [apply("__add__", *level[index : index + 2]) for index in range(0, len(level), 2)]
    condition = Col("condition", BOOL)
    node = If(condition, level[0], apply("__add__", leaves[0], lit("other")), STR)
    (factored,) = factor_branches((node,))
    assert unique_size((factored,)) <= unique_size((node,))
    pending = [(factored, 0)]
    while pending:
        current, depth = pending.pop()
        assert depth < 16
        pending.extend((child, depth + 1) for child in children(current))
    schema = {"condition": BOOL, **dict.fromkeys((leaf.name for leaf in leaves), STR)}
    values = ("x", *("" for _ in leaves[1:]))
    rows = [(True, *values), (False, *values)]
    assert POLARS.evaluate([POLARS.lower(factored).expr], schema, rows) == [["x", "xother"]]


def test_nested_branches_factor_their_shared_suffix() -> None:
    def labelled(text: str) -> str:
        if "a" in text:
            label = "first"
        elif "b" in text:
            label = "second"
        else:
            label = "third"
        size = "long" if len(text) > 3 else "short"
        return label + ": " + text.upper() + ", " + size

    program = trace(labelled, {"text": STR})
    pending = [program.value]
    while pending:
        current = pending.pop()
        if isinstance(current, If):
            assert all(isinstance(branch, If | Lit) for branch in (current.then, current.orelse))
        pending.extend(children(current))
    rows = [("a",), ("b",), ("c",), ("ß",), ("abcd",)]
    assert check_function(labelled, {"text": STR}, POLARS, rows) == []


def test_factored_concatenation_preserves_condition_faults() -> None:
    def selected(text: str, divisor: str) -> str:
        return text + "/" if 1 / int(divisor) else text

    schema = {"text": STR, "divisor": STR}
    program = trace(selected, schema)
    assert isinstance(program.value, Call)
    assert program.fault != FALSE
    rows = [("x", "1"), ("", "-1"), ("x", "0"), ("x", "bad")]
    assert check_function(selected, schema, POLARS, rows) == []


def test_factored_concatenation_preserves_caught_faults() -> None:
    def selected(text: str, value: str) -> str:
        try:
            return text + "/" if int(value) else text
        except ValueError:
            return "invalid"

    schema = {"text": STR, "value": STR}
    rows = [("x", "1"), ("x", "0"), ("", "bad"), ("😀", "-1")]
    assert check_function(selected, schema, POLARS, rows) == []


def test_identical_branches_keep_condition_faults() -> None:
    def selected(text: str, value: str) -> str:
        left = text.strip()
        right = text.strip()
        return left if int(value) else right

    schema = {"text": STR, "value": STR}
    program = trace(selected, schema)
    assert isinstance(program.value, Call)
    assert program.value.op.name == ".strip"
    assert program.fault != FALSE
    assert check_function(selected, schema, POLARS, [("a", "0"), ("b", "1"), ("c", "bad")]) == []


def test_nullable_prefix_keeps_its_empty_branch() -> None:
    def selected(text: str, condition: bool) -> str:
        return text + "/" if condition else text

    schema = {"text": STR.or_none(), "condition": BOOL}
    program = trace(selected, schema)
    assert program.result == STR.or_none()
    assert isinstance(program.value, If)
    rows = [(None, False), (None, True), ("", False), ("x", True)]
    assert check_function(selected, schema, POLARS, rows) == []


@pytest.mark.parametrize(
    ("function", "schema", "rows"),
    [
        (
            lambda text, condition: re.sub("a", "z", text) if condition else re.sub("b", "z", text),
            {"text": STR, "condition": BOOL},
            [("aab", True), ("aab", False)],
        ),
        (
            lambda value, condition: value.strftime("%Y") if condition else value.strftime("%m"),
            {"value": DATETIME, "condition": BOOL},
            [(datetime(2024, 1, 2), True), (datetime(2024, 1, 2), False)],
        ),
    ],
)
def test_literal_options_stay_native(
    function: Callable[..., object], schema: dict[str, Type], rows: list[tuple[object, ...]]
) -> None:
    program = trace(function, schema)
    assert isinstance(program.value, If)
    lowered = POLARS.lower(program.value)
    exact = POLARS.predicate(lowered.exact)
    assert POLARS.evaluate([lowered.expr, exact], schema, rows) == [
        [function(*row) for row in rows],
        [True] * len(rows),
    ]


def test_canonical_branches_preserve_signed_zero() -> None:
    def selected(condition: bool) -> float:
        return 0.0 if condition else -0.0

    schema = {"condition": BOOL}
    program = trace(selected, schema)
    assert isinstance(program.value, If)
    values = POLARS.evaluate([POLARS.build(program)], schema, [(True,), (False,)])[0]
    assert all(isinstance(value, float) for value in values)
    assert [math.copysign(1.0, value) for value in values if isinstance(value, float)] == [
        1.0,
        -1.0,
    ]


@pytest.mark.parametrize("positive", [False, True])
def test_boolean_branches_reduce_to_the_condition(positive: bool) -> None:
    def selected(condition: bool) -> bool:
        return positive if condition else not positive

    program = trace(selected, {"condition": BOOL})
    assert isinstance(program.value, Col if positive else Call)
    assert check_function(selected, {"condition": BOOL}, POLARS, [(True,), (False,)]) == []


def test_effectful_program_does_not_factor_branches() -> None:
    def selected(text: str, condition: bool) -> str:
        random.gauss()
        return text + "/" if condition else text

    program = trace(selected, {"text": STR, "condition": BOOL})
    assert not program.deterministic
    assert isinstance(program.value, If)
