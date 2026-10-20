from collections.abc import Callable, Mapping

import pytest

from udfx.explain import render
from udfx.frontend import trace
from udfx.frontend.merge import merge
from udfx.frontend.state import Return
from udfx.ir import TRUE, Col, If, Node, lit
from udfx.ops import apply
from udfx.types import BOOL, INT, STR, Type

from .test_targets import TARGETS, converttoeuro


def year(date: str | None) -> int | None:
    if date:
        try:
            return int(date[:4])
        except Exception:
            return -1
    return None


def grade(score: int, bonus: int) -> str:
    if score > 50:
        if bonus > 0:
            return "high"
        return "mid"
    return "low"


RENDERED = [
    (
        year,
        {"date": STR.or_none()},
        "if(bool(col date), "
        "if(int_parses(slice(col date, None, 4)), int(slice(col date, None, 4)), -1), None)",
    ),
    (
        grade,
        {"score": INT, "bonus": INT},
        "if(__gt__(col score, 50), if(__gt__(col bonus, 0), 'high', 'mid'), 'low')",
    ),
    (
        converttoeuro,
        {"x": STR.or_none(), "y": STR.or_none()},
        "if(is_none(col x), None, if(is_none(col y), None, if(float_parses(col x), "
        "if(and(and(not(__eq__(col y, 'EUR')), not(__eq__(col y, 'USD'))), "
        "not(__eq__(col y, 'GBP'))), 0.0, "
        "if(__eq__(if(__eq__(col y, 'EUR'), 1.0, if(__eq__(col y, 'USD'), 1.09, 0.85)), 0.0), "
        "0.0, __truediv__(float(col x), "
        "if(__eq__(col y, 'EUR'), 1.0, if(__eq__(col y, 'USD'), 1.09, 0.85))))), 0.0)))",
    ),
]


def atoms_once(node: Node, seen: frozenset[Node] = frozenset()) -> bool:
    if not isinstance(node, If):
        return True
    if node.condition in seen:
        return False
    below = seen | {node.condition}
    return atoms_once(node.then, below) and atoms_once(node.orelse, below)


@pytest.mark.parametrize(("function", "schema", "expected"), RENDERED)
def test_merge_builds_decision_tree(
    function: Callable[..., object], schema: Mapping[str, Type], expected: str
) -> None:
    assert render(trace(function, schema).value) == expected


@pytest.mark.parametrize(
    ("function", "schema"),
    [(function, schema) for function, schema, _ in RENDERED]
    + [(function, schema) for function, schema, _ in TARGETS],
)
def test_atoms_appear_once_per_path(
    function: Callable[..., object], schema: Mapping[str, Type]
) -> None:
    assert atoms_once(trace(function, schema).value)


FIRST = Col("first", BOOL)
SECOND = Col("second", BOOL)
BOTH = apply("and", FIRST, SECOND)


def test_contradictory_literals_keep_chain_order() -> None:
    contradiction = apply("and", BOTH, apply("not", BOTH))
    tree = merge([Return(contradiction, lit(1)), Return(BOTH, lit(2))], [apply("not", BOTH)])
    assert (
        render(tree) == "if(and(and(col first, col second), not(and(col first, col second))), 1, 2)"
    )


def test_unrecognized_literal_is_still_checked() -> None:
    double = apply("not", apply("not", FIRST))
    tree = merge([Return(double, lit(1)), Return(TRUE, lit(0))], [])
    assert render(tree) == "if(not(not(col first)), 1, 0)"
