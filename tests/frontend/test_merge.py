from collections.abc import Callable, Mapping

import pytest

from udfx.explain import render
from udfx.frontend import trace
from udfx.ir import If, Node
from udfx.types import INT, STR, Type

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
    if node.cond in seen:
        return False
    below = seen | {node.cond}
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
