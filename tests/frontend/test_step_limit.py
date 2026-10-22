from __future__ import annotations

import sys
from collections.abc import Callable

import pytest

from udfx.errors import Unsupported
from udfx.frontend import trace
from udfx.types import INT, STR

FIRST_STOP = {
    (3, 12): {6: 37, 7: 17, 8: 8},
    (3, 13): {6: 37, 7: 17, 8: 7},
    (3, 14): {6: 36, 7: 17, 8: 7},
    (3, 15): {6: 36, 7: 17, 8: 7},
}[sys.version_info[:2]]


def branching(branches: int, extra: int) -> Callable[..., object]:
    lines = ["def function(x, s):", "    a = 0"]
    for index in range(branches):
        lines += [f"    if s.startswith('{index}'):", "        a = a + 1"]
        lines += ["        a = a + 0"] * extra
    lines.append("    return a")
    scope: dict[str, Callable[..., object]] = {}
    exec("\n".join(lines), scope)
    return scope["function"]


@pytest.mark.parametrize("branches", sorted(FIRST_STOP))
def test_step_limit_stops_at_the_same_instruction_count(branches: int) -> None:
    schema = {"x": INT, "s": STR}
    trace(branching(branches, FIRST_STOP[branches] - 1), schema)
    with pytest.raises(Unsupported, match="more than 10000 execution paths"):
        trace(branching(branches, FIRST_STOP[branches]), schema)
