from typing import Any

import pytest
from conftest import backends

from udfx.backends.base import Backend
from udfx.ops import Op, all_ops
from udfx.testing import check_op
from udfx.testing.text_traps import REGEX_PATTERNS, STRING_TRAPS
from udfx.types import Kind

CASES = [
    pytest.param(backend, op, id=f"{label}-{op}", marks=pytest.mark.xdist_group(backend.name))
    for label, backend in backends()
    for op in all_ops()
    if op.name.startswith("re.")
]


@pytest.mark.parametrize(("backend", "op"), CASES)
def test_regex_traps(backend: Backend[Any], op: Op) -> None:
    rows = [
        (text, *("" if parameter.kind is Kind.STR else 0 for parameter in op.parameters[1:]))
        for text in STRING_TRAPS
    ]
    for pattern in REGEX_PATTERNS:
        constants: dict[int, object] = {1: pattern, 2: 0}
        if op.name == "re.sub":
            constants.update({3: r"\g<0>-$1", 4: 0})
            if "(" in pattern:
                constants[3] = r"\g<1>0"
        if op.name == "re.split":
            constants[3] = 0
        assert check_op(op, backend, rows, constants) == [], pattern
