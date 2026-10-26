import operator

import pytest

from udfx.backends.rules import fold_predicate
from udfx.ir import FALSE, TRUE, Col, If, lit
from udfx.ops import apply
from udfx.semantics.ranges import proven
from udfx.types import BOOL, FLOAT


@pytest.mark.parametrize("operation", ["__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__"])
@pytest.mark.parametrize("values", [(0.5, 2.0), (0.0, -0.0), (float("nan"), 0.0), (-1.0, 1.0)])
def test_constant_branch_comparisons_use_python_semantics(
    operation: str, values: tuple[float, float]
) -> None:
    reference = getattr(operator, operation)
    condition = Col("condition", BOOL)
    value = If(condition, lit(values[0]), lit(values[1]), FLOAT)
    for left, right in ((value, lit(0.0)), (lit(0.0), value)):
        node = apply(operation, left, right)
        results = [
            reference(item, 0.0) if left is value else reference(0.0, item) for item in values
        ]
        expected = results[0] if results[0] == results[1] else None
        assert proven(node) is expected
        if expected is not None:
            assert fold_predicate(node, lambda _: None) == (TRUE if expected else FALSE)
