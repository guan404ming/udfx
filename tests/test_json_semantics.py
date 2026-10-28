import pytest

from udfx.frontend.trace import json_result
from udfx.graphs import unique_size
from udfx.ir import Call, Col, If, Node
from udfx.ops import apply
from udfx.semantics.facts import json_argument_validated
from udfx.testing.outcome import same_typed
from udfx.types import BOOL, JSON, STR, list_of


@pytest.mark.parametrize(
    ("left", "right", "equal"),
    [
        ('{"x": "中文"}', '{"x":"\\u4e2d\\u6587"}', True),
        ("[1, true, null]", "[1,true,null]", True),
        ('{"x":1,"y":2}', '{"y":2,"x":1}', False),
        ('{"x":1}', '{"x":1.0}', False),
        ('{"x":true}', '{"x":1}', False),
        ("[-0.0]", "[0.0]", False),
        ('{"x":NaN}', '{"x":NaN}', True),
    ],
)
def test_json_comparison_preserves_types_and_order(left: str, right: str, equal: bool) -> None:
    assert same_typed(left, right, JSON) is equal
    assert same_typed([left], [right], list_of(JSON)) is equal


def test_json_conversion_preserves_shared_branches() -> None:
    value: Node = apply("json.value", Col("document", STR))
    for index in range(12):
        value = If(Col(f"condition{index}", BOOL), value, value, JSON)
    converted = json_result(value, STR)
    assert unique_size((converted,)) == unique_size((value,)) + 1
    result = apply("json.result.str", value)
    assert isinstance(result, Call)
    assert json_argument_validated(result)
