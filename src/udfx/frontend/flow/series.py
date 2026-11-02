from __future__ import annotations

from collections.abc import Mapping, Sequence

from udfx.errors import Unsupported
from udfx.frontend.core.bindings import Bindings
from udfx.frontend.core.evaluation import single
from udfx.frontend.core.expressions import node_of
from udfx.frontend.core.series import elementwise, rows
from udfx.frontend.core.values import Item, Series
from udfx.frontend.flow.comprehensions import validate_fault_handlers
from udfx.frontend.flow.frame import Frame, Result
from udfx.ir import FALSE
from udfx.ops import Fault


def construct_series(
    frame: Frame, bindings: Bindings, arguments: Sequence[Item], keywords: Mapping[str, Item]
) -> list[Result]:
    rows()
    if keywords or len(arguments) != 1:
        raise Unsupported("pandas.Series requires one positional argument")
    item = arguments[0]
    if isinstance(item, Series):
        return frame.settle(bindings, single(item))
    value, fault = elementwise(node_of(item))
    if fault == FALSE:
        return frame.settle(bindings, single(Series(value)))
    validate_fault_handlers(frame)
    return frame.settle(bindings, single(Series(value), [Fault(Exception, fault)]))
