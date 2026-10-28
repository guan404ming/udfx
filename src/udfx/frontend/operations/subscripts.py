import operator

from udfx.errors import Unsupported
from udfx.frontend.core.evaluation import Evaluation, always, single
from udfx.frontend.core.expressions import invoke, node_of
from udfx.frontend.core.logic import fold
from udfx.frontend.core.values import Item, Json, Opaque, Row, Slice, Tuple, Value
from udfx.frontend.resolve.constants import GENERICS, annotation, constant, constant_value, lookup
from udfx.ir import Lit


def subscript(container: Item, index: Item, where: str) -> Evaluation:
    if isinstance(container, Json):
        node, faults = invoke("json.item", [node_of(container), node_of(index)], TypeError)
        return single(Json(node), faults)
    return ordinary_subscript(container, index, where)


def ordinary_subscript(container: Item, index: Item, where: str) -> Evaluation:
    if isinstance(container, Opaque) and isinstance(container.value, tuple):
        container = Tuple(tuple(constant(value) for value in container.value))
    if isinstance(container, Tuple):
        position = constant_value(index)
        if not isinstance(position, int):
            raise Unsupported("tuple index must be an integer literal", where)
        try:
            return single(container.items[position])
        except IndexError:
            return always(IndexError)
    if isinstance(container, Row):
        if not (
            isinstance(index, Value)
            and isinstance(index.node, Lit)
            and isinstance(index.node.value, str | int)
        ):
            raise Unsupported("row access requires a constant string or integer key", where)
        try:
            return single(container.column(index.node.value))
        except IndexError:
            return always(IndexError)
    if isinstance(container, Opaque) and any(container.value is generic for generic in GENERICS):
        return single(annotation(operator.getitem, container, index))
    if isinstance(index, Opaque) and isinstance(index.value, slice):
        bounds = (index.value.start, index.value.stop, index.value.step)
        index = Slice(tuple(node_of(constant(bound)) for bound in bounds))
    if isinstance(index, Slice):
        node, faults = invoke("slice", [node_of(container), *index.bounds], TypeError)
    elif isinstance(container, Opaque):
        node, faults = lookup(container.value, node_of(index))
    else:
        node, faults = invoke("__getitem__", [node_of(container), node_of(index)], TypeError)
    return single(Value(node), faults)


def build_slice(items: tuple[Item, ...], where: str) -> Evaluation:
    bounds = tuple(node_of(item) for item in items)
    if len(bounds) == 3 and not isinstance(bounds[2], Lit):
        raise Unsupported("slice step must be a literal", where)
    return single(Slice(bounds))


def cut(container: Item, start: Item, stop: Item) -> Evaluation:
    node, faults = invoke("slice", [node_of(container), node_of(start), node_of(stop)], TypeError)
    return single(Value(fold(node)), faults)
