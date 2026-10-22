from __future__ import annotations

from udfx.errors import Unsupported
from udfx.frontend.core.evaluation import single
from udfx.frontend.core.exceptions import exception_classes, exception_kind
from udfx.frontend.core.expressions import truth
from udfx.frontend.core.values import NULL, Cell, Opaque, Value
from udfx.frontend.decode import ops
from udfx.frontend.flow.frame import Flow, Frame, Produced, Result
from udfx.frontend.flow.machine import evaluates, optional
from udfx.frontend.operations import containers, operators
from udfx.frontend.operations.attributes import load_attribute
from udfx.frontend.operations.formatting import build_string, format_operand
from udfx.frontend.operations.iterables import iterator_of
from udfx.frontend.operations.library import require_empty_keywords
from udfx.frontend.resolve.constants import constant, global_item
from udfx.frontend.resolve.functions import make_function
from udfx.ir import lit

IDENTITY = {"is": False, "is not": True}
MEMBERSHIP = {"in": False, "not in": True}


@evaluates(ops.Constant)
def load_constant(frame: Frame, flow: Flow, op: ops.Constant) -> list[Result]:
    return [Produced(flow.bindings, (constant(op.value),))]


@evaluates(ops.Local)
def local(frame: Frame, flow: Flow, op: ops.Local) -> list[Result]:
    bindings = flow.bindings
    if op.clear:
        return [Produced(bindings.delete(op.name), (bindings.names.get(op.name, NULL),))]
    return [Produced(bindings, (bindings.load(op.name),))]


@evaluates(ops.Cell)
def cell(frame: Frame, flow: Flow, op: ops.Cell) -> list[Result]:
    return [Produced(flow.bindings, (Opaque(Cell(op.name)),))]


@evaluates(ops.Store)
def store(frame: Frame, flow: Flow, op: ops.Store) -> list[Result]:
    if op.cell and op.name in frame.code.co_freevars:
        raise Unsupported(f"nonlocal write to {op.name}", frame.where(op.offset))
    return [Produced(flow.bindings.store(op.name, flow.values[op.source]), ())]


@evaluates(ops.Delete)
def delete(frame: Frame, flow: Flow, op: ops.Delete) -> list[Result]:
    return [Produced(flow.bindings.delete(op.name), ())]


@evaluates(ops.Global)
def global_(frame: Frame, flow: Flow, op: ops.Global) -> list[Result]:
    return [Produced(flow.bindings, (global_item(frame.scope, frame.builtins, op.name),))]


@evaluates(ops.Placeholder)
def placeholder(frame: Frame, flow: Flow, op: ops.Placeholder) -> list[Result]:
    return [Produced(flow.bindings, (Opaque(None),))]


@evaluates(ops.Boolean)
def boolean(frame: Frame, flow: Flow, op: ops.Boolean) -> list[Result]:
    test, faults = truth(flow.values[op.source])
    return frame.settle(flow.bindings, single(Value(test), faults))


@evaluates(ops.Not)
def logical_not(frame: Frame, flow: Flow, op: ops.Not) -> list[Result]:
    return frame.settle(flow.bindings, operators.logical_not(flow.values[op.source]))


@evaluates(ops.Negate)
def negate(frame: Frame, flow: Flow, op: ops.Negate) -> list[Result]:
    return frame.settle(flow.bindings, operators.negation(flow.values[op.source]))


@evaluates(ops.ExceptionMatch)
def exception_match(frame: Frame, flow: Flow, op: ops.ExceptionMatch) -> list[Result]:
    exception, classes = flow.values[op.exception], flow.values[op.classes]
    matched = issubclass(exception_kind(exception), exception_classes(classes))
    return [Produced(flow.bindings, (Value(lit(matched)),))]


@evaluates(ops.BinaryOp)
def binary(frame: Frame, flow: Flow, op: ops.BinaryOp) -> list[Result]:
    left, right = flow.values[op.left], flow.values[op.right]
    evaluation = operators.binary_operation(op.symbol, left, right, frame.where(op.offset))
    return frame.settle(flow.bindings, evaluation)


@evaluates(ops.Compare)
def compare(frame: Frame, flow: Flow, op: ops.Compare) -> list[Result]:
    left, right, where = flow.values[op.left], flow.values[op.right], frame.where(op.offset)
    if op.symbol in IDENTITY:
        evaluation = operators.identity(left, right, IDENTITY[op.symbol], where)
    elif op.symbol in MEMBERSHIP:
        evaluation = operators.membership(left, right, MEMBERSHIP[op.symbol])
    else:
        evaluation = operators.comparison(op.symbol, left, right, where)
    return frame.settle(flow.bindings, evaluation)


@evaluates(ops.Subscript)
def subscript(frame: Frame, flow: Flow, op: ops.Subscript) -> list[Result]:
    container, index = flow.values[op.container], flow.values[op.index]
    return frame.settle(
        flow.bindings, operators.subscript(container, index, frame.where(op.offset))
    )


@evaluates(ops.Cut)
def cut(frame: Frame, flow: Flow, op: ops.Cut) -> list[Result]:
    values = flow.values
    return frame.settle(
        flow.bindings, operators.cut(values[op.container], values[op.start], values[op.stop])
    )


@evaluates(ops.Slice)
def build_slice(frame: Frame, flow: Flow, op: ops.Slice) -> list[Result]:
    items = tuple(flow.values[identifier] for identifier in op.bounds)
    return frame.settle(flow.bindings, operators.build_slice(items, frame.where(op.offset)))


@evaluates(ops.Build)
def build(frame: Frame, flow: Flow, op: ops.Build) -> list[Result]:
    items = tuple(flow.values[identifier] for identifier in op.items)
    where = frame.where(op.offset)
    match op.kind:
        case "tuple":
            evaluation = containers.build_tuple(items)
        case "list":
            evaluation = containers.build_list(items, where)
        case "set":
            evaluation = containers.build_set(items, where)
        case "map":
            evaluation = containers.build_map(items)
        case "keys":
            evaluation = containers.build_keys_map(items, where)
        case _:
            evaluation = build_string(items)
    return frame.settle(flow.bindings, evaluation)


@evaluates(ops.Extend)
def extend(frame: Frame, flow: Flow, op: ops.Extend) -> list[Result]:
    container, source = flow.values[op.container], flow.values[op.source]
    evaluation = containers.extend(container, source, op.kind, frame.where(op.offset))
    return frame.settle(flow.bindings, evaluation)


@evaluates(ops.Unpack)
def unpack(frame: Frame, flow: Flow, op: ops.Unpack) -> list[Result]:
    evaluation = containers.unpack(
        flow.values[op.source], op.before, op.after, op.starred, frame.where(op.offset)
    )
    return frame.settle(flow.bindings, evaluation)


@evaluates(ops.Format)
def format_(frame: Frame, flow: Flow, op: ops.Format) -> list[Result]:
    specification = optional(flow.values, op.specification)
    return frame.settle(
        flow.bindings, format_operand(flow.values[op.source], op.conversion, specification)
    )


@evaluates(ops.Iterate)
def iterate(frame: Frame, flow: Flow, op: ops.Iterate) -> list[Result]:
    iterator, faults = iterator_of(flow.values[op.source])
    return frame.settle(flow.bindings, single(iterator, faults))


@evaluates(ops.Merge)
def merge(frame: Frame, flow: Flow, op: ops.Merge) -> list[Result]:
    require_empty_keywords(flow.values[op.source], frame.where(op.offset))
    return [Produced(flow.bindings, ())]


@evaluates(ops.Attribute)
def attribute(frame: Frame, flow: Flow, op: ops.Attribute) -> list[Result]:
    evaluation = load_attribute(
        flow.values[op.owner], op.name, op.method, frame.snapshot, frame.where(op.offset)
    )
    return frame.settle(flow.bindings, evaluation)


@evaluates(ops.MakeFunction)
def function(frame: Frame, flow: Flow, op: ops.MakeFunction) -> list[Result]:
    values = flow.values
    made = make_function(
        op.code,
        frame,
        optional(values, op.closure),
        optional(values, op.keyword_defaults),
        optional(values, op.defaults),
    )
    return [Produced(flow.bindings, (made,))]


@evaluates(ops.Append, ops.Yield, ops.GeneratorStart)
def unsupported(
    frame: Frame, flow: Flow, op: ops.Append | ops.Yield | ops.GeneratorStart
) -> list[Result]:
    raise Unsupported(op.reason, frame.where(op.offset))
