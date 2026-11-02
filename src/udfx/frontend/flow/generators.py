from __future__ import annotations

from dataclasses import replace

from udfx.errors import Unsupported
from udfx.frontend.core.bindings import Bindings
from udfx.frontend.core.values import Function, Generator, Item, Opaque
from udfx.frontend.decode import ops
from udfx.frontend.flow.collections import consumes
from udfx.frontend.flow.frame import Flow, Frame
from udfx.frontend.resolve.constants import global_item


def handler_key(handler: ops.Handler | None) -> tuple[int, tuple[int | None, ...]] | None:
    if handler is None:
        return None
    arguments = tuple(
        None if argument == handler.exception else argument for argument in handler.edge.arguments
    )
    return handler.edge.target, arguments


def consumed(frame: Frame, following: tuple[ops.Op, ...], name: str) -> bool:
    match following:
        case (
            ops.Constant(value=str(), result=separator),
            ops.Attribute(owner=owner, name="join", method=True, results=(callee, receiver)),
            ops.Local(name=local, result=argument, clear=False, cell=False),
            ops.Call() as call,
            *_,
        ) if owner == separator:
            return (
                local == name
                and call.callee == callee
                and call.receiver == receiver
                and call.arguments == (argument,)
                and not call.keywords
            )
        case (
            ops.Global(name=global_name, result=callee),
            ops.Local(name=local, result=argument, clear=False, cell=False),
            ops.Call() as call,
            *_,
        ):
            target = global_item(frame.scope, frame.builtins, global_name)
            return (
                isinstance(target, Opaque)
                and any(target.value is function for function in (list, sorted, sum))
                and local == name
                and call.callee == callee
                and call.receiver is None
                and call.arguments == (argument,)
                and not call.keywords
            )
        case _:
            return False


def validate(frame: Frame, flow: Flow, position: int) -> None:
    operations = frame.decoded.blocks[flow.block].ops
    store = operations[position]
    previous = operations[position - 1] if position else None
    if not isinstance(store, ops.Store) or not isinstance(previous, ops.Call):
        raise Unsupported("generator must be consumed immediately")
    references = [
        op
        for op in frame.decoded.operations(0)
        if isinstance(op, ops.Store | ops.Local | ops.Delete | ops.Cell) and op.name == store.name
    ]
    following = operations[position + 1 :]
    call = next((op for op in following if isinstance(op, ops.Call)), None)
    if (
        store.cell
        or store.name in frame.code.co_cellvars
        or previous.result != store.source
        or len(references) != 2
        or any(
            op is not store and consumes(op, previous.result) for op in frame.decoded.operations(0)
        )
        or call is None
        or any(
            handler_key(op.handler) != handler_key(previous.handler)
            for op in operations[position:]
            if op.offset <= call.offset
        )
        or not consumed(frame, following, store.name)
    ):
        raise Unsupported("generator must be consumed immediately")


def validate_creation(frame: Frame, flow: Flow, callee: Item, op: ops.Call) -> None:
    generator = isinstance(callee, Function) and callee.code.co_name == "<genexpr>"
    mapping = isinstance(callee, Opaque) and (callee.value is map or callee.value is filter)
    following = frame.decoded.blocks[flow.block].ops[flow.position + 1 :]
    if (
        (generator or mapping)
        and following
        and isinstance(following[0], ops.Store)
        and following[0].source == op.result
    ):
        validate(frame, flow, flow.position + 1)


def store_generator(frame: Frame, flow: Flow, op: ops.Store, item: Generator) -> Bindings:
    validate(frame, flow, flow.position)
    return replace(flow.bindings, names={**flow.bindings.names, op.name: item})
