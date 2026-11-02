from dataclasses import fields, replace

from udfx.errors import Unsupported
from udfx.frontend.core.values import Item, Iterator, Value
from udfx.frontend.decode import graph, ops
from udfx.frontend.flow.comprehensions import (
    comprehension,
    target_names,
    validate_fault_handlers,
)
from udfx.frontend.flow.frame import Flow, Frame
from udfx.ir import Lit
from udfx.ops import Fault


def is_empty_list(item: Item | None) -> bool:
    return (
        isinstance(item, Value)
        and item.container is list
        and isinstance(item.node, Lit)
        and item.node.value == ()
    )


def references(value: object, identifier: int) -> bool:
    if type(value) is int:
        return value == identifier
    if isinstance(value, ops.Edge):
        return identifier in value.arguments
    if isinstance(value, ops.Handler):
        return identifier in value.edge.arguments
    return isinstance(value, tuple) and any(references(item, identifier) for item in value)


def consumes(op: ops.Op, identifier: int) -> bool:
    ignored = {"offset", "weight", "result", "results", "before", "after", "end"}
    if isinstance(op, ops.Constant):
        ignored.add("value")
    return any(
        references(getattr(op, field.name), identifier)
        for field in fields(op)
        if field.name not in ignored
    )


def fresh_list(frame: Frame, flow: Flow, exit: graph.ForEach, name: str) -> bool:
    initial = flow.bindings.load(name)
    for block in frame.decoded.blocks.values():
        for index, op in enumerate(block.ops[:-1]):
            if not (
                isinstance(op, ops.Build)
                and op.kind == "list"
                and not op.items
                and flow.values.get(op.result) is initial
            ):
                continue
            stored = block.ops[index + 1]
            if not (
                isinstance(stored, ops.Store)
                and not stored.cell
                and stored.name == name
                and stored.source == op.result
                and stored.offset < exit.offset
            ):
                return False
            if any(consumes(part, op.result) for part in (*block.ops[index + 2 :], block.exit)):
                return False
            return not any(
                isinstance(part, ops.Local | ops.Cell | ops.Store | ops.Delete)
                and part.name == name
                for part in frame.decoded.operations(stored.offset + 1, exit.offset)
            )
    return False


def append_methods(blocks: tuple[graph.Block, ...], flow: Flow) -> tuple[dict[int, str], set[int]]:
    methods = {}
    owners = set()
    for block in blocks:
        names = {
            op.result: op.name
            for op in block.ops
            if isinstance(op, ops.Local) and not op.cell and not op.clear
        }
        for op in block.ops:
            if (
                isinstance(op, ops.Attribute)
                and op.method
                and op.name == "append"
                and op.owner in names
                and is_empty_list(flow.bindings.names.get(names[op.owner]))
            ):
                methods[op.results[0]] = names[op.owner]
                owners.add(op.owner)
    return methods, owners


def collection_loop(
    frame: Frame, flow: Flow, exit: graph.ForEach, iterator: Iterator
) -> tuple[str, Item, list[Fault]] | None:
    blocks = frame.decoded.span(exit.offset, exit.end)
    methods, owners = append_methods(blocks, flow)
    names = set(methods.values())
    if not names:
        return None
    if len(names) != 1:
        raise Unsupported("collection loops require one local list")
    name = names.pop()
    if not fresh_list(frame, flow, exit, name):
        raise Unsupported("collection loops require a fresh local list without prior uses")
    body = frame.decoded.operations(exit.offset + 1, exit.end)
    targets = target_names(body)
    if any(
        isinstance(op, ops.Local) and op.name in targets
        for op in frame.decoded.operations(exit.end)
    ):
        raise Unsupported("loop locals used after a dynamic loop")
    if any(
        isinstance(op, ops.Local) and op.name == name and op.result not in owners for op in body
    ):
        raise Unsupported("collection loop iterations depend on the accumulator")
    replacements = {}
    for block in blocks:
        for index, op in enumerate(block.ops):
            if not isinstance(op, ops.Call) or op.callee not in methods:
                continue
            if (
                len(op.arguments) != 1
                or op.keywords
                or index != len(block.ops) - 1
                or not isinstance(block.exit, graph.Jump)
                or block.exit.edge.target != flow.block
            ):
                raise Unsupported("collection append must end the iteration")
            replacements[block.offset] = replace(
                block,
                ops=(
                    *block.ops[:index],
                    ops.Append(
                        value=op.arguments[0],
                        offset=op.offset,
                        handler=op.handler,
                        weight=op.weight,
                    ),
                ),
            )
    if not replacements:
        raise Unsupported("collection loops require a direct append call")
    decoded = replace(frame.decoded, blocks={**frame.decoded.blocks, **replacements})
    item, pending = comprehension(replace(frame, decoded=decoded), flow, exit, iterator, ops.Append)
    if pending:
        validate_fault_handlers(frame)
    return name, item, pending
