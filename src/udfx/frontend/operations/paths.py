import inspect
import os
from collections.abc import Callable, Mapping, Sequence
from typing import cast

from udfx.frontend.core.values import Item, Opaque, Tuple, Value
from udfx.frontend.operations.builtins import call
from udfx.frontend.resolve.callables import known_op
from udfx.frontend.resolve.functions import bind_helper
from udfx.ops import Fault
from udfx.semantics.paths import join_path, split_extension, split_path


def call_path(
    callee: Item, arguments: Sequence[Item], keywords: Mapping[str, Item]
) -> tuple[Item, list[Fault]] | None:
    if not isinstance(callee, Opaque):
        return None
    name = known_op(callee.value)
    if name is None or not name.startswith("posixpath."):
        return None
    signature = inspect.signature(cast("Callable[..., object]", callee.value))
    bindings = bind_helper(signature, arguments, keywords)
    items = [
        member
        for item in bindings.values()
        for member in (item.items if isinstance(item, Tuple) else (item,))
    ]
    converted = [call(Opaque(os.fspath), [item]) for item in items]
    nodes = [node for node, _ in converted]
    faults = [fault for _, pending in converted for fault in pending]
    if name == "posixpath.join":
        return Value(join_path(nodes)), faults
    parts = split_extension(nodes[0]) if name == "posixpath.splitext" else split_path(nodes[0])
    if name in {"posixpath.basename", "posixpath.dirname"}:
        return Value(parts[name == "posixpath.basename"]), faults
    return Tuple(tuple(Value(part) for part in parts)), faults
