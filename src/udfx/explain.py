from __future__ import annotations

from collections.abc import Callable, Mapping

from udfx.api import compile
from udfx.backends.base import Backend
from udfx.ir import Call, Col, If, Lit, Node
from udfx.types import Type


def render(node: Node) -> str:
    match node:
        case Col():
            return f"col {node.name}"
        case Lit():
            return repr(node.value)
        case Call(op=op, args=(container, index)) if op.name == "__getitem__":
            return f"{render(container)}[{render(index)}]"
        case Call():
            return f"{node.op.name}({', '.join(render(argument) for argument in node.args)})"
        case If():
            return f"if({render(node.cond)}, {render(node.then)}, {render(node.orelse)})"


def explain[E](
    function: Callable[..., object],
    schema: Mapping[str, Type],
    backend: Backend[E] | str = "polars",
) -> str:
    translation = compile(function, schema, backend)
    name = getattr(function, "__name__", repr(function))
    columns = ", ".join(f"{name}: {column_type}" for name, column_type in schema.items())
    lines = [f"Function: {name}", f"Schema: {{{columns}}}"]
    if translation.unsupported is not None:
        lines.append(f"Unsupported: {translation.unsupported}")
    elif translation.program is not None:
        lines.extend(
            (
                f"Value: {render(translation.program.value)}",
                f"Fault: {render(translation.program.fault)}",
                f"Native: {translation.expr!s}",
            )
        )
    return "\n".join(lines)
