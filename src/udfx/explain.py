from __future__ import annotations

from collections.abc import Callable, Mapping

from udfx.api import compile
from udfx.backends.base import Backend
from udfx.ir import Call, Col, Comprehension, Element, Guarded, If, ListValue, Lit, Node
from udfx.types import Type


def render_name(name: str) -> str:
    return name if name.isidentifier() else repr(name)


def render(node: Node) -> str:
    match node:
        case Col():
            return f"col {render_name(node.name)}"
        case Lit():
            return repr(node.value)
        case Call(op=op, arguments=(container, index)) if op.name == "__getitem__":
            return f"{render(container)}[{render(index)}]"
        case Call():
            return f"{node.op.name}({', '.join(render(argument) for argument in node.arguments)})"
        case Guarded():
            return f"guard({render(node.value)}, {render(node.exact)})"
        case If():
            return f"if({render(node.condition)}, {render(node.then)}, {render(node.orelse)})"
        case ListValue():
            return f"[{', '.join(map(render, node.items))}]"
        case Element():
            return node.name
        case Comprehension():
            return (
                f"[{render(node.value)} for {node.element.name} in {render(node.source)}"
                f" if {render(node.condition)}]"
            )


def explain[E](
    function: Callable[..., object],
    schema: Mapping[str, Type],
    backend: Backend[E] | str = "polars",
) -> str:
    translation = compile(function, schema, backend)
    name = getattr(function, "__name__", repr(function))
    columns = ", ".join(
        f"{render_name(name)}: {column_type}" for name, column_type in schema.items()
    )
    lines = [f"Function: {name}", f"Schema: {{{columns}}}"]
    if translation.unsupported is not None:
        lines.append(f"Unsupported: {translation.unsupported}")
    elif translation.program is not None:
        if translation.program.row_mode:
            lines.append("Mode: row")
        lines.extend(
            (
                f"Value: {render(translation.program.value)}",
                f"Fault: {render(translation.program.fault)}",
                f"Native: {str(translation.expr)!r}",
            )
        )
    return "\n".join(lines)
