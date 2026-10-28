from __future__ import annotations

from collections.abc import Callable, Mapping
from time import perf_counter

from udfx.api import compile
from udfx.backends.base import Backend
from udfx.backends.diagnostics import LoweringDetails, capture_lowering
from udfx.backends.hoisting import size
from udfx.backends.program_lowering import EXPANDED_LIMIT
from udfx.graphs import canonicalize
from udfx.ir import Call, Col, Comprehension, Element, Guarded, If, ListValue, Lit, Node, children
from udfx.types import Type


def render_name(name: str) -> str:
    return name if name.isidentifier() else repr(name)


def render(node: Node) -> str:
    if size(node) > EXPANDED_LIMIT:
        return render_graph(node)

    def visit(part: Node) -> str:
        return render_node(part, visit)

    return visit(node)


def render_graph(node: Node) -> str:
    references: dict[int, str] = {}
    definitions: list[str] = []

    def visit(part: Node) -> str:
        if not children(part):
            return render_node(part, visit)
        if id(part) not in references:
            value = render_node(part, visit)
            name = f"%{len(definitions)}"
            references[id(part)] = name
            definitions.append(f"{name} = {value}")
        return references[id(part)]

    (canonical,) = canonicalize((node,))
    value = visit(canonical)
    return "let " + "; ".join([*definitions, value])


def render_node(node: Node, render: Callable[[Node], str]) -> str:
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
    *,
    result: Type | None = None,
) -> str:
    start = perf_counter()
    with capture_lowering() as details:
        translation = compile(function, schema, backend, result=result)
    elapsed = perf_counter() - start
    name = getattr(function, "__name__", repr(function))
    columns = ", ".join(
        f"{render_name(name)}: {column_type}" for name, column_type in schema.items()
    )
    lines = [
        f"Function: {name}",
        f"Schema: {{{columns}}}",
        f"Backend: {translation.backend}",
        f"Compile time: {elapsed:.6f} s",
    ]
    if translation.unsupported is not None:
        lines.append("Status: unsupported")
        lines.append(f"Unsupported: {translation.unsupported}")
    elif translation.program is not None:
        graph = size(translation.program.value) + size(translation.program.fault) > EXPANDED_LIMIT
        lines.extend(
            execution_lines(details[-1] if details else None, graph=graph)
            if translation.program.deterministic
            else [
                "Status: Python",
                "Python fallback: all rows",
                "Reason: function uses nondeterministic operations",
            ]
        )
        if translation.program.row_mode:
            lines.append("Mode: row")
        lines.extend(
            (
                f"Value: {render(translation.program.value)}",
                f"Fault: {render(translation.program.fault)}",
                "Native: shared expression graph; inspect translation.expr with the engine"
                if graph
                else f"Native: {str(translation.expr)!r}",
            )
        )
    return "\n".join(lines)


def execution_lines(details: LoweringDetails | None, *, graph: bool = False) -> list[str]:
    if details is None:
        return ["Status: compiled", "Python fallback: unknown (backend supplied no diagnostics)"]
    lines = [
        f"Status: {'native' if details.exact is None else 'guarded'}",
        f"Python fallback: {'no' if details.exact is None else 'possible'}",
        "Fault handling: "
        + (
            "native errors"
            if details.native_errors
            else "Python fallback"
            if details.failure
            else "none"
        ),
    ]
    if details.exact is not None:
        lines.extend(
            [
                "Fallback rows: not measured",
                "Fallback when: native guard is false or null",
                "Native guard: part of the shared expression graph"
                if graph
                else f"Native guard: {str(details.exact)!r}",
            ]
        )
        if details.value_guard:
            lines.append("Reason: native values need a Python-equivalence check")
        if details.fault_guard:
            lines.append("Reason: native fault checks need a Python-equivalence check")
        if details.failure and not details.native_errors:
            lines.append("Reason: Python exceptions or integer overflow need Python execution")
    return lines
