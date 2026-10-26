import re
from copy import copy
from dataclasses import replace
from typing import Any

from udfx.backends.base import Backend
from udfx.backends.faults import NativeFaults
from udfx.ir import Col
from udfx.program import Program


class FingerprintFaults(NativeFaults):
    def marker(self, program: Program) -> str:
        return "udfx fault fingerprint:"


def normalize_lambdas(text: str) -> str:
    parts = re.split(r"('(?:\\.|''|[^'\\])*'|`(?:``|[^`])*`)", text)
    variable = r"\b[xyz]_\d+\b"
    parameters = rf"(?:{variable}|\({variable}(?:,\s*{variable}){{1,2}}\))\s*->"
    names = {
        name
        for part in parts[::2]
        for match in re.finditer(parameters, part)
        for name in re.findall(variable, match[0])
    }
    renamed: dict[str, str] = {}
    for index in range(0, len(parts), 2):
        parts[index] = re.sub(
            variable,
            lambda match: (
                renamed.setdefault(match[0], f"element{len(renamed)}")
                if match[0] in names
                else match[0]
            ),
            parts[index],
        )
    return "".join(parts)


def expression_text(expr: Any, engine: str) -> str:
    if engine == "polars":
        return expr.meta.serialize(format="json")
    text = str(expr)
    if engine == "duckdb":
        literals = r"'(?:''|[^'])*'"
        fallback = r'"udfx fallback [0-9a-f]{64} \d+"(?=\()'
        binding = r'"udfx element \d+"'
        renamed: dict[str, str] = {}
        return re.sub(
            rf"{literals}|(?P<fallback>{fallback})|(?P<binding>{binding})",
            lambda match: (
                '"udfx fallback"'
                if match["fallback"]
                else renamed.setdefault(match[0], f'"udfx element {len(renamed)}"')
                if match["binding"]
                else match[0]
            ),
            text,
        )
    if engine == "datafusion":
        return text.removeprefix("Expr(").removesuffix(")")
    if engine == "spark":
        return normalize_lambdas(text.removeprefix("Column<'").removesuffix("'>"))
    return text


def fingerprint(backend: Backend[Any], program: Program) -> str:
    names = {name: f"c{index}" for index, name in enumerate(program.schema)}
    guard = ""

    def column(self: Backend[Any], node: Col) -> Any:
        return backend.column(Col(names[node.name], node.type))

    def fallback(self: Backend[Any], native: Any, exact: Any, program: Program) -> Any:
        nonlocal guard
        guard = f" fallback when not {expression_text(exact, backend.name)}"
        return native

    def native_raise(self: Backend[Any], program: Program) -> Any:
        if backend.name == "duckdb":
            program = replace(
                program, schema={names[name]: type for name, type in program.schema.items()}
            )
        return type(backend).native_raise(self, program)

    normalized = copy(backend)
    normalized.__class__ = type(
        "FingerprintBackend",
        (type(backend),),
        {
            "column": column,
            "fallback": fallback,
            "native_raise": native_raise,
            "native_faults": property(lambda self: FingerprintFaults()),
        },
    )
    return expression_text(normalized.build(program), backend.name) + guard
