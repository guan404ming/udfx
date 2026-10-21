from collections.abc import Sequence
from functools import reduce
from string import Formatter

from udfx.errors import Unsupported
from udfx.ir import Lit, Node, lit
from udfx.ops import Fault, apply
from udfx.types import STR, Kind


def format_value(value: Node, conversion: str | None, specifier: Node) -> tuple[Node, list[Fault]]:
    if not isinstance(specifier, Lit) or not isinstance(specifier.value, str):
        raise Unsupported("dynamic format spec")
    if conversion not in (None, "s", "r"):
        raise Unsupported(f"format conversion !{conversion}")
    if conversion == "r" and value.type.kind is Kind.STR:
        raise Unsupported("format conversion !r for str")
    if value.type.kind is Kind.LIST:
        raise Unsupported(f"formatting {value.type}")
    text = lit("None") if value.type.kind is Kind.NONE else apply("str", value)
    if value.type == STR:
        text = value
    formatted = text if conversion else value
    specification = specifier.value
    if specification and not (
        (specification == "d" and formatted.type.kind is Kind.INT)
        or (specification == "s" and formatted.type.kind is Kind.STR)
    ):
        raise Unsupported(f"format spec {specification!r} for {formatted.type}")
    faults = (
        [Fault(TypeError, apply("is_none", formatted))]
        if specification and formatted.type.nullable
        else []
    )
    return text, faults


def join_strings(parts: Sequence[Node]) -> Node:
    return reduce(lambda left, right: apply("__add__", left, right), parts) if parts else lit("")


def literal_template(template: Node) -> str:
    if not isinstance(template, Lit) or not isinstance(template.value, str):
        raise Unsupported("dynamic format template")
    return template.value


def format_template(template: Node, arguments: Sequence[Node]) -> tuple[Node, list[Fault]]:
    text = literal_template(template)
    try:
        fields = list(Formatter().parse(text))
    except ValueError as error:
        raise Unsupported(f"format template {text!r}: {error}") from error
    parts: list[Node] = []
    faults: list[Fault] = []
    automatic: bool | None = None
    position = 0
    for literal, field, specifier, conversion in fields:
        if literal:
            parts.append(lit(literal))
        if field is None:
            continue
        if field and not field.isdecimal():
            raise Unsupported(f"format field {field!r}")
        if automatic is not None and automatic != (field == ""):
            raise Unsupported("mixed automatic and manual format fields")
        automatic = field == ""
        try:
            index = position if automatic else int(field)
        except ValueError as error:
            raise Unsupported(f"format field index {field!r} out of range") from error
        position += automatic
        if index >= len(arguments):
            raise Unsupported(f"format field index {index} out of range")
        if specifier is not None and ("{" in specifier or "}" in specifier):
            raise Unsupported(f"dynamic format spec {specifier!r}")
        value, raised = format_value(arguments[index], conversion, lit(specifier or ""))
        parts.append(value)
        faults.extend(raised)
    return join_strings(parts), faults


def format_percent(template: Node, arguments: Sequence[Node]) -> tuple[Node, list[Fault]]:
    remaining = literal_template(template)
    parts: list[Node] = []
    faults: list[Fault] = []
    position = 0
    while remaining:
        literal, marker, remaining = remaining.partition("%")
        if literal:
            parts.append(lit(literal))
        if not marker:
            break
        if not remaining:
            raise Unsupported("incomplete printf spec '%'")
        specifier, remaining = remaining[0], remaining[1:]
        if specifier == "%":
            parts.append(lit("%"))
            continue
        if specifier not in ("s", "d"):
            raise Unsupported(f"printf spec '%{specifier}'")
        if position >= len(arguments):
            raise Unsupported("printf argument count")
        conversion, specification = ("s", "") if specifier == "s" else (None, "d")
        value, raised = format_value(arguments[position], conversion, lit(specification))
        parts.append(value)
        faults.extend(raised)
        position += 1
    if position != len(arguments):
        raise Unsupported("printf argument count")
    return join_strings(parts), faults
