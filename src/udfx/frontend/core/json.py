from dataclasses import replace

from udfx.frontend.core.comprehensions import cardinality, permutation_source
from udfx.ir import FALSE, TRUE, Call, Comprehension, Element, Guarded, If, Lit, Node, lit
from udfx.ops import Fault, apply
from udfx.semantics.facts import json_argument_type, json_argument_validated
from udfx.semantics.json import ESCAPES, non_ascii_word_pattern
from udfx.semantics.text import whitespace_pattern
from udfx.types import JSON, STR, Kind, list_of


def known_type(node: Call) -> Node | None:
    name = node.arguments[1]
    kind = json_argument_type(node)
    if kind is not None and isinstance(name, Lit) and json_argument_validated(node):
        return lit(kind == name.value)
    return None


def nonempty_array(node: Call) -> Node | None:
    length, limit = node.arguments
    if isinstance(length, Call) and length.op.name == "len":
        length = cardinality(length) or length
    if isinstance(length, Guarded):
        result = nonempty_array(Call(node.op, (length.value, limit)))
        return None if result is None else Guarded(result, length.exact)
    if (
        limit != lit(0)
        or not isinstance(length, Call)
        or length.op.name != "json.length"
        or json_argument_type(length) != "list"
    ):
        return None
    value = apply("json.text", length.arguments[0])
    empty = apply("re.contains", value, lit(r"\A[ \t\r\n]*\[[ \t\r\n]*\]"), lit(0))
    return apply("not", empty)


def typed_get(node: Call) -> Node | None:
    value = node.arguments[0]
    if (
        isinstance(value, Call)
        and value.op.name == "json.get"
        and value.arguments[1].type.kind is Kind.STR
    ):
        return apply(f"json.get.{node.type}", *value.arguments)
    return None


def dynamic(value: Node) -> Node:
    return value if value.type.kind is Kind.JSON else apply("json.value", value)


def text(value: Node) -> Node:
    if isinstance(value, Call) and value.op.name == "json.value":
        return value.arguments[0]
    return apply("json.text", value)


def encoded(value: Node) -> Node:
    if value.type.kind is Kind.JSON:
        return value
    if value.type.kind is Kind.NONE:
        return lit("null", JSON)
    return apply("json.encode", value)


def iterable_faults(value: Node) -> list[Fault]:
    return [type_fault(value, TypeError, ("list", "str", "dict"))]


def type_fault(value: Node, kind: type[Exception], names: tuple[str, ...]) -> Fault:
    allowed = FALSE
    for name in names:
        allowed = apply("or", allowed, apply("json.is_type", value, lit(name)))
    return Fault(kind, apply("not", allowed))


def array(value: Node) -> Node:
    return apply("json.items" if value.type.kind is Kind.JSON else "json.tokens", value)


def string(value: Node) -> Node:
    if (
        isinstance(value, Call)
        and value.op.name == "json.encode"
        and value.arguments[0].type == STR
    ):
        return value.arguments[0]
    result = apply("json.result.str", dynamic(value))
    exact = (
        apply("json.is_type", value, lit("str"))
        if value.type.kind is Kind.JSON
        else apply("json.string_native", value)
    )
    return Guarded(result, exact)


def decoded_string(value: Node) -> Node:
    trimmed = apply(".strip", value, lit(" \t\r\n"))
    decoded = apply("slice", trimmed, lit(1), lit(-1))
    decoded = apply(".replace", decoded, lit("\\\\"), lit("\x00"))
    for escaped, replacement in ESCAPES.items():
        if escaped != "\\\\":
            decoded = apply(".replace", decoded, lit(escaped), lit(replacement))
    result = apply(".replace", decoded, lit("\x00"), lit("\\"))
    return Guarded(result, apply("json.string_native", value))


def strings(value: Node) -> Node:
    if (
        isinstance(value, Call)
        and value.op.name == "json.dumps"
        and value.arguments[0].type == list_of(STR)
    ):
        return value.arguments[0]
    return apply("json.result.list[str]", dynamic(value))


def decoded_strings(value: Node) -> Node:
    tokens = array(value)
    element = Element("json string", STR)
    decoded = Comprehension(tokens, element, string(element), TRUE, FALSE)
    plain = Guarded(
        apply("slice", apply(".split", value, lit('"')), lit(1), lit(None), lit(2)),
        apply("json.array_native", value),
    )
    return If(apply("__contains__", value, lit("\\")), decoded, plain, list_of(STR))


def packed(values: Node) -> Node:
    if (
        isinstance(values, Comprehension)
        and isinstance(values.value, Call)
        and values.value.op.name == "json.encode"
        and values.value.arguments[0].type == STR
    ):
        return apply("json.dumps", replace(values, value=values.value.arguments[0]), lit(False))
    joined = apply(".join", lit(", "), values)
    return apply("__add__", apply("__add__", lit("["), joined), lit("]"))


def serialized_strings(value: Node, ensure_ascii: Node) -> Node:
    element = Element("json item", value.type.item or STR)
    items = Comprehension(value, element, apply("json.dumps", element, ensure_ascii), TRUE, FALSE)
    result = packed(items)
    return If(apply("is_none", value), lit("null"), result, STR) if value.type.nullable else result


def word_source(value: Node) -> Node | None:
    source = value
    while (
        isinstance(source, Comprehension)
        and source.value == source.element
        and source.condition == TRUE
        and source.fault == FALSE
    ):
        source = source.source
    if (
        not isinstance(source, Call)
        or source.op.name not in {".split", ".rsplit"}
        or len(source.arguments) != 1
    ):
        return None
    return source.arguments[0]


def serialized_ascii_lists(value: Node, ensure_ascii: Node) -> Node:
    if ensure_ascii != TRUE:
        return serialized_strings(value, ensure_ascii)
    source = word_source(value)
    if source is not None:
        exact = apply("not", apply("re.contains", source, lit(non_ascii_word_pattern()), lit(0)))
    else:
        source = permutation_source(value)
        if not isinstance(source, Call) or source.op.name != "json.result.list[str]":
            return serialized_strings(value, ensure_ascii)
        source = text(source.arguments[0])
        exact = apply("and", apply(".isascii", source), apply("json.array_native", source))
    output = serialized_strings(value, FALSE)
    output = apply(".replace", output, lit(chr(127)), lit(r"\u007f"))
    return Guarded(output, exact)


def serialized_words(value: Node, ensure_ascii: Node) -> Node:
    text = word_source(value)
    if text is None:
        return serialized_strings(value, ensure_ascii)
    normalized = apply("re.sub", text, lit(whitespace_pattern()), lit(0), lit(" "), lit(0))
    encoded = apply("json.dumps", apply(".strip", normalized, lit(" ")), ensure_ascii)
    separated = apply(".replace", encoded, lit(" "), lit('", "'))
    result = apply("__add__", apply("__add__", lit("["), separated), lit("]"))
    return If(apply("bool", apply(".strip", text)), result, lit("[]"), STR)
