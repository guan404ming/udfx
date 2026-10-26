from udfx.ir import FALSE, TRUE, Comprehension, Element, Guarded, If, Node, lit
from udfx.ops import Fault, apply
from udfx.semantics.json import ESCAPES
from udfx.types import STR


def iterable_faults(value: Node) -> list[Fault]:
    return [type_fault(value, TypeError, ("list", "str", "dict"))]


def type_fault(value: Node, kind: type[Exception], names: tuple[str, ...]) -> Fault:
    allowed = FALSE
    for name in names:
        allowed = apply("or", allowed, apply("json.is_type", value, lit(name)))
    return Fault(kind, apply("not", allowed))


def array(value: Node) -> Node:
    return apply("json.tokens", value)


def string(value: Node) -> Node:
    trimmed = apply(".strip", value, lit(" \t\r\n"))
    tokens = apply("re.findall", trimmed, lit(r'\\["\\/bfnrt]|[^"\\]+'), lit(0))
    element = Element("json escape", STR)
    decoded: Node = element
    for escaped, replacement in ESCAPES.items():
        decoded = If(apply("__eq__", element, lit(escaped)), lit(replacement), decoded, STR)
    pieces = Comprehension(tokens, element, decoded, TRUE, FALSE)
    result = apply(".join", lit(""), pieces)
    return Guarded(result, apply("json.is_type", value, lit("str")))


def strings(value: Node) -> Node:
    tokens = array(value)
    element = Element("json string", STR)
    return Comprehension(tokens, element, string(element), TRUE, FALSE)


def packed(values: Node) -> Node:
    joined = apply(".join", lit(", "), values)
    return apply("__add__", apply("__add__", lit("["), joined), lit("]"))
