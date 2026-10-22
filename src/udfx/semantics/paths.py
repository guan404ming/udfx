from collections.abc import Sequence

from udfx.ir import If, Node, lit
from udfx.ops.registry import apply
from udfx.types import INT, STR


def split_path(path: Node) -> tuple[Node, Node]:
    boundary = apply("__add__", apply(".rfind", path, lit("/")), lit(1))
    head = apply("slice", path, lit(None, INT.or_none()), boundary)
    tail = apply("slice", path, boundary, lit(None, INT.or_none()))
    stripped = apply(".rstrip", head, lit("/"))
    return If(apply("bool", stripped), stripped, head, STR), tail


def split_extension(path: Node) -> tuple[Node, Node]:
    separator = apply(".rfind", path, lit("/"))
    dot = apply(".rfind", path, lit("."))
    prefix = apply("slice", path, apply("__add__", separator, lit(1)), dot)
    found = apply(
        "and", apply("__gt__", dot, separator), apply("bool", apply(".lstrip", prefix, lit(".")))
    )
    root = apply("slice", path, lit(None, INT.or_none()), dot)
    extension = apply("slice", path, dot, lit(None, INT.or_none()))
    return If(found, root, path, STR), If(found, extension, lit(""), STR)


def join_path(parts: Sequence[Node]) -> Node:
    path = parts[0]
    for part in parts[1:]:
        separator = If(
            apply("or", apply("not", apply("bool", path)), apply(".endswith", path, lit("/"))),
            lit(""),
            lit("/"),
            STR,
        )
        path = If(
            apply(".startswith", part, lit("/")),
            part,
            apply("__add__", apply("__add__", path, separator), part),
            STR,
        )
    return path
