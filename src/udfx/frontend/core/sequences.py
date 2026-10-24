import re
from collections.abc import Sequence

from udfx.ir import Call, Guarded, If, Lit, Node, lit
from udfx.ops.registry import apply
from udfx.semantics.text import (
    has_prefix_suffix_overlap,
    non_whitespace_pattern,
    whitespace_pattern,
)
from udfx.types import INT


def unit_stride(steps: Sequence[Node]) -> bool:
    return not steps or (isinstance(steps[0], Lit) and steps[0].value in (None, 1))


def full_stride(node: Node) -> tuple[Node, int] | None:
    if not isinstance(node, Call) or node.op.name != "slice" or len(node.arguments) != 4:
        return None
    source, start, stop, step = node.arguments
    if (
        isinstance(start, Lit)
        and start.value is None
        and isinstance(stop, Lit)
        and stop.value is None
        and isinstance(step, Lit)
        and isinstance(step.value, int)
        and step.value
    ):
        return source, step.value
    return None


def stride_length(source: Node, step: int) -> Node:
    length = apply("len", source)
    if abs(step) == 1:
        return length
    return apply("__floordiv__", apply("__add__", length, lit(abs(step) - 1)), lit(abs(step)))


def limited_split(node: Node) -> tuple[Node, Node, Node, bool] | None:
    if not isinstance(node, Call) or node.op.name not in {".split", ".rsplit"}:
        return None
    if len(node.arguments) != 3:
        return None
    source, separator, maximum = node.arguments
    if (
        not isinstance(separator, Lit)
        or not isinstance(separator.value, str)
        or not separator.value
        or has_prefix_suffix_overlap(separator.value)
    ):
        return None
    pieces = apply(".split", source, separator)
    size = apply("len", pieces)
    count = apply("min", apply("max", maximum, lit(0)), apply("__sub__", size, lit(1)))
    length = If(apply("__lt__", maximum, lit(0)), size, apply("__add__", count, lit(1)), INT)
    return pieces, separator, length, node.op.name == ".rsplit"


def partition_item(pieces: Node, separator: Node, length: Node, right: bool, index: Node) -> Node:
    position = If(apply("__lt__", index, lit(0)), apply("__add__", index, length), index, INT)
    count = apply("__sub__", length, lit(1))
    if right:
        offset = apply("__sub__", apply("len", pieces), length)
        prefix = apply("slice", pieces, lit(0), apply("__add__", offset, lit(1)))
        remainder = apply(".join", separator, prefix)
        item = apply("__getitem__", pieces, apply("__add__", offset, position))
        return If(apply("__eq__", position, lit(0)), remainder, item, item.type)
    suffix = apply("slice", pieces, count, lit(None, INT.or_none()))
    remainder = apply(".join", separator, suffix)
    item = apply("__getitem__", pieces, position)
    return If(apply("__lt__", position, count), item, remainder, item.type)


def sequence_length(name: str, length: Node) -> Node | None:
    if name == "len":
        return length
    if name == "bool":
        return apply("bool", length)
    return None


def simplify_sequence(node: Call) -> Node:
    total = sum_partition(node)
    if total is not None:
        return total
    if node.op.name in {"len", "bool", "__getitem__"}:
        partition = limited_split(node.arguments[0])
        if partition is not None:
            pieces, separator, length, right = partition
            size = sequence_length(node.op.name, length)
            if size is not None:
                return size
            return partition_item(pieces, separator, length, right, node.arguments[1])
        sliced = full_stride(node.arguments[0])
        if sliced is not None:
            source, step = sliced
            length = stride_length(source, step)
            size = sequence_length(node.op.name, length)
            if size is not None:
                return size
            index = node.arguments[1]
            position = If(
                apply("__lt__", index, lit(0)), apply("__add__", index, length), index, INT
            )
            offset = apply("__mul__", position, lit(abs(step)))
            if step < 0:
                offset = apply("__sub__", apply("__sub__", apply("len", source), lit(1)), offset)
            return apply("__getitem__", source, offset)
    return join_prefix(node)


def join_prefix(node: Call) -> Node:
    if node.op.name != ".join":
        return node
    separator, sliced = node.arguments
    if not isinstance(separator, Lit) or not isinstance(separator.value, str):
        return node
    if not isinstance(sliced, Call) or sliced.op.name != "slice":
        return node
    source, start, stop, *steps = sliced.arguments
    if not isinstance(source, Call) or source.op.name != ".split" or len(source.arguments) != 2:
        return node
    if (
        source.arguments[1] != separator
        or len(separator.value) != 1
        or not isinstance(start, Lit)
        or start.value not in (None, 0)
        or not isinstance(stop, Lit)
        or not isinstance(stop.value, int)
        or not 0 < stop.value <= 64
        or not unit_stride(steps)
    ):
        return node
    escaped = re.escape(separator.value)
    pattern = rf"[^{escaped}]*(?:{escaped}[^{escaped}]*){{0,{stop.value - 1}}}"
    return matched_text("re.match", source.arguments[0], pattern)


def sum_partition(node: Call) -> Node | None:
    if node.op.name != "__add__" or node.type != INT:
        return None
    combined, last = node.arguments
    if not isinstance(combined, Call) or combined.op.name != "__add__":
        return None
    first, middle = combined.arguments
    if not isinstance(middle, Call) or middle.op.name != "sum":
        return None
    sliced = middle.arguments[0]
    if not isinstance(sliced, Call) or sliced.op.name != "slice":
        return None
    source, start, stop, *steps = sliced.arguments
    if (
        not isinstance(start, Lit)
        or start.value != 1
        or not isinstance(stop, Lit)
        or stop.value != -1
        or steps
        or source.type.item != INT
        or first != apply("__getitem__", source, lit(0))
        or last != apply("__getitem__", source, lit(-1))
    ):
        return None
    return Guarded(apply("sum", source), apply("__ge__", apply("len", source), lit(2)))


def word_slice(source: Node) -> tuple[Node, int, int] | None:
    start, stop = 0, 0
    if isinstance(source, Call) and source.op.name == "slice":
        source, begin, end, *steps = source.arguments
        if (
            not isinstance(begin, Lit)
            or not isinstance(begin.value, int | type(None))
            or not isinstance(end, Lit)
            or not isinstance(end.value, int | type(None))
            or not unit_stride(steps)
        ):
            return None
        start, stop = begin.value or 0, end.value or 0
        if not 0 <= start <= 64 or not -64 <= stop <= 0 or end.value == 0:
            return None
    if not isinstance(source, Call) or source.op.name != ".split" or len(source.arguments) != 1:
        return None
    return source.arguments[0], start, stop


def join_words(node: Call) -> Node | None:
    if node.op.name != ".join" or node.arguments[0] != lit(" "):
        return None
    sliced = word_slice(node.arguments[1])
    if sliced is None:
        return None
    source, start, stop = sliced
    spaced = apply(
        "re.sub",
        apply(".strip", source),
        lit(whitespace_pattern()),
        lit(0),
        lit(" "),
        lit(0),
    )
    patterns = []
    if start:
        patterns.append(rf"\A(?:[^ ]+ ?){{1,{start}}}")
    if stop:
        patterns.append(rf"(?: ?[^ ]+){{1,{-stop}}}\Z")
    if patterns:
        spaced = apply("re.sub", spaced, lit("|".join(patterns)), lit(0), lit(""), lit(0))
    return spaced


def word_item(node: Call) -> Node | None:
    if node.op.name != "__getitem__":
        return None
    source, index = node.arguments
    if (
        not isinstance(source, Call)
        or source.op.name != ".split"
        or len(source.arguments) != 1
        or not isinstance(index, Lit)
        or index.value not in (0, -1)
    ):
        return None
    text = source.arguments[0]
    pattern = non_whitespace_pattern()
    if index.value == -1:
        text = apply(".rstrip", text)
        pattern += r"\Z"
    return matched_text("re.search", text, pattern)


def matched_text(operation: str, text: Node, pattern: str) -> Node:
    matched = apply(operation, text, lit(pattern), lit(0))
    return apply("coalesce", matched, lit(""))
