import re

from udfx.frontend.core.sequences import unit_stride
from udfx.ir import Call, Guarded, If, Lit, Node, lit
from udfx.ops.registry import apply
from udfx.semantics.casing import folded_case, lowercase_pattern
from udfx.semantics.ranges import integer_bounds
from udfx.semantics.regex_patterns import literal_suffix
from udfx.semantics.regex_translation import union_patterns
from udfx.semantics.text import CASE_METHODS
from udfx.types import INT, INT64_MAX, STR, Kind


def case_chain(node: Call) -> tuple[Node, list[str]]:
    source = node.arguments[0]
    methods = [node.op.name.removeprefix(".")]
    while isinstance(source, Call) and source.op.name in {f".{method}" for method in CASE_METHODS}:
        methods.append(source.op.name.removeprefix("."))
        source = source.arguments[0]
    return source, methods


def guarded_case_chain(node: Call) -> Node:
    source, methods = case_chain(node)
    if source == node.arguments[0] or isinstance(source, Lit):
        return node
    if folded_case(methods) == ():
        return apply(".casefold", source)
    return Guarded(apply(".lower", source), apply(".isascii", source))


def folded_case_chain(node: Call) -> Node:
    source, methods = case_chain(node)
    if source == node.arguments[0] or isinstance(source, Lit):
        return node
    replacements = folded_case(methods)
    if replacements is None:
        return guarded_case_chain(node)
    for original, replacement in replacements:
        source = apply(".replace", source, lit(original), lit(replacement))
    return apply(".casefold", source)


def simplify_text(node: Call) -> Node:
    if node.op.name in {"__eq__", "__ne__"}:
        left, right = node.arguments
        source = right if left == lit("") else left if right == lit("") else None
        if source is not None and not isinstance(source, Lit) and source.type == STR:
            test = apply("bool", source)
            return apply("not", test) if node.op.name == "__eq__" else test
    if node.op.name == ".casefold":
        source, methods = case_chain(node)
        if source != node.arguments[0] and folded_case(methods) == ():
            return apply(".casefold", source)
    if node.op.name == ".lower":
        guarded = guarded_case_chain(node)
        if guarded != node:
            return guarded
    if node.op.name in {".lower", ".upper", ".title"}:
        argument = node.arguments[0]
        if (
            isinstance(argument, Call)
            and argument.op.name in {".strip", ".lstrip", ".rstrip"}
            and len(argument.arguments) == 1
        ):
            return apply(argument.op.name, apply(node.op.name, argument.arguments[0]))
    return searched_slice(node)


def searched_slice(node: Call) -> Node:
    suffix = last_split(node)
    if suffix != node or node.op.name != "slice":
        return suffix
    text, start, stop, *steps = node.arguments
    if not unit_stride(steps) or not isinstance(stop, Call):
        return node
    if stop.op.name not in {".find", ".rfind"} or stop.arguments[0] != text:
        return node
    separator = stop.arguments[1]
    if (
        not isinstance(separator, Lit)
        or not isinstance(separator.value, str)
        or len(separator.value) != 1
    ):
        return node
    escaped = re.escape(separator.value)
    substitute = ""
    if stop.op.name == ".find" and isinstance(start, Lit) and start.value in (None, 0):
        pattern = rf"{escaped}[\s\S]*\Z"
    elif stop.op.name == ".rfind" and start == apply(
        "__add__", apply(".find", text, separator), lit(1)
    ):
        pattern = (
            rf"\A[^{escaped}]*{escaped}([\s\S]*){escaped}[^{escaped}]*\Z"
            rf"|\A[^{escaped}]*{escaped}[^{escaped}]*\Z"
        )
        substitute = r"\1"
    else:
        return node
    selected = apply("re.sub", text, lit(pattern), lit(0), lit(substitute), lit(0))
    missing = apply("slice", text, lit(None, INT.or_none()), lit(-1))
    return If(apply("__contains__", text, separator), selected, missing, node.type)


def last_split(node: Call) -> Node:
    if node.op.name != "slice":
        return node
    text, start, stop, *steps = node.arguments
    if not unit_stride(steps):
        return node
    if not isinstance(stop, Lit) or stop.value is not None:
        return node
    if not isinstance(start, Call) or start.op.name != "__add__":
        return node
    search, offset = start.arguments
    if not isinstance(search, Call) or search.op.name != ".rfind":
        return node
    source, pattern = search.arguments
    if (
        source == text
        and isinstance(pattern, Lit)
        and isinstance(pattern.value, str)
        and len(pattern.value) == 1
        and isinstance(offset, Lit)
        and offset.value == 1
    ):
        return apply("last_split", text, pattern)
    return node


def joined_split(node: Call) -> Node | None:
    def parts(value: Node) -> list[Node]:
        if isinstance(value, Call) and value.op.name == "__add__" and value.type.kind is Kind.STR:
            return [part for argument in value.arguments for part in parts(argument)]
        return [value]

    values = parts(node)
    if len(values) < 3 or len(values) % 2 == 0:
        return None
    separator = values[1]
    if not isinstance(separator, Lit) or not isinstance(separator.value, str):
        return None
    if any(value != separator for value in values[1::2]):
        return None
    source = None
    for index, value in enumerate(values[::2]):
        if not isinstance(value, Call) or value.op.name != "__getitem__":
            return None
        items, position = value.arguments
        if position != lit(index) or (source is not None and source != items):
            return None
        source = items
    if not isinstance(source, Call) or source.op.name != ".split" or len(source.arguments) != 2:
        return None
    text, original = source.arguments
    if not isinstance(original, Lit) or not isinstance(original.value, str) or not original.value:
        return None
    count = len(values) // 2 + 1
    if integer_bounds(apply("len", source), INT64_MAX) != (count, count):
        return None
    return apply(".replace", text, original, separator)


def removed_suffix(node: Call) -> Node | None:
    text, pattern, flags, replacement, count = node.arguments
    if (
        replacement != lit("")
        or not isinstance(pattern, Lit)
        or not isinstance(pattern.value, str)
        or not isinstance(flags, Lit)
        or not isinstance(flags.value, int)
        or not isinstance(count, Lit)
        or not isinstance(count.value, int)
        or count.value < 0
    ):
        return None
    separator = literal_suffix(pattern.value, flags.value)
    return None if separator is None else apply("split_item", text, lit(separator), lit(0))


def lowercase_contains(node: Call) -> Node | None:
    source, needle = node.arguments
    if (
        not isinstance(source, Call)
        or source.op.name != ".lower"
        or not isinstance(needle, Lit)
        or not isinstance(needle.value, str)
    ):
        return None
    pattern = lowercase_pattern(needle.value)
    if pattern is None:
        return None
    return apply("re.contains", source.arguments[0], lit(pattern), lit(0))


def combined_search(node: Call) -> Node | None:
    left, right = node.arguments
    if not all(
        isinstance(value, Call) and value.op.name == "re.contains" for value in (left, right)
    ):
        return None
    assert isinstance(left, Call) and isinstance(right, Call)
    text, first, flags = left.arguments
    other, second, other_flags = right.arguments
    if (
        text != other
        or flags != other_flags
        or not isinstance(flags, Lit)
        or not isinstance(flags.value, int)
        or not isinstance(first, Lit)
        or not isinstance(first.value, str)
        or not isinstance(second, Lit)
        or not isinstance(second.value, str)
    ):
        return None
    pattern = union_patterns(first.value, second.value, flags.value)
    return None if pattern is None else apply("re.contains", text, lit(pattern), flags)
