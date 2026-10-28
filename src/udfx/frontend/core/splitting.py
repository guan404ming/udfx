import re

from udfx.ir import FALSE, Call, Comprehension, Lit, Node, lit
from udfx.ops import apply
from udfx.semantics.ranges import proven


def literal_split(node: Node) -> tuple[Node, str] | None:
    if not isinstance(node, Call) or node.op.name != ".split" or len(node.arguments) != 2:
        return None
    value, separator = node.arguments
    if (
        not isinstance(separator, Lit)
        or not isinstance(separator.value, str)
        or not separator.value
    ):
        return None
    return value, separator.value


def split_item(node: Call) -> Node | None:
    source = literal_split(node.arguments[0])
    if source is None:
        return None
    value, separator = source
    return apply("split_item", value, lit(separator), node.arguments[1])


def split_length_greater(node: Call) -> Node | None:
    length, limit = node.arguments
    if (
        not isinstance(length, Call)
        or length.op.name != "len"
        or not isinstance(limit, Lit)
        or not isinstance(limit.value, int)
        or limit.value < 0
    ):
        return None
    source = literal_split(length.arguments[0])
    if source is None:
        return None
    known = proven(node)
    if known is not None:
        return lit(known)
    value, separator = source
    if len(separator) != 1 or limit.value > 64:
        return apply("__ge__", apply(".count", value, lit(separator)), limit)
    escaped = re.escape(separator)
    pattern = rf"\A(?:[^{escaped}]*{escaped}){{{limit.value}}}"
    return apply("re.contains", value, lit(pattern), lit(0))


def join_filtered_words(node: Call) -> Node | None:
    separator, values = node.arguments
    if (
        not isinstance(separator, Lit)
        or not isinstance(separator.value, str)
        or len(separator.value) != 1
        or not isinstance(values, Comprehension)
        or values.value != values.element
        or values.fault != FALSE
    ):
        return None
    condition = values.condition
    if not isinstance(condition, Call) or condition.op.name not in {"__gt__", "__ge__"}:
        return None
    length, limit = condition.arguments
    if (
        not isinstance(length, Call)
        or length.op.name != "len"
        or length.arguments != (values.element,)
        or not isinstance(limit, Lit)
        or type(limit.value) is not int
    ):
        return None
    minimum = limit.value + (condition.op.name == "__gt__")
    source = literal_split(values.source)
    if not 1 <= minimum <= 64 or source is None or source[1] != separator.value:
        return None
    escaped = re.escape(separator.value)
    token = f"[^{escaped}]{{0,{minimum - 1}}}"
    first = f"[^{escaped}]{{1,{minimum - 1}}}"
    pattern = (
        rf"(?:\A{first}|{escaped}{token})(?:{escaped}{token})*(?:{escaped}|\Z)"
        if minimum > 1
        else escaped + "{2,}"
    )
    replacement = separator.value.replace("\\", "\\\\")
    changed = apply("re.sub", source[0], lit(pattern), lit(0), lit(replacement), lit(0))
    return apply(".strip", changed, separator)
