from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from udfx.errors import Unsupported
from udfx.frontend.core.evaluation import Evaluation
from udfx.frontend.core.expressions import invoke
from udfx.frontend.core.values import Item, Method, Series, Value
from udfx.frontend.operations.builtins import call
from udfx.frontend.operations.series import settled
from udfx.frontend.resolve.constants import constant_value
from udfx.ir import Node, lit
from udfx.ops import Fault
from udfx.semantics.regex_replacement import replacement
from udfx.semantics.regex_translation import FLAGS, translate
from udfx.types import INT

PYTHON_METHODS = {
    "lower",
    "upper",
    "strip",
    "lstrip",
    "rstrip",
    "title",
    "capitalize",
    "swapcase",
    "casefold",
    "startswith",
    "endswith",
    "find",
    "rfind",
    "zfill",
    "center",
    "ljust",
    "rjust",
    "isalnum",
    "isalpha",
    "isdigit",
    "isspace",
    "islower",
    "isupper",
    "istitle",
    "isnumeric",
    "isdecimal",
}

type Invoked = tuple[Node, list[Fault]]
type TextHandler = Callable[[str, Node, Sequence[Item], Mapping[str, Item]], Invoked]


def bound(
    arguments: Sequence[Item], keywords: Mapping[str, Item], names: Sequence[str]
) -> dict[str, Item]:
    if len(arguments) > len(names) or not set(keywords) <= set(names):
        raise Unsupported(f"series method arguments {', '.join(keywords)}")
    found = dict(zip(names, arguments, strict=False))
    if found.keys() & keywords.keys():
        raise Unsupported("series method argument given twice")
    return {**found, **keywords}


def value(found: Mapping[str, Item], name: str, default: object = None) -> object:
    return constant_value(found[name]) if name in found else default


def flags_of(found: Mapping[str, Item], name: str) -> int:
    flags = value(found, "flags", 0)
    if not isinstance(flags, int) or flags & ~FLAGS:
        raise Unsupported(f"series str.{name} needs literal regex flags")
    return flags


def length(
    name: str, node: Node, arguments: Sequence[Item], keywords: Mapping[str, Item]
) -> Invoked:
    bound(arguments, keywords, ())
    return invoke("len", [node], TypeError)


def sliced(
    name: str, node: Node, arguments: Sequence[Item], keywords: Mapping[str, Item]
) -> Invoked:
    found = bound(arguments, keywords, ("start", "stop", "step"))
    start, stop = value(found, "start"), value(found, "stop")
    if value(found, "step") is not None or not all(
        limit is None or type(limit) is int for limit in (start, stop)
    ):
        raise Unsupported("series str.slice needs literal integer bounds and no step")
    return invoke("slice", [node, lit(start, INT.or_none()), lit(stop, INT.or_none())], TypeError)


def searched(
    name: str, node: Node, arguments: Sequence[Item], keywords: Mapping[str, Item]
) -> Invoked:
    names = ("pat", "case", "flags", "na", "regex") if name == "contains" else ("pat", "flags")
    found = bound(arguments, keywords, names)
    pattern, flags = value(found, "pat"), flags_of(found, name)
    if not isinstance(pattern, str):
        raise Unsupported(f"series str.{name} needs a literal pattern")
    if value(found, "case", True) is not True or value(found, "na") is not None:
        raise Unsupported(f"series str.{name} supports only case=True and na=None")
    if value(found, "regex", True) is False:
        if flags:
            raise Unsupported("series str.contains flags need regex=True")
        return invoke("__contains__", [node, lit(pattern)], TypeError)
    translate(pattern, flags, "rust")
    operation = "re.contains" if name == "contains" else "re.count"
    return invoke(operation, [node, lit(pattern), lit(flags)], TypeError)


def replaced(
    name: str, node: Node, arguments: Sequence[Item], keywords: Mapping[str, Item]
) -> Invoked:
    found = bound(arguments, keywords, ("pat", "repl", "n", "case", "flags", "regex"))
    pattern, text, count = value(found, "pat"), value(found, "repl"), value(found, "n", -1)
    flags = flags_of(found, name)
    if not (isinstance(pattern, str) and isinstance(text, str) and type(count) is int):
        raise Unsupported("series str.replace needs literal strings and count")
    if value(found, "case") is not None:
        raise Unsupported("series str.replace supports only the default case")
    if value(found, "regex", False) is False:
        if flags or count >= 0:
            raise Unsupported("series str.replace without regex supports only n=-1 and no flags")
        return invoke(".replace", [node, lit(pattern), lit(text)], TypeError)
    translate(pattern, flags, "rust")
    replacement(pattern, flags, text, "rust")
    nodes = [node, lit(pattern), lit(flags), lit(text), lit(max(count, 0))]
    return invoke("re.sub", nodes, TypeError)


def python_method(
    name: str, node: Node, arguments: Sequence[Item], keywords: Mapping[str, Item]
) -> Invoked:
    if keywords:
        raise Unsupported(f"series str.{name} with keywords")
    return call(Method(name), [Value(node), *arguments])


HANDLERS: dict[str, TextHandler] = {
    "len": length,
    "slice": sliced,
    "contains": searched,
    "count": searched,
    "replace": replaced,
    **dict.fromkeys(PYTHON_METHODS, python_method),
}


def text_method(
    receiver: Series, name: str, arguments: Sequence[Item], keywords: Mapping[str, Item]
) -> Evaluation:
    handler = HANDLERS.get(name)
    if handler is None:
        raise Unsupported(f"series str.{name}")
    return settled([receiver], *handler(name, receiver.node, arguments, keywords))
