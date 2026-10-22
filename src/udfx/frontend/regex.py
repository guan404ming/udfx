from __future__ import annotations

import inspect
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from udfx.errors import Unsupported
from udfx.frontend.binding import bind
from udfx.frontend.calls import invoke, known_op, node_of, none_test, truth
from udfx.frontend.constants import constant, constant_value
from udfx.frontend.logic import conjunction
from udfx.frontend.values import Item, Match, Method, Opaque, Value
from udfx.ir import FALSE, Lit, Node, lit
from udfx.ops import Fault
from udfx.semantics.regex_translation import replacement, translate


def literal(item: Item, name: str, kind: type | tuple[type, ...]) -> Any:
    try:
        value = constant_value(item)
    except Unsupported as error:
        raise Unsupported(f"regex {name} must be a literal") from error
    if not isinstance(value, kind):
        raise Unsupported(f"regex {name} has an unsupported literal type")
    return value


def call_regex(
    callee: Item,
    arguments: Sequence[Item],
    keywords: Mapping[str, Item],
    path: Node,
) -> tuple[Item, list[Fault]] | None:
    if isinstance(callee, Method) and arguments and isinstance(arguments[0], Match):
        match = arguments[0]
        groups = arguments[1:]
        if (
            callee.name != "group"
            or keywords
            or len(groups) > 1
            or (groups and literal(groups[0], "group", int) != 0)
        ):
            raise Unsupported("regex matches support only group() and group(0)")
        if conjunction(path, none_test(match)) != FALSE:
            raise Unsupported("regex group() needs a path that proves the match is not None")
        return Value(match.node), []
    if not isinstance(callee, Opaque):
        return None
    if callee.value is bool and len(arguments) == 1 and isinstance(arguments[0], Match):
        if keywords:
            raise Unsupported("bool takes no keyword arguments")
        node, faults = truth(arguments[0])
        return Value(node), faults
    name = known_op(callee.value)
    if name is None or not name.startswith("re."):
        return None
    return invoke_regex(name, cast("Callable[..., object]", callee.value), arguments, keywords)


def invoke_regex(
    name: str,
    function: Callable[..., object],
    arguments: Sequence[Item],
    keywords: Mapping[str, Item],
) -> tuple[Item, list[Fault]]:
    signature = inspect.signature(function)
    bindings = bind(signature, arguments, keywords)
    for key, parameter in signature.parameters.items():
        if key not in bindings and parameter.default is not inspect.Parameter.empty:
            bindings[key] = constant(parameter.default)
    pattern, flags = pattern_arguments(function, bindings)
    translated = translate(pattern, flags, "rust")
    if name == "re.compile":
        return Opaque(re.compile(pattern, flags)), []
    for key, default in (("pos", 0), ("endpos", sys.maxsize)):
        if key in bindings and literal(bindings[key], key, int) != default:
            raise Unsupported(f"regex {key} is unsupported")
    if name == "re.findall" and translated.groups > 1:
        raise Unsupported("regex findall supports at most one capture group")
    if name == "re.split" and translated.groups:
        raise Unsupported("regex split does not support capture groups")
    nodes = [node_of(bindings["string"]), lit(pattern), lit(int(flags))]
    if name == "re.sub":
        substitute = node_of(bindings["repl"])
        if isinstance(substitute, Lit) and isinstance(substitute.value, str):
            replacement(pattern, flags, substitute.value, "rust")
        nodes.extend((substitute, lit(literal(bindings["count"], "count", int))))
    elif name == "re.split":
        nodes.append(lit(literal(bindings["maxsplit"], "maxsplit", int)))
    node, faults = invoke(name, nodes, TypeError)
    return (
        Match(node) if name in {"re.search", "re.match", "re.fullmatch"} else Value(node)
    ), faults


def pattern_arguments(
    function: Callable[..., object], bindings: Mapping[str, Item]
) -> tuple[str, int]:
    owner = getattr(function, "__self__", None)
    if "self" in bindings:
        owner = literal(bindings["self"], "pattern", re.Pattern)
    pattern = (
        owner
        if isinstance(owner, re.Pattern)
        else literal(bindings["pattern"], "pattern", (str, re.Pattern))
    )
    flags = literal(bindings.get("flags", Value(lit(0))), "flags", int)
    if flags & ~int(re.IGNORECASE | re.MULTILINE | re.DOTALL):
        raise Unsupported("regex flags support only IGNORECASE, MULTILINE, and DOTALL")
    if isinstance(pattern, re.Pattern):
        if flags:
            raise Unsupported("regex compiled patterns do not accept flags")
        flags, pattern = pattern.flags, pattern.pattern
    if not isinstance(pattern, str):
        raise Unsupported("regex patterns must be strings")
    return pattern, flags
