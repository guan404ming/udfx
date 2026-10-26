from __future__ import annotations

import inspect
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from udfx.errors import Unsupported
from udfx.frontend.core.expressions import invoke, node_of, none_test, truth
from udfx.frontend.core.logic import conjunction
from udfx.frontend.core.values import Item, Match, Method, Opaque, Value
from udfx.frontend.resolve.callables import known_op
from udfx.frontend.resolve.constants import constant, constant_value
from udfx.frontend.resolve.functions import bind
from udfx.ir import FALSE, Comprehension, Element, Lit, Node, lit
from udfx.ops import Fault, apply
from udfx.semantics.regex_patterns import bounded_runs, group_prefix, position_safe
from udfx.semantics.regex_replacement import replacement
from udfx.semantics.regex_translation import FLAGS, translate
from udfx.types import INT, STR


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
        if callee.name not in {"group", "start"} or keywords or len(groups) > 1:
            raise Unsupported("regex matches support group(0) and start(integer)")
        if conjunction(path, none_test(match)) != FALSE:
            raise Unsupported(
                f"regex {callee.name}() needs a path that proves the match is not None"
            )
        group = literal(groups[0], "group", int) if groups else 0
        if callee.name == "start":
            pattern = group_prefix(match.pattern, match.flags, group, match.mode)
            prefix = apply(
                "re.sub", match.source, lit(pattern), lit(match.flags), lit(r"\g<1>"), lit(0)
            )
            return Value(apply("__add__", match.position, apply("len", prefix))), []
        if group != 0:
            raise Unsupported("regex matches support only group() and group(0)")
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
    owner = getattr(function, "__self__", None)
    source = re.Pattern.search if function is re.Pattern.match else function
    if isinstance(owner, re.Pattern) and name == "re.match":
        source = owner.search
    signature = inspect.signature(source)
    bindings = bind(signature, arguments, keywords)
    for key, parameter in signature.parameters.items():
        if key not in bindings and parameter.default is not inspect.Parameter.empty:
            bindings[key] = constant(parameter.default)
    pattern, flags = pattern_arguments(function, bindings)
    if name == "re.findall" and (runs := bounded_runs(pattern, flags)) is not None:
        nodes = regex_arguments(name, bindings, runs[0], flags, 0)
        source, faults = invoke(name, nodes, TypeError)
        element = Element("regex run", STR)
        matched = apply("re.fullmatch", element, lit(runs[1]), lit(flags))
        predicate = apply("__ne__", matched, lit(None))
        return Value(Comprehension(source, element, element, predicate, FALSE)), faults
    translated = translate(pattern, flags, "rust")
    if name == "re.compile":
        return Opaque(re.compile(pattern, flags)), []
    nodes = regex_arguments(name, bindings, pattern, flags, translated.groups)
    node, faults = invoke(name, nodes, TypeError)
    if name in {"re.search", "re.match", "re.fullmatch"}:
        position = match_position(bindings)
        return Match(node, nodes[0], pattern, flags, name, position), faults
    return Value(node), faults


def regex_arguments(
    name: str, bindings: Mapping[str, Item], pattern: str, flags: int, groups: int
) -> list[Node]:
    if "endpos" in bindings and literal(bindings["endpos"], "endpos", int) != sys.maxsize:
        raise Unsupported("regex endpos is unsupported")
    if name == "re.findall" and groups > 1:
        raise Unsupported("regex findall supports at most one capture group")
    if name == "re.split" and groups:
        raise Unsupported("regex split does not support capture groups")
    nodes = [node_of(bindings["string"]), lit(pattern), lit(int(flags))]
    position = match_position(bindings)
    if position != lit(0):
        if name not in {"re.search", "re.match", "re.fullmatch"} or not position_safe(
            pattern, flags
        ):
            raise Unsupported("regex positions require an unanchored pattern")
        nodes[0] = apply("slice", nodes[0], position, lit(None, INT.or_none()))
    if name == "re.sub":
        substitute = node_of(bindings["repl"])
        if isinstance(substitute, Lit) and isinstance(substitute.value, str):
            replacement(pattern, flags, substitute.value, "rust")
        nodes.extend((substitute, lit(literal(bindings["count"], "count", int))))
    elif name == "re.split":
        nodes.append(lit(literal(bindings["maxsplit"], "maxsplit", int)))
    return nodes


def match_position(bindings: Mapping[str, Item]) -> Node:
    position = node_of(bindings.get("pos", constant(0)))
    if position == lit(0):
        return position
    if position.type != INT:
        raise Unsupported("regex positions require an integer")
    length = apply("len", node_of(bindings["string"]))
    return apply("min", length, apply("max", lit(0), position))


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
    if flags & ~FLAGS:
        raise Unsupported("regex flags support only IGNORECASE, MULTILINE, DOTALL, and UNICODE")
    if isinstance(pattern, re.Pattern):
        if flags:
            raise Unsupported("regex compiled patterns do not accept flags")
        flags, pattern = pattern.flags, pattern.pattern
    if not isinstance(pattern, str):
        raise Unsupported("regex patterns must be strings")
    return pattern, flags
