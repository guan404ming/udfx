from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields, replace

from udfx.graphs import map_children, substitute
from udfx.ir import Call, Comprehension, Element, Guarded, If, Lit, Node, lit
from udfx.ops.registry import apply
from udfx.types import Kind, Type

type Literal = Callable[[Call, int], bool]


@dataclass(frozen=True)
class Entry:
    node: If
    receiver: Node
    suffixes: tuple[str, ...]
    value: Node
    truthy: bool


def dispatch_suffixes(roots: Sequence[Node], literal: Literal) -> tuple[Node, ...]:
    seen: dict[int, Node] = {}

    def visit(node: Node) -> Node:
        if id(node) not in seen:
            entries, rest = chain(node)
            seen[id(node)] = (
                rebuild(entries, visit(rest), visit, literal)
                if entries
                else map_children(node, visit)
            )
        return seen[id(node)]

    return tuple(visit(root) for root in roots)


def entry(node: Node) -> Entry | None:
    if not isinstance(node, If):
        return None
    condition, value, truthy = node.condition, node.then, False
    if (
        isinstance(condition, Call)
        and condition.op.name == "bool"
        and condition.arguments[0] is value
        and isinstance(value, If)
        and isinstance(value.orelse, Lit)
        and value.orelse.value is None
    ):
        condition, value, truthy = value.condition, value.then, True
    found = endings(condition)
    if found is None:
        return None
    return Entry(node, found[0], found[1], value, truthy)


def endings(condition: Node) -> tuple[Node, tuple[str, ...]] | None:
    if (
        isinstance(condition, If)
        and isinstance(condition.then, Lit)
        and condition.then.value is True
    ):
        left, right = endings(condition.condition), endings(condition.orelse)
        if left is None or right is None or left[0] is not right[0]:
            return None
        return left[0], left[1] + right[1]
    if (
        not isinstance(condition, Call)
        or condition.op.name != ".endswith"
        or condition.arguments[0].type.kind is not Kind.STR
        or not isinstance(condition.arguments[1], Lit)
        or not isinstance(condition.arguments[1].value, str)
    ):
        return None
    return condition.arguments[0], (condition.arguments[1].value,)


def chain(node: Node) -> tuple[list[Entry], Node]:
    entries = []
    while (found := entry(node)) is not None:
        entries.append(found)
        node = found.node.orelse
    return entries, node


def rebuild(
    entries: list[Entry], rest: Node, visit: Callable[[Node], Node], literal: Literal
) -> Node:
    entries = [replace(item, value=visit(item.value)) for item in entries]
    groups: list[list[Entry]] = []
    for item in entries:
        group = groups[-1] if groups else []
        if (
            group
            and item.receiver is group[0].receiver
            and item.truthy == group[0].truthy
            and not any(
                suffix != other and suffix.endswith(other)
                for suffix in item.suffixes
                for previous in group
                for other in previous.suffixes
            )
            and generalize([*group, item], literal) is not None
        ):
            group.append(item)
        else:
            groups.append([item])
    for group in reversed(groups):
        node = group[0].node
        if len(group) == 1:
            rest = If(visit(node.condition), visit(node.then), rest, node.type)
        else:
            rest = dispatch(group, rest, visit(group[0].receiver), literal)
    return rest


def dispatch(group: list[Entry], rest: Node, receiver: Node, literal: Literal) -> Node:
    found = generalize(group, literal)
    assert found is not None
    template, holes = found
    suffixes = [suffix for item in group for suffix in item.suffixes]
    pattern = "(?:" + "|".join(dict.fromkeys(map(re.escape, suffixes))) + r")\Z"
    match = apply("re.search", receiver, lit(pattern), lit(re.UNICODE.value))
    replacements = {
        id(hole): lookup(
            match,
            suffixes,
            tuple(value for item, value in zip(group, values, strict=True) for _ in item.suffixes),
            hole,
        )
        for hole, values in holes
    }
    value = substitute(template, replacements)
    if group[0].truthy:
        value = Guarded(value, apply("bool", value))
    return If(apply("is_none", match), rest, value, group[0].node.type)


def lookup(match: Node, suffixes: list[str], values: tuple[object, ...], hole: Element) -> Node:
    if hole.type.kind is Kind.INT and all(
        type(value) is int and value == len(suffix)
        for value, suffix in zip(values, suffixes, strict=True)
    ):
        return apply("len", match)
    if hole.type.kind is Kind.INT and all(
        type(value) is int and value == -len(suffix)
        for value, suffix in zip(values, suffixes, strict=True)
    ):
        return apply("__neg__", apply("len", match))
    cases = list(dict(reversed(list(zip(suffixes, values, strict=True)))).items())[::-1]
    node: Node = Lit(cases[-1][1], hole.type)
    for suffix, value in reversed(cases[:-1]):
        node = If(apply("__eq__", match, lit(suffix)), Lit(value, hole.type), node, hole.type)
    return node


def generalize(
    group: list[Entry], literal: Literal
) -> tuple[Node, list[tuple[Element, tuple[object, ...]]]] | None:
    holes: dict[tuple[Type, tuple[object, ...]], Element] = {}
    memo: dict[tuple[int, ...], Node | None] = {}

    def visit(parts: tuple[Node, ...]) -> Node | None:
        key = tuple(map(id, parts))
        if key not in memo:
            memo[key] = unify(parts)
        return memo[key]

    def unify(parts: tuple[Node, ...]) -> Node | None:
        first = parts[0]
        if all(part is first for part in parts):
            return first
        if all(isinstance(part, Lit) and part.type == first.type for part in parts):
            values = tuple(part.value for part in parts if isinstance(part, Lit))
            if len({repr(value) for value in values}) == 1:
                return first
            return holes.setdefault(
                (first.type, values), Element(f"udfx suffix {len(holes)}", first.type)
            )
        if isinstance(first, Comprehension | Lit) or any(
            type(part) is not type(first) for part in parts
        ):
            return None
        changes: dict[str, object] = {}
        for field in fields(first):
            values = [getattr(part, field.name) for part in parts]
            if isinstance(values[0], Node):
                found = visit(tuple(values))
                if found is None:
                    return None
                changes[field.name] = found
            elif isinstance(values[0], tuple) and values[0] and isinstance(values[0][0], Node):
                if any(len(value) != len(values[0]) for value in values):
                    return None
                items = [visit(items) for items in zip(*values, strict=True)]
                if any(item is None for item in items):
                    return None
                changes[field.name] = tuple(items)
            elif any(value != values[0] for value in values):
                return None
        result = replace(first, **changes)
        if isinstance(result, Call) and any(
            isinstance(argument, Element) and argument in hole_set() and literal(result, index)
            for index, argument in enumerate(result.arguments)
        ):
            return None
        return result

    def hole_set() -> set[Element]:
        return set(holes.values())

    template = visit(tuple(item.value for item in group))
    if template is None:
        return None
    return template, [(hole, values) for (_, values), hole in holes.items()]
