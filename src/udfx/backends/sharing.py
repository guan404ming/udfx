from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from udfx.backends.hoisting import worth_binding
from udfx.backends.lowering_context import bindings_enabled
from udfx.backends.rules import Lowered, Rule, RuleResult
from udfx.ir import Call, Col, If
from udfx.types import Kind

if TYPE_CHECKING:
    from udfx.backends.base import Backend

SHARED = "udfx argument"


def lowered[E](result: RuleResult[E]) -> Lowered[E]:
    return cast("Lowered[E]", result) if isinstance(result, Lowered) else Lowered(result)


def call_rule[E](
    backend: Backend[E],
    node: Call,
    rule: Rule[E],
    arguments: Sequence[Any],
    keywords: Mapping[str, Any],
) -> RuleResult[E]:
    direct = rule(*arguments, **keywords)
    candidates = [
        index
        for index, argument in enumerate(node.arguments)
        if isinstance(argument, Call | If) and argument.type.kind is not Kind.LIST
    ]
    if (
        not candidates
        or not backend.binds
        or not bindings_enabled(backend.binds)
        or backend.rules.linear(node)
    ):
        return direct
    placeholders = {
        index: backend.column(Col(f"{SHARED} {index}", node.arguments[index].type))
        for index in candidates
    }
    probe = lowered(rule(*substituted(arguments, placeholders), **keywords))
    if probe.prepare is not None or probe.recover_overflow is not None:
        return direct
    result = lowered(direct)

    def share(part: Callable[[Lowered[E]], Any]) -> Any:
        value = part(probe)
        if value is None or value is False:
            return part(result)
        text = str(value)
        uses = {index: text.count(str(placeholders[index])) for index in candidates}
        shared = [
            index for index in candidates if worth_binding(node.arguments[index], uses[index])
        ]
        if not shared:
            return part(result)
        return bound(
            backend,
            shared,
            list(arguments),
            lambda values: part(lowered(rule(*values, **keywords))),
        )

    return Lowered(
        share(lambda value: value.expr),
        share(lambda value: value.exact),
        share(lambda value: value.overflow),
        guarantees=result.guarantees,
    )


def substituted(arguments: Sequence[Any], replacements: Mapping[int, Any]) -> list[Any]:
    return [replacements.get(index, argument) for index, argument in enumerate(arguments)]


def bound[E](
    backend: Backend[E], indexes: Sequence[int], values: list[Any], body: Callable[[list[Any]], E]
) -> E:
    if not indexes:
        return body(values)
    index, rest = indexes[0], indexes[1:]
    return backend.bind(
        values[index],
        lambda item: bound(backend, rest, substituted(values, {index: item}), body),
    )
