from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from udfx.backends.registration import Lowered, RuleResult
from udfx.errors import Unsupported
from udfx.ir import FALSE, TRUE, Call, Comprehension, Element, If, Lit
from udfx.ops import OpKey
from udfx.types import Kind

if TYPE_CHECKING:
    from udfx.backends.base import Backend


def lower_if[E](backend: Backend[E], node: If, elements: Mapping[Element, E]) -> Lowered[E]:
    condition = backend.lower(node.condition, elements)
    then, orelse = backend.lower(node.then, elements), backend.lower(node.orelse, elements)
    expr = backend.select(condition.expr, then.expr, orelse.expr)
    exact = backend.exact(condition)
    if then.exact is not None or orelse.exact is not None:
        branch = backend.select(
            condition.expr,
            then.exact if then.exact is not None else backend.literal(TRUE),
            orelse.exact if orelse.exact is not None else backend.literal(TRUE),
        )
        exact = backend.conjunction(exact, branch)
    overflow = None
    if then.overflow is not None or orelse.overflow is not None:
        overflow = backend.select(
            condition.expr,
            then.overflow if then.overflow is not None else backend.literal(FALSE),
            orelse.overflow if orelse.overflow is not None else backend.literal(FALSE),
        )
    return Lowered(expr, exact, overflow)


def lower_call[E](backend: Backend[E], node: Call, elements: Mapping[Element, E]) -> Lowered[E]:
    rule = backend.rules.get(node.op.key)
    if rule is None:
        raise Unsupported(f"{backend.name} has no rule for {node.op.key}")
    options = backend.rules.options(node.op.key)
    arguments = [
        Lowered(cast("Any", argument.value))
        if index in options.literals and isinstance(argument, Lit)
        else backend.lower(argument, elements)
        for index, argument in enumerate(node.arguments)
    ]
    result: RuleResult[E] = rule(
        *(argument.expr for argument in arguments),
        **{name: fact(node) for name, fact in options.facts.items()},
    )
    lowered = cast("Lowered[E]", result) if isinstance(result, Lowered) else Lowered(result)
    return Lowered(
        lowered.expr,
        backend.conjunction(*(backend.exact(argument) for argument in arguments), lowered.exact),
        lowered.overflow,
    )


def lower_comprehension[E](
    backend: Backend[E], node: Comprehension, elements: Mapping[Element, E]
) -> Lowered[E]:
    source = backend.lower(node.source, elements)

    def parts(element: E) -> tuple[Lowered[E], Lowered[E], E]:
        scope = {**elements, node.element: element}
        condition = backend.lower(node.condition, scope)
        value = backend.lower(node.value, scope)
        fault = backend.lower(node.fault, scope)
        exact = backend.conjunction(
            backend.exact(condition),
            backend.select(condition.expr, backend.exact_expression(value), backend.literal(TRUE)),
            backend.exact(fault),
            backend.negate(fault.expr),
        )
        return condition, value, cast("E", exact)

    try:
        if isinstance(node.source, Lit) and isinstance(node.source.value, tuple | list):
            chunks: list[E] = []
            exacts: list[E] = []
            for item in node.source.value:
                element = backend.lower(Lit(item, node.element.type), elements)
                condition, value, exact = parts(element.expr)
                chunks.append(
                    backend.select(
                        condition.expr,
                        backend.list_value([value.expr], node.type),
                        backend.literal(Lit((), node.type)),
                    )
                )
                exacts.append(exact)
            result = backend.literal(Lit((), node.type))
            concatenate = backend.rules.get(OpKey("__add__", (Kind.LIST, Kind.LIST)))
            if concatenate is None:
                raise Unsupported(f"{backend.name} has no list concatenation")
            for chunk in chunks:
                result = cast("E", concatenate(result, chunk))
            return Lowered(result, backend.conjunction(backend.exact(source), *exacts))
        filtered = backend.list_filter(source.expr, lambda element: parts(element)[0].expr)
        value = backend.list_transform(filtered, lambda element: parts(element)[1].expr)
        flags = backend.list_transform(source.expr, lambda element: parts(element)[2])
        every = backend.rules.get(OpKey("all", (Kind.LIST,)))
        if every is None:
            raise Unsupported(f"{backend.name} has no list reduction")
        return Lowered(value, backend.conjunction(backend.exact(source), cast("E", every(flags))))
    except Unsupported:
        if node.requires_native:
            raise
        return Lowered(backend.literal(Lit((), node.type)), backend.literal(FALSE))
