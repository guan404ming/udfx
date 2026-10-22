from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from udfx.backends.registration import Lowered, RuleResult
from udfx.backends.rules import fold_predicate
from udfx.errors import Unsupported
from udfx.ir import FALSE, TRUE, Call, Comprehension, Element, If, Lit
from udfx.ops import OpKey
from udfx.program import Program
from udfx.types import Kind

if TYPE_CHECKING:
    from udfx.backends.base import Backend


def lower_program[E](backend: Backend[E], program: Program) -> Lowered[E]:
    value = backend.lower(program.value)
    expr, exact, failure = value.expr, value.exact, value.overflow
    predicate = fold_predicate(program.fault)
    if predicate != FALSE:
        fault = backend.lower(predicate)
        exact = backend.conjunction(exact, backend.exact(fault))
        failure = backend.disjunction(failure, fault.expr)
    if failure is not None:
        raised = backend.native_raise(program)
        if raised is None:
            exact = backend.conjunction(exact, backend.negate(failure))
        else:
            condition = failure if exact is None else backend.both(exact, failure)
            expr = backend.select(condition, raised, expr)
    return Lowered(expr, exact)


def lower_if[E](backend: Backend[E], node: If, elements: Mapping[Element, E]) -> Lowered[E]:
    predicate = fold_predicate(node.condition)
    if predicate in (TRUE, FALSE):
        return backend.lower(node.then if predicate == TRUE else node.orelse, elements)
    condition = backend.lower(predicate, elements)
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
    if node.type.kind is Kind.BOOL:
        folded = fold_predicate(node)
        if folded != node:
            return backend.lower(folded, elements)
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
    guards = [
        guard(argument.value if isinstance(argument, Lit) else arguments[index].expr)
        for index, guard in options.guards.items()
        for argument in (node.arguments[index],)
    ]
    return Lowered(
        lowered.expr,
        backend.conjunction(
            *(backend.exact(argument) for argument in arguments), lowered.exact, *guards
        ),
        lowered.overflow,
    )


def lower_comprehension[E](
    backend: Backend[E], node: Comprehension, elements: Mapping[Element, E]
) -> Lowered[E]:
    source = backend.lower(node.source, elements)

    guarded, recorded = False, False

    def parts(element: E) -> tuple[Lowered[E], Lowered[E], E | None]:
        nonlocal guarded
        scope = {**elements, node.element: element}
        condition = backend.lower(node.condition, scope)
        value = backend.lower(node.value, scope)
        fault = backend.lower(node.fault, scope)
        value_exact = backend.exact(value)
        if node.condition != TRUE and value_exact is not None:
            value_exact = backend.select(condition.expr, value_exact, backend.literal(TRUE))
        exact = backend.conjunction(
            backend.exact(condition),
            value_exact,
            backend.exact(fault),
            backend.negate(fault.expr) if node.fault != FALSE else None,
        )
        guarded = guarded or exact is not None
        return condition, value, exact

    def body(element: E) -> E:
        nonlocal recorded
        condition, value, exact = parts(element)
        if node.condition == TRUE and exact is None:
            return value.expr
        recorded = True
        fields = {"udfx value": value.expr}
        if node.condition != TRUE:
            fields["udfx condition"] = condition.expr
        if exact is not None:
            fields["udfx exact"] = exact
        return backend.record(fields)

    try:
        if isinstance(node.source, Lit) and isinstance(node.source.value, tuple | list):
            chunks: list[E] = []
            exacts: list[E | None] = []
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
        packed = backend.list_transform(source.expr, body)
        if not recorded:
            return Lowered(packed, backend.exact(source))
        filtered = (
            backend.list_filter(packed, lambda element: backend.field(element, "udfx condition"))
            if node.condition != TRUE
            else packed
        )
        value = backend.list_transform(
            filtered, lambda element: backend.field(element, "udfx value")
        )
        if not guarded:
            return Lowered(value, backend.exact(source))
        flags = backend.list_transform(packed, lambda element: backend.field(element, "udfx exact"))
        every = backend.rules.get(OpKey("all", (Kind.LIST,)))
        if every is None:
            raise Unsupported(f"{backend.name} has no list reduction")
        return Lowered(value, backend.conjunction(backend.exact(source), cast("E", every(flags))))
    except Unsupported:
        if node.requires_native:
            raise
        return Lowered(backend.literal(Lit((), node.type)), backend.literal(FALSE))
