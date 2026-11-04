from __future__ import annotations

import operator
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import reduce
from typing import Any, ClassVar, Literal, cast

from udfx.backends.faults import FallbackError, NativeFaults, decode_fallback_error
from udfx.backends.lowering.context import (
    distinct_guards,
    factored_guards,
    lower_shared,
    remember_guards,
)
from udfx.backends.lowering.nodes import lower_node
from udfx.backends.lowering.programs import build_smallest, lower_program
from udfx.backends.registration.shared import Lowered, RuleSet
from udfx.backends.rules import Scope
from udfx.errors import Unsupported
from udfx.ir import FALSE, TRUE, Col, Lit, Node
from udfx.program import Program
from udfx.types import Kind, Type

RESERVED = "udfx "
OUTPUT = f"{RESERVED}out"
RAISED = threading.local()
Row = tuple[object, ...]
Gather = Callable[[Sequence[int]], Sequence[Sequence[object]]]


def transpose_rows(rows: Sequence[Row], width: int) -> list[tuple[object, ...]]:
    return list(zip(*rows, strict=True)) if rows else [() for _ in range(width)]


def output_names(count: int) -> list[str]:
    return [f"{OUTPUT} {index}" for index in range(count)]


class Backend[E](ABC):
    name: ClassVar[str]
    rules: ClassVar[RuleSet[Any]]
    graphs: ClassVar[bool] = False
    choose_graphs: ClassVar[bool] = False
    binds: ClassVar[bool] = False
    hoisted: ClassVar[frozenset[Kind]] = frozenset({Kind.STR, Kind.INT, Kind.LIST})
    indexes: ClassVar[bool] = False
    size_limit: ClassVar[int | None] = None

    def supports_graphs(self) -> bool:
        return self.graphs

    def run_fallback[T](self, run: Callable[[], T]) -> T:
        try:
            return run()
        except Exception as error:
            RAISED.error = error
            raise FallbackError(error) from error

    def evaluate_pending_rows(
        self, program: Program, pending: Sequence[int], gather: Gather
    ) -> list[object]:
        if not pending:
            return []
        columns = gather(pending)
        rows = zip(*columns, strict=True) if columns else [()] * len(pending)
        return self.run_fallback(lambda: program.run(rows))

    def fallback_values(
        self, program: Program, length: int, pending: Sequence[int], gather: Gather
    ) -> list[object]:
        values: list[object] = [None] * length
        for index, value in zip(
            pending, self.evaluate_pending_rows(program, pending, gather), strict=True
        ):
            values[index] = value
        return values

    def execute[T](self, run: Callable[[], T]) -> T:
        RAISED.error = None
        try:
            return run()
        except Exception as error:
            raise self.unwrap(error) from None
        finally:
            RAISED.error = None

    @property
    def native_faults(self) -> NativeFaults:
        return NativeFaults()

    def unwrap(self, error: BaseException) -> BaseException:
        return (
            getattr(RAISED, "error", None)
            or self.native_faults.decode(str(error))
            or decode_fallback_error(str(error))
            or error
        )

    def native_raise(self, program: Program) -> E | None:
        return None

    @abstractmethod
    def column(self, node: Col) -> E: ...

    @abstractmethod
    def literal(self, node: Lit) -> E: ...

    @abstractmethod
    def select(self, condition: E, then: E, orelse: E) -> E: ...

    def select_many(self, cases: Sequence[tuple[E, E]], orelse: E) -> E:
        return reduce(lambda value, case: self.select(*case, value), reversed(cases), orelse)

    def select_value(self, value: E, cases: Sequence[tuple[E, E]], orelse: E) -> E | None:
        return None

    def bind(self, value: E, body: Callable[[E], E]) -> E:
        raise Unsupported(f"{self.name} cannot bind expressions")

    def bind_sequence(self, value: E, steps: Sequence[Callable[[E], E]]) -> E:
        return reduce(self.bind, steps, value)

    def record(self, values: Sequence[E]) -> E:
        raise Unsupported(f"{self.name} cannot construct expression records")

    def field(self, value: E, index: int) -> E:
        raise Unsupported(f"{self.name} cannot read expression records")

    def both(self, left: E, right: E) -> E:
        return cast("Any", left) & right

    def negate(self, value: E) -> E:
        return ~cast("Any", value)

    def list_value(self, items: Sequence[E], type: Type) -> E:
        raise Unsupported(f"{self.name} has no list constructor")

    def list_transform(self, source: E, transform: Callable[[E], E]) -> E:
        raise Unsupported(f"{self.name} has no list transform")

    def list_filter(self, source: E, predicate: Callable[[E], E]) -> E:
        raise Unsupported(f"{self.name} has no list filter")

    def list_distinct(self, source: E, transform: Callable[[E], E]) -> E | None:
        return None

    def list_indexed(
        self, source: E, value: Callable[[E, E], E], condition: Callable[[E, E], E] | None
    ) -> E:
        raise Unsupported(f"{self.name} has no indexed list transform")

    @abstractmethod
    def fallback(self, native: E, exact: E, program: Program) -> E: ...

    @abstractmethod
    def evaluate(
        self,
        exprs: Sequence[E],
        schema: Mapping[str, Type],
        rows: Sequence[Row],
        reduced: bool = False,
    ) -> list[list[object]]: ...

    def lower(self, node: Node, elements: Scope[E] | None = None) -> Lowered[E]:
        return lower_shared(self, node, {} if elements is None else elements)

    def lower_node(self, node: Node, elements: Scope[E]) -> Lowered[E]:
        return lower_node(self, node, elements)

    def fold_name(self, name: str) -> str:
        return name

    def check_names(self, names: Iterable[str]) -> None:
        seen: dict[str, str] = {}
        for name in names:
            folded = self.fold_name(name)
            if folded.startswith(RESERVED):
                raise Unsupported(f"column names starting with {RESERVED!r} are reserved")
            first = seen.setdefault(folded, name)
            if first != name:
                raise Unsupported(
                    f"{self.name} column names are case-insensitive: {first!r} and {name!r} collide"
                )

    def build(self, program: Program) -> E:
        return build_smallest(self, program)

    def build_candidate(self, program: Program) -> E:
        return self.compose(self.lower_program(program), program)

    def expression_cost(self, expr: E) -> int:
        return len(str(expr))

    def compose(self, value: Lowered[E], program: Program) -> E:
        if value.exact is False:
            return self.fallback(
                self.literal(Lit(None, program.result.or_none())), self.literal(FALSE), program
            )
        return value.consume(
            lambda part: (
                part.expr
                if part.exact is None
                else self.fallback(part.expr, self.predicate(part.exact), program)
            )
        )

    def lower_program(self, program: Program) -> Lowered[E]:
        return lower_program(self, program)

    def exact(self, lowered: Lowered[E]) -> E | Literal[False] | None:
        return self.conjunction(
            lowered.exact,
            self.negate(lowered.overflow) if lowered.overflow is not None else None,
        )

    def exact_expression(self, lowered: Lowered[E]) -> E:
        return self.predicate(self.exact(lowered))

    def predicate(self, value: E | bool | None) -> E:
        if value is None or value is True:
            return self.literal(TRUE)
        return self.literal(FALSE) if value is False else value

    def select_exact(
        self, condition: E, then: E | Literal[False] | None, orelse: E | Literal[False] | None
    ) -> E | Literal[False] | None:
        if then is orelse:
            return then
        shared, first, second = factored_guards(then, orelse)
        if not shared:
            return self.select(condition, self.predicate(then), self.predicate(orelse))
        branch = self.select_exact(condition, self.conjunction(*first), self.conjunction(*second))
        return self.conjunction(*shared, branch)

    def disjunction(self, *parts: E | None) -> E | None:
        present = [part for part in parts if part is not None]
        return reduce(operator.or_, present) if present else None

    def conjunction(self, *parts: E | bool | None) -> E | Literal[False] | None:
        if any(part is False for part in parts):
            return False
        guards = distinct_guards(parts)
        return remember_guards(self.combine_guards(guards), guards) if guards else None

    def combine_guards(self, guards: Sequence[E]) -> E:
        return reduce(self.both, guards)
