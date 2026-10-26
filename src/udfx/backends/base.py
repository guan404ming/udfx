from __future__ import annotations

import operator
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import reduce
from typing import Any, ClassVar, cast

from udfx.backends.faults import NativeFaults, decode_fallback_error, encode_fallback_error
from udfx.backends.lowering_context import lower_shared
from udfx.backends.nodes import lower_node
from udfx.backends.program_lowering import lower_program
from udfx.backends.registration import Lowered, RuleSet
from udfx.backends.rules import Scope
from udfx.errors import Unsupported
from udfx.ir import (
    FALSE,
    TRUE,
    Col,
    Lit,
    Node,
)
from udfx.program import Program
from udfx.types import Type

RESERVED = "udfx "
OUTPUT = f"{RESERVED}out"
RAISED = threading.local()
Row = tuple[object, ...]
Gather = Callable[[list[int]], Sequence[Sequence[object]]]


def transpose_rows(rows: Sequence[Row], width: int) -> list[tuple[object, ...]]:
    return list(zip(*rows, strict=True)) if rows else [() for _ in range(width)]


def output_names(count: int) -> list[str]:
    return [f"{OUTPUT} {index}" for index in range(count)]


class FallbackError(RuntimeError):
    def __init__(self, error: Exception | str) -> None:
        super().__init__(error if isinstance(error, str) else encode_fallback_error(error))


class Backend[E](ABC):
    name: ClassVar[str]
    rules: ClassVar[RuleSet[Any]]
    graphs: ClassVar[bool] = False
    binds: ClassVar[bool] = False
    indexes: ClassVar[bool] = False

    def supports_graphs(self) -> bool:
        return self.graphs

    def run_fallback_rows(self, program: Program, rows: Iterable[Sequence[object]]) -> list[object]:
        try:
            return program.run(rows)
        except Exception as error:
            RAISED.error = error
            raise FallbackError(error) from error

    def evaluate_pending_rows(
        self, program: Program, pending: list[int], gather: Gather
    ) -> list[object]:
        if not pending:
            return []
        columns = gather(pending)
        rows = zip(*columns, strict=True) if columns else [()] * len(pending)
        return self.run_fallback_rows(program, rows)

    def fallback_values(
        self, program: Program, length: int, pending: list[int], gather: Gather
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

    def list_indexed(
        self, source: E, value: Callable[[E, E], E], condition: Callable[[E, E], E] | None
    ) -> E:
        raise Unsupported(f"{self.name} has no indexed list transform")

    @abstractmethod
    def fallback(self, native: E, exact: E, program: Program) -> E: ...

    @abstractmethod
    def evaluate(
        self, exprs: Sequence[E], schema: Mapping[str, Type], rows: Sequence[Row]
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
        self.check_names(program.schema)
        value = self.lower_program(program)
        return (
            value.expr if value.exact is None else self.fallback(value.expr, value.exact, program)
        )

    def lower_program(self, program: Program) -> Lowered[E]:
        return lower_program(self, program)

    def exact(self, lowered: Lowered[E]) -> E | None:
        return self.conjunction(
            lowered.exact,
            self.negate(lowered.overflow) if lowered.overflow is not None else None,
        )

    def exact_expression(self, lowered: Lowered[E]) -> E:
        exact = self.exact(lowered)
        return exact if exact is not None else self.literal(TRUE)

    def disjunction(self, *parts: E | None) -> E | None:
        present = [part for part in parts if part is not None]
        return reduce(operator.or_, present) if present else None

    def conjunction(self, *parts: E | bool | None) -> E | None:
        if any(part is False for part in parts):
            return self.literal(FALSE)
        present = [cast("E", part) for part in parts if part is not None and part is not True]
        return reduce(self.both, present) if present else None
