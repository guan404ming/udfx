from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, ClassVar, cast

from udfx.backends.faults import FALLBACK_MARKER, NativeFaults, decode_fallback_error
from udfx.backends.lowering import lower_call, lower_comprehension, lower_if, lower_program
from udfx.backends.registration import Lowered, RuleSet
from udfx.backends.rules import fold_predicate
from udfx.errors import Unsupported
from udfx.ir import (
    FALSE,
    TRUE,
    Call,
    Col,
    Comprehension,
    Element,
    Guarded,
    If,
    ListValue,
    Lit,
    Node,
)
from udfx.program import Program
from udfx.types import Kind, Type, fits

RESERVED = "udfx "
OUTPUT = f"{RESERVED}out"
Row = tuple[object, ...]
Gather = Callable[[list[int]], Sequence[Sequence[object]]]


def transpose_rows(rows: Sequence[Row], width: int) -> list[tuple[object, ...]]:
    return list(zip(*rows, strict=True)) if rows else [() for _ in range(width)]


def output_names(count: int) -> list[str]:
    return [f"{OUTPUT} {index}" for index in range(count)]


RAISED = threading.local()


class FallbackError(RuntimeError):
    def __init__(self, error: Exception | str) -> None:
        super().__init__(
            error if isinstance(error, str) else f"{FALLBACK_MARKER}{type(error).__name__}: {error}"
        )


class Backend[E](ABC):
    name: ClassVar[str]
    rules: ClassVar[RuleSet[Any]]

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

    def record(self, fields: Mapping[str, E]) -> E:
        raise Unsupported(f"{self.name} has no record constructor")

    def field(self, record: E, name: str) -> E:
        raise Unsupported(f"{self.name} has no record fields")

    @abstractmethod
    def fallback(self, native: E, exact: E, program: Program) -> E: ...

    @abstractmethod
    def evaluate(
        self, exprs: Sequence[E], schema: Mapping[str, Type], rows: Sequence[Row]
    ) -> list[list[object]]: ...

    def lower(self, node: Node, elements: Mapping[Element, E] | None = None) -> Lowered[E]:
        elements = elements or {}
        match node:
            case Guarded():
                value = self.lower(node.value, elements)
                predicate = fold_predicate(node.exact)
                if predicate == TRUE:
                    return value
                exact = self.lower(predicate, elements)
                return Lowered(
                    value.expr,
                    self.conjunction(value.exact, self.exact(exact), exact.expr),
                    value.overflow,
                )
            case Element():
                return Lowered(elements[node])
            case ListValue():
                items = [self.lower(item, elements) for item in node.items]
                return Lowered(
                    self.list_value([item.expr for item in items], node.type),
                    self.conjunction(*(self.exact(item) for item in items)),
                )
            case Comprehension():
                return lower_comprehension(self, node, elements)
            case Col():
                return Lowered(self.column(node))
            case Lit() if (
                node.type.kind is Kind.LIST
                and node.type.item is not None
                and isinstance(node.value, tuple | list)
            ):
                return self.lower(
                    ListValue(tuple(Lit(item, node.type.item) for item in node.value), node.type),
                    elements,
                )
            case Lit() if node.type.kind is Kind.INT and not fits(node.value, node.type):
                return Lowered(self.literal(Lit(0, node.type)), overflow=self.literal(TRUE))
            case Lit():
                return Lowered(self.literal(node))
            case If():
                return lower_if(self, node, elements)
            case Call():
                return lower_call(self, node, elements)

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
        value = lower_program(self, program)
        return (
            value.expr if value.exact is None else self.fallback(value.expr, value.exact, program)
        )

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
        if not present:
            return None
        combined = present[0]
        for part in present[1:]:
            combined = cast("Any", combined) | part
        return combined

    def conjunction(self, *parts: E | bool | None) -> E | None:
        if any(part is False for part in parts):
            return self.literal(FALSE)
        present = [cast("E", part) for part in parts if part is not None and part is not True]
        if not present:
            return None
        combined = present[0]
        for part in present[1:]:
            combined = self.both(combined, part)
        return combined
