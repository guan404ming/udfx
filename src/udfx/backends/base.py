from __future__ import annotations

import builtins
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, ClassVar, cast

from udfx.backends.registration import Lowered, RuleResult, RuleSet
from udfx.errors import Unsupported
from udfx.ir import FALSE, TRUE, Call, Col, Comprehension, Element, If, ListValue, Lit, Node
from udfx.ops import OpKey
from udfx.program import Program
from udfx.semantics.numbers import bounded_integer_division
from udfx.semantics.text import maximum_length
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
MARKER = "udfx raised "


class FallbackError(RuntimeError):
    def __init__(self, error: Exception | str) -> None:
        super().__init__(
            error if isinstance(error, str) else f"{MARKER}{type(error).__name__}: {error}"
        )


def decode_fallback_error(text: str) -> Exception | None:
    _, found, detail = text.partition(MARKER)
    name, _, message = detail.partition(": ")
    kind = getattr(builtins, name.strip(), None)
    if not found or not isinstance(kind, type) or not issubclass(kind, Exception):
        return None
    try:
        return kind(message.partition("\n")[0])
    except TypeError:
        return None


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

    def unwrap(self, error: BaseException) -> BaseException:
        return getattr(RAISED, "error", None) or decode_fallback_error(str(error)) or error

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

    @abstractmethod
    def fallback(self, native: E, exact: E, program: Program) -> E: ...

    @abstractmethod
    def evaluate(
        self, exprs: Sequence[E], schema: Mapping[str, Type], rows: Sequence[Row]
    ) -> list[list[object]]: ...

    def lower(self, node: Node, elements: Mapping[Element, E] | None = None) -> Lowered[E]:
        elements = elements or {}
        match node:
            case Element():
                return Lowered(elements[node])
            case ListValue():
                items = [self.lower(item, elements) for item in node.items]
                return Lowered(
                    self.list_value([item.expr for item in items], node.type),
                    self._all(*(item.exact for item in items)),
                )
            case Comprehension():
                return self._lower_comprehension(node, elements)
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
                return Lowered(self.literal(Lit(0, node.type)), self.literal(FALSE))
            case Lit():
                return Lowered(self.literal(node))
            case If():
                return self._lower_if(node, elements)
            case Call():
                return self._lower_call(node, elements)

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
        value = self.lower(program.value)
        exact = value.exact
        if program.fault != FALSE:
            fault = self.lower(program.fault)
            exact = self._all(exact, fault.exact, self.negate(fault.expr))
        if exact is None:
            return value.expr
        return self.fallback(value.expr, exact, program)

    def _lower_if(self, node: If, elements: Mapping[Element, E]) -> Lowered[E]:
        condition = self.lower(node.condition, elements)
        then, orelse = self.lower(node.then, elements), self.lower(node.orelse, elements)
        expr = self.select(condition.expr, then.expr, orelse.expr)
        if then.exact is None and orelse.exact is None:
            return Lowered(expr, condition.exact)
        branch = self.select(condition.expr, self._exact(then), self._exact(orelse))
        return Lowered(expr, self._all(condition.exact, branch))

    def _lower_call(self, node: Call, elements: Mapping[Element, E]) -> Lowered[E]:
        rule = self.rules.get(node.op.key)
        if rule is None:
            raise Unsupported(f"{self.name} has no rule for {node.op.key}")
        arguments = [
            Lowered(cast("Any", argument.value))
            if (
                node.op.name.startswith("re.")
                or node.op.name
                in {".strftime", ".translate", "datetime.strptime", "datetime_parses"}
            )
            and index > 0
            and isinstance(argument, Lit)
            else self.lower(argument, elements)
            for index, argument in enumerate(node.arguments)
        ]
        operands = [argument.expr for argument in arguments]
        result: RuleResult[E]
        if node.op.name == "slice":
            for index in range(1, len(node.arguments)):
                bound = node.arguments[index]
                if isinstance(bound, Lit):
                    operands[index] = cast("Any", bound.value)
            result = rule(*operands, result=node.type)
        elif node.op.name in {"int", "int_parses"} and node.arguments[0].type.kind is Kind.STR:
            result = rule(*operands, maximum_length=maximum_length(node.arguments[0]))
        else:
            result = rule(*operands)
        lowered = cast("Lowered[E]", result) if isinstance(result, Lowered) else Lowered(result)
        if bounded_integer_division(node):
            lowered = Lowered(lowered.expr)
        return Lowered(
            lowered.expr, self._all(*(argument.exact for argument in arguments), lowered.exact)
        )

    def _lower_comprehension(
        self, node: Comprehension, elements: Mapping[Element, E]
    ) -> Lowered[E]:
        source = self.lower(node.source, elements)

        def parts(element: E) -> tuple[Lowered[E], Lowered[E], E]:
            scope = {**elements, node.element: element}
            condition = self.lower(node.condition, scope)
            value = self.lower(node.value, scope)
            fault = self.lower(node.fault, scope)
            exact = self._all(
                condition.exact,
                self.select(condition.expr, self._exact(value), self.literal(TRUE)),
                fault.exact,
                self.negate(fault.expr),
            )
            return condition, value, cast("E", exact)

        try:
            if isinstance(node.source, Lit) and isinstance(node.source.value, tuple | list):
                chunks: list[E] = []
                exacts: list[E] = []
                for item in node.source.value:
                    element = self.lower(Lit(item, node.element.type), elements)
                    condition, value, exact = parts(element.expr)
                    chunks.append(
                        self.select(
                            condition.expr,
                            self.list_value([value.expr], node.type),
                            self.literal(Lit((), node.type)),
                        )
                    )
                    exacts.append(exact)
                result = self.literal(Lit((), node.type))
                concatenate = self.rules.get(OpKey("__add__", (Kind.LIST, Kind.LIST)))
                if concatenate is None:
                    raise Unsupported(f"{self.name} has no list concatenation")
                for chunk in chunks:
                    result = cast("E", concatenate(result, chunk))
                return Lowered(result, self._all(source.exact, *exacts))
            filtered = self.list_filter(source.expr, lambda element: parts(element)[0].expr)
            value = self.list_transform(filtered, lambda element: parts(element)[1].expr)
            flags = self.list_transform(source.expr, lambda element: parts(element)[2])
            every = self.rules.get(OpKey("all", (Kind.LIST,)))
            if every is None:
                raise Unsupported(f"{self.name} has no list reduction")
            return Lowered(value, self._all(source.exact, cast("E", every(flags))))
        except Unsupported:
            if node.requires_native:
                raise
            return Lowered(self.literal(Lit((), node.type)), self.literal(FALSE))

    def _exact(self, lowered: Lowered[E]) -> E:
        return lowered.exact if lowered.exact is not None else self.literal(TRUE)

    def _all(self, *parts: E | None) -> E | None:
        present = [part for part in parts if part is not None]
        if not present:
            return None
        combined = present[0]
        for part in present[1:]:
            combined = self.both(combined, part)
        return combined
