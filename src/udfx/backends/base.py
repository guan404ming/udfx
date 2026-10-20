from __future__ import annotations

import builtins
import operator
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, cast

from udfx.errors import Unsupported
from udfx.ir import FALSE, TRUE, Call, Col, If, Lit, Node
from udfx.ops.registry import OpKey
from udfx.program import Program
from udfx.semantics.literals import (
    TextPrimitives,
    float_parses,
    int_parses,
    parse_float,
    parse_int,
)
from udfx.semantics.numbers import (
    Binary,
    OrderPrimitives,
    extremum,
    float_compare,
    float_extremum,
    widened_exact,
)
from udfx.types import Kind, Type, fits

RESERVED = "udfx "
OUT = f"{RESERVED}out"
Row = tuple[object, ...]
Gather = Callable[[list[int]], Sequence[Sequence[object]]]


@dataclass(frozen=True)
class Lowered[E]:
    expr: E
    exact: E | None = None


type RuleResult[E] = E | Lowered[E]
type Rule[E] = Callable[..., RuleResult[E]]
type BinaryRule[E] = Callable[[E, E], RuleResult[E]]


BOOLEAN_RULES: dict[OpKey, Rule[Any]] = {
    OpKey("not", (Kind.BOOL,)): operator.invert,
    OpKey("and", (Kind.BOOL, Kind.BOOL)): operator.and_,
    OpKey("or", (Kind.BOOL, Kind.BOOL)): operator.or_,
}


def transpose_rows(rows: Sequence[Row], width: int) -> list[tuple[object, ...]]:
    return list(zip(*rows, strict=True)) if rows else [() for _ in range(width)]


def output_names(count: int) -> list[str]:
    return [f"{OUT} {index}" for index in range(count)]


class RuleSet[E]:
    def __init__(self) -> None:
        self._rules: dict[OpKey, Rule[E]] = dict(BOOLEAN_RULES)

    def __call__[**P](
        self, name: str, *kinds: Kind
    ) -> Callable[[Callable[P, RuleResult[E]]], Callable[P, RuleResult[E]]]:
        key = OpKey(name, kinds)

        def decorate(rule: Callable[P, RuleResult[E]]) -> Callable[P, RuleResult[E]]:
            if key in self._rules:
                raise ValueError(f"duplicate rule {key}")
            self._rules[key] = rule
            return rule

        return decorate

    def get(self, key: OpKey) -> Rule[E] | None:
        return self._rules.get(key)

    def keys(self) -> frozenset[OpKey]:
        return frozenset(self._rules)


def register_float_variants[E](
    rules: RuleSet[E], name: str, rule: BinaryRule[E], to_float: Callable[[E], E]
) -> None:
    rules(name, Kind.FLOAT, Kind.FLOAT)(rule)
    rules(name, Kind.INT, Kind.FLOAT)(lambda left, right: rule(to_float(left), right))
    rules(name, Kind.FLOAT, Kind.INT)(lambda left, right: rule(left, to_float(right)))


def register_widened_variants[E](
    rules: RuleSet[E], name: str, rule: Binary, to_float: Callable[[E], E], order: OrderPrimitives
) -> None:
    rules(name, Kind.INT, Kind.FLOAT)(
        lambda left, right: Lowered(rule(to_float(left), right), widened_exact(order, left))
    )
    rules(name, Kind.FLOAT, Kind.INT)(
        lambda left, right: Lowered(rule(left, to_float(right)), widened_exact(order, right))
    )


def register_comparison[E](
    rules: RuleSet[E],
    name: str,
    compare: Binary,
    to_float: Callable[[E], E],
    order: OrderPrimitives,
) -> None:
    def ordered(left: E, right: E) -> E:
        return float_compare(order, compare, left, right)

    rules(name, Kind.INT, Kind.INT)(compare)
    rules(name, Kind.FLOAT, Kind.FLOAT)(ordered)
    register_widened_variants(rules, name, ordered, to_float, order)


def register_equality[E](
    rules: RuleSet[E], equal: Binary, to_float: Callable[[E], E], order: OrderPrimitives
) -> None:
    def unequal(left: Any, right: Any) -> Any:
        return ~equal(left, right)

    def float_unequal(left: Any, right: Any) -> Any:
        return ~float_compare(order, equal, left, right)

    register_comparison(rules, "__eq__", equal, to_float, order)
    rules("__ne__", Kind.INT, Kind.INT)(unequal)
    rules("__ne__", Kind.FLOAT, Kind.FLOAT)(float_unequal)
    register_widened_variants(rules, "__ne__", float_unequal, to_float, order)


def register_extremum[E](
    rules: RuleSet[E], name: str, compare: Binary, order: OrderPrimitives
) -> None:
    rules(name, Kind.FLOAT, Kind.FLOAT)(
        lambda left, right: float_extremum(order, compare, left, right)
    )
    for kind in (Kind.INT, Kind.STR):
        rules(name, kind, kind)(lambda left, right: extremum(order, compare, left, right))


def register_text_conversions[E](rules: RuleSet[E], text: TextPrimitives) -> None:
    def register(name: str, parse: Callable[[TextPrimitives, Any], tuple[Any, Any]]) -> None:
        rules(name, Kind.STR)(lambda value: Lowered(*parse(text, value)))

    register("int_parses", int_parses)
    register("int", parse_int)
    register("float_parses", float_parses)
    register("float", parse_float)


RAISED = threading.local()
MARKER = "udfx raised "


class FallbackError(RuntimeError):
    def __init__(self, error: Exception) -> None:
        super().__init__(f"{MARKER}{type(error).__name__}: {error}")
        self.error = error


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
        rows = [tuple(column[position] for column in columns) for position in range(len(pending))]
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

    @abstractmethod
    def fallback(self, native: E, exact: E, program: Program) -> E: ...

    @abstractmethod
    def evaluate(
        self, exprs: Sequence[E], schema: Mapping[str, Type], rows: Sequence[Row]
    ) -> list[list[object]]: ...

    def lower(self, node: Node) -> Lowered[E]:
        match node:
            case Col():
                return Lowered(self.column(node))
            case Lit() if node.type.kind is Kind.INT and not fits(node.value, node.type):
                return Lowered(self.literal(Lit(0, node.type)), self.literal(FALSE))
            case Lit():
                return Lowered(self.literal(node))
            case If():
                return self._lower_if(node)
            case Call():
                return self._lower_call(node)

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

    def _lower_if(self, node: If) -> Lowered[E]:
        condition = self.lower(node.condition)
        then, orelse = self.lower(node.then), self.lower(node.orelse)
        expr = self.select(condition.expr, then.expr, orelse.expr)
        if then.exact is None and orelse.exact is None:
            return Lowered(expr, condition.exact)
        branch = self.select(condition.expr, self._exact(then), self._exact(orelse))
        return Lowered(expr, self._all(condition.exact, branch))

    def _lower_call(self, node: Call) -> Lowered[E]:
        rule = self.rules.get(node.op.key)
        if rule is None:
            raise Unsupported(f"{self.name} has no rule for {node.op.key}")
        arguments = [self.lower(argument) for argument in node.arguments]
        result: RuleResult[E] = rule(*(argument.expr for argument in arguments))
        lowered = cast("Lowered[E]", result) if isinstance(result, Lowered) else Lowered(result)
        return Lowered(
            lowered.expr, self._all(*(argument.exact for argument in arguments), lowered.exact)
        )

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
