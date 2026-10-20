from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, cast

from udfx.errors import Unsupported
from udfx.ir import FALSE, TRUE, Call, Col, If, Lit, Node
from udfx.ops.registry import OpKey
from udfx.program import Program
from udfx.types import Kind, Type

Row = tuple[object, ...]


@dataclass(frozen=True)
class Lowered[E]:
    expr: E
    exact: E | None = None


class RuleSet[E]:
    def __init__(self) -> None:
        self._rules: dict[OpKey, Callable[..., E | Lowered[E]]] = {}

    def __call__(
        self, name: str, *kinds: Kind
    ) -> Callable[[Callable[..., E | Lowered[E]]], Callable[..., E | Lowered[E]]]:
        key = OpKey(name, kinds)

        def decorate(rule: Callable[..., E | Lowered[E]]) -> Callable[..., E | Lowered[E]]:
            if key in self._rules:
                raise ValueError(f"duplicate rule {key}")
            self._rules[key] = rule
            return rule

        return decorate

    def get(self, key: OpKey) -> Callable[..., E | Lowered[E]] | None:
        return self._rules.get(key)

    def keys(self) -> frozenset[OpKey]:
        return frozenset(self._rules)


class Backend[E](ABC):
    name: ClassVar[str]
    rules: ClassVar[RuleSet[Any]]
    raised: BaseException | None = None

    def patch(self, program: Program, rows: Iterable[Sequence[object]]) -> list[object]:
        try:
            return program.run(rows)
        except Exception as error:
            self.raised = error
            raise

    def fallback_values(
        self, program: Program, flags: Sequence[object], columns: Sequence[Sequence[object]]
    ) -> list[object]:
        pending = [index for index, flag in enumerate(flags) if not flag]
        rows = [tuple(column[index] for column in columns) for index in pending]
        values: list[object] = [None] * len(flags)
        for index, value in zip(pending, self.patch(program, rows), strict=True):
            values[index] = value
        return values

    def execute[T](self, run: Callable[[], T]) -> T:
        self.raised = None
        try:
            return run()
        except Exception as error:
            raise self.unwrap(error) from None
        finally:
            self.raised = None

    def unwrap(self, error: BaseException) -> BaseException:
        return self.raised or error

    @abstractmethod
    def column(self, node: Col) -> E: ...

    @abstractmethod
    def literal(self, node: Lit) -> E: ...

    @abstractmethod
    def select(self, condition: E, then: E, orelse: E) -> E: ...

    @abstractmethod
    def both(self, left: E, right: E) -> E: ...

    @abstractmethod
    def negate(self, value: E) -> E: ...

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
            case Lit():
                return Lowered(self.literal(node))
            case If():
                return self._lower_if(node)
            case Call():
                return self._lower_call(node)

    def build(self, program: Program) -> E:
        value = self.lower(program.value)
        exact = value.exact
        if program.fault != FALSE:
            fault = self.lower(program.fault)
            exact = self._all(exact, fault.exact, self.negate(fault.expr))
        if exact is None:
            return value.expr
        return self.fallback(value.expr, exact, program)

    def _lower_if(self, node: If) -> Lowered[E]:
        condition = self.lower(node.cond)
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
        args = [self.lower(arg) for arg in node.args]
        result: E | Lowered[E] = rule(*(arg.expr for arg in args))
        lowered = cast("Lowered[E]", result) if isinstance(result, Lowered) else Lowered(result)
        return Lowered(lowered.expr, self._all(*(arg.exact for arg in args), lowered.exact))

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
