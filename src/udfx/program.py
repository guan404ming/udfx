from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence, Sized
from dataclasses import dataclass
from typing import Any

from udfx.ir import FALSE, Node
from udfx.semantics.scalars import python_scalar
from udfx.types import INT64_MAX, INT64_MIN, Kind, Type, fits

PLAIN = {Kind.STR: str, Kind.FLOAT: float, Kind.BOOL: bool, Kind.INT: int}
NUMPY_KINDS = {Kind.FLOAT: "f", Kind.INT: "i", Kind.BOOL: "b"}


class Row:
    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, object]) -> None:
        self._values = values

    def __getitem__(self, name: str | int) -> object:
        values = object.__getattribute__(self, "_values")
        return tuple(values.values())[name] if isinstance(name, int) else values[name]

    def __getattribute__(self, name: str) -> object:
        try:
            return object.__getattribute__(self, "_values")[name]
        except KeyError:
            raise AttributeError(name) from None


@dataclass(frozen=True)
class Reduction:
    method: str
    values: tuple[Node, ...]
    present: Node
    type: Type


@dataclass(frozen=True)
class Program:
    function: Callable[..., object]
    schema: Mapping[str, Type]
    value: Node
    fault: Node = FALSE
    row_mode: bool = False
    deterministic: bool = True
    series: Callable[..., Any] | None = None
    reductions: tuple[Reduction, ...] | None = None

    @property
    def result(self) -> Type:
        return self.value.type

    def arguments(self, row: Sequence[object]) -> Sequence[object]:
        return (Row(dict(zip(self.schema, row, strict=True))),) if self.row_mode else row

    def run(self, rows: Iterable[Sequence[object]]) -> list[object]:
        if self.series is not None:
            columns = [list(column) for column in zip(*rows, strict=True)]
            return self.checked(self.series_result(columns).tolist()) if columns else []
        function = self.function
        if self.row_mode:
            rows = (self.arguments(row) for row in rows)
        return self.checked(function(*row) for row in rows)

    def batch(self, columns: Sequence[Sized]) -> Any:
        found = self.series_result(columns)
        if found.dtype.kind == NUMPY_KINDS.get(self.result.kind):
            return found.to_numpy()
        return self.checked(found.tolist())

    def series_result(self, columns: Sequence[Sized]) -> Any:
        series = self.series
        assert series is not None
        found = series(self.function(*(series(column) for column in columns)))
        expected = len(columns[0])
        if len(found) != expected:
            raise ValueError(f"series UDF returned {len(found)} rows for {expected} inputs")
        return found

    def checked(self, values: Iterable[object]) -> list[object]:
        checked: list[object] = []
        result = self.result
        plain = PLAIN.get(result.kind)
        for found in values:
            value = found
            if type(value) is not plain or (plain is int and not INT64_MIN <= value <= INT64_MAX):  # pyright: ignore[reportOperatorIssue]
                value = python_scalar(value)
                if not fits(value, result):
                    raise OverflowError(f"udfx result {value!r} does not fit {result}")
            checked.append(value)
        return checked
