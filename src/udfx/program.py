from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from udfx.ir import FALSE, Node
from udfx.semantics.scalars import python_scalar
from udfx.types import INT64_MAX, INT64_MIN, Kind, Type, fits

PLAIN = {Kind.STR: str, Kind.FLOAT: float, Kind.BOOL: bool, Kind.INT: int}


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
class Program:
    function: Callable[..., object]
    schema: Mapping[str, Type]
    value: Node
    fault: Node = FALSE
    row_mode: bool = False

    @property
    def result(self) -> Type:
        return self.value.type

    def arguments(self, row: Sequence[object]) -> Sequence[object]:
        return (Row(dict(zip(self.schema, row, strict=True))),) if self.row_mode else row

    def run(self, rows: Iterable[Sequence[object]]) -> list[object]:
        values: list[object] = []
        result = self.result
        function = self.function
        if self.row_mode:
            rows = (self.arguments(row) for row in rows)
        plain = PLAIN.get(result.kind)
        for row in rows:
            value = function(*row)
            if type(value) is not plain or (
                isinstance(value, int) and not INT64_MIN <= value <= INT64_MAX
            ):
                value = python_scalar(value)
                if not fits(value, result):
                    raise OverflowError(f"udfx result {value!r} does not fit {result}")
            values.append(value)
        return values
