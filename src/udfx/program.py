from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from udfx.ir import FALSE, Node
from udfx.semantics.scalars import python_scalar
from udfx.types import Type, fits


class Row:
    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, object]) -> None:
        self._values = values

    def __getitem__(self, name: str) -> object:
        return object.__getattribute__(self, "_values")[name]

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
        for row in rows:
            value = python_scalar(self.function(*self.arguments(row)))
            if not fits(value, self.result):
                raise OverflowError(f"udfx result {value!r} does not fit {self.result}")
            values.append(value)
        return values
