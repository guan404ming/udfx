from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from udfx.ir import FALSE, Node
from udfx.types import Type, fits


@dataclass(frozen=True)
class Program:
    function: Callable[..., object]
    schema: Mapping[str, Type]
    value: Node
    fault: Node = FALSE

    @property
    def result(self) -> Type:
        return self.value.type

    def run(self, rows: Iterable[Sequence[object]]) -> list[object]:
        values: list[object] = []
        for row in rows:
            value = self.function(*row)
            if not fits(value, self.result):
                raise OverflowError(f"udfx result {value!r} does not fit {self.result}")
            values.append(value)
        return values
