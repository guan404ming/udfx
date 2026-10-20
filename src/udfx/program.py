from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from udfx.ir import FALSE, Node
from udfx.types import Type


@dataclass(frozen=True)
class Program:
    function: Callable[..., object]
    schema: Mapping[str, Type]
    value: Node
    fault: Node = FALSE

    @property
    def result(self) -> Type:
        return self.value.type
