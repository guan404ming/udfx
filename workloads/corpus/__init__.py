from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

from udfx.types import Type


@dataclass(frozen=True)
class Case:
    name: str
    function: Callable[..., object]
    schema: Mapping[str, Type]
    expect: Literal["supported", "optional"]
