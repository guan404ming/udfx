from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from udfx.types import Type


@dataclass(frozen=True)
class Join:
    table: Path
    left: str
    right: str
    category: str


@dataclass(frozen=True)
class Workload:
    suite: str
    name: str
    function: Callable[..., object] | None
    schema: Mapping[str, Type]
    result: Type
    table: Path
    columns: Mapping[str, str]
    category: str | None = None
    constants: Mapping[str, object] = field(default_factory=dict)
    join: Join | None = None
    native: Mapping[str, Callable[..., Any]] = field(default_factory=dict)
    origin: str = "custom"
    exclusion: str | None = None
    error: str | None = None
    deterministic: bool = True

    @property
    def identifier(self) -> str:
        return f"{self.suite}/{self.name}"
