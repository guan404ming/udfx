from collections.abc import Callable
from typing import Any

import pyarrow

from bench.data import Input
from udfx.backends.base import Backend
from workloads.models import Workload


class Adapter:
    backend: Backend[Any]
    primary: str

    def __init__(self, workload: Workload, data: Input, threads: int) -> None:
        self.workload = workload
        self.data = data
        self.threads = threads

    def python(self) -> dict[str, Any]:
        raise NotImplementedError

    def execute(self, expr: Any, mode: str) -> pyarrow.Table:
        raise NotImplementedError

    def native(self) -> Any | None:
        factory = self.workload.native.get(self.backend.name)
        return None if factory is None else factory()

    def actions(
        self, expressions: dict[str, Any], mode: str
    ) -> dict[str, Callable[[], pyarrow.Table]]:
        return {
            name: lambda expr=expr: self.execute(expr, mode) for name, expr in expressions.items()
        }

    def settings(self) -> dict:
        return {"threads": self.threads, "memory_input": "Arrow", "query_input": "Parquet"}

    def close(self) -> None:
        pass
