from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import CodeType

from udfx.errors import Unsupported
from udfx.frontend.core.bindings import Bindings
from udfx.frontend.core.evaluation import Evaluation, Raise, guard
from udfx.frontend.core.values import Item
from udfx.frontend.decode import decode
from udfx.frontend.decode.graph import Decoded
from udfx.frontend.resolve.frozen import Snapshot


@dataclass(frozen=True)
class Produced:
    bindings: Bindings
    items: tuple[Item, ...]


Result = Produced | Raise


@dataclass(frozen=True)
class Flow:
    block: int
    position: int
    bindings: Bindings
    values: Mapping[int, Item]


@dataclass(frozen=True)
class Frame:
    decoded: Decoded
    scope: Mapping[str, object]
    builtins: Mapping[str, object]
    caller: tuple[Frame, Bindings] | None
    snapshot: Snapshot
    depth: int
    catching: bool = True

    @classmethod
    def from_code(
        cls,
        code: CodeType,
        scope: Mapping[str, object],
        builtins: Mapping[str, object],
        caller: tuple[Frame, Bindings] | None,
        snapshot: Snapshot,
    ) -> Frame:
        depth = caller[0].depth + 1 if caller else 0
        return cls(decode(code), scope, builtins, caller, snapshot, depth)

    @property
    def code(self) -> CodeType:
        return self.decoded.code

    def where(self, offset: int) -> str:
        return f"{self.code.co_qualname}+{offset}"

    def locate(self, bindings: Bindings, home: Frame) -> Bindings:
        if self is home:
            return bindings
        if self.caller is None:
            raise Unsupported("helper escapes its defining function")
        frame, outer = self.caller
        return frame.locate(outer, home)

    def settle(self, bindings: Bindings, evaluation: Evaluation) -> list[Result]:
        path, results = guard(bindings.path, evaluation.faults)
        outcomes: list[Result] = list(results)
        if path is not None:
            outcomes.append(Produced(bindings.under(path), evaluation.items))
        return outcomes
