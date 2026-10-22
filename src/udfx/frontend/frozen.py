from __future__ import annotations

from types import CellType, FunctionType, ModuleType

from udfx.frontend.callables import known_op
from udfx.frontend.constants import constant
from udfx.frontend.values import Item
from udfx.semantics.functions import code_names


class Snapshot:
    def __init__(self) -> None:
        self.values: dict[int, object] = {}
        self.modules: dict[ModuleType, dict[str, object]] = {}

    def members(self, module: ModuleType) -> dict[str, object]:
        return self.modules.get(module, vars(module))

    def attribute(self, module: ModuleType, name: str) -> object:
        value = self.value(self.members(module)[name])
        if module in self.modules:
            vars(module)[name] = value
        return value

    def value(self, value: object) -> object:
        if id(value) in self.values:
            return self.values[id(value)]
        if isinstance(value, FunctionType) and known_op(value) is None:
            return self.function(value)
        if isinstance(value, ModuleType):
            frozen = ModuleType(value.__name__)
            self.modules[frozen] = vars(value).copy()
            self.values[id(value)] = frozen
            self.values[id(frozen)] = frozen
            return frozen
        if isinstance(value, dict | list | set):
            copied = value.copy()
            self.values[id(value)] = copied
            self.values[id(copied)] = copied
            return copied
        return value

    def function(self, function: FunctionType) -> FunctionType:
        existing = self.values.get(id(function))
        if isinstance(existing, FunctionType):
            return existing
        frozen = FunctionType(
            function.__code__,
            {"__builtins__": function.__builtins__},
            function.__name__,
            function.__defaults__,
            tuple(_freeze_cell(cell) for cell in function.__closure__)
            if function.__closure__
            else None,
        )
        self.values[id(function)] = frozen
        self.values[id(frozen)] = frozen
        frozen.__globals__.update(
            (name, self.value(function.__globals__[name]))
            for name in code_names(function.__code__)
            if name in function.__globals__
        )
        frozen.__defaults__ = tuple(self.value(value) for value in function.__defaults__ or ())
        frozen.__kwdefaults__ = {
            name: self.value(value) for name, value in (function.__kwdefaults__ or {}).items()
        }
        for cell in frozen.__closure__ or ():
            try:
                cell.cell_contents = self.value(cell.cell_contents)
            except ValueError:
                continue
        return frozen


def _freeze_cell(cell: CellType) -> CellType:
    try:
        return CellType(cell.cell_contents)
    except ValueError:
        return CellType()


def closure(function: FunctionType) -> dict[str, Item]:
    items: dict[str, Item] = {}
    for name, cell in zip(function.__code__.co_freevars, function.__closure__ or (), strict=True):
        try:
            items[name] = constant(cell.cell_contents)
        except ValueError:
            continue
    return items
