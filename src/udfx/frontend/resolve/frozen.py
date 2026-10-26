from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from types import BuiltinFunctionType, CellType, FunctionType, ModuleType

from udfx.errors import Unsupported
from udfx.frontend.core.values import Item
from udfx.frontend.resolve.callables import known_op
from udfx.frontend.resolve.constants import constant
from udfx.frontend.resolve.imports import (
    STANDARD_IMPORT,
    FrozenModule,
    ImportTable,
    capture_imports,
)
from udfx.semantics.functions import code_names


class Snapshot:
    def __init__(self) -> None:
        self.values: dict[int, object] = {}
        self.modules: dict[ModuleType, dict[str, object]] = {}
        self.imports: dict[tuple[str, tuple[str, ...]], ModuleType] = {}
        self.importer = ImportTable(self.imports)

    def imported(
        self, name: str, level: object, names: object, scope: Mapping[str, object]
    ) -> object:
        if level != 0 or scope.get("__import__") is not self.importer:
            raise Unsupported("imports require the standard absolute import mechanism")
        if names is not None and not (
            isinstance(names, tuple) and all(isinstance(item, str) for item in names)
        ):
            raise Unsupported("import names require a constant tuple")
        members = tuple(str(item) for item in names or ())
        target = name if members else name.split(".", maxsplit=1)[0]
        module = sys.modules.get(target)
        if not isinstance(sys.modules.get(name), ModuleType) or not isinstance(module, ModuleType):
            raise Unsupported(f"import requires an already loaded module: {name}")
        frozen = self.value(module)
        if not isinstance(frozen, ModuleType):
            raise Unsupported(f"import requires a module: {name}")
        self.imports[name, members] = frozen
        return frozen

    def members(self, module: ModuleType) -> dict[str, object]:
        return self.modules.get(module, vars(module))

    def callable(self, function: FunctionType) -> Callable[..., object]:
        if not self.imports:
            return function
        functions = list(
            {value for value in self.values.values() if isinstance(value, FunctionType)}
        )
        return capture_imports(function, functions, self.imports, list(self.modules))

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
            frozen = FrozenModule(value.__name__)
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
        namespace = dict(function.__builtins__)
        if (
            isinstance(STANDARD_IMPORT, BuiltinFunctionType)
            and namespace.get("__import__") is STANDARD_IMPORT
        ):
            namespace["__import__"] = self.importer
        frozen = FunctionType(
            function.__code__,
            {"__builtins__": namespace},
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
