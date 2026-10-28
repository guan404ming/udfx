from __future__ import annotations

import builtins
from collections.abc import Mapping, Sequence
from functools import partial
from types import CellType, CodeType, FunctionType, ModuleType
from typing import Any

STANDARD_IMPORT = builtins.__import__


class FrozenModule(ModuleType):
    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return restore_module, (self.__name__, vars(self))


def restore_module(name: str, members: dict[str, object]) -> FrozenModule:
    module = FrozenModule(name)
    vars(module).update(members)
    return module


class ImportTable:
    def __init__(self, modules: Mapping[tuple[str, tuple[str, ...]], ModuleType]) -> None:
        self.modules = modules

    def __call__(
        self,
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> ModuleType:
        frozen = self.modules.get((name, tuple(fromlist or ()))) if level == 0 else None
        return (
            frozen
            if frozen is not None
            else STANDARD_IMPORT(name, globals, locals, fromlist, level)
        )


class ImportedFunction(partial):
    def __new__(cls, function: FunctionType, importer: ImportTable) -> ImportedFunction:
        return super().__new__(cls, function)

    def __init__(self, function: FunctionType, importer: ImportTable) -> None:
        self.function = function
        self.importer = importer

    @property
    def __code__(self) -> CodeType:
        return self.function.__code__

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        function = self.function
        namespace = {**function.__builtins__, "__import__": self.importer}
        return restore_function, (
            function.__code__,
            function.__globals__,
            namespace,
            (
                function.__name__,
                function.__defaults__,
                function.__closure__,
                function.__kwdefaults__,
            ),
        )


def restore_function(
    code: CodeType,
    scope: dict[str, Any],
    namespace: dict[str, Any],
    attributes: tuple[
        str, tuple[object, ...] | None, tuple[CellType, ...] | None, dict[str, object] | None
    ],
) -> FunctionType:
    name, defaults, closure, keywords = attributes
    function = FunctionType(code, {**scope, "__builtins__": namespace}, name, defaults, closure)
    function.__kwdefaults__ = keywords
    return function


def capture_imports(
    function: FunctionType,
    functions: Sequence[FunctionType],
    modules: Mapping[tuple[str, tuple[str, ...]], ModuleType],
    frozen_modules: Sequence[ModuleType],
) -> ImportedFunction:
    importer = ImportTable(modules)
    wrapped = {id(member): ImportedFunction(member, importer) for member in functions}
    for module in frozen_modules:
        vars(module).update(
            (name, wrapped.get(id(value), value)) for name, value in vars(module).items()
        )
    for member in functions:
        member.__globals__.update(
            (name, wrapped.get(id(value), value)) for name, value in member.__globals__.items()
        )
        for cell in member.__closure__ or ():
            try:
                cell.cell_contents = wrapped.get(id(cell.cell_contents), cell.cell_contents)
            except ValueError:
                continue
    return wrapped[id(function)]
