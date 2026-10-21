from __future__ import annotations

import ast
import builtins
import dis
import importlib
from collections import deque
from types import CodeType, FunctionType

from bench.mine.definitions import bind_function, strip_definition
from bench.mine.models import Binding, Candidate

MODULES = {"math", "re", "string", "datetime", "unicodedata", "numpy", "pandas"}
HELPER_DEPTH = 8


def global_names(code: CodeType) -> set[str]:
    names = {
        instruction.argval
        for instruction in dis.get_instructions(code)
        if instruction.opname in {"LOAD_GLOBAL", "LOAD_NAME"}
    }
    for value in code.co_consts:
        if isinstance(value, CodeType):
            names.update(global_names(value))
    return names


def notebook_bindings(trees: list[ast.Module]) -> dict[str, Binding]:
    definitions: dict[str, Binding] = {}
    constants: dict[str, Binding] = {}
    imports: dict[str, Binding] = {}
    for tree in trees:
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                definitions[node.name] = node
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        constants[target.id] = node.value
            elif (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.value is not None
            ):
                constants[node.target.id] = node.value
            elif isinstance(node, ast.Import | ast.ImportFrom):
                imports.update(import_bindings(node))
    return {**imports, **constants, **definitions}


def import_bindings(node: ast.Import | ast.ImportFrom) -> dict[str, Binding]:
    if isinstance(node, ast.Import):
        return {
            alias.asname or alias.name: ast.Import(names=[alias])
            for alias in node.names
            if alias.name in MODULES
        }
    if node.level or node.module not in MODULES:
        return {}
    return {
        alias.asname or alias.name: ast.ImportFrom(module=node.module, names=[alias], level=0)
        for alias in node.names
        if alias.name != "*"
    }


def select_bindings(
    node: ast.FunctionDef | ast.Lambda, available: dict[str, Binding]
) -> dict[str, Binding]:
    pending = deque((name, 0) for name in sorted(global_names(bind_function(node).__code__)))
    selected: dict[str, Binding] = {}
    while pending:
        name, depth = pending.popleft()
        if name in selected or name == "__builtins__" or depth > HELPER_DEPTH:
            continue
        binding = available.get(name)
        if binding is None:
            continue
        if isinstance(binding, ast.FunctionDef):
            stripped = strip_definition(binding)
            assert isinstance(stripped, ast.FunctionDef)
            function = bind_function(stripped)
            selected[name] = stripped
            pending.extend((name, depth + 1) for name in sorted(global_names(function.__code__)))
        elif isinstance(binding, ast.expr):
            try:
                ast.literal_eval(binding)
            except (ValueError, TypeError, SyntaxError, RecursionError):
                continue
            selected[name] = binding
        else:
            selected[name] = binding
    return selected


def bind_candidate(candidate: Candidate) -> FunctionType:
    function = bind_function(candidate.node)
    namespace: dict[str, object] = {"__builtins__": builtins}
    helpers: dict[str, FunctionType] = {}
    for name, binding in candidate.bindings.items():
        if name == "__builtins__":
            continue
        if isinstance(binding, ast.FunctionDef):
            helpers[name] = bind_function(binding)
        elif isinstance(binding, ast.expr):
            namespace[name] = ast.literal_eval(binding)
        else:
            module_name = (
                binding.names[0].name if isinstance(binding, ast.Import) else binding.module
            )
            if module_name not in MODULES or (
                isinstance(binding, ast.ImportFrom) and binding.level
            ):
                continue
            assert module_name is not None
            try:
                module = importlib.import_module(module_name)
            except ImportError as error:
                raise ModuleNotFoundError(f"missing module {module_name}") from error
            if isinstance(binding, ast.Import):
                namespace[name] = module
            elif binding.names[0].name in vars(module):
                namespace[name] = vars(module)[binding.names[0].name]
    for name, helper in helpers.items():
        namespace[name] = FunctionType(helper.__code__, namespace, helper.__name__)
    return FunctionType(function.__code__, namespace, function.__name__)
