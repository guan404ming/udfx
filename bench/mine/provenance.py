from __future__ import annotations

import ast

from bench.mine.bindings import MODULES, global_names, import_bindings
from bench.mine.definitions import bind_function
from bench.mine.models import Binding, Candidate


def binding_origin(binding: Binding) -> str | None:
    if isinstance(binding, ast.Import):
        return binding.names[0].name
    if isinstance(binding, ast.ImportFrom):
        return "." * binding.level + (binding.module or "")
    return None


def notebook_origins(
    trees: list[ast.Module], available: dict[str, Binding]
) -> dict[str, str | None]:
    origins: dict[str, str | None] = {}
    pending: list[ast.AST] = list(reversed(trees))
    while pending:
        node = pending.pop()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            origins[node.name] = None
        elif isinstance(node, ast.Import | ast.ImportFrom):
            origins.update(
                (name, binding_origin(binding)) for name, binding in import_bindings(node).items()
            )
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            origins[node.id] = None
        elif not isinstance(
            node, ast.Lambda | ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp
        ):
            pending.extend(reversed(list(ast.iter_child_nodes(node))))
    origins.update((name, binding_origin(binding)) for name, binding in available.items())
    return origins


def candidate_origins(
    node: ast.FunctionDef | ast.Lambda,
    bindings: dict[str, Binding],
    origins: dict[str, str | None],
) -> dict[str, str | None]:
    names = global_names(bind_function(node).__code__)
    for binding in bindings.values():
        if isinstance(binding, ast.FunctionDef):
            names.update(global_names(bind_function(binding).__code__))
    return {name: origins[name] for name in sorted(names) if name in origins}


def third_party_names(candidate: Candidate, reason: str) -> list[str]:
    for prefix in ("unknown global ", "missing module "):
        if reason.startswith(prefix):
            name = reason.removeprefix(prefix)
            if prefix == "missing module " and name in MODULES:
                return []
            origin = candidate.name_origins.get(name, "")
            return [name] if origin is not None and origin not in MODULES else []
    return []
