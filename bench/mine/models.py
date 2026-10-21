from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass, field

Binding = ast.FunctionDef | ast.expr | ast.Import | ast.ImportFrom


@dataclass
class Counts:
    data_file: str = ""
    notebooks: int = 0
    cells: int = 0
    magic_cells: int = 0
    skipped_cells: Counter[str] = field(default_factory=Counter)
    call_sites: Counter[str] = field(default_factory=Counter)
    unresolved: Counter[str] = field(default_factory=Counter)
    skipped_definitions: Counter[str] = field(default_factory=Counter)
    csv_error: str | None = None
    collection_seconds: float = 0
    compile_seconds: float = 0


@dataclass
class Candidate:
    kernel_id: str
    first_cell: int
    node: ast.FunctionDef | ast.Lambda
    occurrences: int = 1
    kind: str = "element"
    bindings: dict[str, Binding] = field(default_factory=dict)
    name_origins: dict[str, str | None] = field(default_factory=dict)


@dataclass
class Result:
    kernel_id: str
    occurrences: int
    source: str
    ok: bool = False
    schema: dict[str, str] | None = None
    reason: str | None = None
    category: str | None = None
    kind: str = "element"
    third_party_names: list[str] = field(default_factory=list)
