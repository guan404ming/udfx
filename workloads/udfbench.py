import builtins
import inspect
from pathlib import Path
from typing import Any

from udfx.types import FLOAT, INT, STR, Type
from workloads.models import Workload
from workloads.storage import digest, read_manifest

TEXT = STR.or_none()
NUMBER = FLOAT.or_none()
INTEGER = INT.or_none()

BINDINGS: dict[str, tuple[str, dict[str, str], dict[str, Type], Type]] = {
    "addnoise": ("artifacts", {"val": "9"}, {"val": INTEGER}, NUMBER),
    "clean": ("artifact_authorlists", {"val": "1"}, {"val": TEXT}, TEXT),
    "cleandate": ("artifacts", {"pubdate": "4"}, {"pubdate": TEXT}, TEXT),
    "converttoeuro": ("artifact_charges", {"x": "1", "y": "2"}, {"x": NUMBER, "y": TEXT}, NUMBER),
    "extractclass": ("projects", {"project": "4"}, {"project": TEXT}, TEXT),
    "extractcode": ("projects", {"project": "4"}, {"project": TEXT}, TEXT),
    "extractday": ("artifacts", {"arg": "4"}, {"arg": TEXT}, INTEGER),
    "extractfunder": ("projects", {"project": "4"}, {"project": TEXT}, TEXT),
    "extractid": ("projects", {"project": "4"}, {"project": TEXT}, TEXT),
    "extractmonth": ("artifacts", {"arg": "4"}, {"arg": TEXT}, INTEGER),
    "extractprojectid": ("projects", {"input": "4"}, {"input": TEXT}, TEXT),
    "extractyear": ("artifacts", {"arg": "4"}, {"arg": TEXT}, INTEGER),
    "filterstopwords": ("artifact_abstracts", {"words": "1"}, {"words": TEXT}, TEXT),
    "frequentterms": ("artifact_abstracts", {"input": "1"}, {"input": TEXT, "N": INT}, TEXT),
    "jaccard": (
        "artifact_authorlists",
        {"arg1": "1", "arg2": "next_authorlist"},
        {"arg1": TEXT, "arg2": TEXT},
        NUMBER,
    ),
    "jpack": ("artifact_abstracts", {"input": "1"}, {"input": TEXT}, TEXT),
    "jsoncount": ("artifact_authorlists", {"jval": "1"}, {"jval": TEXT}, INTEGER),
    "jsonparse": (
        "artifact_authors",
        {"json_content": "6"},
        {"json_content": TEXT, "key1": STR},
        TEXT,
    ),
    "jsort": ("artifact_authorlists", {"jval": "1"}, {"jval": TEXT}, TEXT),
    "jsortvalues": ("artifact_authorlists", {"jval": "1"}, {"jval": TEXT}, TEXT),
    "keywords": ("artifact_abstracts", {"input": "1"}, {"input": TEXT}, TEXT),
    "log_10": ("projects", {"x": "21"}, {"x": NUMBER}, FLOAT),
    "lowerize": ("artifacts", {"val": "1"}, {"val": TEXT}, TEXT),
    "removeshortterms": ("artifact_authorlists", {"jval": "1"}, {"jval": TEXT}, TEXT),
    "stem": ("artifact_abstracts", {"input": "1"}, {"input": TEXT}, TEXT),
}
CONSTANTS = {"frequentterms": {"N": 5}, "jsonparse": {"key1": "value"}}
CATEGORIES = {"artifacts": "12", "projects": "3", "artifact_charges": "2"}


def load(root: Path) -> list[Workload]:
    entries = [
        entry for entry in read_manifest("udfbench")["files"] if entry["path"].endswith(".py")
    ]
    names = {Path(entry["path"]).stem for entry in entries}
    if names != BINDINGS.keys():
        raise ValueError(
            "Every upstream scalar file must have an explicit schema and input binding"
        )
    workloads = []
    for entry in entries:
        name = Path(entry["path"]).stem
        table, columns, schema, result = BINDINGS[name]
        function = None
        error = None
        path = root / "udfbench" / "source" / entry["path"]
        try:
            if digest(path) != entry["sha256"]:
                raise ValueError(f"Checksum mismatch: {path}")
            namespace: dict[str, Any] = {"__name__": f"udfbench_{name}"}
            exec(builtins.compile(path.read_bytes(), str(path), "exec"), namespace)
            function = namespace[name]
            if tuple(inspect.signature(function).parameters) != tuple(schema):
                raise ValueError(f"Schema does not match the original signature: {name}")
        except Exception as failure:
            error = f"{type(failure).__name__}: {failure}"
        workloads.append(
            Workload(
                "udfbench",
                name,
                function,
                schema,
                result,
                root / "udfbench" / f"{table}.parquet",
                columns,
                CATEGORIES.get(table),
                CONSTANTS.get(name, {}),
                origin="upstream",
                deterministic=name != "addnoise",
                error=error,
            )
        )
    return workloads
