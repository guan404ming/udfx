import datetime
import inspect
import math
from pathlib import Path

import pyarrow

from udfx.types import BOOL, DATE, FLOAT, INT, STR
from workloads import tpch
from workloads.functions import load_function
from workloads.models import Workload
from workloads.storage import digest, read_manifest

TYPES = {"bool": BOOL, "date32": DATE, "float64": FLOAT, "int64": INT, "string": STR}
IMPORTS = {"datetime": datetime, "math": math}


def date_inputs(tables: dict[str, pyarrow.Table]) -> None:
    if not {"lineitem_orders", "customer", "supplier", "nation"} <= tables.keys():
        return
    customers = tpch.join(tables["lineitem_orders"], tables["customer"], "o_custkey", "c_custkey")
    shipping = tpch.join(customers, tables["supplier"], "l_suppkey", "s_suppkey")
    for alias, key in (("n1", "s_nationkey"), ("n2", "c_nationkey")):
        nation = tables["nation"].rename_columns(
            [f"{alias}.{name}" for name in tables["nation"].column_names]
        )
        shipping = tpch.join(shipping, nation, key, f"{alias}.n_nationkey")
    tables["froid_q8"] = shipping
    if "region" in tables:
        regional = tpch.join(
            customers,
            tables["supplier"],
            ["l_suppkey", "c_nationkey"],
            ["s_suppkey", "s_nationkey"],
        )
        regional = tpch.join(regional, tables["nation"], "s_nationkey", "n_nationkey")
        tables["froid_q4"] = tpch.join(regional, tables["region"], "n_regionkey", "r_regionkey")


def case_tables(cases: set[str]) -> set[str]:
    bindings = {
        f"qure/{Path(entry['path']).with_suffix('')}": entry["binding"]
        for entry in read_manifest("qure")["files"]
        if "binding" in entry
    }
    if unknown := cases - bindings.keys():
        raise ValueError(f"Unknown QURE cases: {', '.join(sorted(unknown))}")
    tables = {bindings[case]["table"] for case in cases}
    return tables | ({"lineitem_orders"} if tables & {"froid_q4", "froid_q8"} else set())


def prepare(paths: list[Path], selected: set[str] | None = None) -> dict[str, pyarrow.Table]:
    tables = tpch.load([path for path in paths if path.suffix == ".tbl"])
    for name, right, key, foreign_key in (
        ("lineitem_orders", "orders", "l_orderkey", "o_orderkey"),
        ("lineitem_part", "part", "l_partkey", "p_partkey"),
        ("lineitem_partsupp", "partsupp", "l_suppkey", "ps_suppkey"),
    ):
        if (selected is None or name in selected) and "lineitem" in tables and right in tables:
            tables[name] = tpch.join(tables["lineitem"], tables[right], key, foreign_key)
    date_inputs(tables)
    return tables


def load(
    root: Path, cases: set[str] | None = None, *, load_functions: bool = True
) -> list[Workload]:
    entries = [entry for entry in read_manifest("qure")["files"] if "binding" in entry]
    selected = (
        None
        if cases is None
        else {name.removeprefix("qure/") for name in cases if name.startswith("qure/")}
    )
    names = {str(Path(entry["path"]).with_suffix("")) for entry in entries}
    if selected is not None and selected - names:
        raise ValueError(f"Unknown QURE cases: {sorted(selected - names)}")
    workloads = []
    for entry in entries:
        name = str(Path(entry["path"]).with_suffix(""))
        if selected is not None and name not in selected:
            continue
        binding = entry["binding"]
        adaptations = entry.get("adaptations", {})
        exclusion = binding.get("exclusion")
        schema = (
            {} if exclusion else {name: TYPES[type] for name, type in binding["schema"].items()}
        )
        function = None
        error = None
        if load_functions and exclusion is None:
            path = root / "qure" / "source" / entry["path"]
            try:
                if digest(path) != entry["sha256"]:
                    raise ValueError(f"Checksum mismatch: {path}")
                additions = {
                    name: getattr(IMPORTS[module], name)
                    for module, names in adaptations.get("imports", {}).items()
                    for name in names
                }
                function = load_function(path, binding["function"], bindings=additions)
                if tuple(inspect.signature(function).parameters) != tuple(binding["schema"]):
                    raise ValueError(f"Schema does not match the original signature: {name}")
            except Exception as failure:
                error = f"{type(failure).__name__}: {failure}"
        workloads.append(
            Workload(
                "qure",
                name,
                function,
                schema,
                TYPES[binding["result"]],
                root / "qure" / f"{binding['table']}.parquet",
                binding["columns"],
                origin="adapted" if adaptations else "upstream",
                exclusion=exclusion,
                error=error,
            )
        )
    return workloads
