from datetime import date
from pathlib import Path

import pyarrow
import pytest

from udfx.types import DATE
from workloads import qure, tpch
from workloads.storage import digest, read_manifest


def table_file(root: Path, name: str, rows: list[dict[str, object]]) -> Path:
    path = root / f"{name}.tbl"
    values = []
    for row in rows:
        fields = []
        for field in tpch.SCHEMAS[name]:
            default = (
                "1995-02-03"
                if pyarrow.types.is_date32(field.type)
                else "NULL"
                if pyarrow.types.is_string(field.type)
                else 1
            )
            fields.append(str(row.get(field.name, default)))
        values.append("|".join(fields) + "|")
    path.write_text("\n".join(values) + "\n")
    return path


def source_entry(root: Path, source: str) -> dict:
    path = root / "qure/source/python/q1.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return {
        "path": "python/q1.py",
        "input": "qure.code",
        "sha256": digest(path),
        "binding": {
            "function": "udf_0",
            "table": "part",
            "columns": {"value": "p_size"},
            "schema": {"value": "int64"},
            "result": "int64",
        },
    }


@pytest.mark.parametrize("name", tpch.SCHEMAS)
def test_tpch_reads_explicit_types_and_preserves_dates(tmp_path: Path, name: str) -> None:
    table = tpch.read(table_file(tmp_path, name, [{}]))
    assert table.schema == tpch.SCHEMAS[name]
    for field in table.schema:
        if pyarrow.types.is_date32(field.type):
            assert table[field.name].to_pylist() == [date(1995, 2, 3)]
        if pyarrow.types.is_string(field.type):
            assert table[field.name].to_pylist() == ["NULL"]


def test_tpch_rejects_malformed_rows_and_implicit_inputs(tmp_path: Path) -> None:
    path = table_file(tmp_path, "region", [{}])
    path.write_text("1|name|comment|unexpected\n")
    with pytest.raises(ValueError, match="empty field"):
        tpch.read(path)
    path.write_text("1|name|comment\n")
    with pytest.raises(pyarrow.ArrowInvalid):
        tpch.read(path)
    with pytest.raises(ValueError, match="Unknown TPC-H"):
        tpch.read(tmp_path / "unknown.tbl")
    with pytest.raises(ValueError, match="Supply local"):
        qure.prepare([])
    path = table_file(tmp_path, "region", [{}])
    with pytest.raises(ValueError, match="Duplicate TPC-H"):
        tpch.load([path, path])


def test_qure_joins_keep_source_order_and_supplier_only_multiplicity(tmp_path: Path) -> None:
    paths = [
        table_file(
            tmp_path,
            "lineitem",
            [
                {"l_orderkey": 2, "l_partkey": 20, "l_suppkey": 2},
                {"l_orderkey": 1, "l_partkey": 10, "l_suppkey": 1},
            ],
        ),
        table_file(
            tmp_path,
            "orders",
            [{"o_orderkey": 1, "o_totalprice": 10}, {"o_orderkey": 2, "o_totalprice": 20}],
        ),
        table_file(
            tmp_path,
            "part",
            [{"p_partkey": 10, "p_size": 3}, {"p_partkey": 20, "p_size": 7}],
        ),
        table_file(
            tmp_path,
            "partsupp",
            [
                {"ps_partkey": 99, "ps_suppkey": 2, "ps_supplycost": 30},
                {"ps_partkey": 20, "ps_suppkey": 2, "ps_supplycost": 10},
                {"ps_partkey": 10, "ps_suppkey": 1, "ps_supplycost": 20},
            ],
        ),
        table_file(
            tmp_path,
            "customer",
            [{"c_custkey": 1, "c_nationkey": 3}],
        ),
        table_file(
            tmp_path,
            "supplier",
            [{"s_suppkey": 2, "s_nationkey": 4}, {"s_suppkey": 1, "s_nationkey": 3}],
        ),
        table_file(
            tmp_path,
            "nation",
            [{"n_nationkey": 3, "n_name": "FRANCE"}, {"n_nationkey": 4, "n_name": "GERMANY"}],
        ),
        table_file(tmp_path, "region", [{"r_name": "ASIA"}]),
    ]
    tables = qure.prepare(paths)
    assert tables["lineitem_orders"]["o_totalprice"].to_pylist() == [20.0, 10.0]
    assert tables["lineitem_part"]["p_size"].to_pylist() == [7, 3]
    joined = tables["lineitem_partsupp"]
    assert joined["l_orderkey"].to_pylist() == [2, 2, 1]
    assert joined["ps_supplycost"].to_pylist() == [30.0, 10.0, 20.0]
    assert joined["l_shipdate"].type == pyarrow.date32()
    assert tables["froid_q4"]["l_orderkey"].to_pylist() == [1]
    assert tables["froid_q4"]["r_name"].to_pylist() == ["ASIA"]
    assert tables["froid_q8"]["l_orderkey"].to_pylist() == [2, 1]
    assert tables["froid_q8"]["n1.n_name"].to_pylist() == ["GERMANY", "FRANCE"]
    assert tables["froid_q8"]["n2.n_name"].to_pylist() == ["FRANCE", "FRANCE"]


def test_qure_manifest_keeps_all_cases_and_local_provenance() -> None:
    manifest = read_manifest("qure")
    sources = [entry for entry in manifest["files"] if "binding" in entry]
    assert {entry["path"] for entry in sources} == {
        *(f"python/q{number}.py" for number in range(1, 28)),
        *(f"froid/q{number}.py" for number in range(1, 10)),
    }
    assert all("exclusion" not in entry["binding"] for entry in sources)
    adaptations = {
        entry["path"]: entry["adaptations"]["imports"]
        for entry in sources
        if "adaptations" in entry
    }
    assert adaptations == {
        "python/q2.py": {"math": ["exp", "sqrt"]},
        "python/q14.py": {"math": ["acos", "atan2", "cos", "sin", "sqrt"]},
        "froid/q5.py": {"datetime": ["timedelta"]},
        "froid/q9.py": {"datetime": ["timedelta"]},
    }
    assert all(len(entry["sha256"]) == 64 and entry["input"] == "qure.code" for entry in sources)
    license = next(entry for entry in manifest["files"] if entry["path"] == "License.pdf")
    assert len(license["sha256"]) == 64
    tables = [entry for entry in manifest["files"] if entry["input"] == "qure.tpch"]
    assert {entry["path"] for entry in tables} == {f"{name}.tbl" for name in tpch.SCHEMAS}
    assert all("sha256" not in entry for entry in tables)


def test_qure_loads_original_callable_and_filters_full_identifiers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "query-ran"
    entry = source_entry(
        tmp_path,
        "OFFSET = 7\ndef udf_0(value):\n    return value + OFFSET\n"
        f"open({str(marker)!r}, 'w').write('query')\n",
    )
    monkeypatch.setattr(qure, "read_manifest", lambda name: {"files": [entry]})
    workloads = qure.load(tmp_path, {"qure/python/q1", "flights/year_month"})
    assert len(workloads) == 1
    workload = workloads[0]
    assert workload.identifier == "qure/python/q1"
    assert workload.origin == "upstream"
    assert workload.error is None
    assert workload.function is not None and workload.function(4) == 11
    assert not marker.exists()
    assert qure.load(tmp_path, {"flights/year_month"}) == []
    with pytest.raises(ValueError, match="Unknown QURE"):
        qure.load(tmp_path, {"qure/python/missing"})


@pytest.mark.parametrize(
    "source,reason",
    [
        ("def udf_0(value):\n    return missing_helper(value)\n", "missing_helper"),
        ("def udf_0(other):\n    return other\n", "signature"),
        ("def udf_0(value):\n  return (\n", "SyntaxError"),
    ],
)
def test_qure_source_defects_are_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str, reason: str
) -> None:
    entry = source_entry(tmp_path, source)
    monkeypatch.setattr(qure, "read_manifest", lambda name: {"files": [entry]})
    workload = qure.load(tmp_path)[0]
    assert workload.error is not None and reason in workload.error
    assert workload.exclusion is None


def test_qure_checksum_failure_and_inventory_only_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = source_entry(tmp_path, "def udf_0(value):\n    return value\n")
    monkeypatch.setattr(qure, "read_manifest", lambda name: {"files": [entry]})
    path = tmp_path / "qure/source/python/q1.py"
    path.write_text("def udf_0(value):\n    return value + 1\n")
    workload = qure.load(tmp_path)[0]
    assert workload.error is not None and "Checksum mismatch" in workload.error
    path.unlink()
    inventory = qure.load(tmp_path, load_functions=False)[0]
    assert inventory.function is None and inventory.error is None
    entry["binding"]["schema"] = {"value": "date32"}
    inventory = qure.load(tmp_path, load_functions=False)[0]
    assert inventory.schema == {"value": DATE}
    assert inventory.exclusion is None


def test_qure_marks_explicit_import_adaptation_and_preserves_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = source_entry(tmp_path, "def udf_0(value):\n    return ceil(value)\n")
    entry["adaptations"] = {"imports": {"math": ["ceil"]}}
    monkeypatch.setattr(qure, "read_manifest", lambda name: {"files": [entry]})
    workload = qure.load(tmp_path)[0]
    assert workload.origin == "adapted"
    assert workload.error is None and workload.function is not None
    assert workload.function(1.25) == 2
    assert digest(tmp_path / "qure/source/python/q1.py") == entry["sha256"]
    assert qure.load(tmp_path, load_functions=False)[0].origin == "adapted"


def test_qure_prepare_skips_joins_without_their_tables(tmp_path: Path) -> None:
    tables = qure.prepare([table_file(tmp_path, "lineitem", [{}])])
    assert set(tables) == {"lineitem"}
