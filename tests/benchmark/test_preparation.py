import csv
import json
from pathlib import Path
from zipfile import ZipFile

import pyarrow
import pytest
from pyarrow import parquet

from workloads.prepare import FLIGHT_COLUMNS, PREPARERS, flights, main, nyc311, prepare, udfbench
from workloads.storage import digest, verify_tables


def flight_archive(path: Path, cities: tuple[str, str] = ("Atlanta", "Atlanta")) -> Path:
    source = path.with_suffix(".csv")
    with source.open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(FLIGHT_COLUMNS)
        for city in cities:
            writer.writerow(["2024-01-01", "AA", "ATL", city, "ORD", "", "2.5", "0", "600"])
    with ZipFile(path, "w") as archive:
        archive.write(source, "flights.csv")
    return path


def test_flight_preparation_preserves_nulls_and_builds_lookup(tmp_path: Path) -> None:
    tables = flights([flight_archive(tmp_path / "flights.zip")])
    assert tables["flights"].num_rows == 2
    assert tables["flights"]["departure_time"].to_pylist() == [None, None]
    assert tables["flights"]["arrival_delay"].to_pylist() == [2.5, 2.5]
    assert tables["airports"].to_pylist() == [{"airport": "ATL", "city": "Atlanta"}]
    with pytest.raises(ValueError, match="one city"):
        flights([flight_archive(tmp_path / "different.zip", ("Atlanta", "Different"))])


def test_nyc_timestamps_keep_microseconds_and_nulls(tmp_path: Path) -> None:
    path = tmp_path / "requests.csv"
    path.write_text(
        "unique_key,created_date,closed_date,agency,complaint_type,descriptor,incident_zip,borough\n1,2019-01-01T00:00:00.123456,,NYPD,Noise,,00123,QUEENS\n"
    )
    table = nyc311([path])["requests"]
    assert table["incident_zip"].to_pylist() == ["00123"]
    assert table["closed_date"].null_count == 1
    created = table["created_date"].to_pylist()[0]
    assert created is not None and created.microsecond == 123456


def test_udfbench_preparation_keeps_original_values(tmp_path: Path) -> None:
    path = tmp_path / "artifact_authorlists.parquet"
    parquet.write_table(pyarrow.table({"0": ["a", "b"], "1": ['["A"]', None]}), path)
    table = udfbench([path])["artifact_authorlists"]
    assert table["1"].to_pylist() == ['["A"]', None]
    assert table["next_authorlist"].to_pylist() == [None, '["A"]']


def test_explicit_prepare_checks_source_and_records_row_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = flight_archive(tmp_path / "flights.zip")
    source = {"files": [{"url": path.as_uri(), "sha256": digest(path), "path": "flights.zip"}]}
    monkeypatch.setattr("workloads.prepare.read_manifest", lambda _: source)
    root = tmp_path / "cache"
    prepare("flights", root, rows=1)
    metadata = verify_tables("flights", root)
    assert metadata["tables"]["flights"]["rows"] == 1
    assert metadata["sampling"] == {"method": "source prefix", "row_limit": 1}
    assert json.loads((root / "flights/manifest.json").read_text())["tables"]


def test_prepare_command_does_not_run_without_valid_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    monkeypatch.setattr("workloads.prepare.prepare", lambda *args: calls.append(args))
    with pytest.raises(SystemExit):
        main(["--rows", "0"])
    assert calls == []
    main(["--suites", "flights", "--root", str(tmp_path), "--rows", "3"])
    assert calls == [("flights", tmp_path, 3)]


def test_sampling_keeps_the_complete_dimension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "workloads.prepare.read_manifest", lambda _: {"files": [], "dimension_tables": ["airports"]}
    )
    tables = {
        "flights": pyarrow.table({"value": [1, 2, 3]}),
        "airports": pyarrow.table({"value": [4, 5, 6]}),
    }
    monkeypatch.setitem(PREPARERS, "flights", lambda _: tables)
    prepare("flights", tmp_path, rows=1)
    metadata = verify_tables("flights", tmp_path)
    assert metadata["tables"]["flights"]["rows"] == 1
    assert metadata["tables"]["airports"]["rows"] == 3
