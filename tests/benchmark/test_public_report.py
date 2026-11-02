import json
from pathlib import Path

from bench.artifacts import write
from bench.report import main, public_results
from tests.benchmark.test_reporting import artifact


def test_public_export_omits_diagnostics_and_materials(tmp_path: Path) -> None:
    result = artifact()
    secret = "private input and source content"
    result["datasets"] = {"source": secret}
    result["results"][0]["data"] = {"value": secret}
    result["results"][0]["origin"] = "adapted"
    result["results"][1]["reason"] = secret
    exported = public_results(result)
    assert secret not in json.dumps(exported)
    assert result["results"][1]["reason"] == secret
    assert exported["results"][0]["comparisons"] == result["results"][0]["comparisons"]
    assert exported["results"][0]["origin"] == "adapted"
    path = tmp_path / "result.json"
    markdown = tmp_path / "public.md"
    csv = tmp_path / "public.csv"
    write(path, result)
    main([str(path), "--public", "--markdown", str(markdown), "--csv", str(csv)])
    assert secret not in markdown.read_text()
    assert secret not in csv.read_text()
    assert "0.50x" in markdown.read_text()
    assert "unsupported" in csv.read_text()
    assert "adapted" in markdown.read_text()
    assert "adapted" in csv.read_text()
