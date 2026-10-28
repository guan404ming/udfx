from pathlib import Path
from typing import Any

import pytest
from conftest import backend_params

from bench.data import prepare
from udfx import compile
from udfx.backends.base import Backend
from udfx.testing.equivalence import check_translation
from workloads.storage import DEFAULT_CACHE, verify_tables
from workloads.udfbench import load

pytestmark = pytest.mark.public

SUPPORTED = {
    "clean",
    "cleandate",
    "converttoeuro",
    "extractclass",
    "extractcode",
    "extractday",
    "extractfunder",
    "extractid",
    "extractmonth",
    "extractprojectid",
    "extractyear",
    "filterstopwords",
    "frequentterms",
    "jaccard",
    "jpack",
    "jsoncount",
    "jsort",
    "jsortvalues",
    "keywords",
    "log_10",
    "lowerize",
    "removeshortterms",
}


@pytest.mark.parametrize("backend", backend_params())
def test_original_udfbench_sources(backend: Backend[Any], tmp_path: Path) -> None:
    if not (DEFAULT_CACHE / "udfbench/manifest.json").exists():
        pytest.skip(
            "Prepare UDFBench explicitly with python -m workloads.prepare --suites udfbench"
        )
    verify_tables("udfbench", DEFAULT_CACHE)
    workloads = load(DEFAULT_CACHE)
    assert len(workloads) == 25
    supported = set()
    for workload in workloads:
        if workload.exclusion or not workload.deterministic:
            continue
        assert workload.function is not None, workload.error
        translation = compile(workload.function, workload.schema, backend)
        if translation.unsupported is not None:
            continue
        data = prepare(workload, tmp_path, rows=32)
        rows = list(zip(*(data.table[name].to_pylist() for name in workload.schema), strict=True))
        assert check_translation(translation, backend, rows) == [], workload.name
        assert translation.program is not None
        lowered = backend.lower_program(translation.program)
        if lowered.exact is not None:
            flags = backend.evaluate([lowered.exact], workload.schema, rows)[0]
            assert any(flag is True for flag in flags), workload.name
        supported.add(workload.name)
    assert supported >= SUPPORTED
    if backend.supports_graphs():
        assert "stem" in supported
    assert sum(not workload.deterministic for workload in workloads) == 1


@pytest.mark.parametrize("backend", backend_params())
def test_original_jsonparse_with_result_contract(backend: Backend[Any], tmp_path: Path) -> None:
    if not (DEFAULT_CACHE / "udfbench/manifest.json").exists():
        pytest.skip("Prepare UDFBench with python -m workloads.prepare --suites udfbench")
    workload = next(workload for workload in load(DEFAULT_CACHE) if workload.name == "jsonparse")
    assert workload.function is not None, workload.error
    translation = compile(workload.function, workload.schema, backend, result=workload.result)
    assert translation.ok, translation.unsupported
    data = prepare(workload, tmp_path, rows=1000)
    rows = list(zip(*(data.table[name].to_pylist() for name in workload.schema), strict=True))
    assert check_translation(translation, backend, rows) == []
