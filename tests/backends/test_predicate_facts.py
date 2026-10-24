from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.backends.spark import SparkBackend
from udfx.ir import Col
from udfx.ops import apply
from udfx.semantics.assumptions import assuming
from udfx.testing import check_function
from udfx.types import STR


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("truth", [False, True])
def test_ascii_predicates_reuse_branch_results(backend: Backend[Any], truth: bool) -> None:
    if not isinstance(backend, SparkBackend):
        pytest.skip("Spark ASCII predicate branches")
    source = Col("text", STR)
    samples = ["", "abc", "123", "١٢", "²", "½", "Ⅻ", "1²", "①", "\t", "\uff11"]
    predicate = "isnumeric" if truth else "isdecimal"
    rows = [(text,) for text in samples if getattr(text, predicate)() == truth]
    methods = ("isdecimal", "isdigit") if truth else ("isdigit", "isnumeric")
    expressions = []
    for method in methods:
        with assuming(apply(f".{predicate}", source), truth):
            expression = backend.lower(apply(f".{method}", source)).expr
        fast, unicode = str(expression).split(" ELSE ", 1)
        assert "rlike(" not in fast
        assert "rlike(" in unicode
        expressions.append(expression)
    assert backend.evaluate(expressions, {"text": STR}, rows) == [
        [getattr(text, method)() for (text,) in rows] for method in methods
    ]
    assert "rlike(" in str(backend.lower(apply(".isdigit", source)).expr).split(" ELSE ", 1)[0]


@pytest.mark.parametrize("backend", backend_params())
def test_predicate_facts_keep_subjects_and_faults_separate(backend: Backend[Any]) -> None:
    def classify(left: str, right: str) -> bool:
        return right.isdigit() if left.isdecimal() else right.isnumeric()

    schema = {"left": STR.or_none(), "right": STR.or_none()}
    values = (None, "", "1", "²", "½", "abc")
    rows = [(left, right) for left in values for right in values]
    assert check_function(classify, schema, backend, rows) == []
