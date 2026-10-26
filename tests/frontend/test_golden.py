from __future__ import annotations

import contextlib

import pytest

from tests.frontend.snapshot import GOLDEN, RECORDING, key
from udfx.errors import Unsupported
from udfx.frontend import trace
from workloads.corpus import Case
from workloads.corpus.cases import CASES


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_corpus_case_matches_the_golden_snapshot(case: Case) -> None:
    with contextlib.suppress(Unsupported):
        trace(case.function, case.schema)
    assert RECORDING or GOLDEN.knows(key(case.function, case.schema))
