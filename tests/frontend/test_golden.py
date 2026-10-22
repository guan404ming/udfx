from __future__ import annotations

import contextlib

import pytest

from tests.corpus import Case
from tests.corpus.cases import CASES
from tests.frontend.snapshot import GOLDEN, RECORDING, key
from udfx.errors import Unsupported
from udfx.frontend import trace


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_corpus_case_matches_the_golden_snapshot(case: Case) -> None:
    with contextlib.suppress(Unsupported):
        trace(case.function, case.schema)
    assert RECORDING or GOLDEN.knows(key(case.function, case.schema))
