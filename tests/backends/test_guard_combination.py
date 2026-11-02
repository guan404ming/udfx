from itertools import product

import pyarrow as pa
import pytest
from datafusion import Expr, col, functions, lit, udf

from udfx.backends.datafusion import DataFusionBackend
from udfx.backends.diagnostics import capture_lowering
from udfx.backends.faults import encode_fallback_error
from udfx.frontend import trace
from udfx.ir import Col, Guarded, If
from udfx.ir import lit as literal
from udfx.ops import apply
from udfx.program import Program
from udfx.testing import check_function
from udfx.testing.strategies import INT_TRAPS
from udfx.types import BOOL, INT, INT64_MAX, STR


def test_combined_guards_keep_exact_rows() -> None:
    backend = DataFusionBackend()
    schema = {f"guard{index}": BOOL.or_none() for index in range(4)}
    guards = [backend.column(Col(name, kind)) for name, kind in schema.items()]
    rows = list(product((False, True, None), repeat=4))
    combined = backend.conjunction(*guards)
    assert combined is not None
    result = backend.evaluate([backend.predicate(combined)], schema, rows)[0]
    assert [value is True for value in result] == [
        all(value is True for value in row) for row in rows
    ]
    assert backend.combine_guards([guards[0]]) is guards[0]
    assert backend.conjunction(None, True) is None
    assert backend.conjunction(guards[0], False) is False
    assert backend.conjunction(None, guards[0], guards[0], True) is guards[0]


def test_eager_guards_share_string_work() -> None:
    backend = DataFusionBackend()
    frame = backend.context.from_pydict({"flag": [True, False], "text": ["AZ", "anything"]})
    lowered = functions.lower(col("text"))
    combined = backend.conjunction(
        col("flag"),
        functions.starts_with(lowered, lit("a")),
        functions.ends_with(lowered, lit("z")),
    )
    assert combined is not None
    selected = frame.select(backend.predicate(combined).alias("value"))
    assert selected.execution_plan().display_indent().count("lower(") == 1
    assert selected.to_arrow_table()["value"].to_pylist() == [True, False]


def test_eager_guards_keep_overflow_and_caught_faults() -> None:
    def calculate(enabled: bool, numerator: int, divisor: int) -> int:
        if not enabled:
            return 0
        try:
            return numerator // divisor + 1
        except ZeroDivisionError:
            return -1

    backend = DataFusionBackend()
    schema = {"enabled": BOOL, "numerator": INT.or_none(), "divisor": INT.or_none()}
    rows = list(product((False, True), (*INT_TRAPS, None), (-1, 0, 1, None)))
    assert check_function(calculate, schema, backend, rows) == []


def test_inactive_conversion_guard_stays_native() -> None:
    def calculate(enabled: bool, text: str) -> int:
        if enabled:
            return int(text) + INT64_MAX
        return 0

    backend = DataFusionBackend()
    schema = {"enabled": BOOL, "text": STR.or_none()}
    values = (None, "", "invalid", "-1", "0", "1", str(INT64_MAX + 1))
    rows = list(product((False, True), values))
    assert check_function(calculate, schema, backend, rows) == []
    with capture_lowering() as details:
        backend.build(trace(calculate, schema))
    exact = details[-1].exact
    assert isinstance(exact, Expr) or exact is None or exact is False
    frame = backend.context.from_arrow(
        pa.table({"enabled": [False] * len(values), "text": pa.array(values, pa.string())})
    )
    assert frame.select(backend.predicate(exact).alias("value")).to_arrow_table()[
        "value"
    ].to_pylist() == [True] * len(values)


def test_combined_guards_keep_native_faults_in_the_selected_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = DataFusionBackend()
    raised = []

    def fail(values: pa.Array) -> pa.Array:
        if len(values):
            raised.append(len(values))
            raise RuntimeError(encode_fallback_error(OverflowError("native fault")))
        return pa.array([], pa.int64())

    failure = udf(fail, [pa.int64()], pa.int64(), "volatile", "guard_fault_probe")
    monkeypatch.setattr(backend, "native_raise", lambda program: failure(col("value")))
    schema = {"flag": BOOL, "value": INT, "left": BOOL.or_none(), "right": BOOL.or_none()}
    value = If(
        Col("flag", BOOL),
        apply(
            "__add__",
            Guarded(Col("value", INT), Col("left", BOOL.or_none())),
            Guarded(literal(1), Col("right", BOOL.or_none())),
        ),
        literal(0),
        INT,
    )
    program = Program(lambda flag, value, left, right: value + 1 if flag else 0, schema, value)
    expr = backend.build(program)
    rows = [(False, INT64_MAX, True, True), (False, INT64_MAX, False, None), (True, 0, True, True)]
    assert backend.evaluate([expr], schema, rows) == [[0, 0, 1]]
    assert not raised
    for left, right in ((False, True), (True, None)):
        with pytest.raises(OverflowError):
            backend.evaluate([expr], schema, [(True, INT64_MAX, left, right)])
        assert not raised
    with pytest.raises(OverflowError, match="native fault"):
        backend.evaluate([expr], schema, [(True, INT64_MAX, True, True)])
    assert raised == [1]
