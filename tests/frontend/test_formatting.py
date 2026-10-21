# ruff: noqa: F523, PLW0108, UP030, UP031, UP032
from collections.abc import Callable

import pytest

from udfx.backends.polars import PolarsBackend
from udfx.errors import Unsupported
from udfx.frontend import trace
from udfx.ir import FALSE, Col, Lit
from udfx.ops import apply
from udfx.testing import check_function
from udfx.types import BOOL, FLOAT, INT, STR, Type


@pytest.mark.parametrize("type", [STR, INT, FLOAT, BOOL])
@pytest.mark.parametrize("function", [lambda value: f"{value}", lambda value: f"{value!s}"])
def test_f_string_reuses_str(function: Callable[..., str], type: Type) -> None:
    schema = {"value": type.or_none()}
    program = trace(function, schema)
    assert program.value == apply("str", Col("value", schema["value"]))
    assert program.fault == FALSE


@pytest.mark.parametrize("type", [INT, FLOAT, BOOL])
def test_f_string_repr_reuses_str(type: Type) -> None:
    program = trace(lambda value: f"{value!r}", {"value": type.or_none()})
    assert program.value == apply("str", Col("value", type.or_none()))
    assert program.fault == FALSE


@pytest.mark.parametrize(
    ("function", "type", "rows"),
    [
        (lambda value: f"<{value}>:{value!s}", STR, [("",), ("é{}",), (None,)]),
        (lambda value: f"{value:d}", INT, [(0,), (-42,), (None,)]),
        (lambda value: f"{value:s}", STR, [("",), ("é",), (None,)]),
        (lambda value: f"{value:}", FLOAT, [(-0.0,), (float("nan"),), (None,)]),
        (lambda value: f"{value!s:s}", BOOL, [(True,), (False,), (None,)]),
        (lambda value: f"{value!r:s}", INT, [(12,), (-12,), (None,)]),
    ],
)
def test_f_string_matches_python(
    function: Callable[..., str], type: Type, rows: list[tuple[object, ...]]
) -> None:
    assert check_function(function, {"value": type.or_none()}, PolarsBackend(), rows) == []


@pytest.mark.parametrize(
    ("function", "expected"),
    [(lambda: f"{None}", "None"), (lambda: f"{{{123:d}}}", "{123}")],
)
def test_literal_f_string_folds(function: Callable[..., str], expected: str) -> None:
    assert trace(function, {}).value == Lit(expected, STR)


@pytest.mark.parametrize(
    ("function", "schema", "reason"),
    [
        (lambda value: f"{value!r}", {"value": STR}, "conversion !r for str"),
        (lambda value: f"{value!a}", {"value": INT}, "conversion !a"),
        (lambda value: f"{value:04d}", {"value": INT}, "format spec '04d'"),
        (lambda value: f"{value:.2f}", {"value": FLOAT}, "format spec '.2f'"),
        (lambda value: f"{value:d}", {"value": FLOAT}, "format spec 'd'"),
        (lambda value: f"{value:s}", {"value": INT}, "format spec 's'"),
        (lambda value: f"{value!s:d}", {"value": INT}, "format spec 'd'"),
        (
            lambda value, specifier: f"{value:{specifier}}",
            {"value": INT, "specifier": STR},
            "dynamic format spec",
        ),
    ],
)
def test_unsupported_f_string_names_reason(
    function: Callable[..., str], schema: dict[str, Type], reason: str
) -> None:
    with pytest.raises(Unsupported) as caught:
        trace(function, schema)
    assert reason in caught.value.reason


@pytest.mark.parametrize(
    "function",
    [
        lambda text, number: "{} {}".format(text, number),
        lambda text, number: "{1}-{0}-{1}".format(text, number),
        lambda text, number: "{{{0!s:s}}}:{1!r}".format(text, number),
        lambda text, number: "{0:s}-{1:d}".format(text, number),
        lambda text, number: "{1!r:s}".format(text, number),
        lambda text, number: "plain".format(text, number),
        lambda text, number: "".format(text, number),
    ],
)
def test_format_template_matches_python(function: Callable[..., str]) -> None:
    schema = {"text": STR.or_none(), "number": INT.or_none()}
    rows = [("é{}", 42), ("", -5), (None, 0), ("x", None)]
    assert check_function(function, schema, PolarsBackend(), rows) == []


@pytest.mark.parametrize(
    ("function", "expected"),
    [
        (lambda: "{0!r:>5}".format("x"), "  'x'"),
        (lambda: "{}-{:d}".format("x", 42), "x-42"),
    ],
)
def test_literal_format_template_folds(function: Callable[..., str], expected: str) -> None:
    assert trace(function, {}).value == Lit(expected, STR)


@pytest.mark.parametrize(
    ("template", "type", "reason"),
    [
        ("{0!r}", STR, "conversion !r for str"),
        ("{0!a}", INT, "conversion !a"),
        ("{0!q}", INT, "conversion !q"),
        ("{0:04d}", INT, "format spec '04d'"),
        ("{0:.2f}", FLOAT, "format spec '.2f'"),
        ("{0:d}", STR, "format spec 'd'"),
        ("{0:s}", INT, "format spec 's'"),
        ("{0!r:d}", INT, "format spec 'd'"),
        ("{0:{0}}", INT, "dynamic format spec"),
        ("{name}", STR, "format field 'name'"),
        ("{0.real}", INT, "format field '0.real'"),
        ("{0[0]}", STR, "format field '0[0]'"),
        ("{-1}", INT, "format field '-1'"),
        ("{}-{0}", INT, "mixed automatic and manual"),
        ("{1}", INT, "format field index 1 out of range"),
        ("{" + "9" * 5000 + "}", INT, "format field index"),
        ("{", INT, "format template"),
    ],
)
def test_unsupported_format_template_names_reason(template: str, type: Type, reason: str) -> None:
    with pytest.raises(Unsupported) as caught:
        trace(lambda value: template.format(value), {"value": type})
    assert reason in caught.value.reason


def test_dynamic_format_template_is_unsupported() -> None:
    with pytest.raises(Unsupported, match="dynamic format template"):
        trace(lambda template, value: template.format(value), {"template": STR, "value": INT})


def test_format_keyword_is_unsupported() -> None:
    with pytest.raises(Unsupported, match="keyword arguments"):
        trace(lambda value: "{value}".format(value=value), {"value": INT})


@pytest.mark.parametrize("type", [STR, INT, FLOAT, BOOL])
def test_percent_string_reuses_str(type: Type) -> None:
    program = trace(lambda value: "%s" % value, {"value": type.or_none()})
    assert program.value == apply("str", Col("value", type.or_none()))
    assert program.fault == FALSE


@pytest.mark.parametrize(
    "function",
    [
        lambda text, number: "%s-%s" % (text, number),
        lambda text, number: "%s-%d%%" % (text, number),
        lambda text, number: "%d" % number,
        lambda text, number: "%s-%s-%d" % (text, None, number),
    ],
)
def test_percent_format_matches_python(function: Callable[..., str]) -> None:
    schema = {"text": STR.or_none(), "number": INT.or_none()}
    rows = [("é{}%", 42), ("", -5), (None, 0), ("x", None)]
    assert check_function(function, schema, PolarsBackend(), rows) == []


def test_percent_stored_tuple_matches_python() -> None:
    def function(text: str, number: int) -> str:
        arguments = (text, number)
        return "%s-%d" % arguments

    assert (
        check_function(
            function, {"text": STR, "number": INT}, PolarsBackend(), [("é", 12), ("", -1)]
        )
        == []
    )


@pytest.mark.parametrize(
    ("function", "expected"),
    [
        (lambda: "%s-%d" % ("x", 42), "x-42"),
        (lambda: "%d-%d" % (1, 2), "1-2"),
        (lambda: "%s" % None, "None"),
        (lambda: "%%" % (), "%"),
    ],
)
def test_literal_percent_format_folds(function: Callable[..., str], expected: str) -> None:
    assert trace(function, {}).value == Lit(expected, STR)


@pytest.mark.parametrize(
    ("template", "type", "reason"),
    [
        ("%d", FLOAT, "format spec 'd' for float"),
        ("%d", STR, "format spec 'd' for str"),
        ("%r", INT, "printf spec '%r'"),
        ("%a", INT, "printf spec '%a'"),
        ("%f", FLOAT, "printf spec '%f'"),
        ("%x", INT, "printf spec '%x'"),
        ("%04d", INT, "printf spec '%0'"),
        ("%*s", STR, "printf spec '%*'"),
        ("%(name)s", STR, "printf spec '%('"),
        ("%", STR, "incomplete printf spec"),
        ("%s%s", STR, "printf argument count"),
        ("literal", STR, "printf argument count"),
    ],
)
def test_unsupported_percent_format_names_reason(template: str, type: Type, reason: str) -> None:
    with pytest.raises(Unsupported) as caught:
        trace(lambda value: template % value, {"value": type})
    assert reason in caught.value.reason


def test_dynamic_percent_template_is_unsupported() -> None:
    with pytest.raises(Unsupported, match="dynamic format template"):
        trace(lambda template, value: template % value, {"template": STR, "value": INT})


@pytest.mark.parametrize("container", [["x"], {"x"}, frozenset({"x"})])
def test_percent_does_not_unpack_other_containers(container: object) -> None:
    with pytest.raises(Unsupported, match="constant container"):
        trace(lambda: "%s" % container, {})
