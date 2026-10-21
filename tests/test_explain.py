import pytest

from udfx import compile, explain
from udfx.explain import render
from udfx.ir import Col, If, lit
from udfx.ops import apply
from udfx.types import INT, STR


def project_name(project: str | None) -> str | None:
    return project.split("::")[0] if project else None


def loop(value: int) -> int:
    while value > 0:
        value = value - 1
    return value


def test_supported_report() -> None:
    schema = {"project": STR.or_none()}
    report = explain(project_name, schema)
    translation = compile(project_name, schema)

    assert "Function: project_name" in report
    assert "Schema: {project: str?}" in report
    assert "bool(col project)" in report
    assert ".split(col project, '::')[0]" in report
    assert translation.program is not None
    assert f"Value: {render(translation.program.value)}" in report
    assert f"Fault: {render(translation.program.fault)}" in report
    assert f"Native: {str(translation.expr)!r}" in report


def test_unsupported_report() -> None:
    report = explain(loop, {"value": INT})

    assert "Function: loop" in report
    assert "Schema: {value: int}" in report
    assert "Unsupported: loops and comprehensions at loop+" in report
    assert "JUMP_BACKWARD" not in report


def test_render() -> None:
    column = Col("project", STR)
    node = If(
        apply("bool", column),
        apply("__getitem__", apply(".split", column, lit("::")), lit(0)),
        lit(None),
        STR.or_none(),
    )

    assert render(node) == "if(bool(col project), .split(col project, '::')[0], None)"


def test_report_escapes_control_characters() -> None:
    report = explain(lambda: "line\n\v", {})
    assert "Value: 'line\\n\\x0b'" in report
    assert len(report.splitlines()) == 5
    assert "\\n" in report.splitlines()[-1]
    assert "\\x0b" in report.splitlines()[-1]
    unsupported = explain(lambda column: column, {"column\n\v": STR})
    assert "Schema: {'column\\n\\x0b': str}" in unsupported
    assert len(unsupported.splitlines()) == 3


@pytest.mark.parametrize("name", ["first name", "line\n\v", "a'b", 'a"b', "a\\b"])
def test_column_names_are_escaped(name: str) -> None:
    assert render(Col(name, STR)) == f"col {name!r}"
    report = explain(lambda column: column, {name: STR})
    assert f"Schema: {{{name!r}: str}}" in report
    assert len(report.splitlines()) == 3
