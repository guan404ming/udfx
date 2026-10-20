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
    assert f"Native: {translation.expr!s}" in report


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
