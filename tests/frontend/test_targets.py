import pytest

from udfx.frontend import trace
from udfx.frontend.logic import choose, conj, neg
from udfx.ir import FALSE, Call, Col, If, Lit, Node, lit
from udfx.ops import apply
from udfx.types import BOOL, FLOAT, INT, STR

pytestmark = pytest.mark.usefixtures("stand_ins")

NOTHING = lit(None)


def extractfunder(project: str | None) -> str | None:
    if project:
        try:
            if "::" in project:
                return project.split("::")[0]
            else:
                return None
        except:  # noqa: E722
            return None
    else:
        return None


def extractyear(arg: str | None) -> int | None:
    if arg:
        try:
            return int(arg[: arg.find("-")])
        except:  # noqa: E722
            return -1
    else:
        return None


def converttoeuro(x: float | None, y: str | None) -> float | None:
    rates = {"EUR": 1.0, "USD": 1.09, "GBP": 0.85}
    if x is not None and y is not None:
        try:
            return float(x) / rates[y]
        except:  # noqa: E722
            return 0.0
    else:
        return None


def removeshort(name: str) -> bool:
    def keep(word: str) -> bool:
        return len(word) > 2

    return keep(name)


def op_names(node: Node) -> set[str]:
    match node:
        case Call(op=op, args=args):
            return {op.name}.union(*(op_names(arg) for arg in args))
        case If(cond=condition, then=then, orelse=orelse):
            return op_names(condition) | op_names(then) | op_names(orelse)
        case _:
            return set()


def test_extractfunder() -> None:
    project = Col("project", STR.or_none())
    program = trace(extractfunder, {"project": project.type})
    truthy = apply("bool", project)
    found = apply("__contains__", project, Lit("::", STR))
    first = apply("__getitem__", apply(".split", project, Lit("::", STR)), Lit(0, INT))
    expected = choose(
        [(conj(truthy, found), first), (conj(truthy, neg(found)), NOTHING), (neg(truthy), NOTHING)]
    )
    assert program.value == expected
    assert program.fault == FALSE


def test_extractyear() -> None:
    arg = Col("arg", STR.or_none())
    program = trace(extractyear, {"arg": arg.type})
    truthy = apply("bool", arg)
    head = apply("slice", arg, Lit(None, INT.or_none()), apply(".find", arg, Lit("-", STR)))
    empty = neg(apply("bool", head))
    expected = choose(
        [
            (conj(truthy, empty), Lit(-1, INT)),
            (conj(truthy, neg(empty)), apply("int", head)),
            (neg(truthy), NOTHING),
        ]
    )
    assert program.value == expected
    assert program.fault == FALSE


def test_converttoeuro() -> None:
    program = trace(converttoeuro, {"x": FLOAT.or_none(), "y": STR.or_none()})
    assert program.result == FLOAT.or_none()
    assert program.fault == FALSE
    assert {"__eq__", "float", "__truediv__", "is_none"} <= op_names(program.value)


def test_removeshort() -> None:
    program = trace(removeshort, {"name": STR})
    assert program.value == apply("__gt__", apply("len", Col("name", STR)), Lit(2, INT))
    assert program.result == BOOL
    assert program.fault == FALSE
