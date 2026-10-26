from collections.abc import Callable, Sequence

from datafusion import Expr, SessionContext, functions

from udfx.backends.datafusion.rules import array_lambda, column


def bind(value: Expr, body: Callable[[Expr], Expr]) -> Expr:
    return functions.array_element(
        functions.array_transform(functions.make_array(value), array_lambda(body)), 1
    )


def bind_sequence(value: Expr, steps: Sequence[Callable[[Expr], Expr]]) -> Expr:
    values = functions.make_array(value)
    for step in steps:
        values = functions.array_transform(values, array_lambda(step))
    return functions.array_element(values, 1)


def supported(context: SessionContext) -> bool:
    frame = context.from_pydict({"values": [[1, 2]]})
    records = functions.array_transform(
        column("values"),
        array_lambda(lambda value: functions.named_struct([("value", value)])),
    )
    result = bind(
        records,
        lambda values: functions.array_transform(
            values, array_lambda(lambda record: record["value"])
        ),
    )
    try:
        frame.select(result.alias("result")).execution_plan()
    except Exception:
        return False
    return True
