import operator
from collections.abc import Callable
from functools import partial

from udfx.ir import Node
from udfx.ops.registry import Fault, Op, apply, register
from udfx.ops.strings import out_of_range
from udfx.semantics.json import encode, has_type
from udfx.semantics.json_values import (
    RESULTS,
    convert,
    document,
    encoded,
    get,
    get_result,
    item,
    items,
    length,
    missing,
    raises,
    result,
    result_name,
    serialize,
    truth,
)
from udfx.types import BOOL, FLOAT, INT, JSON, STR, Type, list_of

OPERATIONS: list[Op] = []


def operation(op: Op) -> None:
    OPERATIONS.append(register(op))


def checked(
    name: str,
    parameters: tuple[Type, ...],
    output: Type,
    reference: Callable[..., object],
    kinds: tuple[type[Exception], ...],
) -> None:

    def faults(*arguments: Node) -> tuple[Fault, ...]:
        return tuple(Fault(kind, apply(f"{name}.{kind.__name__}", *arguments)) for kind in kinds)

    operation(Op(name, parameters, output, reference, faults))
    for kind in kinds:
        operation(Op(f"{name}.{kind.__name__}", parameters, BOOL, partial(raises, reference, kind)))


operation(Op("json.value", (STR,), JSON, document))
operation(Op("json.text", (JSON,), STR, document))
operation(Op("json.is_type", (JSON, STR), BOOL, has_type))
operation(Op("json.none", (JSON,), BOOL, missing))
operation(Op("json.truth", (JSON,), BOOL, truth))
operation(Op("json.serialize", (JSON, BOOL), STR, serialize))
operation(Op("json.dumps", (list_of(STR.or_none()).or_none(), BOOL), STR, encode))
register(Op("len", (list_of(JSON),), INT, len))
register(Op("bool", (list_of(JSON).or_none(),), BOOL, bool))
register(Op("__getitem__", (list_of(JSON), INT), JSON, operator.getitem, out_of_range))
for key in (STR, INT, FLOAT, BOOL):
    checked("json.get", (JSON, key.or_none(), JSON), JSON, get, (AttributeError,))
for output in (STR, STR.or_none()):
    operation(
        Op(
            f"json.get.{output}",
            (JSON, STR.or_none(), JSON),
            output,
            partial(get_result, type=output),
            lambda value, key, default: (
                Fault(AttributeError, apply("json.get.AttributeError", value, key, default)),
            ),
        )
    )
for key in (STR, INT):
    checked("json.item", (JSON, key.or_none()), JSON, item, (TypeError, KeyError, IndexError))
checked("json.items", (JSON,), list_of(JSON), items, (TypeError,))
checked("json.length", (JSON,), INT, length, (TypeError,))
for type in RESULTS:
    operation(Op("json.encode", (type.or_none(),), JSON, encoded))
for type in (STR, INT, FLOAT, BOOL):
    checked(
        f"json.convert.{type}",
        (JSON,),
        type,
        partial(convert, name=type.kind.value),
        (TypeError, ValueError, OverflowError) if type in (INT, FLOAT) else (),
    )
for type in RESULTS:
    for output in (type, type.or_none()):
        operation(
            Op(
                result_name(output),
                (JSON,),
                output,
                partial(result, type=output),
            )
        )
