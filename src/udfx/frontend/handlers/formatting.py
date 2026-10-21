import dis

from udfx.frontend.calls import node_of
from udfx.frontend.context import Context
from udfx.frontend.formatting import format_value, join_strings
from udfx.frontend.logic import fold
from udfx.frontend.machine import emit, handles
from udfx.frontend.state import State, Step
from udfx.frontend.values import Value
from udfx.ir import lit


@handles("FORMAT_VALUE")
def format_(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    flags = instruction.arg or 0
    specifier = lit("")
    if flags & 4:
        (item,), state = state.pop()
        specifier = node_of(item)
    (item,), rest = state.pop()
    conversion = (None, "s", "r", "a")[flags & 3]
    node, faults = format_value(node_of(item), conversion, specifier)
    return emit(context, rest, Value(fold(node)), faults)


@handles("BUILD_STRING")
def build_string(context: Context, state: State, instruction: dis.Instruction) -> list[Step]:
    items, rest = state.pop(instruction.arg or 0)
    return [rest.push(Value(fold(join_strings([node_of(item) for item in items])))).advance()]
