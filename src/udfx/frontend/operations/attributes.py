from __future__ import annotations

import re
from datetime import datetime
from types import ModuleType

from udfx.errors import Unsupported
from udfx.frontend.core.evaluation import Evaluation, single
from udfx.frontend.core.expressions import invoke
from udfx.frontend.core.values import (
    NULL,
    Counter,
    CounterKey,
    Duration,
    Item,
    Json,
    Match,
    Method,
    Opaque,
    Row,
    Set,
    Value,
    describe,
)
from udfx.frontend.operations.datetimes import duration_attribute
from udfx.frontend.resolve.constants import constant
from udfx.frontend.resolve.frozen import Snapshot
from udfx.semantics.datetimes import ATTRIBUTES
from udfx.types import Kind

CLASS_METHODS = {(datetime, "strptime"): datetime.strptime, (str, "maketrans"): str.maketrans}


def load_attribute(
    owner: Item, name: str, method: bool, snapshot: Snapshot, where: str
) -> Evaluation:
    match owner:
        case Counter() if name == "get" and not method:
            return single(CounterKey(owner))
        case Row():
            attribute = owner.column(name)
        case Opaque(value=ModuleType() as module) if name in snapshot.members(module):
            attribute = constant(snapshot.attribute(module, name))
        case Opaque(value=type() as python_type) if (python_type, name) in CLASS_METHODS:
            attribute = Opaque(CLASS_METHODS[python_type, name])
        case Opaque(value=dict()) if name in {"get", "keys", "values", "items"} and method:
            return Evaluation((Method(name), owner))
        case Opaque(value=re.Pattern() as pattern):
            attribute = Opaque(getattr(pattern, name, None))
        case Value(node=node) if node.type.kind is Kind.DATETIME and name in ATTRIBUTES:
            if method:
                raise Unsupported(f"datetime attribute {name} is not callable", where)
            result, faults = invoke(f".{name}", [node], AttributeError)
            return single(Value(result), faults)
        case Duration(node=node) if name in {"days", "seconds"} and not method:
            result, faults = duration_attribute(node, name)
            return single(Value(result), faults)
        case Value() | Match() | Duration() | Set() | Json() if method:
            return Evaluation((Method(name), owner))
        case Value():
            raise Unsupported(f"attribute access on a column value (attribute load {name})", where)
        case _:
            raise Unsupported(f"attribute load {name} of {describe(owner)}", where)
    return Evaluation((attribute, NULL)) if method else single(attribute)
