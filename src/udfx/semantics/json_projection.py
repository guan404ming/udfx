from collections.abc import Callable
from typing import Any

from udfx.semantics.json_lowering import JsonPrimitives, accepted, get, result, safe
from udfx.types import Type


def get_result(
    primitives: JsonPrimitives,
    value: Any,
    key: Any,
    default: Any,
    type: Type,
    *,
    validated: bool = False,
    default_validated: bool = False,
    default_none: bool = False,
    source: Any = None,
    indexes: tuple[int, ...] = (),
    consume: Callable[[Any, Any], Any] | None = None,
) -> Any:
    if primitives.typed_field is None:
        selected, exact = get(
            primitives,
            value,
            key,
            default,
            validated=validated,
            default_validated=default_validated,
            default_none=default_none,
        )
        output, typed = result(primitives, selected, type, True)
        return (output, exact & typed) if consume is None else consume(output, exact & typed)
    text = primitives.text
    document = safe(primitives, value, validated)

    def finish(output: Any, present: Any, exact: Any) -> Any:
        exact = exact & accepted(primitives, value, validated) & primitives.unique(document)
        exact = exact & accepted(primitives, default, default_validated)
        if not (default_none and type.nullable):
            fallback, typed = result(
                primitives, safe(primitives, default, default_validated), type, True
            )
            output = text.when(present, output, fallback)
            exact = exact & text.when(present, text.literal(True), typed)
        return (output, exact) if consume is None else consume(output, exact)

    arguments = document if source is None else source, key, type, indexes
    if consume is not None and primitives.project_field is not None:
        return primitives.project_field(*arguments, finish)
    return finish(*primitives.typed_field(*arguments))
