from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from types import CodeType
from typing import cast

from udfx.errors import Unsupported
from udfx.frontend.values import Item, Opaque, Tuple


def helper_signature(
    code: CodeType, defaults: Sequence[object], keyword_defaults: Mapping[str, object]
) -> inspect.Signature:
    parameters: list[inspect.Parameter] = []
    first_default = code.co_argcount - len(defaults)
    for index, name in enumerate(code.co_varnames[: code.co_argcount + code.co_kwonlyargcount]):
        default = inspect.Parameter.empty
        if index < code.co_posonlyargcount:
            kind = inspect.Parameter.POSITIONAL_ONLY
        elif index < code.co_argcount:
            kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
        else:
            kind = inspect.Parameter.KEYWORD_ONLY
            default = keyword_defaults.get(name, default)
        if first_default <= index < code.co_argcount:
            default = defaults[index - first_default]
        parameters.append(inspect.Parameter(name, kind, default=default))
    index = code.co_argcount + code.co_kwonlyargcount
    if code.co_flags & inspect.CO_VARARGS:
        parameters.insert(
            code.co_argcount,
            inspect.Parameter(code.co_varnames[index], inspect.Parameter.VAR_POSITIONAL),
        )
        index += 1
    if code.co_flags & inspect.CO_VARKEYWORDS:
        parameters.append(inspect.Parameter(code.co_varnames[index], inspect.Parameter.VAR_KEYWORD))
    return inspect.Signature(parameters)


def bind_helper(
    signature: inspect.Signature, arguments: Sequence[Item], keywords: Mapping[str, Item]
) -> dict[str, Item]:
    bound = bind(signature, arguments, keywords)
    bindings: dict[str, Item] = {}
    for name, parameter in signature.parameters.items():
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            bindings[name] = Tuple(cast("tuple[Item, ...]", bound.get(name, ())))
        elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
            if bound.get(name):
                raise Unsupported("nonempty variable keyword arguments")
            bindings[name] = Opaque({})
        else:
            bindings[name] = bound.get(name, cast("Item", parameter.default))
    return bindings


def bind[Value](
    signature: inspect.Signature, arguments: Sequence[Value], keywords: Mapping[str, Value]
) -> dict[str, Value]:
    try:
        return signature.bind(*arguments, **keywords).arguments
    except TypeError as error:
        raise Unsupported(f"invalid call arguments: {error}") from error
