from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from types import CodeType

from udfx.errors import Unsupported


def helper_signature(
    code: CodeType, defaults: Sequence[object], keyword_defaults: Mapping[str, object]
) -> inspect.Signature:
    if code.co_flags & (inspect.CO_VARARGS | inspect.CO_VARKEYWORDS):
        raise Unsupported(f"{code.co_qualname} takes variable parameters")
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
    return inspect.Signature(parameters)


def bind[Value](
    signature: inspect.Signature, arguments: Sequence[Value], keywords: Mapping[str, Value]
) -> dict[str, Value]:
    try:
        return signature.bind(*arguments, **keywords).arguments
    except TypeError as error:
        raise Unsupported(f"invalid call arguments: {error}") from error
