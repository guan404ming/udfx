import json
from collections.abc import Callable
from typing import Any

from udfx.types import BOOL, FLOAT, INT, STR, Type, fits, list_of

SCALARS = (STR, INT, FLOAT, BOOL)
RESULTS = (*SCALARS, *(list_of(item) for type in SCALARS for item in (type, type.or_none())))


def encoded(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def document(value: str) -> str:
    return value


def get(value: str, key: object, default: str) -> str:
    return encoded(json.loads(value).get(key, json.loads(default)))


def item(value: str, key: Any) -> str:
    return encoded(json.loads(value)[key])


def items(value: str) -> list[str]:
    return [encoded(item) for item in json.loads(value)]


def length(value: str) -> int:
    return len(json.loads(value))


def truth(value: str) -> bool:
    return bool(json.loads(value))


def missing(value: str) -> bool:
    return json.loads(value) is None


def serialize(value: str, ensure_ascii: bool) -> str:
    return json.dumps(json.loads(value), ensure_ascii=ensure_ascii)


def convert(value: str, name: str) -> Any:
    return {"str": str, "int": int, "float": float, "bool": bool}[name](json.loads(value))


def result(value: str, type: Type) -> Any:
    output = json.loads(value)
    if not fits(output, type):
        raise OverflowError(f"udfx result {output!r} does not fit {type}")
    return output


def get_result(value: str, key: object, default: str, type: Type) -> Any:
    return result(get(value, key, default), type)


def raises(function: Callable[..., object], kind: type[Exception], *arguments: Any) -> bool:
    try:
        function(*arguments)
    except kind:
        return True
    except Exception:
        return False
    return False


def result_name(type: Type) -> str:
    return f"json.result.{type}"
