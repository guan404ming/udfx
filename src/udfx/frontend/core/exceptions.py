from __future__ import annotations

from udfx.errors import Unsupported
from udfx.frontend.core.values import Item, Opaque, describe


def exception_classes(item: Item) -> tuple[type[BaseException], ...]:
    candidates = item.value if isinstance(item, Opaque) else None
    members = candidates if isinstance(candidates, tuple) else (candidates,)
    classes = tuple(
        member
        for member in members
        if isinstance(member, type) and issubclass(member, BaseException)
    )
    if not classes or len(classes) != len(members):
        raise Unsupported(f"exception match against {describe(item)}")
    return classes


def exception_kind(item: Item) -> type[Exception]:
    match item:
        case Opaque(value=type() as kind) if issubclass(kind, Exception):
            return kind
        case _:
            raise Unsupported(f"expected an exception, got {describe(item)}")
