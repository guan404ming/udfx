from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import cast

INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1


class Kind(StrEnum):
    STR = "str"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    NONE = "none"
    LIST = "list"


@dataclass(frozen=True)
class Type:
    kind: Kind
    nullable: bool = False
    item: Type | None = None

    def __str__(self) -> str:
        base = f"list[{self.item}]" if self.kind is Kind.LIST else self.kind.value
        return f"{base}?" if self.nullable and self.kind is not Kind.NONE else base

    def or_none(self) -> Type:
        return Type(self.kind, True, self.item)

    def strict(self) -> Type:
        return Type(self.kind, False, self.item)


STR = Type(Kind.STR)
INT = Type(Kind.INT)
FLOAT = Type(Kind.FLOAT)
BOOL = Type(Kind.BOOL)
NONE = Type(Kind.NONE, nullable=True)


def list_of(item: Type) -> Type:
    return Type(Kind.LIST, item=item)


def type_of(value: object) -> Type | None:
    match value:
        case None:
            return NONE
        case bool():
            return BOOL
        case int():
            return INT
        case float():
            return FLOAT
        case str():
            return STR
        case _:
            return None


def unify(a: Type, b: Type) -> Type | None:
    if a.kind is Kind.NONE:
        return b.or_none()
    if b.kind is Kind.NONE:
        return a.or_none()
    if a.strict() != b.strict():
        return None
    return a.or_none() if a.nullable or b.nullable else a


def fits(value: object, type: Type) -> bool:
    if value is None:
        return type.nullable
    match type.kind:
        case Kind.INT:
            return isinstance(value, int) and INT64_MIN <= value <= INT64_MAX
        case Kind.LIST if type.item is not None and isinstance(value, list):
            item = type.item
            return all(fits(element, item) for element in cast("list[object]", value))
        case _:
            return True
