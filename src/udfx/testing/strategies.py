from __future__ import annotations

import struct
from collections.abc import Mapping, Sequence
from datetime import datetime

from hypothesis import strategies as st

from udfx.testing.text_traps import REGEX_PATTERNS, STRING_TRAPS
from udfx.types import INT64_MAX, INT64_MIN, Kind, Type

__all__ = ["STRING_TRAPS"]

LIST_TRAPS = (("b", "a", "b"), ("\ue000", "\U00010000"), ("z", "Z", "ä", "a", ""))

INTEGER_LIST_TRAPS = (
    (2**53,),
    (2**53 + 1,),
    (2**52, 2**52),
    (2**52 + 1, -(2**52)),
    (INT64_MAX, 1, -1),
    (INT64_MIN, -1, 1),
    (INT64_MAX, 1),
    (INT64_MIN, -1),
)

FLOAT_LIST_TRAPS = (
    (-0.0, 0.0),
    (0.0, -0.0),
    (float("nan"), float("nan")),
    (1.0, float("-nan")),
    tuple(struct.unpack("!d", bytes.fromhex("7ff8000000000001"))[0] for _index in range(2)),
)

PREDICATE_TRAPS = ("Aa " * 10000,)
INTEGER_TEXT_TRAPS = ("\u0660_" * 4300 + "\u0660",)
CHARACTER_TRAPS = ("\ud7ff\ue000\U00010000😀\U0010ffff\n", r"\ud800\udfff\ud83d\ude00")
SPLIT_TRAPS = (r"\E", r"\Q\E", r"\.^$|?*+()[]{}", "[", ".")
CLUSTER_TRAPS = (
    *(
        text
        for point in "\u0d4e\u0e33\u0eb3\uff9e\uff9f\U000111c2\U000113d1\U0001193f\U00011941"
        "\U00011a3a\U00011a84\U00011d46\U00011f02\u1acf\U00011b61"
        "\U00016d63\U00016d67\U00016d68\U00016d69\U00016d6a"
        for text in (f"0 {point}", f"{point} 0")
    ),
    "\U00016d63\U00016d67",
    "\uac00\u1161",
    "\uac01\u11a8",
)

INT_TRAPS = (0, 1, -1, 2, -2, 3, 5, INT64_MIN, INT64_MAX, 10, 1000, 2**53, 2**53 + 1, -(2**53) - 1)
SLICE_TRAPS = (-65, -64, -10, -4, -1, 1, 4, 64, 65, 2**31 - 1, 2**31, 2**32 - 1, 2**32)

DATETIME_FORMATS = (
    "%Y",
    "%Y-%m",
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y%m%d",
    "%d/%m/%y",
    "%Y %j",
    "%H:%M:%S",
    "%f",
    "%y",
    "%%",
    "",
    "literal",
    "day %% %d.%m.%Y",
    "\x00%Y\n%m\t%d",
    "date 年 %Y",
    "%Y %Y",
    "%Y %y",
)

DATETIME_TEXT_TRAPS = (
    "0000-01-01",
    "24",
    "24-01",
    "24-01-01",
    "10000-01-01",
    "9999-12-31 24:00:00.000000",
    "0001-01-01",
    "9999-12-31",
    "1969-12-31",
    "2024-02-29",
    "2023-02-29",
    "2024-2-9",
    "２０２４-01-01",
    "2024-0٢-01",
    "2024-02-2０",
    "2024-02-０1",
    "2024-02- 9",
    "2024-01-01junk",
    "2024-01-01\n",
    "2024-02-29 12:34:56.123456",
    "2024-01-01 00:00:00.1",
    "2024-01-01 24:00:00.000000",
    "2024-01-01 00:00:60.000000",
    "01/02/68",
    "01/02/69",
    "2023 366",
    "2024 367",
)

DATETIME_TRAPS = (
    datetime.min,
    datetime.max,
    datetime(1970, 1, 1),
    datetime(1969, 12, 31, 23, 59, 59, 999999),
    datetime(2000, 2, 29, 12, 34, 56),
    datetime(1900, 3, 1),
    datetime(1600, 2, 29, 0, 0, 0, 999999),
    datetime(2024, 3, 10, 2, 30),
    datetime(2024, 11, 3, 1, 30),
)

FLOAT_TRAPS = (
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5,
    -0.5,
    -0.1,
    0.1,
    0.7,
    2.5,
    -7.0,
    2.0**53,
    2.0**63,
    -(2.0**63),
    float("nan"),
    float("inf"),
    float("-inf"),
    1e308,
    5e-324,
    709.782712893384,
    709.7827128933841,
    -745.1332191019411,
    -1.0000000000000002,
    -0.9999999999999999,
    1e16,
    1e-4,
    1e-5,
    1e-3,
    1e7,
    -float("nan"),
    900719925474100.5,
    -900719925474100.5,
    16.055,
    56294995342131.5,
)


def values(type: Type) -> st.SearchStrategy[object]:
    base = _base(type)
    return st.one_of(st.none(), base) if type.nullable else base


def constant_values(type: Type) -> st.SearchStrategy[object]:
    if type.kind is Kind.INT:
        return st.one_of(values(type), st.sampled_from(SLICE_TRAPS))
    if type.kind is Kind.STR:
        return st.one_of(values(type), st.sampled_from((*REGEX_PATTERNS, *DATETIME_FORMATS)))
    return values(type)


def extra_traps(type: Type) -> Sequence[object]:
    if type.kind is Kind.STR:
        return (
            *PREDICATE_TRAPS,
            *INTEGER_TEXT_TRAPS,
            *CHARACTER_TRAPS,
            *SPLIT_TRAPS,
            *CLUSTER_TRAPS,
            *DATETIME_TEXT_TRAPS,
        )
    if type.kind is Kind.DATETIME:
        return DATETIME_TRAPS
    if type.kind is Kind.FLOAT:
        return FLOAT_TRAPS
    if type.kind is Kind.INT:
        return INT_TRAPS
    return ()


def rows(schema: Mapping[str, Type]) -> st.SearchStrategy[tuple[object, ...]]:
    return st.tuples(*(values(type) for type in schema.values()))


def batches(
    schema: Mapping[str, Type], size: int = 40
) -> st.SearchStrategy[list[tuple[object, ...]]]:
    return st.lists(rows(schema), min_size=1, max_size=size)


def _base(type: Type) -> st.SearchStrategy[object]:
    match type.kind:
        case Kind.DATETIME:
            return st.one_of(st.sampled_from(DATETIME_TRAPS), st.datetimes())
        case Kind.STR:
            return st.one_of(st.sampled_from(STRING_TRAPS), st.text(max_size=12))
        case Kind.INT:
            return st.one_of(
                st.sampled_from(INT_TRAPS), st.integers(min_value=INT64_MIN, max_value=INT64_MAX)
            )
        case Kind.FLOAT:
            return st.one_of(st.sampled_from(FLOAT_TRAPS), st.floats())
        case Kind.BOOL:
            return st.booleans()
        case Kind.NONE:
            return st.none()
        case Kind.LIST:
            if type.item is None:
                raise TypeError("list type needs an item type")
            items = st.lists(values(type.item), max_size=5)
            if type.item.kind is Kind.INT:
                return st.one_of(st.sampled_from(INTEGER_LIST_TRAPS).map(list), items)
            if type.item.kind is Kind.FLOAT:
                return st.one_of(st.sampled_from(FLOAT_LIST_TRAPS).map(list), items)
            return (
                st.one_of(st.sampled_from(LIST_TRAPS).map(list), items)
                if type.item.kind is Kind.STR
                else items
            )
