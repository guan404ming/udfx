from __future__ import annotations

from collections.abc import Mapping

from hypothesis import strategies as st

from udfx.types import INT64_MAX, INT64_MIN, Kind, Type

STRING_TRAPS = (
    "",
    " ",
    "\t",
    " ",
    " ",
    "İ",
    "ß",
    "ǅ",
    "١٢",
    "\U0001d7d9_\U0001d7da",
    "\x85١٢\u3000",
    "1_2",
    " 12 ",
    "+7",
    "-0",
    "::",
    "a::b::c",
    "2020-01-02",
    "2020/01",
    "[]",
    '["b a", "c"]',
    "123456",
    "\x1c",
    " \x1c a \x85 ",
    "aa",
    "aaa",
    "$0",
    "a😀b",
    "\r\n",
    "a\u0301b",
    "1__2",
    "_1",
    "1_0.0_1e1_0",
    "1.5",
    "12.0",
    "0x1A",
    "0b1",
    "0o7",
    ".5",
    "5.",
    "1e5",
    "1e",
    "-Infinity",
    "nan",
    "\x1c12",
    "１２",
    "9223372036854775808",
    "AΣ\u0897b",
    "A\u0295Σ",
    "+nan",
    "ΑΣ",
    "  a\t b  c \n",
    "a_b_c_d_e",
    "ababa",
    "aba",
    "👩\u200d👩\u200d👧\u200d👦",
    "🇹🇼",
    "\u1100\u1161\u11a8",
)

INT_TRAPS = (0, 1, -1, 2, -2, INT64_MIN, INT64_MAX, 10, 1000, 2**53, 2**53 + 1, -(2**53) - 1)
SLICE_TRAPS = (-10, 2**31 - 1, 2**31, 2**32 - 1, 2**32)

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


def rows(schema: Mapping[str, Type]) -> st.SearchStrategy[tuple[object, ...]]:
    return st.tuples(*(values(type) for type in schema.values()))


def batches(
    schema: Mapping[str, Type], size: int = 40
) -> st.SearchStrategy[list[tuple[object, ...]]]:
    return st.lists(rows(schema), min_size=1, max_size=size)


def _base(type: Type) -> st.SearchStrategy[object]:
    match type.kind:
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
            return st.lists(values(type.item), max_size=5)
