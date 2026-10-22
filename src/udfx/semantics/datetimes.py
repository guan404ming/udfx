from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from udfx.errors import Unsupported

ATTRIBUTES = ("year", "month", "day", "hour", "minute", "second", "microsecond")
DIRECTIVES = {
    "Y": ("year", len(datetime.min.strftime("%Y"))),
    "m": ("month", 2),
    "d": ("day", 2),
    "H": ("hour", 2),
    "M": ("minute", 2),
    "S": ("second", 2),
    "f": ("microsecond", 6),
    "j": ("doy", 3),
    "y": ("year", 2),
}
EPOCH_ORDINAL = datetime(1970, 1, 1).toordinal()


MICROSECONDS_PER_SECOND = 1000000
SECONDS_PER_DAY = 86400
MICROSECONDS_PER_DAY = SECONDS_PER_DAY * MICROSECONDS_PER_SECOND
EPOCH = datetime(1970, 1, 1)


def format_parts(format: str) -> list[tuple[str, bool]]:
    parts: list[tuple[str, bool]] = []
    index = 0
    while index < len(format):
        end = format.find("%", index)
        if end < 0:
            parts.append((format[index:], False))
            break
        if end > index:
            parts.append((format[index:end], False))
        directive = format[end + 1 : end + 2]
        if directive == "%":
            parts.append(("%", False))
        elif directive in DIRECTIVES:
            parts.append((directive, True))
        else:
            raise Unsupported(f"unsupported datetime directive %{directive}")
        index = end + 2
    return parts


def parse_parts(format: str) -> list[tuple[str, bool]]:
    parts = format_parts(format)
    directives = [part for part, directive in parts if directive]
    if len(set(directives)) != len(directives):
        raise Unsupported("repeated strptime directives")
    return parts


def strptime_parses(text: str, format: str) -> bool:
    try:
        datetime.strptime(text, format)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class DatetimePrimitives:
    literal: Callable[[Any], Any]
    part: Callable[[Any, str], Any]
    pad: Callable[[Any, int], Any]
    concatenate: Callable[[Sequence[Any]], Any]
    extract: Callable[[Any, str, int], Any]
    parse: Callable[[Any], Any]
    to_integer: Callable[[Any], Any]
    select: Callable[[Any, Any, Any], Any]
    is_null: Callable[[Any], Any]


def datetime_field(primitives: DatetimePrimitives, value: Any, name: str) -> Any:
    if name in {"weekday", "isoweekday", "toordinal"}:
        ordinal = primitives.part(value, "epoch_day") + primitives.literal(EPOCH_ORDINAL)
        if name == "toordinal":
            return ordinal
        weekday = (ordinal - primitives.literal(1)) % primitives.literal(7)
        return weekday + primitives.literal(1) if name == "isoweekday" else weekday
    return primitives.part(value, name)


def format_datetime(primitives: DatetimePrimitives, value: Any, format: Any) -> tuple[Any, Any]:
    if not isinstance(format, str):
        return primitives.literal(""), primitives.literal(False)
    try:
        parts = format_parts(format)
    except Unsupported:
        return primitives.literal(""), primitives.literal(False)
    result: list[Any] = []
    for part, directive in parts:
        if not directive:
            result.append(primitives.literal(part))
            continue
        field, width = DIRECTIVES[part]
        number = primitives.part(value, field)
        if part == "y":
            number = number % primitives.literal(100)
        result.append(primitives.pad(number, width))
    return primitives.concatenate(result or [primitives.literal("")]), primitives.literal(True)


def parse_datetime(primitives: DatetimePrimitives, text: Any, format: Any) -> tuple[Any, Any]:
    fallback = primitives.parse(primitives.literal("")), primitives.literal(False)
    if not isinstance(format, str):
        return fallback
    try:
        parts = parse_parts(format)
    except Unsupported:
        return fallback
    directives = [part for part, directive in parts if directive]
    if "j" in directives:
        return fallback
    if "y" in directives and "Y" in directives:
        return fallback
    pattern = (
        "\\A"
        + "".join(
            f"([0-9]{{{4 if part == 'Y' else DIRECTIVES[part][1]}}})"
            if directive
            else "".join("\\" + point if point in r"\.^$|?*+()[]{}" else point for point in part)
            for part, directive in parts
        )
        + "\\z"
    )
    fields = {
        part: primitives.extract(text, pattern, index) for index, part in enumerate(directives, 1)
    }
    year = fields.get("Y", primitives.literal("1900"))
    if "y" in fields:
        short_year = primitives.to_integer(fields["y"])
        century = primitives.select(
            short_year <= primitives.literal(68),
            primitives.literal(2000),
            primitives.literal(1900),
        )
        year = primitives.pad(short_year + century, 4)
    source = primitives.concatenate(
        [
            year,
            primitives.literal("-"),
            fields.get("m", primitives.literal("01")),
            primitives.literal("-"),
            fields.get("d", primitives.literal("01")),
            primitives.literal(" "),
            fields.get("H", primitives.literal("00")),
            primitives.literal(":"),
            fields.get("M", primitives.literal("00")),
            primitives.literal(":"),
            fields.get("S", primitives.literal("00")),
            primitives.literal("."),
            fields.get("f", primitives.literal("000000")),
        ]
    )
    parsed = primitives.parse(source)
    year = primitives.part(parsed, "year")
    valid = (year >= primitives.literal(1)) & (year <= primitives.literal(9999))
    parsed = primitives.select(valid, parsed, fallback[0])
    rendered, _ = format_datetime(primitives, parsed, format)
    return parsed, ~primitives.is_null(parsed) & (rendered == text)


def duration_microseconds(value: timedelta) -> int:
    return (
        value.days * MICROSECONDS_PER_DAY
        + value.seconds * MICROSECONDS_PER_SECOND
        + value.microseconds
    )


MINIMUM_TIMESTAMP = duration_microseconds(datetime.min - EPOCH)
MAXIMUM_TIMESTAMP = duration_microseconds(datetime.max - EPOCH)
MAXIMUM_SHIFT = MAXIMUM_TIMESTAMP - MINIMUM_TIMESTAMP + 1


def datetime_difference(left: datetime, right: datetime) -> int:
    return duration_microseconds(left - right)


def datetime_shift(value: datetime, microseconds: int) -> datetime:
    return value + timedelta(microseconds=microseconds)


def datetime_shift_overflows(value: datetime, microseconds: int) -> bool:
    timestamp = duration_microseconds(value - EPOCH)
    return (
        microseconds < MINIMUM_TIMESTAMP - timestamp or microseconds > MAXIMUM_TIMESTAMP - timestamp
    )


@dataclass(frozen=True)
class DatetimeArithmeticPrimitives:
    literal: Callable[[Any], Any]
    to_microseconds: Callable[[Any], Any]
    from_microseconds: Callable[[Any], Any]
    select: Callable[[Any, Any, Any], Any]


def shift_overflows(primitives: DatetimeArithmeticPrimitives, timestamp: Any, offset: Any) -> Any:
    minimum = primitives.literal(MINIMUM_TIMESTAMP) - timestamp
    maximum = primitives.literal(MAXIMUM_TIMESTAMP) - timestamp
    return (offset < minimum) | (offset > maximum)


def shift_datetime(primitives: DatetimeArithmeticPrimitives, value: Any, offset: Any) -> Any:
    timestamp = primitives.to_microseconds(value)
    safe = primitives.select(
        shift_overflows(primitives, timestamp, offset), primitives.literal(0), offset
    )
    return primitives.from_microseconds(timestamp + safe)
