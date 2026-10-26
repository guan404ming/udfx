from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FrequencyPrimitives:
    literal: Callable[[object], Any]
    distinct: Callable[[Any], Any]
    transform: Callable[[Any, Callable[[Any], Any]], Any]
    count: Callable[[Any, Any], Any]
    position: Callable[[Any, Any], Any]
    record: Callable[[Any, Any, Any], Any]
    sort: Callable[[Any], Any]
    key: Callable[[Any], Any]


def ordered_frequencies(primitives: FrequencyPrimitives, values: Any) -> Any:
    ranked = primitives.transform(
        primitives.distinct(values),
        lambda key: primitives.record(
            primitives.literal(0) - primitives.count(values, key),
            primitives.position(values, key),
            key,
        ),
    )
    return primitives.transform(primitives.sort(ranked), primitives.key)
