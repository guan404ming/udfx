from collections.abc import Iterator, Sequence

from udfx.backends.base import Row
from udfx.ops import Op
from udfx.semantics.text import NATIVE_PADDING_LIMIT, PADDING_METHODS

OUTPUT_BYTES = 16 * 1024 * 1024


def allocation_batches(op: Op, rows: Sequence[Row]) -> Iterator[tuple[int, Sequence[Row]]]:
    if op.name.removeprefix(".") not in PADDING_METHODS:
        yield 0, rows
        return
    width = len(op.parameters) + 1
    start = size = 0
    for index, row in enumerate(rows):
        required = 0
        for offset in range(0, len(row), width):
            text, length = row[offset : offset + 2]
            if not row[offset + width - 1] or not isinstance(text, str):
                continue
            padding = (
                length if isinstance(length, int) and 0 < length <= NATIVE_PADDING_LIMIT else 0
            )
            required += 4 * (len(text) + max(len(text), padding))
        if size + required > OUTPUT_BYTES and start < index:
            yield start, rows[start:index]
            start, size = index, 0
        size += required
    if start < len(rows):
        yield start, rows[start:]
