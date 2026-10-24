# DuckDB

## Install

Install `udfx[duckdb]`.

## Run a query

Build and execute on the same connection object.

```python
import duckdb
import udfx
from udfx.backends.duckdb import DuckDBBackend
from udfx.types import STR


def prefix(text: str) -> str:
    return text[:3]


connection = duckdb.connect()
backend = DuckDBBackend(connection)
translation = udfx.compile(prefix, {"text": STR}, backend)
assert translation.ok, translation.unsupported
frame = connection.sql("SELECT * FROM (VALUES ('hello'), ('world')) AS inputs(text)")
print(backend.execute(lambda: frame.select(translation.expr.alias("prefix")).fetchall()))
connection.close()
```

The result contains `hel` and `wor`.

Each thread must use its own connection or cursor.
