# Python support

udfx supports a subset of Python. Always check `translation.ok` before using its expression.

## Supported patterns

- Arithmetic, comparisons, conversions, and selected `math` and NumPy scalar functions.
- String operations, literal regex patterns, lists, string sets, and naive datetimes.
- `json.dumps` for strings, nulls, and nested string lists, with default formatting.
- `json.loads`, nested JSON indexing, dictionary `get`, type checks, and scalar conversions.
- Branches, exception handlers, helpers, and constant container lookups.
- Local imports of loaded modules, constant dictionary merges, and constant `frozenset` values.
- `heapq.nlargest` over a string `Counter`, with `counter.get` as its key.
- Bounded loops and supported comprehensions or generator reductions.
- Row inputs through `row["name"]` or `row.name`.

## Unsupported patterns

Unbounded loops, mutation, I/O, timezone-aware datetimes, and unsupported calls can reject
compilation. Read `translation.unsupported` for the reason.

Large helper pipelines can compile as shared expression graphs on Polars, DuckDB, and Spark.
DataFusion needs [engine fixes](backends/datafusion.md#large-functions) for these large graphs.
Successful compilation does not guarantee a speedup.

## Semantics

### Results and errors

Each row uses a native result where it matches Python, or runs the original function.
A raising row fails the query. This behavior is checked by tests, not a formal proof.

JSON output supports `ensure_ascii=False`. With the default `ensure_ascii=True`, non-ASCII
strings use Python fallback to preserve Unicode escapes.

DuckDB and Spark also lower JSON object lookups, array indexing, and typed scalar results
natively. Keys can come from columns. Other engines retain the native string and string-array
paths and use Python for dynamic objects.

Rows use fallback when an engine cannot preserve Python semantics. This includes duplicate
keys, some numeric values, and unsupported JSON paths. Spark currently keeps JSON documents
with floating-point numbers on Python. Set a [result type](api.md#json-results) for a dynamic
JSON return. Native coverage does not guarantee a speedup.

`backend.execute(run)` restores Python exceptions. DuckDB and Spark restore builtin exception
types and `JSONDecodeError` through engine error messages.

### Inputs and captured values

Declare nullable columns with `.or_none()`. udfx does not validate input data against the schema.
Globals and closure constants are frozen at compile time for native and fallback paths.
Supported local imports use the same snapshot, including in Spark workers.

Positional row access follows pandas 2.x behavior. Use named fields for portable code.

Next: [compile and types](api.md) or [inspect fallback conditions](explain.md).
