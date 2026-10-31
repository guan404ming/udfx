# Supported Patterns

udfx supports a subset of Python. Always check `translation.ok` before using its expression.

## Supported

| Category | Supported patterns |
| --- | --- |
| Numbers | Arithmetic, comparisons, conversions, and selected `math` and NumPy scalar functions |
| Values | String operations, literal regex patterns, lists, string sets, and naive datetimes |
| JSON encoding | `json.dumps` for strings, nulls, and nested string lists, with default formatting |
| JSON decoding | `json.loads`, nested JSON indexing, dictionary `get`, type checks, and scalar conversions |
| Control flow | Branches, exception handlers, helpers, and constant container lookups |
| Bindings | Local imports of loaded modules, constant dictionary merges, and constant `frozenset` values |
| Ranking | `heapq.nlargest` over a string `Counter`, with `counter.get` as its key |
| Iteration | Bounded loops and supported comprehensions or generator reductions |
| Row inputs | `row["name"]` or `row.name` |

## Unsupported

Unbounded loops, mutation, I/O, timezone-aware datetimes, and unsupported calls can reject
compilation. Read `translation.unsupported` for the reason.

Large helper pipelines can compile as shared expression graphs on Polars, DuckDB, and Apache Spark.
Apache DataFusion needs engine fixes for these graphs. See
[Apache DataFusion > Large functions](backends/datafusion.md#large-functions).
Successful compilation does not guarantee a speedup.

## Semantics

### Results

Each row uses a native result where it matches Python, or runs the original function.
A raising row fails the query. This behavior is checked by tests, not a formal proof.

JSON output supports `ensure_ascii=False`. Some backends use Python fallback when native
encoding cannot preserve Python Unicode escapes.

DuckDB and Apache Spark also lower JSON object lookups, array indexing, and typed scalar results
natively. Keys can come from columns. Other engines retain the native string and string-array
paths and use Python for dynamic objects.

Rows use fallback when an engine cannot preserve Python semantics. This includes duplicate
keys, some numeric values, and unsupported JSON paths. Apache Spark currently keeps JSON documents
with floating-point numbers on Python. Dynamic JSON returns need a result type.
See [udfx.compile > JSON results](compile.md#json-results).
Native coverage does not guarantee a speedup.

`backend.execute(run)` restores Python exceptions. DuckDB and Apache Spark restore builtin exception
types and `JSONDecodeError` through engine error messages.

### Captured Inputs

Declare nullable columns with `.or_none()`. udfx does not validate input data against the schema.
Globals and closure constants are frozen at compile time for native and fallback paths.
Supported local imports use the same snapshot, including in Apache Spark workers.

Positional row access follows pandas 2.x behavior. Use named fields for portable code.

Compile a function with [udfx.compile](compile.md) or inspect its execution paths with [udfx.explain](explain.md).
