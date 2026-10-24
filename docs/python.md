# Python support

udfx supports a subset of Python. Always check `translation.ok` before using its expression.

## Supported patterns

- Arithmetic, comparisons, conversions, and selected `math` and NumPy scalar functions.
- String operations, literal regex patterns, lists, string sets, and naive datetimes.
- Branches, exception handlers, helpers, and constant container lookups.
- Bounded loops and supported comprehensions or generator reductions.
- Row inputs through `row["name"]` or `row.name`.

## Unsupported patterns

Unbounded loops, mutation, I/O, timezone-aware datetimes, and unsupported calls can reject
compilation. Read `translation.unsupported` for the reason.

## Semantics

### Results and errors

Each row uses a native result where it matches Python, or runs the original function.
A raising row fails the query. This behavior is checked by tests, not a formal proof.

`backend.execute(run)` restores Python exceptions. DuckDB and Spark restore builtin exception
types through engine error messages.

### Inputs and captured values

Declare nullable columns with `.or_none()`. udfx does not validate input data against the schema.
Globals and closure constants are frozen at compile time for native and fallback paths.

Positional row access follows pandas 2.x behavior. Use named fields for portable code.

Next: [compile and types](api.md) or [inspect fallback conditions](explain.md).
