# Supported Patterns

udfx supports a subset of Python. Always check `translation.ok` before using its expression.

## Supported

| Category | Supported patterns |
| --- | --- |
| Numbers | Arithmetic, comparisons, conversions, and selected `math` and NumPy scalar functions |
| Values | String operations, literal regex patterns, lists, string sets, dates, and naive datetimes |
| JSON encoding | `json.dumps` for strings, nulls, and nested string lists, with default formatting |
| JSON decoding | `json.loads`, nested JSON indexing, dictionary `get`, type checks, and scalar conversions |
| Control flow | Branches, exception handlers, helpers, and constant container lookups |
| Bindings | Local imports of loaded modules, constant dictionary merges, and constant `frozenset` values |
| Ranking | `heapq.nlargest` over a string `Counter`, with `counter.get` as its key |
| Iteration | Bounded loops and supported comprehensions or generator reductions |
| Row inputs | `row["name"]` or `row.name` |

## Unsupported

These reject compilation. Read `translation.unsupported` for the reason.

- Unbounded loops, mutation, and I/O
- Timezone-aware datetimes
- Unsupported calls

Large helper pipelines compile as shared expression graphs:

| Engine | Large pipelines |
| --- | --- |
| Polars, DuckDB | Shared graphs |
| Apache Spark | Shared graphs, or Python |
| Apache DataFusion | Needs engine fixes. See [Large functions](backends/datafusion.md#large-functions) |

Successful compilation does not guarantee a speedup.

## Semantics

### Results

- Each row uses the native result where it matches Python. Other rows run the original function.
- A raising row fails the query. Tests check this behavior. It is not a formal proof.
- `backend.execute(run)` restores Python exceptions. DuckDB and Apache Spark restore builtin
  exception types and `JSONDecodeError` from engine error messages.

### JSON

| JSON feature | DuckDB, Apache Spark | Polars, Apache DataFusion |
| --- | --- | --- |
| Strings and string arrays | Native | Native |
| Object lookups, array indexing, typed scalars | Native, keys can come from columns | Python |

- `ensure_ascii=False` works. Some backends use Python when native encoding cannot keep Python
  Unicode escapes.
- Duplicate keys, some numeric values, and unsupported paths use Python.
  Apache Spark keeps documents with floating-point numbers on Python.
- Dynamic JSON returns need a result type. See [udfx.compile > JSON results](compile.md#json-results).

### Captured Inputs

- Declare nullable columns with `.or_none()`. udfx does not validate input data against the schema.
- Globals, closure constants, and supported local imports are frozen at compile time.
  Native, fallback, and Apache Spark worker paths use the same snapshot.
- Positional row access follows pandas 2.x. Use named fields for portable code.

Compile a function with [udfx.compile](compile.md) or inspect its execution paths with [udfx.explain](explain.md).
