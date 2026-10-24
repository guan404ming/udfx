# Compiler structure

udfx compiles a Python function and a declared schema into one engine expression.

## Compilation and execution

1. **Frontend** decodes bytecode, follows branches, and traces values and faults into an IR.
2. **Operations and semantics** define Python behavior, exceptions, and shared simplifications.
3. **Lowering** applies shared rules and builds expressions through a backend.
4. **Execution** uses native results where they match Python and runs fallback rows in Python.

Backends can raise Python-equivalent errors natively when supported. A raising row fails the query.

## Source layout

| Directory | Responsibility |
| --- | --- |
| `src/udfx/frontend/` | Python tracing; independent of engines |
| `src/udfx/ops/` | Operation contracts and fault predicates |
| `src/udfx/semantics/` | Shared Python facts and policies |
| `src/udfx/backends/` | Shared lowering and four engine implementations |
| `src/udfx/testing/` | Python oracle and equivalence checks |

See the [source layout and contracts](https://github.com/guan404ming/udfx/blob/main/AGENTS.md)
before adding an operation or backend.
