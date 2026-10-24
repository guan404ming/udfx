# How it works

## 1. Compile your function

Pass a Python function, its input types (the schema), and a backend to `udfx.compile`.
udfx translates supported operations into an engine expression. For example,
`text.strip().lower()` becomes string operations for your chosen engine.
Compilation does not process your input rows.

## 2. Run it in your query

Check `translation.ok`, then pass `translation.expr` to your dataframe query.
Your engine executes the expression when you run the query. Native operations avoid
calling the Python function for each row they handle.

## 3. Preserve Python behavior

When native behavior can differ from Python, udfx adds checks to choose the correct path.
Rows that fail those checks run the original Python function as a fallback.
If udfx cannot compile the function, `translation.ok` is false and `translation.unsupported`
explains why.

Use [explain](explain.md) to inspect native execution and fallback conditions.

Next: [run your first query](quickstart.md).
