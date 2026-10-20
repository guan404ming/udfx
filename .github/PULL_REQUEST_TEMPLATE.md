## Summary

## Checklist

- [ ] `uv run ruff check . && uv run ruff format --check .` passes
- [ ] `uv run pyright` passes
- [ ] `uv run pytest` passes, including conformance tests
- [ ] The change adds no comments or docstrings
- [ ] If I found a semantic gap, I added trap values to `src/udfx/testing/strategies.py`
