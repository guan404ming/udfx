# Development

## Set up and check

```sh
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
uv run pytest -m corpus
```

Spark needs Java 21. Use `UDFX_TESTS=full uv run pytest` for all engine configurations.
The default test run uses the fast tier.

After editing Spark runtime Java sources, run
`uv run python tools/build_spark_runtime.py` with JDK 21.
Commit the sources, bundled JAR, and runtime metadata together.
CI checks them with `uv run python tools/build_spark_runtime.py --check`.
Package installation does not require a Java compiler.

## Preview the docs

```sh
uv run --group docs zensical serve
uv run --group docs zensical build --strict
```

Open `http://localhost:8000`. The build writes static files to `site/`.
Dependencies use the existing `pyproject.toml` and `uv.lock`.

Keep one topic per page. Aim for 150–250 words and stay below 350 words, including examples.
Put detailed measurement records in linked reports. Keep `zensical.toml` navigation in sync.

Use one H1 per page. Use H2 for sections and H3 for subsections. Do not skip heading levels.

## Publish the docs

Pushes to `main` publish the site to [GitHub Pages](https://guan404ming.github.io/udfx/).
Pull requests build the docs without deploying them.

## Make a change

Keep Python semantics in shared modules and engine calls in each backend.
Follow the [compiler structure](architecture.md) and the repository's
[contribution rules](https://github.com/guan404ming/udfx/blob/main/CONTRIBUTING.md).

Add trap values when a change fixes a semantic gap. Keep each code file at 250 lines or fewer.
