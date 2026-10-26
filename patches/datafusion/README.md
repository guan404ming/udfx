# DataFusion graph planning patches

These unreleased engine patches enable large native expression graphs, including the original
UDFBench `stem` function. They do not change udfx's installed dependencies automatically.

## Source versions

| Project | Version | Commit |
| --- | --- | --- |
| Apache DataFusion | 55.2.0 | `500744d5c25499055e3b2ca46b022bc425989203` |
| DataFusion Python | 55.0.0 | `a2882b9fcfe2a366ded6c94f6d50057b69bd8889` |

`core.patch` fixes nested lambda argument resolution, type propagation during coercion,
projection schemas, and constant folding with captured variables. It also repairs Unnest
functional dependencies that incorrectly preserve row uniqueness after list expansion.
`python.patch` removes the late binding-only analyzer and adds regression tests.

## Build

Use separate checkouts at the exact commits above. The tested build uses Rust 1.97.0,
CPython 3.12.13, Maturin 1.15.0, and Protobuf compiler 30.2.

```sh
git -C /path/to/datafusion apply /path/to/udfx/patches/datafusion/core.patch
git -C /path/to/datafusion-python apply /path/to/udfx/patches/datafusion/python.patch
cp /path/to/udfx/patches/datafusion/Cargo.lock /path/to/datafusion-python/Cargo.lock
python /path/to/udfx/patches/datafusion/link_core.py /path/to/datafusion /path/to/datafusion-python
```

The helper links the binding to local core crates. It does not download or install anything.
From the DataFusion Python checkout, build and install into a separate environment:

```sh
PROTOC=/path/to/protoc PYO3_PYTHON=/path/to/env/bin/python \
CARGO_BUILD_JOBS=16 CARGO_PROFILE_RELEASE_LTO=false \
CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 CARGO_PROFILE_RELEASE_OPT_LEVEL=2 \
/path/to/env/bin/maturin build --release --out /path/to/wheels
uv pip install --python /path/to/env/bin/python --no-deps --reinstall /path/to/wheels/datafusion-55.0.0-cp310-abi3-linux_x86_64.whl
```

## Validation

On Linux with CPython 3.12, the patched build passed:

- 10,478 DataFusion Rust tests and the SQL logic suite.
- 1,645 DataFusion Python tests, including doctests.
- 5,943 udfx full-tier tests and 990 corpus tests across four backends.
- The public UDFBench correctness tests across four backends and Spark ANSI off.

The original `stem` function completed through `udfx.compile()` on 32 public input rows and
61 edge cases. All 93 rows used native execution and matched Python. UDFBench compilation
coverage is 23/24 on this build; `jsonparse` remains unsupported. The released DataFusion
Python 55.0.0 wheel remains at 22/24.

These results establish correctness on the tested inputs. Large plans remain expensive to
prepare. They do not establish a throughput speedup or full-suite results on Python 3.13 or 3.14.

To run the upstream suites, initialize each checkout's test-data submodules first. From the
udfx checkout, use the patched environment and prepare UDFBench data explicitly:

```sh
PYTHONPATH=src:. /path/to/env/bin/python -m workloads.prepare --suites udfbench --rows 32
UDFX_TESTS=full PYTHONPATH=src:. /path/to/env/bin/python -m pytest
UDFX_TESTS=full PYTHONPATH=src:. /path/to/env/bin/python -m pytest -m corpus
UDFX_TESTS=full PYTHONPATH=src:. /path/to/env/bin/python -m pytest -m public
```

Set `PYSPARK_PYTHON` to the same interpreter when Spark workers need an explicit path.
