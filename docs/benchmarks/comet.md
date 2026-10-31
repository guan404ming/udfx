# Apache DataFusion Comet benchmark record

Measured on 2026-10-24. The comparison uses the same udfx expressions with Apache Spark alone,
released Apache DataFusion Comet, and an experimental Apache DataFusion Comet patch. The patch improves expression coverage
but does not improve the geometric mean of these workloads. It is not part of the udfx package.

## Method

The Linux host uses a Ryzen 9 9950X, Apache Spark 4.1.3, Apache DataFusion Comet 1.1.0, Python 3.12.13,
PyArrow 26.0.0, and Java 21. Each application uses `local[8]`, an 8 GiB driver,
4 GiB off-heap memory, UTC, and ANSI mode. Apache DataFusion Comet uses strict floating point and
`sparkToColumnar`. Incompatible expression opt-ins remain disabled.
A pre-existing vLLM service remains active at approximately 22% of one CPU core.
There are no concurrent test or build jobs during timing. Apache Spark does not use the GPU.

The timing set contains 24 cases chosen before the initial Apache DataFusion Comet experiment. Each has
1,000,000 rows in eight Apache Parquet files. Each case runs against Apache Spark cache and warm Apache Parquet.
Every timed action creates a fresh `sum(hash(expression))` query and collects the result.
There is one warmup and five samples per case and input layout. Setup, compilation of the
Python function, data generation, cache population, and correctness checks are outside timing.
Apache Spark query construction, planning, execution, and result collection are inside timing.

Application order is Apache Spark, released Apache DataFusion Comet, patched Apache DataFusion Comet, patched Apache DataFusion Comet, released Apache DataFusion Comet,
Apache Spark. Each case uses the median of five samples, followed by the median of its two application
medians. The summary is the geometric mean of the 24 case ratios. A ratio above 1 is faster
than Apache Spark with udfx. This is not a comparison against Python UDF execution.
These runs do not establish cold I/O, distributed-cluster, join, or confidence-interval results.

## Results

| Input | Released Apache DataFusion Comet / Apache Spark speedup | Patched Apache DataFusion Comet / Apache Spark speedup | Patched cases above 1 |
| --- | ---: | ---: | ---: |
| Apache Spark cache | 0.898x | 0.855x | 4/24 |
| Warm Apache Parquet | 1.018x | 1.005x | 13/24 |

The patched cache result takes about 17% more time than Apache Spark. The Apache Parquet result is effectively
flat: the two application ratios are 0.994x and 1.018x. The small positive aggregate ratio is
not evidence of a reliable general speedup.

There are useful individual gains on Apache Parquet: `upper` is 1.33x, `int_arithmetic` is 1.26x,
and `lower_strip` is 1.43x. There are also regressions: `jaccard` is 0.46x and
`datetime_age_days` is 0.82x. See [all per-case measurements](comet.csv).

## Why more Apache DataFusion Comet coverage is not faster

1. Small actions have a substantial fixed cost. In the scan/hash control probe, 1 million rows
   take roughly 24–54 ms before the larger expression is included. More native operators
   cannot remove Apache Spark scheduling and query setup.
2. Input conversion matters. An 8-million-row cached float query records 39.8 ms of summed
   task conversion time in `CometSparkColumnarToColumnar`. The cached datetime query records
   155 ms in `CometSparkRowToColumnar`. These are summed task metrics, not wall-clock additions.
   The corresponding Apache Parquet plans read directly through `CometNativeScan`.
3. Apache DataFusion Comet projections can call JVM codegen through Apache Arrow and JNI. Moving a projection into
   Apache DataFusion Comet does not make each expression a Rust kernel. The patched datetime path executes in
   Apache DataFusion Comet but is slower than released Apache DataFusion Comet's Apache Spark projection fallback on the same new udfx code.
4. Removing a fallback changes the surrounding execution plan. For example, admitting
   `raise_error` lets the `jaccard` projection move into Apache DataFusion Comet, but its measured time more than
   doubles. That coverage gain does not meet a performance regression gate.

The size probe also shows real opportunities. On the original udfx code, warm Apache Parquet at
8 million rows gives Apache DataFusion Comet 2.14x for `float_arithmetic` and 3.20x for `lower_strip`.
That probe uses five samples in one application per engine. Its 8-million-row input repeats
the same 1-million-row dataset in 64 files. These are selected scaling examples, not suite means.

The official [plan guide](https://datafusion.apache.org/comet/user-guide/1.1/understanding-comet-plans.html)
and [conversion guide](https://datafusion.apache.org/comet/user-guide/1.1/tuning/transitions.html)
describe these execution boundaries. The measurements above come from the captured plans and runs.

## udfx changes

Datetime arithmetic now uses timestamp-without-time-zone intervals. It reconstructs microseconds
from integer seconds and the exact fractional-second component. This avoids two epoch conversions
and a large-decimal conversion of the complete duration. All backends use an explicit difference
primitive in the existing datetime arithmetic record.

This also fixes a real DST error. With `America/Los_Angeles`, Python expects 77400 for
`(datetime(1970, 1, 1) - datetime(2024, 3, 10, 2, 30)).seconds`. The old implementation returns
73800 because Apache Spark's `timestamp_diff` implicitly converts the naive input through a zoned
timestamp. The regression fails on the original source snapshot and passes after this change.
Datetime shifts use the same naive interval conversion and retain their overflow guards.

A paired 8-million-row probe compares the old expression with the new interval calculation
inside the same Apache Spark application. Cached time falls from 267.6 to 223.4 ms (1.20x), and warm
Apache Parquet from 327.3 to 284.4 ms (1.15x). This is a focused expression probe, not a new full Apache Spark KPI.

## Compatibility

All 165 cases pass output comparison on 4096 rows per case in each of three forms: pickled
Python UDF, Apache Arrow Python UDF, and udfx. Both released and patched Apache DataFusion Comet pass this check.

| Plan coverage on the new udfx code | Released Apache DataFusion Comet | Patched Apache DataFusion Comet |
| --- | ---: | ---: |
| Cases with `CometProject` | 94/165 | 152/165 |
| Cases with Python execution | 11/165 | 11/165 |

The earlier udfx code had 96 Apache DataFusion Comet projections. Correcting datetime arithmetic changes the
expressions sent to Apache DataFusion Comet, so the released engine now falls back for those two projections.
The experimental patch adds `RaiseError` and ANSI interval-second extraction through the existing
codegen dispatcher. It adds scalar and array `NullType` support and rejects nested null fields
that Apache Arrow's struct writer cannot allocate. Python error guards remain intact.

The patch also fixes task-cache cleanup. Registering a listener inside `computeIfAbsent` lets a
completed task remove the same map entry from inside that computation. The original code fails
with `TaskCompletionListenerException: Recursive update`. The patch inserts the entry first and
registers cleanup after insertion. The regression test fails before the patch and passes after it.

The two remaining non-Python projection fallbacks are `datetime_shift_forward` and
`datetime_shift_backward`: Apache DataFusion Comet does not support their `timestampaddinterval` expression.
The other 11 cases retain Python execution. These are normal engine fallbacks, not failed results.

The patch is based on Apache DataFusion Comet tag `1.1.0`, commit
`992c806a7e38c2e88bd018aa5774164b0850e1fa`. Only JVM code changes. The build reuses the exact released
Linux amd64 native library. No Rust code or protobuf changes are required.
The tested jar SHA-256 is
`0c82d77d6724d5b6c6b5a5b80414205daa0d3f74f61cc112af5c84f6b255eebc`.
The patch has not been submitted or released upstream. Its performance regressions make it an
experimental compatibility patch, not a recommended default deployment.

## Validation

The local Python 3.12 full tier passes 5530 tests with 123 skips. It uses Apache Spark 4.2.0, Polars 2.0.0,
DuckDB 1.5.6, Apache DataFusion 55.0.0, and PyArrow 26.0.0. Ruff and Pyright pass.
The separate full corpus run passes 990 tests across the installed backends.

On Apache Spark 4.1.3 with the Apache DataFusion Comet error, null, and task-cache changes, 1031 udfx tests pass with
20 skips. After interval-second dispatch is added, another 96 datetime tests pass, including
ANSI on/off and the timezone regression. The final Apache DataFusion Comet JVM suites pass 69 tests, including
generated-code checks, lazy error branches, null output, interval seconds, and completed-task cleanup.

This does not run the full Apache DataFusion Comet project suite, Apache Spark's upstream SQL CI shard, or a new native
Rust build. It does not establish compatibility with other Apache Spark/Apache DataFusion Comet version pairs or clusters.
