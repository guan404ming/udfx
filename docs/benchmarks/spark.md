# Spark benchmark record

udfx can remove Python UDF execution, but native expressions can also be slower. Measure the
query and data distribution you intend to deploy. A correct fallback does not guarantee a
speedup.

## Recorded Spark development run

The [measurements](spark.csv) contain all 165 cases from each of three complete
sweeps. Each case used one million rows and the best of three timings. Each sweep measured
both pickled and Arrow Python UDFs and selected the faster baseline. Output values were
checked before timing. No cases were skipped.

The machine used an AMD Ryzen 9 9950X with 32 logical CPUs, CPython 3.12.13, Spark 4.2.0,
and Java 21. Spark ran locally; the GPU was not used. These are results from a development
snapshot before the release-packaging and explain changes, not results for a published tag.
The benchmark used `local[*]`, an 8 GB driver, UTC, and `TIMESTAMP_NTZ`. It timed a hash-and-sum
action over cached input. Input creation, compilation, and result checks were outside the timed
action. These timings therefore do not represent end-to-end ingestion latency.

| Metric | Result |
|---|---:|
| Cases slower than the selected Python UDF baseline | 16 / 165 |
| Geometric mean speedup | 1.425827x |
| Deduplicated geometric mean speedup | 1.423287x |
| Distinct expression groups | 153 |
| Cases with no Python UDF in the plan | 154 / 165 |

For each case, divide Python time by udfx time in each sweep, then take the median of those
three ratios. The table uses those medians. For the deduplicated mean, group cases by their
normalized expression fingerprint, compute each group's geometric mean, and give each group
equal weight. The CSV stores the SHA-256 digest of each fingerprint.

Individual sweeps had 18, 17, and 16 slower cases. The aggregate does not imply that every
case improves: `total_numbers` was about 0.211x. These fixed benchmark inputs do not measure
coverage of arbitrary Python UDFs, first-call compile cost, or distributed production jobs.

The measured source and benchmark manifest digest was
`275cc84ee1ca55fa158a3a54437442ddf35401cc3ab80f0117e69629bfb7ed42`.

## Reproduce the archived method

```sh
git checkout 64a5cfe
uv sync --locked
uv run python -m bench.speed --engines spark --rows 1000000 --repeats 3
```

Use a separate checkout for this archived runner. The current public-data runner is described
in [benchmark methods](../benchmarking.md).
Run the archived benchmark three times on an otherwise idle machine to compare with the method above.
Do not reuse a baseline cache when comparing revisions. Keep engine versions, input sizes,
session settings, and hardware fixed. Report slower cases together with the geometric mean.

To inspect one function before benchmarking, use [udfx.explain](../explain.md). Its compile time
and fallback status describe compilation. They are not query-runtime measurements.
