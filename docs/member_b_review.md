# Member B implementation and acceptance review

Member B's sorting/searching, ranking and statistics modules are implemented and
verified together as of checkpoint 7. This record maps their requirements to
code and tests. Benchmark measurements, notebook, report and video evidence
remain part of later project checkpoints.

## Requirement coverage

| Requirement | Implementation | Verification |
| --- | --- | --- |
| Stable and multi-key sorting | `sort_search.stable_sort`, `multi_key_sort` | Axis/shape checks, stable ties, directions, NaNs, extreme integers, strides in `test_sort_search.py` |
| Partial top-k | `sort_search.topk` | Partition cutoff, earliest boundary ties, k=0/n, NaNs last, indices and Python references |
| Educational selection | `sort_search.quickselect` | Explicit three-way partition loop, every-rank sorted references, duplicates, NaNs, read-only input and RNG preservation |
| Binary search | `sort_search.binary_search` | Left insertion/existence, empty/scalar/shaped queries, NaNs, mixed numeric precision and bisect references |
| Tie-aware ranking | `rank.rank` | Five methods, hand-calculated ties, independent counting, large integers, omitted NaNs and ordinal stability in `test_rank.py` |
| Percentiles | `rank.percentile` | Five interpolation modes, query shapes, NumPy references, missing slices, subnormal and finite-overflow regressions |
| Descriptive statistics | `stats.mean`, `median`, `std`, `var`, `minimum`, `maximum` | NumPy references, axes, NaN policies, ddof, shapes, precision and errors in `test_stats.py` |
| Quantiles | `stats.quantile` | Shared kernel, probabilities 0–1, axes, missing slices and nearest-index probability precision |
| Histograms | `stats.histogram` | Non-NaN counts, ranges/edges, final-edge inclusion, integer precision, empty samples and infinity policy |
| Streaming statistics | `stats.RunningStats` | Update/merge equivalence, population variance, empty/NaN input, finite-range checks and atomic failures in `test_running_stats.py` |
| Consistent cross-module behaviour | Public APIs above | Score analysis, missing/empty data, large integer centering, reordered rows and independent feature streams in `test_member_b_integration.py` |

Paths in the table refer to the `numcompute/` package or `tests/` folder.
Docstrings describe shapes, parameters, errors and complexity. Package imports
use Python and NumPy only; pytest is a development dependency.

## Running-statistics design

The retained state is count n, mean and centered moment M2. Batch moments use
deviations from a local reference value to avoid summing a huge common offset.
For two nonempty summaries A and B, let delta = mean_B - mean_A and n = n_A+n_B:

```text
mean = mean_A + delta * n_B / n
M2   = M2_A + M2_B + delta**2 * n_A * n_B / n
population_variance = M2 / n
```

The implementation scales the correction before squaring to reduce avoidable
overflow and checks finite results before committing state. Updating never
retains a batch. Merge snapshots do not share mutable state; self-merge repeats
observations. Empty updates/merges are no-ops, and an empty summary reports NaN
mean/variance without warnings.

Algorithm reference: [John D. Cook's presentation of Welford and parallel moment
formulas](https://www.johndcook.com/blog/skewness_kurtosis/). NumCompute exposes
population variance (M2/n); that reference's example returns sample variance.

## Numerical and API limits

- Statistics operate on real numeric/boolean arrays; unsupported object,
  complex and scalar sample inputs raise errors.
- RunningStats summarizes one-dimensional batches. Use one accumulator per
  matrix feature, as verified by the integration tests.
- Floating calculations use at least float64. Large integers can round on
  conversion; center in integer arithmetic first when preserving tiny offsets
  beyond floating precision is necessary. Ordering/selection retains raw dtypes.
- Floating moment arithmetic is finite-range and order-dependent. RunningStats
  rejects infinity and raises OverflowError atomically when finite moments
  cannot be represented. Its centered moment may overflow before the normalized
  variance would. Ordinary floating underflow/rounding still applies.
- Batch NaN omission/propagation follows documented NumPy reduction semantics.
  Streaming always omits NaNs. Empty summaries return NaN; empty extrema and
  out-of-range selection indices raise errors.
- This review claims no measured speedup or final rubric score.

## Verification and remaining submission evidence

```sh
python -m pytest tests --doctest-modules numcompute -W error
```

Observed result: **499 unit-test cases plus 18 docstring tests pass** (517 combined
cases), with unexpected warnings treated as errors. Seven integration cases
exercise workflows across all three Member B modules. There are 28 executable
docstring examples and seven README Python examples.

Counts cover currently implemented code, including shared validation and the
original package tests. A/C modules still need implementations and tests. Later
checkpoints must add reproducible benchmarks and measured tables, an executed
end-to-end notebook, concise testing/design report sections, and the group's
recorded demonstration. Actual contributions are recorded in the README and Git.
