# Development record

## Baseline

The repository was inspected and fetched from `origin` before implementation.
The baseline is commit `9eb8c77` on `main`, with a clean working tree and two
passing package tests. All ten functional modules contained only docstrings.
Existing commits remain intact. Work continues on the user-requested branch
`numcompute-branch2`.

## Checkpoint 1 — Environment and validation foundation

- Aligned packaging with the agreed Python 3.10+ requirement.
- Added a lightweight `test` installation extra while retaining the existing
  `dev` extra for pytest and Jupyter.
- Implemented numeric, categorical, label, sample-length and callable validation.
- Preserved integer precision and caller inputs; documented potential sharing
  of memory between input and returned arrays.
- Added shape/type errors, independent NaN/infinity controls, and tests for
  narrowing conversions that overflow to infinity.
- Added 45 validation test cases alongside the two original package tests.
- Documented installation, current API availability, planned functionality,
  benchmark status, and actual team contributions in the README.

Validation commands:

```sh
python -m pip install -e ".[test]"
python -m pytest
python -m pip check
git diff --check
```

Observed environment: macOS arm64, Python 3.11.13, NumPy 2.4.6, pytest 9.1.1.
The test suite reports **47 passed** and the dependency check reports no broken
requirements. These results validate the foundation only, not the unfinished
toolkit. No algorithm timing results exist yet.

## Checkpoint 2 — Stable and multi-key sorting

Before starting, `origin` was fetched and inspected. Remote `main` remained at
`9eb8c77`; no new teammate changes needed integration. The working tree was clean
on `numcompute-branch2`.

- Added `stable_sort(a, axis=-1)` using NumPy's stable sort, including negative
  axes and flattening with `axis=None`.
- Added `multi_key_sort(X, keys, descending=None)` using stable `np.lexsort`,
  reversing the comparison-key sequence so the first listed column is primary.
- Descending keys use bounded integer order codes rather than negating source
  values. This preserves unsigned integers and minimum signed integer values.
- Both APIs preserve dtypes and caller data. NaNs sort last; complete ties
  preserve original order. Signed-zero and row-ID tests verify stability.
- Added 67 sorting cases, including independent Python reference comparisons,
  mixed directions, NaNs/infinities, integer extremes, strided inputs, empty
  arrays, and validation of axes, keys and direction flags.
- Documented shapes, NaNs, input mutation, exceptions and complexity for both
  public functions. Python loops in multi-key sorting visit sort-key metadata
  only; all numerical row ordering uses NumPy operations.

Validation commands:

```sh
python -m pytest
python -m pytest --doctest-modules numcompute/sort_search.py
git diff --check
```

Results: **114 unit-test cases passed**, plus **two docstring examples passed**.
The README sorting example was also executed successfully. The environment is
unchanged from checkpoint 1. No performance measurements are claimed.

## Checkpoint 3 — Top-k selection and binary search

Fetched `origin` before starting; remote `main` remained at `9eb8c77` with no
new teammate changes. Work continued from a clean `numcompute-branch2`.

- Added `topk(values, k, largest=True, return_indices=True)`. Partial partitioning
  identifies a numeric cutoff; a separate scan chooses the earliest boundary
  ties. Only the selected values are sorted. NaNs fill output positions only
  after all valid values. The implementation supports k=0 and k=n, preserves
  dtypes, avoids negating original values, and returns original input indices.
- Added `binary_search(sorted_array, x)` returning left insertion positions and
  existence flags for scalar or shaped queries. NaNs never count as matches;
  out-of-range insertion positions are guarded before gathering candidate data.
- Investigated mixed-dtype precision before implementation. NumPy's usual
  int64/float64 or int64/uint64 promotion can round distinct integers into equal
  floats. Search uses an exact Python numeric comparison fallback through
  NumPy object arrays for those dtype combinations. Trailing NaNs are excluded
  from object search to preserve NaNs-last insertion semantics. The extra
  linear conversion cost is documented, rather than claiming logarithmic total
  time for that path. Compatible numeric ndarray searches avoid that conversion.
- Added 34 top-k and 24 binary-search test cases. Coverage includes independent
  stable Python sort/bisect references, cutoff ties, signed zero, extreme
  integers, signed/unsigned/float mixtures, NaNs/infinities, empty data,
  scalar/multidimensional queries, invalid arguments, and read-only strided data.

Validation commands:

```sh
python -m pytest tests --doctest-modules numcompute/sort_search.py
git diff --check
```

Results: **172 unit-test cases and four docstring tests passed** (176 test cases
in the combined run). The docstring tests execute six examples across the four
sorting/search APIs. All three README Python examples also execute successfully.
The environment is unchanged. No benchmark results are claimed.

## Checkpoint 4 — Educational quickselect

Fetched `origin` before starting; remote `main` remained at `9eb8c77` with no
new teammate changes. The working tree on `numcompute-branch2` was clean.

- Implemented `quickselect(a, k)` for a zero-based ascending order statistic.
  The function validates the one-dimensional nonempty input and index before
  partitioning a private copy, preserving dtype and caller data.
- NaNs occupy the final ranks, matching the module's ascending ordering. They
  are separated before pivot comparisons; infinities remain valid values.
- Explicit iterative Dutch-flag partitioning creates less-than, equal-to, and
  greater-than pivot groups, then visits only the group containing k. Complete
  tie groups need only one partition pass. There is no recursion or delegation
  to a sort/partition library implementation.
- Pivots come from a local NumPy generator seeded with 42, leaving global random
  state unchanged. Documented average time is O(n), worst-case time O(n²), and
  additional storage O(n) for the private copy/mask. Educational Python loops
  are intentional. Tied zeros have no guaranteed original occurrence/sign.
- Added 33 cases covering every rank for sorted, reversed, duplicate, random,
  boolean, float32, signed-zero, NaN and infinity reference arrays; extreme
  signed/unsigned integers; a 2,000-element equal group; empty and invalid
  inputs; list/read-only/strided input preservation; and global RNG state.
- Checked consistency with top-k selection and binary search and added working
  README and docstring examples.

Validation commands:

```sh
python -m pytest tests --doctest-modules numcompute/sort_search.py
git diff --check
```

Results: **205 unit-test cases and five docstring tests passed** (210 combined
test cases). The docstring tests execute eight examples across all five public
sorting/search functions. All four README Python examples execute successfully.
The environment is unchanged. No performance measurements are claimed.

## Checkpoint 5 — Tie-aware ranking and percentile interpolation

Fetched `origin` before starting; remote `main` remained at `9eb8c77` with no
new teammate changes. Work started from a clean `numcompute-branch2`.

- Implemented `rank(data, method='average')` with average, dense, ordinal, min,
  and max methods. A stable sort plus vectorized tie-group boundaries preserves
  ordinal tie order and avoids creating false ties between large integers.
  NaNs retain their positions and do not consume ranks. Output is float64.
- Implemented `percentile(data, q, interpolation='linear')` with linear, lower,
  higher, midpoint and nearest methods. Public data is flattened; scalar and
  vector percentage queries have explicit output shapes. Nearest half-indices
  round to even. Empty/all-NaN samples warn and return NaN; empty queries return
  an empty array without warnings. Interpolation uses at least float64 output.
- Added a shared internal axis-aware kernel for the upcoming statistics
  quantiles. It uses sorted copies, per-slice non-NaN counts, vectorized index
  maths, and bound gathering. It does not delegate interpolation to percentile
  or quantile library functions, and has no Python loops over sample values.
- Defined infinity handling: exact endpoints/equal infinities are retained,
  finite-to-infinite interior blends return the infinity, and blends spanning
  opposite infinities warn and return NaN.
- Added 26 rank and 71 percentile/kernel cases, including hand-calculated tie
  results, independent counting references, NumPy finite/NaN references, axes,
  shapes, missing slices, strides, read-only inputs, and numerical edge cases.

### Debugging evidence

An explicit small-value check exposed a bug in the first interpolation formula:
for the smallest positive float64 `s`, the midpoint of `[s, 2*s]` returned `s`
instead of `2*s`. Both linear and midpoint regression tests failed. The cause
was rounding each weighted endpoint before adding them. Interpolation now uses
the nearer endpoint and a finite difference when safe; it falls back to convex
weights when that difference overflows for opposite finite extremes. Both
regression tests and the finite-overflow tests pass. The corrected feature and
regression tests were committed together, preserving a passing feature commit.

Validation commands:

```sh
python -m pytest tests --doctest-modules numcompute/rank.py numcompute/sort_search.py
git diff --check
```

Results: **302 unit-test cases and seven docstring tests passed** (309 combined
test cases). The docstring tests execute 12 examples. All five README Python
examples execute successfully. Python source lines remain within 100 characters.
The environment is unchanged. No benchmark results are claimed.

## Checkpoint 6 — Axis-aware statistics, quantiles and histograms

Fetched `origin` before starting; remote `main` remained at `9eb8c77` and there
were no new teammate changes. Work started from a clean `numcompute-branch2`.

- Implemented mean, median, standard deviation, variance, minimum and maximum
  with one integer axis or flattened reduction and an explicit NaN policy.
  Shared validation rejects unsupported axes/flags, and invalid ddof types or
  negative corrections. Floating calculations promote to at least float64;
  extrema retain original dtypes, including large signed/unsigned integers.
- Preserved NumPy's warning and result semantics for empty/all-NaN slices,
  non-finite arithmetic and insufficient sample counts. Documented the
  differences between NaN omission and propagation. Empty extrema raise errors.
- Implemented quantile probabilities in [0, 1] using the checkpoint 5 kernel,
  including scalar/vector queries, negative axes, empty shapes and interpolation
  modes. Probability inputs are now passed directly to the shared kernel.
- Implemented histograms with NumPy-generated equal-width edges or explicit
  strictly increasing edges. NaNs are omitted; out-of-range data is excluded;
  the final right edge is included. Counts use vectorized binary lookup and
  bincount. Reusing exact binary search protects mixed integer/float boundaries.
- Added 156 cases: 90 reductions, 34 quantiles and 32 histograms, with NumPy
  references, hand-calculated values, shape/dtype checks, read-only/strided data,
  error paths, NaN/inf handling and numerical precision cases.

### Precision evidence

A pre-implementation check showed that converting quantile probabilities to
percentages and back can change nearest-index selection. For n=4 and q=5/6,
direct index rounding selects index 2; the percentage roundtrip selects index 3.
The shared kernel now accepts an internal scale of 1 for probabilities, avoiding
that conversion. A regression test compares the half-index queries directly to
NumPy quantile results, and all existing percentile tests still pass.

Histogram tests also exercise observations above 2**53 against float bin edges,
where rounding the observation before lookup could assign it to the wrong bin.
The implementation keeps the original numeric values through exact lookup.

Validation command:

```sh
python -m pytest tests --doctest-modules numcompute -W error
git diff --check
```

Results: **458 unit-test cases and 15 docstring tests passed** (473 combined
test cases), with unexpected warnings treated as errors. Docstrings execute
20 examples. All six README Python examples execute successfully. Changed
Python files remain within 100 characters per line. The environment is unchanged.
No performance measurements are claimed.


## Checkpoint 7 — Welford running statistics and Member B integration

Fetched `origin` before starting; remote `main` remained at `9eb8c77` with no new
teammate work to integrate. Work started from a clean `numcompute-branch2`.

- Implemented `RunningStats` with three scalar state values and read-only count,
  mean and population variance. Empty state returns NaN summaries without
  warnings; singleton variance is zero. Updates accept one-dimensional batches,
  omit NaNs, reject infinity and leave caller arrays unchanged/unretained.
- Vectorized centered batch moments combine through the parallel Welford
  formula. Shifting before computing batch moments avoids summing a large
  common offset; merging scales the correction before squaring a mean delta.
- `update` and `merge` return self. Empty inputs are no-ops, merging preserves
  its source, and self-merge repeats observations. Candidate state is computed
  and checked before assignment; invalid input or unrepresentable moments leave
  existing summaries unchanged. Finite-range/roundoff limits are documented.
- Added 34 running-statistics cases for chunked/batch equivalence, merges,
  NaNs/empty data, large offsets, finite limits, dtypes, state preservation,
  read-only input, self-merge and input errors.
- Added seven Member B workflow cases covering score analysis, missing/empty
  data, exact large-integer ordering with centered streams, stable multi-key row
  ordering and per-feature streaming versus axis reductions.
- Added a Member B requirement-to-test map and an explicit record of numerical
  limits and outstanding benchmark/demo/report evidence.

Numerical checks include a constant stream at float64's maximum (finite mean,
zero variance), mean deltas whose square overflows before a valid merge weight
is applied, and a large-offset stream with a known small variance. Tests also
confirm atomic rejection when a centered moment itself cannot be represented,
even where a normalized variance could be finite. No performance claim is made.

Validation command:

```sh
python -m pytest tests --doctest-modules numcompute -W error
git diff --check
```

Results: **499 unit-test cases and 18 docstring tests passed** (517 combined
cases), with unexpected warnings treated as errors. Docstrings execute 28
examples; all seven README Python examples execute successfully. Changed Python
files remain within 100 characters per line. The environment is unchanged.

## Design decisions and limitations

- Numeric validation defaults to `dtype=None` because distinct integers above
  2**53 can become equal when converted to floating point.
- Empty arrays are valid at the validation layer. Each algorithm must document
  whether it can compute a meaningful result from them.
- Categorical object arrays require element type inspection; the validation-only
  loop is documented. Core numerical computations will use NumPy operations.
- Encoders must define their own NaN category and within-feature comparison
  rules; metrics must align label types and reject unusable empty samples.
- The remaining A/C algorithm implementations, numerical
  utilities, benchmarks, the notebook, report, and video materials remain
  scheduled for later checkpoints.

## Next checkpoint recorded after checkpoint 7

Checkpoint 8 implements CSV loading and chunked input with tests and docstrings.
Before starting it, fetch the remote and inspect new contributions.
Each checkpoint ends with local commits and a pause for user review. Branch
publication and the final pull request remain scheduled after final verification.

## Post-merge integration and audit closure — 25 October 2026

The group merged PRs #1, #2, #4 and #3, reaching main a3c6c52. Rugved then
requested that the audit fixes be developed directly on local main. The earlier
checkpoint entries remain a historical record; their future-work statements
do not describe the current implementation state.

Small follow-up commits fixed transformer protocol dispatch, fitted-state
consistency, distance cancellation, activation infinities, extreme/tiny scaler
inputs, exact mixed numeric labels, regression overflow and masked integer CSV
loading. Further commits completed public API contracts, mixed-data integration
tests, benchmark methodology and automated repository checks.

The full suite now passes 764 tests/doctests without skips on both NumPy 1.24.4
and 2.4.6 with Python 3.11.13. A wheel was installed and imported outside the
checkout. Both README examples and the full 18-case/five-round benchmark run
completed. New tests reproduce genuine failures; no artificial buggy commits
were introduced. See [audit closure](audit_fixes.md) for counterexamples, test
results, measurements and remaining verification limitations.

The requested notebook edit and later fresh-kernel run were declined. The team
notebook remains unchanged; README and integration tests contain the combined
numeric/categorical example. Post-fix notebook execution and hosted CI remain
pending. The report and video are the next user-directed phase.
