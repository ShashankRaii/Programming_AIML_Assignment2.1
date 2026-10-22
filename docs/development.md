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

## Design decisions and limitations

- Numeric validation defaults to `dtype=None` because distinct integers above
  2**53 can become equal when converted to floating point.
- Empty arrays are valid at the validation layer. Each algorithm must document
  whether it can compute a meaningful result from them.
- Categorical object arrays require element type inspection; the validation-only
  loop is documented. Core numerical computations will use NumPy operations.
- Encoders must define their own NaN category and within-feature comparison
  rules; metrics must align label types and reject unusable empty samples.
- Educational quickselect, the remaining algorithm implementations, numerical
  utilities, benchmarks, the notebook, report, and video materials remain
  scheduled for later checkpoints.

## Next checkpoint

Checkpoint 4 implements educational quickselect with tests and
docstrings. Before starting it, fetch the remote and inspect new contributions.
Each checkpoint ends with local commits and a pause for user review. Branch
publication and the final pull request remain scheduled after final verification.
