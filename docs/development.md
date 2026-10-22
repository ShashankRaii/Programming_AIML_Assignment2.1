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

## Design decisions and limitations

- Numeric validation defaults to `dtype=None` because distinct integers above
  2**53 can become equal when converted to floating point.
- Empty arrays are valid at the validation layer. Each algorithm must document
  whether it can compute a meaningful result from them.
- Categorical object arrays require element type inspection; the validation-only
  loop is documented. Core numerical computations will use NumPy operations.
- Encoders must define their own NaN category and within-feature comparison
  rules; metrics must align label types and reject unusable empty samples.
- Selection, searching, the remaining algorithm implementations, numerical
  utilities, benchmarks, the notebook, report, and video materials remain
  scheduled for later checkpoints.

## Next checkpoint

Checkpoint 3 implements top-k selection and binary search with tests and
docstrings. Before starting it, fetch the remote and inspect new contributions.
Each checkpoint ends with local commits and a pause for user review. Branch
publication and the final pull request remain scheduled after final verification.
