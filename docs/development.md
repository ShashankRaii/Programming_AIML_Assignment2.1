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

## Design decisions and limitations

- Numeric validation defaults to `dtype=None` because distinct integers above
  2**53 can become equal when converted to floating point.
- Empty arrays are valid at the validation layer. Each algorithm must document
  whether it can compute a meaningful result from them.
- Categorical object arrays require element type inspection; the validation-only
  loop is documented. Core numerical computations will use NumPy operations.
- Encoders must define their own NaN category and within-feature comparison
  rules; metrics must align label types and reject unusable empty samples.
- Numerical utilities, algorithm implementations, benchmarks, the notebook,
  final report, and video materials remain scheduled for later checkpoints.

## Next checkpoint

Checkpoint 2 implements stable sorting and multi-key sorting with tests and
docstrings. Before starting it, fetch the remote and inspect new contributions.
Each checkpoint ends with local commits and a pause for user review. Branch
publication and the final pull request remain scheduled after final verification.
