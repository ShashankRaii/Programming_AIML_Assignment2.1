# NumCompute

NumCompute is a scientific computing toolkit for Programming AIML Assignment 2.1,
built with Python and NumPy. The project will provide reusable data loading,
preprocessing, sorting, ranking, statistics, metrics, numerical differentiation,
and pipeline components.

**Current status: checkpoint 3 completed.** Packaging, shared input validation,
stable/multi-key sorting, top-k selection and binary search are implemented and
tested. Educational quickselect and the remaining functional modules are
scheduled for later checkpoints. Benchmarks and the end-to-end demo are pending.

## Installation

Use Python **3.10 or newer**. NumPy is the only runtime dependency. From the
repository root, create an environment and install the test dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
python -m pytest
```

Confirm that `python3 --version` reports Python 3.10+ before creating the
environment. On Windows, activate it with `.venv\Scripts\Activate.ps1` in
PowerShell. For notebook development, install `python -m pip install -e ".[dev]"`.
For package use without test/notebook tools, use `python -m pip install -e .`.

## Current usage

Validation preserves numeric dtypes unless an explicit conversion is requested:

```python
import numpy as np
from numcompute.utils import check_array, check_categories, check_labels

values = np.array([2**53, 2**53 + 1], dtype=np.int64)
validated = check_array(values, allow_nan=False)
assert validated[0] != validated[1]

features = check_array([[1, 2], [3, 4]], ensure_2d=True, dtype=float)
categories = check_categories([["red"], ["blue"]])
labels = check_labels(["positive", "negative"])
```

Validation does not modify input data. Returned arrays can share memory with
their inputs; any later algorithm that writes to a working array must copy it.
Numeric validation accepts real numbers and booleans, permits NaNs and infinity
by default, and provides independent switches to reject either. Numeric strings
are rejected rather than silently parsed. Categorical validation preserves string
and numeric atoms, allows NaNs for an encoder to handle, and rejects infinities
and unsupported objects. Label validation rejects missing/non-finite labels and
requires one dimension. Individual algorithms will define stricter requirements
where needed.

Sorting returns independent copies, preserving the input dtype and caller data:

```python
from numcompute.sort_search import multi_key_sort, stable_sort

assert stable_sort([3, 1, 2, 1]).tolist() == [1, 1, 2, 3]
rows = [[2, 1], [1, 2], [1, 3]]
ordered = multi_key_sort(rows, keys=[0, 1], descending=[False, True])
assert ordered.tolist() == [[1, 3], [1, 2], [2, 1]]
```

`stable_sort` supports integer axes and `axis=None` to flatten.
`multi_key_sort` takes a numeric matrix and column keys in priority order, with
the first listed key primary. Negative column indices are accepted; repeated
columns are rejected. Directions are a boolean sequence with one flag per key.
Complete ties preserve original row order. NaNs sort last in either direction;
later keys resolve ties between NaNs in earlier keys. Descending sorting uses
bounded order codes so that unsigned and extreme signed integers remain safe.

Top-k uses partial selection, returns sorted values with original indices, and
prefers earlier indices when values tie:

```python
from numcompute.sort_search import binary_search, topk

selected, indices = topk([3, 5, 5, 1], 2)
assert selected.tolist() == [5, 5]
assert indices.tolist() == [1, 2]
assert topk([3, 5, 1], 2, largest=False, return_indices=False).tolist() == [1, 3]

assert binary_search([1, 3, 3, 5], 3) == (1, True)
positions, found = binary_search([1, 3, 3, 5], [0, 3, 6])
assert positions.tolist() == [0, 1, 4]
assert found.tolist() == [False, True, False]
```

`topk` accepts `k=0` through the full input length and returns NaNs only after
all valid values. Its returned indices address the supplied input, including
when that input is a slice. It preserves the value dtype and caller data.

`binary_search` requires an ascending one-dimensional source, with NaNs last.
It returns the left insertion position plus an existence flag, preserving the
query shape; scalar queries return Python `int`/`bool` values. NaN queries always
report no match. The function does not scan the source to validate sortedness.
For mixed numeric dtypes that would lose integer precision under normal NumPy
promotion, it uses exact object-array comparisons. That fallback adds linear
conversion time and memory; ordinary compatible ndarray searches retain binary
search complexity. Inputs are never modified.

## API and implementation status

Import validation helpers from `numcompute.utils` and sorting/search functions from
`numcompute.sort_search`.

| Component | Public interface | Status |
| --- | --- | --- |
| Numeric validation | `check_array(X, ensure_2d=False, allow_nan=True, dtype=None, *, allow_inf=True)` | Implemented |
| Sample alignment | `check_same_length(a, b)` | Implemented |
| Categorical validation | `check_categories(X, ensure_2d=True)` | Implemented |
| Label validation | `check_labels(y)` | Implemented |
| Function validation | `check_callable(function)` | Implemented |
| Stable sorting | `stable_sort(a, axis=-1)` | Implemented |
| Multi-key sorting | `multi_key_sort(X, keys, descending=None)` | Implemented |
| Top-k selection | `topk(values, k, largest=True, return_indices=True)` | Implemented |
| Binary search | `binary_search(sorted_array, x)` | Implemented |
| Educational quickselect | Zero-based kth-value selection | Planned: checkpoint 4 |
| Ranking | Tie methods and percentile interpolation | Planned: checkpoint 5 |
| Statistics | Reductions, quantiles, histogram, streaming statistics | Planned: checkpoints 6–7 |
| Data loading | CSV and chunked loading | Planned: checkpoint 8 |
| Preprocessing | Scalers, imputer, one-hot encoder | Planned: checkpoint 9 |
| Pipelines | Pipeline and feature union | Planned: checkpoint 10 |
| Metrics | Classification, MSE, ROC/AUC | Planned: checkpoint 11 |
| Differentiation | Gradient and Jacobian | Planned: checkpoint 12 |
| Numerical utilities and benchmarking | Stable activations, distances, timing harness | Planned: checkpoint 13 |
| Demonstration | End-to-end notebook and sample data | Planned: checkpoint 14 |

## Testing and development

```sh
python -m pytest
python -m pytest tests/test_validation.py
python -m pytest tests/test_sort_search.py
python -m pytest --doctest-modules numcompute/sort_search.py
python -m pip check
git diff --check
```

Checkpoint 3 has **172 passing unit-test cases**: 67 sorting cases, 34 top-k
cases, 24 binary-search cases, 45 validation cases, and two original package
smoke tests. Four docstring tests (six examples) also pass. Coverage includes
stable ties and cutoff selection, extreme k, insertion boundaries, scalar and
shaped queries, mixed sort directions, large and mixed numeric dtypes, NaNs,
infinities, empty/ragged inputs, errors, strided/read-only arrays and preservation
of caller data. Independent Python sorting and bisect references check ordering
and search results. Later algorithms still need their own tests.

Every feature checkpoint includes tests and docstrings describing parameters,
return shapes, errors, and time/space complexity. Genuine defects receive
regression tests and explanatory fixes. See [the development record](docs/development.md)
for the baseline and completed checkpoint evidence.

## Benchmarks and submission materials

Measured benchmarks are pending. The later benchmark runner will compare
equivalent loop and vectorized computations using fixed random seeds, correctness
checks, repeated timings, and recorded environment details. The performance table
will report operation, input size, loop time, vectorized time, and speedup; no
performance claims are made yet.

The final submission will also include `demo/quickstart.ipynb`, a technical report
of at most four pages, and a demo video of at most five minutes. The submission
deadline is **26 October 2026, 9:00 AM ACDT**.

## Team responsibilities and actual contributions

Planned module ownership and completed work are recorded separately so that the
table does not attribute unfinished work to a teammate.

| Member | Planned responsibility | Completed contribution |
| --- | --- | --- |
| A — Shashank Rai | I/O, preprocessing, pipeline, sample data, demo | Original project scaffold, initial packaging and two package smoke tests, as recorded in the original Git history |
| B — Rugved Wankhade | Sorting/searching, ranking, statistics, README, testing report section | Checkpoints 1–3: Python/test setup, validation helpers, stable/multi-key sorting, top-k, binary search, tests and documentation, developed with Codex assistance |
| C — name to confirm | Utilities, metrics, differentiation, benchmarks | No implemented module contribution identified in the reviewed remote snapshot |
| All | Review, integration, report, video | Pending |

The agreed completion plan also allows Member B's development work to cover
unfinished A/C modules. This table will be updated for actual contributions as
they arrive. Local checkpoint commits are developed on `numcompute-branch2`;
the completed branch will be pushed for review after final verification.
