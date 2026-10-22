# NumCompute

NumCompute is a scientific computing toolkit for Programming AIML Assignment 2.1,
built with Python and NumPy. The project will provide reusable data loading,
preprocessing, sorting, ranking, statistics, metrics, numerical differentiation,
and pipeline components.

**Current status: checkpoint 1 completed.** Packaging and shared input validation
are implemented and tested. The other functional modules are placeholders;
their APIs, examples, benchmark results, and demo will be documented as they are
implemented. This README does not claim the whole assignment is complete.

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

## API and implementation status

Import the implemented helpers from `numcompute.utils`.

| Component | Public interface | Status |
| --- | --- | --- |
| Numeric validation | `check_array(X, ensure_2d=False, allow_nan=True, dtype=None, *, allow_inf=True)` | Implemented |
| Sample alignment | `check_same_length(a, b)` | Implemented |
| Categorical validation | `check_categories(X, ensure_2d=True)` | Implemented |
| Label validation | `check_labels(y)` | Implemented |
| Function validation | `check_callable(function)` | Implemented |
| Sorting and searching | Stable/multi-key sort, top-k, quickselect, binary search | Planned: checkpoints 2–4 |
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
python -m pip check
git diff --check
```

Checkpoint 1 has **47 passing test cases**: 45 validation cases and two original
package smoke tests. Coverage includes large integers, explicit dtype conversion,
NaNs, infinities, empty arrays, ragged input, shape errors, non-contiguous views,
categories, labels, callables, and preservation of caller data. These tests cover
the current implementation; later algorithms still need their own tests.

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
| B — Rugved Wankhade | Sorting/searching, ranking, statistics, README, testing report section | Checkpoint 1: Python requirement/test installation updates, validation helpers and tests, README and development record, developed with Codex assistance |
| C — name to confirm | Utilities, metrics, differentiation, benchmarks | No implemented module contribution identified in the reviewed remote snapshot |
| All | Review, integration, report, video | Pending |

The agreed completion plan also allows Member B's development work to cover
unfinished A/C modules. This table will be updated for actual contributions as
they arrive. Local checkpoint commits are developed on `numcompute-branch2`;
the completed branch will be pushed for review after final verification.
