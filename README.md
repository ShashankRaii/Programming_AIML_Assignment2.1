# NumCompute

NumCompute is a scientific computing toolkit for Programming AIML Assignment 2.1,
built with Python and NumPy. The project will provide reusable data loading,
preprocessing, sorting, ranking, statistics, metrics, numerical differentiation,
and pipeline components.

**Current status: checkpoint 7 completed.** Packaging, shared input validation,
all planned sorting/search APIs, tie-aware ranking and percentile interpolation
are implemented and tested, alongside descriptive statistics, quantiles and
histograms and mergeable running statistics. Member B's three modules have
integration checks. The remaining modules, benchmarks and end-to-end demo are
scheduled for later checkpoints.

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

Quickselect returns a single zero-based kth smallest value:

```python
from numcompute.sort_search import quickselect

values = [3, 1, 4, 1, 5]
assert int(quickselect(values, 2)) == 3
assert values == [3, 1, 4, 1, 5]
```

`quickselect(a, k)` requires `0 <= k < len(a)` and returns a NumPy scalar with
the input's inferred dtype. NaNs rank after every valid value, so selecting a
trailing NaN rank returns NaN. It uses explicit iterative three-way partitioning
on a private copy, with reproducible local pivots and no change to global random
state. Average time is O(n), worst-case time is O(n²), and the copy requires O(n)
additional space. Python loops are intentional for this educational algorithm.
Equal values are equivalent; unlike the stable sorting APIs, quickselect does
not promise a particular original occurrence or sign for tied zeros.

Ranking assigns one-based ranks, with configurable tie handling. Percentiles
ignore NaNs and support scalar or vector percentage queries:

```python
from numcompute.rank import percentile, rank

assert rank([30, 10, 10, 20]).tolist() == [4, 1.5, 1.5, 3]
assert rank([30, 10, 10, 20], method="dense").tolist() == [3, 1, 1, 2]
assert rank([30, 10, 10, 20], method="ordinal").tolist() == [4, 1, 2, 3]
assert percentile([10, 20, 30, 40], [0, 25, 50, 100]).tolist() == [10, 17.5, 25, 40]
```

`rank` takes one-dimensional data and supports average, dense, ordinal, min and
max tie methods. NaNs remain at their input positions and do not consume ranks.
Ordinal ties retain input order. All ranks are float64; input values are sorted
in their own dtype to avoid creating false ties between distinct large integers.

`percentile` flattens numeric input and accepts scalar or one-dimensional `q`
in 0–100. It supports linear, lower, higher, midpoint and nearest interpolation;
nearest rounds exact half-indices to even. Results use at least float64, so
large integer outputs may round to floating precision. Empty/all-NaN samples
produce NaN with a RuntimeWarning; empty queries produce an empty array silently.
Exact infinite endpoints and equal infinities are retained. An interior blend
between opposite infinities is undefined and returns NaN with a RuntimeWarning.
Interpolation handles finite extremes and subnormal midpoint rounding without
modifying inputs. The shared internal kernel also supports one reduction axis,
used by the statistics quantile API.

Statistics use explicit axes, NaN policies and output shapes:

```python
import numpy as np
from numcompute import stats

data = np.array([[1.0, np.nan], [3.0, 5.0]])
assert stats.mean(data, axis=0).tolist() == [2, 5]
assert stats.var(data, axis=0).tolist() == [1, 0]
assert stats.quantile(data, [0, 0.5, 1], axis=0).tolist() == [[1, 5], [2, 5], [3, 5]]
counts, edges = stats.histogram([0, 1, 2, 3, np.nan], bins=[0, 1, 3])
assert counts.tolist() == [1, 3]
assert edges.tolist() == [0, 1, 3]
```

`mean`, `median`, `std`, `var`, `minimum` and `maximum` accept `axis=None` or a
single integer axis, including negative indices. None reduces all values to a
scalar; an integer removes that dimension. `ignore_nan=True` is the default;
False propagates NaNs in affected slices. Floating calculations use at least
float64; extrema retain the input dtype. Variance/std accept a nonnegative
integer `ddof`, with 0 for population and 1 for sample estimates. NumPy's warning
and NaN/infinity behaviour applies to undefined reductions and insufficient
sample counts. Empty minimum/maximum reductions raise ValueError.

`quantile` accepts probabilities in 0–1, with the same interpolation and NaN
rules as `percentile`. A vector query prepends its length to the remaining axis
shape; a scalar query simply removes the reduction axis. The shared kernel
accepts probabilities directly to avoid nearest-index rounding errors from a
probability-to-percentage roundtrip.

`histogram` flattens data and drops NaNs. Choose a positive integer bin count or
strictly increasing explicit edges. Bins include their left edges; the final bin
also includes its right edge. A finite range controls automatic edges; explicit
edges take precedence. Infinity in data needs a finite range or explicit edges.
Empty/all-NaN data yields zero counts, with automatic default edges spanning
0–1. Bin lookup reuses exact numeric comparisons from binary search and avoids
silently rounding large integer observations onto floating bin boundaries.

Running statistics summarize a one-dimensional stream without retaining data:

```python
import numpy as np
from numcompute.stats import RunningStats

running = RunningStats().update([1, 3, np.nan])
other = RunningStats().update([5, 7])
assert running.merge(other) is running
assert running.count == 4
assert float(running.mean) == 4
assert float(running.variance) == 5
assert other.count == 2
```

`RunningStats` exposes read-only `count`, `mean` and population `variance`
(ddof=0). An empty accumulator has count 0 and NaN mean/variance, without
warnings; a singleton has zero variance. Empty/all-NaN batches are no-ops.
Updates take one-dimensional batches; create a separate accumulator per feature
for tabular data. Both `update` and `merge` return self. Merging leaves the source
unchanged; self-merging counts the observations twice.

Vectorized batch moments are combined using the parallel Welford formula.
Only three scalars are retained. Updates use O(n) work and temporary space;
merges use O(1) work and space. Infinity is rejected. Updates/merges that cannot
keep a finite centered moment raise OverflowError and preserve existing state.
This includes cases where the unnormalized moment overflows even though a
variance could be representable. Floating conversion and accumulation have the
usual precision limits, and different batch groupings may differ slightly.

## API and implementation status

Import validation helpers from `numcompute.utils`, sorting/search functions from
`numcompute.sort_search`, ranking/percentile functions from `numcompute.rank`,
and statistical functions from `numcompute.stats`.

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
| Educational quickselect | `quickselect(a, k)` | Implemented |
| Ranking | `rank(data, method='average')` | Implemented |
| Percentiles | `percentile(data, q, interpolation='linear')` | Implemented |
| Descriptive statistics | `mean`, `median`, `std`, `var`, `minimum`, `maximum` | Implemented |
| Quantiles | `quantile(data, q, axis=None, interpolation='linear')` | Implemented |
| Histograms | `histogram(data, bins=10, range=None)` | Implemented |
| Streaming statistics | `RunningStats`, `update(batch)`, `merge(other)`, read-only summary properties | Implemented |
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
python -m pytest tests/test_rank.py
python -m pytest tests/test_stats.py
python -m pytest tests/test_running_stats.py tests/test_member_b_integration.py
python -m pytest tests --doctest-modules numcompute -W error
python -m pip check
git diff --check
```

Checkpoint 7 has **499 passing unit-test cases**: 67 sorting cases, 34 top-k
cases, 24 binary-search cases, 33 quickselect cases, 26 rank cases, 71 percentile
cases, 90 descriptive-statistics cases, 34 quantile cases, 32 histogram cases,
34 running-statistics cases, seven integration cases, 45 validation cases, and
two original package smoke tests. Eighteen docstring tests (28 examples) also
pass. The full run treats unexpected warnings as errors.
Coverage includes
stable ties and cutoff selection, extreme k, insertion boundaries, scalar and
shaped queries, mixed sort directions, large and mixed numeric dtypes, NaNs,
infinities, empty/ragged inputs, errors, strided/read-only arrays and preservation
of caller data. Independent Python sorting and bisect references check ordering
and search results. Quickselect tests check every rank on reference arrays,
large equal-value groups, global random state preservation, and agreement with
top-k and binary search. Rank tests include hand-calculated and independent
counting references. Percentile tests compare with NumPy references, exercise the
shared axis kernel, and cover empty/all-NaN slices, infinity handling, finite
overflow and subnormal midpoint regression cases. Statistics tests cover scalar,
matrix and 3-D reduction shapes, NaN policies, degrees of freedom, empty slices,
quantile probability rounding, bin-edge inclusion, and precise large integer
histogram boundaries. Running-statistics checks cover batch/stream agreement,
merging, state preservation, large offsets, finite limits and invalid input.
Integration checks exercise all three Member B modules together. Later
algorithms still need their own tests.

Every feature checkpoint includes tests and docstrings describing parameters,
return shapes, errors, and time/space complexity. Genuine defects receive
regression tests and explanatory fixes. See [the development record](docs/development.md)
for the baseline and completed checkpoint evidence.

The [Member B acceptance review](docs/member_b_review.md) maps completed
requirements to implementation and tests, and identifies the benchmark/demo
evidence still needed for submission.

## Benchmarks and submission materials

Measured benchmarks are pending. The later benchmark runner will compare
equivalent loop and vectorized computations using fixed random seeds, correctness
checks, repeated timings, and recorded environment details. The performance table
will report operation, input size, loop time, vectorized time, and speedup; no
performance claims are made yet.

The final submission will also include `demo/quickstart.ipynb`, a technical report
of at most four pages, and a demo video of at most five minutes.

## Team responsibilities and actual contributions

Planned module ownership and completed work are recorded separately so that the
table does not attribute unfinished work to a teammate.

| Member | Planned responsibility | Completed contribution |
| --- | --- | --- |
| A — Shashank Rai | I/O, preprocessing, pipeline, sample data, demo | Original project scaffold, initial packaging and two package smoke tests, as recorded in the original Git history |
| B — Rugved Wankhade | Sorting/searching, ranking, statistics, README, testing report section | Checkpoints 1–7: Python/test setup, validation, sorting/searching, ranking, interpolation, batch/streaming statistics, quantiles, histograms, unit/integration tests and documentation, developed with Codex assistance |
| C — Rejith Jacob | Utilities, metrics, differentiation, benchmarks | No implemented module contribution identified in the reviewed remote snapshot |
| All | Review, integration, report, video | Pending |
