# NumCompute

A modular scientific computing toolkit for Programming AIML Assignment 2.1,
built with **plain Python and NumPy**. All required modules, the weather-data
demo, tests and reproducible benchmarks are implemented. The merged code has
also received an integration audit and focused numerical/API fixes.

**Verification:** 764 unit/doctest checks pass with warnings treated as errors,
on Python 3.11.13 with both NumPy 1.24.4 and 2.4.6. The merged demo baseline
passed a fresh-kernel run; post-fix notebook execution is configured in CI and
remains to be verified. The report and video are separate submission deliverables.

## Install and verify

Use Python **3.10 or newer**. NumPy is the only runtime dependency; pytest and
Jupyter are development tools. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest tests --doctest-modules numcompute -W error
python -m pip check
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. For library
use install `python -m pip install .`; for tests without Jupyter use
`python -m pip install -e ".[test]"`.

Open the demonstration with:

```sh
jupyter notebook demo/quickstart.ipynb
```

Or execute it without changing the tracked notebook:

```sh
python -m ipykernel install --user --name numcompute
jupyter nbconvert --execute --to notebook --ExecutePreprocessor.kernel_name=numcompute --output quickstart.executed.ipynb --output-dir /tmp demo/quickstart.ipynb
```

Choose an existing temporary output directory on Windows in place of `/tmp`.
The notebook resolves data paths from either `demo/` or the repository root.
It covers CSV/missing data, preprocessing/pipelines, selection/search, ranking,
statistics, evaluation metrics, finite differences and benchmark comparisons.

## Mixed-data workflow

Load numeric and categorical columns separately. Their original CSV row order
is retained, so their processed outputs can be concatenated safely:

```python
import numpy as np
from numcompute.io import load_csv, read_header
from numcompute.pipeline import Pipeline, ThresholdClassifier
from numcompute.preprocessing import OneHotEncoder, SimpleImputer, StandardScaler
from numcompute.metrics import accuracy

path = "data/adelaide_weather_2015.csv"
header = read_header(path)
numeric_columns = [header.index(c) for c in ["MaxTemp", "Rainfall", "Humidity3pm"]]
X = load_csv(path, skip_header=1, usecols=numeric_columns)
categories = load_csv(path, skip_header=1,
                      usecols=header.index("WindGustDir"), dtype=str)
y = (load_csv(path, skip_header=1, usecols=header.index("RainTomorrow"), dtype=str)
     .ravel() == "Yes").astype(int)

numeric = Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler())])
categorical = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent", missing_values="")),
    ("encode", OneHotEncoder()),
])
features = np.hstack([numeric.fit_transform(X), categorical.fit_transform(categories)])
assert features.shape[0] == len(y)
assert np.isfinite(features).all()

example = Pipeline([
    ("impute", SimpleImputer()),
    ("classify", ThresholdClassifier(feature=2)),
])
predictions = example.fit(X, y).predict(X)
print("In-sample API demonstration accuracy:", accuracy(y, predictions))
```

The tiny `ThresholdClassifier` and `MeanRegressor` demonstrate the estimator
protocol. Their in-sample scores are not evidence of predictive generalization;
implementing a trained ML model is not an assignment requirement.

```python
import numpy as np
from numcompute.sort_search import topk, quickselect, binary_search, multi_key_sort
from numcompute.rank import rank, percentile
from numcompute.stats import RunningStats, quantile, histogram

values, indices = topk([3, 5, 5, 1], 2)
assert indices.tolist() == [1, 2]  # Earlier original indices win ties.
assert quickselect([3, 1, 4, 1, 5], 2) == 3
assert binary_search([1, 3, 3, 5], 3) == (1, True)
assert multi_key_sort([[1, 2], [1, 3]], [0, 1], [False, True]).tolist() == [[1, 3], [1, 2]]
assert rank([30, 10, 10, 20]).tolist() == [4, 1.5, 1.5, 3]
assert percentile([10, 20, 30, 40], 25) == 17.5
assert quantile([1, np.nan, 3], .5) == 2
assert histogram([0, 1, 2, 3, np.nan], bins=[0, 1, 3])[0].tolist() == [1, 3]
summary = RunningStats().update([1, 3, np.nan]).merge(RunningStats().update([5, 7]))
assert (summary.count, summary.mean, summary.variance) == (4, 4, 5)
```

## Public API overview

Import functions from their submodules; package functionality uses only NumPy
and the Python standard library. Public docstrings specify shapes, exceptions,
NaN/mutation rules and computational costs.

| Module | Interfaces |
| --- | --- |
| `io` | `read_header`, `load_csv`, `load_csv_chunks`: delimiters, selected columns, headers, missing fill/skip and chunked reading |
| `preprocessing` | `StandardScaler`, `MinMaxScaler`, `SimpleImputer`, `OneHotEncoder`; `fit`, `transform`, `fit_transform`, fitted-state and feature-count checks |
| `pipeline` | `Pipeline`, `FeatureUnion`, transformer/estimator base classes, `NotFittedError`, example estimators |
| `sort_search` | `stable_sort`, `multi_key_sort`, `topk`, `quickselect`, `binary_search` |
| `rank` | `rank` (average/dense/ordinal/min/max), `percentile` (linear/lower/higher/midpoint/nearest) |
| `stats` | `mean`, `median`, `std`, `var`, `minimum`, `maximum`, `quantile`, `histogram`, mergeable `RunningStats` |
| `metrics` | `accuracy`, `precision`, `recall`, `f1`, `confusion_matrix`, `mse`, `mae`, `rmse`, `r2`, `roc_curve`, `auc`, `UndefinedMetricWarning` |
| `optim` | Central/forward `grad` and `jacobian`, optional batched-callable evaluation, Armijo `line_search` |
| `utils` | Numeric/category/label/callable validation, `softmax`, `sigmoid`, `logsumexp`, `euclidean_distances`, `batch_iterator` |
| `benchmarking` | `benchmark`, `compare`, `environment_info`, reproducible `SEED` |

Key contracts and limitations:

- Validation preserves numeric dtypes by default. Algorithms preserve inputs;
  validation arrays and batching slices may share memory. Computations needing
  floating point document float64 conversion and its integer precision limits.
- Stable ties preserve original order; the first listed multi-key sort key is
  primary. Top-k uses partial selection, accepts `k=0`/`k=n`, returns sorted
  values/indices and puts NaNs last in either direction. Quickselect partitions
  a private copy: average O(n), worst O(n²), O(n) copy storage.
- Binary search requires ascending input with trailing NaNs. It returns the
  left insertion position/existence, preserves query shapes, and never matches
  NaN. Unsafe mixed numeric dtypes use exact object comparisons. Classification
  labels use the same exact-comparison policy before encoding/counting.
- Ranking omits NaNs and retains NaN positions. Percentiles flatten their input
  and take percentages 0–100; quantiles take probabilities 0–1 and accept a
  single integer axis or None. Vector queries prepend a query dimension.
- Descriptive reductions accept one axis or None and default to ignoring NaNs.
  `std`/`var` accept nonnegative integer `ddof`; NumPy warnings apply to undefined
  reductions. Histograms omit NaNs and include the final right edge.
- RunningStats retains count/mean/centered sum of squares, omits NaNs, rejects
  infinity, and exposes population variance. Empty batches do nothing; empty
  state has NaN mean/variance. Unrepresentable centered moments raise an atomic
  OverflowError. Floating-point batch groupings can differ slightly.
- Scalers reject infinity and retain NaNs. MinMaxScaler preserves subnormal
  input ranges and uses halves only for overflow-prone differences. Constant
  features use a unit data range; all-NaN fitted features remain NaN. Extreme
  standard deviations may be representable while their squared variance is not.
- Pipeline supports minimal `fit(X)` and `fit_transform(X)` methods as well as
  supervised positional/keyword `y`. Signature dispatch never retries an
  internal TypeError. Pre-fit operations raise NotFittedError, a RuntimeError
  that retains ValueError/AttributeError compatibility. Custom stages define
  their own data/mutation contracts.
- Undefined classification ratios return zero with UndefinedMetricWarning;
  binary/macro/micro modes are available. ROC collapses tied score thresholds.
  Regression errors rescale before reducing to avoid unnecessary overflow;
  genuinely unrepresentable results become infinity.
- Softmax splits mass equally among positive infinities. NaN or entirely
  negative-infinity slices return NaNs; logsumexp propagates NaN and returns
  the appropriate infinity. Distances reject non-finite inputs and use bounded
  broadcast blocks with a stable hypot norm, avoiding squared-norm cancellation.
- Gradient/Jacobian calls ordinarily evaluate each perturbed point in a justified
  loop. `vectorized=True` requires an explicitly batched callable. Perturbation
  storage is O(n²), so large derivative dimensions need bounded workloads.

## Reproducible performance

```sh
python benchmark/run_benchmarks.py --repeats 5 --out benchmark
python benchmark/run_benchmarks.py --quick --repeats 3 --out /tmp/numcompute-benchmarks
```

The full run measures n=1,000/10,000/100,000 and gradients in 100/500/2,000
dimensions. It uses seed 42, an output-agreement check, one warmup and one call
per timing round. Input/list preparation is outside timing. The table uses the
best of five rounds; the [CSV](benchmark/results.csv) also records mean/std.
Gradient rows compare ordinary versus batched dispatch of the same NumPy
objective. Distance rows compare n points with 10 reference points.

Measured on Python 3.11.13, NumPy 2.4.6, macOS/Darwin 27.0.0, arm64, 8 CPUs:

| Operation | Size | Loop (s) | NumCompute (s) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| StandardScaler | 100,000 | 0.13700 | 0.01071 | 12.8x |
| Top-k, k=10 | 100,000 | 0.00376 | 0.00102 | 3.7x |
| Average rank | 100,000 | 0.02721 | 0.00858 | 3.2x |
| Running mean/variance | 100,000 | 0.00694 | 0.00016 | 43.4x |
| Distances to 10 points | 100,000 | 0.37234 | 0.04781 | 7.8x |
| Central gradient | 2,000 | 0.04742 | 0.04205 | 1.1x |

See the [complete results and environment](benchmark/results.md). Timings vary
with machine/load; vectorization is not a universal speed guarantee. The stable
distance implementation trades some speed for accurate nearby large vectors.
The derivative comparison also exposes batching overhead and quadratic memory.

## Testing and debugging evidence

The full suite covers empty/singleton arrays, ties, extreme k, NaNs/infinities,
large integers, strided/read-only arrays, constant features and caller-input
preservation. References include Python sorting/bisect, NumPy reductions,
hand-counted metrics and analytic derivatives. Integration tests combine CSV
loading, numeric/category preprocessing, a pipeline, metrics and streaming stats.

[Audit closure](docs/audit_fixes.md) records reproducible failing-before and
passing-after examples. [Development history](docs/development.md) and the
[Member B acceptance review](docs/member_b_review.md) preserve checkpoint evidence.

[GitHub verification](.github/workflows/verify.yml) checks Linux Python 3.10 with
NumPy 1.24.4, Linux Python 3.13 with current NumPy, and Windows Python 3.11.
A separate job executes the notebook, smoke-checks benchmarks, builds a wheel
and imports it outside the source checkout. Local verification is recorded in
the audit closure; configured CI jobs are not claimed as passed until GitHub runs them.

## Demo data and credit

The committed `data/adelaide_weather_2015.csv` contains 360 Adelaide days from
2015, 11 columns, missing cells, categorical wind/rain columns and a rain-tomorrow
label. It is a subset of `weatherAUS.csv` (Rain in Australia / R `rattle`).
Daily observations are credited to the Commonwealth of Australia, Bureau of
Meteorology: https://www.bom.gov.au/climate/data. The original source file is
excluded from Git. Rebuild the subset with:

```sh
python data/make_adelaide_weather.py path/to/weatherAUS.csv
```

## Contributions and submission

| Member | Implemented contribution |
| --- | --- |
| A — Shashank Rai | Scaffold/packaging; CSV loading/chunking; preprocessing; pipeline/feature union and example estimators; associated tests; weather dataset/rebuild script; all eight notebook sections and branch integration |
| B — Rugved Wankhade | Shared validation; sorting/search/top-k/quickselect; ranking/interpolation; batch/streaming statistics; unit/integration tests, README and checkpoint/debugging records, with Codex assistance. Post-merge audit fixes, API contracts, mixed-data integration tests, CI and refreshed measurements were also completed |
| C — Rejith Jacob | Numerical utilities; classification/regression/ROC/AUC metrics; gradients/Jacobian/line search; benchmark harness/script and initial measurements; associated tests |
