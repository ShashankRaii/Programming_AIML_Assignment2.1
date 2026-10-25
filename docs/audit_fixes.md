# Post-merge audit fixes and verification

Completed locally on 25 October 2026, directly on `main` at the user's request.
Baseline: merged main `a3c6c52`. These changes preserve the original team history
and add small, coherent commits. They do not claim report/video completion.

## Audit closure

| Finding | Resolution | Regression evidence |
| --- | --- | --- |
| Minimal transformer protocol rejected | Inspect/bind fit signatures before invoking; support fit(X), fit_transform(X), positional and keyword y. Never retry an internal TypeError. | `test_pipeline.py`: minimal protocols across Pipeline/FeatureUnion, keyword-only targets, single-call internal errors |
| Pre-fit exception mismatch | NotFittedError now inherits RuntimeError and preserves existing ValueError/AttributeError compatibility. Pipeline tracks its own completed fit. Union rejects non-matrix outputs. | Fitted-state and output-dimension regression cases |
| Distance cancellation | Replace the squared-norm identity with direct differences and stable hypot reductions in bounded broadcast blocks. Reject NaN/inf inputs explicitly. | Nearby 1e8-valued points, large/small distances, block boundaries, input preservation and empty features |
| Scaler overflow/underflow | Center moments at a safe midrange and normalize before summation/squaring. Use direct min-max differences unless halves are necessary. Define constant-feature extrapolation and all-NaN features. | Constant float64 maximum, opposite extremes, subnormal ranges, extreme target ranges and finite extrapolation |
| Classification label precision | Align unsafe mixed numeric arrays through exact Python numeric comparisons before equality, union/encoding and positive-label comparisons. | Distinct integers beyond 2**53, signed/unsigned classes, explicit label order and ROC positive labels |
| Integer CSV missing values | Retain genfromtxt's missing-cell mask and parse directly into the requested dtype; skip before returning and reject unrepresentable integer fills clearly. | Whole/chunked integer reads, exact large integers, unsigned fills and text gaps |
| Regression overflow | Normalize residuals before reductions and rescale in a safe order. RMSE no longer takes sqrt of an already overflowing MSE. R2 has a scaled fallback. | Representable MSE 1.125e308, RMSE 1e200, opposite extreme residuals and extreme R2 |
| Activations at infinity | Define NaN/+inf/-inf policies, use masked exponentials and validate axis types. | Infinity mass sharing, undefined slices, empty reductions and extreme finite inputs under -W error |
| Public documentation incomplete | Complete preprocessing/composition method parameters, returns/shapes, exceptions, complexity and mutation/NaN contracts. | Runnable doctests, shape/error regression tests and manual contract review |
| Stale README and CSV examples | Replace outdated status/API/test/performance statements; make CSV examples self-contained. Add mixed numeric/category concatenation and in-sample metric qualification to README. | Both README Python examples execute; CSV documentation tests now run without skips |
| Missing automated checks | Add Linux minimum/current NumPy, Windows tests, fresh notebook execution, quick benchmarks and wheel-install smoke verification. | Workflow YAML parses; hosted runs remain pending publication |

The team's notebook was left unchanged because permission to edit it was
declined. The combined-feature example is instead implemented in the README
and `test_workflow_integration.py`, including alignment, finite outputs,
classification/count invariants and streaming/batch agreement.

## Genuine debugging evidence

Focused regression tests were first run against the unfixed implementations.
They reproduced seven failing pipeline cases, four utility cases, two scaler
cases, three metric cases and two CSV cases. The fixes were then made and the
relevant suites rerun before committing. Passing-before cases also protect
existing behavior and ensure internal exceptions are not retried.

Examples suitable for the testing section of the report:

1. **Lost distance:** points `[1e8, 1e8]` and `[1e8+1, 1e8]` returned distance
   zero. The squared-norm identity cancelled the small separation. Direct
   differences plus a stable hypot reduction now return one.
2. **False perfect accuracy:** integer labels `[2**53, 2**53+1]` compared with
   two floating `2**53` predictions returned accuracy 1.0. Exact label alignment
   now returns 0.5 and retains the two classes in the confusion matrix.
3. **Integer gaps:** an integer CSV with one incomplete row could not be skipped
   because NaN was converted to integer first. A missing-cell mask now permits
   skipping while preserving large integer values exactly.
4. **API composition:** a transformer with the specified `fit(X)` signature
   raised TypeError when a pipeline unconditionally supplied y. Signature
   binding now selects a compatible call without swallowing errors inside fit.

Earlier interpolation/subnormal and quantile-probability debugging evidence is
retained in `development.md` and the original checkpoint commits.

## Verification actually performed

| Check | Result |
| --- | --- |
| Full current-NumPy suite in a fresh environment | **764 passed, zero skips**, Python 3.11.13 / NumPy 2.4.6 / pytest 9.1.1, unexpected warnings treated as errors |
| Minimum NumPy compatibility in a separate environment | **764 passed, zero skips**, Python 3.11.13 / NumPy 1.24.4 |
| Dependency consistency | No broken requirements |
| Wheel | Built and installed in the temporary environment; imports and numerical smoke checks passed from outside the source checkout |
| README examples | Both complete Python examples passed, including mixed features and the pipeline/metrics example |
| Benchmark command | Full 18-case run completed, equality checks enabled; five timed rounds, one call per round, one warmup, seed 42 |
| Workflow configuration | YAML parsed and matrix/jobs checked locally |
| Notebook at merged baseline | 28 code cells executed successfully during the earlier audit |
| Notebook against updated wheel | **Pending:** permission to start the fresh verification kernel was declined. No post-fix notebook pass is claimed. CI includes the command for this check. |
| Hosted Linux/Windows/Python matrix | **Pending:** configuration committed locally; GitHub has not executed these new local commits |

The minimum-NumPy run verifies NumPy 1.24.4 on the available Python 3.11 runtime;
it does not independently verify Python 3.10 or Windows. These are covered by the
configured hosted matrix once published. The tracked notebook has not been
silently replaced with audit outputs.

## Measurements and tradeoffs

The updated benchmark CSV/Markdown contain measurements of the corrected code,
not the original Windows run. The CSV includes mean/std in addition to minimum
durations. README quotes a representative subset and links to the complete data.

- Machine: Darwin 27.0.0, arm64, eight logical CPUs.
- Python 3.11.13, NumPy 2.4.6.
- Main sizes: 1,000 / 10,000 / 100,000 rows; five numeric scaler features;
  distances to ten reference points in three dimensions.
- Gradient dimensions: 100 / 500 / 2,000, bounding quadratic perturbation storage.
- Five timed rounds, one call/round, one untimed warmup, seed 42.
- Every pair of implementations is checked for output agreement before timing.
- Input allocation/list conversion are outside the timed section.
- At 100,000 rows: scaler 12.8x, top-k 3.7x, rank 3.2x, running moments 43.4x
  and distances 7.8x versus the selected loop baselines.
- The 2,000-dimensional gradient batch achieved 1.1x. This compares callable
  dispatch strategies; both objectives use NumPy. Do not present it as a
  pure-Python scalar arithmetic versus NumPy comparison.

Stable distances deliberately trade speed for accuracy. Timing minima are not
universal speed guarantees; the saved environment/settings and variability
columns allow the group to interpret and reproduce the measurements honestly.

## Remaining group actions

1. Run the configured notebook check against the updated package and inspect
   the outputs. Publish the commits when ready, then inspect hosted checks.
2. Prepare Member B's report contribution using real design/debugging evidence;
   keep the combined report at most four pages.
3. Prepare and record the group demonstration; keep it at most five minutes.
   README contains the in-sample qualification for the example estimator even
   though the unchanged notebook has only its original comments.
4. Complete final group review, contribution credits for report/video work,
   sharing/access checks and the course submission.

No new optional numerical algorithms are needed before these deliverables.
