"""Acceptance workflows across Member B's sorting, ranking and statistics APIs."""

import numpy as np
import pytest

from numcompute import stats
from numcompute.rank import percentile, rank
from numcompute.sort_search import binary_search, multi_key_sort, quickselect, stable_sort, topk


def test_score_analysis_combines_selection_ranks_distribution_and_streaming():
    scores = np.array([3.0, np.nan, -1, 7, 3, 1])
    before = scores.copy()
    scores.flags.writeable = False
    ordered = stable_sort(scores)
    np.testing.assert_array_equal(ordered, [-1, 1, 3, 3, 7, np.nan])
    np.testing.assert_array_equal(rank(scores, "ordinal"), [3, np.nan, 1, 5, 4, 2])
    values, indices = topk(scores, 3)
    np.testing.assert_array_equal(values, [7, 3, 3])
    np.testing.assert_array_equal(indices, [3, 0, 4])
    assert quickselect(scores, 2) == 3
    assert np.isnan(quickselect(scores, 5))
    positions, found = binary_search(ordered, [3, -2, np.nan])
    np.testing.assert_array_equal(positions, [2, 0, 5])
    np.testing.assert_array_equal(found, [True, False, False])

    edges = stats.quantile(scores, [0, 0.5, 1])
    np.testing.assert_array_equal(edges, percentile(scores, [0, 50, 100]))
    counts, _ = stats.histogram(scores, bins=edges)
    np.testing.assert_array_equal(counts, [2, 3])
    first = stats.RunningStats().update(scores[:3])
    second = stats.RunningStats().update(scores[3:])
    first.merge(second)
    assert first.count == int(counts.sum()) == 5
    np.testing.assert_allclose(first.mean, stats.mean(scores), rtol=1e-12)
    np.testing.assert_allclose(first.variance, stats.var(scores), rtol=1e-12)
    np.testing.assert_array_equal(scores, before)


def test_all_nan_workflow_preserves_consistent_missing_data_policies():
    data = np.full(3, np.nan)
    assert np.isnan(stable_sort(data)).all()
    assert np.isnan(rank(data)).all()
    selected, indices = topk(data, 2)
    assert np.isnan(selected).all()
    np.testing.assert_array_equal(indices, [0, 1])
    assert np.isnan(quickselect(data, 0))
    assert binary_search(stable_sort(data), np.nan) == (0, False)
    with pytest.warns(RuntimeWarning, match="all-NaN"):
        assert np.isnan(stats.quantile(data, 0.5))
    with pytest.warns(RuntimeWarning):
        assert np.isnan(stats.mean(data))
    counts, _ = stats.histogram(data, bins=3)
    assert counts.sum() == 0
    running = stats.RunningStats().update(data)
    assert running.count == 0
    assert np.isnan(running.mean) and np.isnan(running.variance)


def test_empty_workflow_has_explicit_shapes_and_selection_errors():
    data = np.empty(0)
    assert stable_sort(data).shape == rank(data).shape == (0,)
    values, indices = topk(data, 0)
    assert values.shape == indices.shape == (0,)
    assert binary_search(data, 1) == (0, False)
    with pytest.raises(ValueError, match="nonempty"):
        quickselect(data, 0)
    assert stats.quantile(data, []).shape == (0,)
    counts, edges = stats.histogram(data, bins=3)
    assert counts.shape == (3,) and edges.shape == (4,) and counts.sum() == 0
    running = stats.RunningStats().update(data).merge(stats.RunningStats())
    assert running.count == 0 and np.isnan(running.variance)


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_large_integer_ordering_and_centered_streaming_keep_small_differences(dtype):
    base = 2**53
    data = np.array([base + 3, base, base + 1, base + 1], dtype=dtype)
    before = data.copy()
    np.testing.assert_array_equal(rank(data), [4, 1, 2.5, 2.5])
    ordered = stable_sort(data)
    assert binary_search(ordered, dtype(base + 1)) == (1, True)
    values, indices = topk(data, 2)
    np.testing.assert_array_equal(indices, [0, 2])
    np.testing.assert_array_equal(values, data[indices])
    assert quickselect(data, 2) == dtype(base + 1)
    edges = np.array([base, base + 1, base + 4], dtype=dtype)
    counts, _ = stats.histogram(data, bins=edges)
    np.testing.assert_array_equal(counts, [1, 3])

    # Center in integer arithmetic before floating accumulation when tiny
    # differences beyond float64's integer precision are the quantity of interest.
    centered = data - stats.minimum(data)
    running = stats.RunningStats().update(centered[:2]).update(centered[2:])
    assert running.count == 4
    assert running.mean == 1.25 and running.variance == 1.1875
    np.testing.assert_allclose(running.variance, stats.var(centered))
    np.testing.assert_array_equal(data, before)


def test_multi_key_row_order_preserves_column_distributions_and_streaming_results():
    records = np.array([[2, 5, 0], [1, 3, 1], [2, 4, 2], [1, 3, 3], [1, 1, 4]])
    ordered = multi_key_sort(records, [0, 1], [False, True])
    np.testing.assert_array_equal(ordered[:, 2], [1, 3, 4, 0, 2])
    np.testing.assert_array_equal(stats.mean(ordered, axis=0), stats.mean(records, axis=0))
    for method in ["linear", "lower", "higher", "midpoint", "nearest"]:
        np.testing.assert_array_equal(
            stats.quantile(ordered[:, :2], [0, 0.5, 1], axis=0, interpolation=method),
            stats.quantile(records[:, :2], [0, 0.5, 1], axis=0, interpolation=method),
        )
    original_stream = stats.RunningStats().update(records[:, 1])
    sorted_stream = stats.RunningStats().update(ordered[:2, 1]).update(ordered[2:, 1])
    np.testing.assert_allclose(sorted_stream.mean, original_stream.mean)
    np.testing.assert_allclose(sorted_stream.variance, original_stream.variance)


def test_per_column_streams_match_axis_reductions_with_an_all_nan_feature():
    data = np.array([[1, 6, np.nan], [np.nan, 4, np.nan], [3, 2, np.nan], [5, np.nan, np.nan]])
    streams = []
    for column in range(data.shape[1]):
        first = stats.RunningStats().update(data[:2, column])
        second = stats.RunningStats().update(data[2:, column])
        streams.append(first.merge(second))
    np.testing.assert_array_equal([stream.count for stream in streams], [3, 3, 0])
    with pytest.warns(RuntimeWarning):
        expected_mean = stats.mean(data, axis=0)
    with pytest.warns(RuntimeWarning):
        expected_var = stats.var(data, axis=0)
    np.testing.assert_allclose([stream.mean for stream in streams], expected_mean, equal_nan=True)
    np.testing.assert_allclose(
        [stream.variance for stream in streams], expected_var, equal_nan=True
    )
