"""Batch/stream equivalence, merging and state safety for Welford statistics."""

import numpy as np
import pytest

from numcompute.stats import RunningStats


def snapshot(running):
    """Read the public summary for state-preservation checks."""
    return running.count, running.mean, running.variance


def test_running_stats_empty_singleton_and_population_variance():
    running = RunningStats()
    assert running.count == 0
    assert type(running.count) is int
    assert np.isnan(running.mean) and np.isnan(running.variance)
    assert running.update([3]) is running
    assert snapshot(running) == (1, 3, 0)
    running.update([1])
    assert snapshot(running) == (2, 2, 1)  # Population, not sample variance 2.


@pytest.mark.parametrize("chunk_size", [1, 2, 7, 32, 257])
def test_running_stats_chunked_updates_match_numpy(chunk_size):
    data = np.random.default_rng(42).normal(size=257)
    data[::11] = np.nan
    running = RunningStats()
    for start in range(0, data.size, chunk_size):
        running.update(data[start:start + chunk_size])
    assert running.count == np.count_nonzero(~np.isnan(data))
    np.testing.assert_allclose(running.mean, np.nanmean(data), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(running.variance, np.nanvar(data), rtol=1e-12)


def test_running_stats_large_offset_retains_small_variance():
    data = 1e12 + np.arange(20, dtype=float)
    running = RunningStats().update(data[:3]).update(data[3:8]).update(data[8:])
    assert running.mean == 1e12 + 9.5
    np.testing.assert_allclose(running.variance, 33.25, rtol=1e-12)


def test_running_stats_constant_near_maximum_avoids_sum_overflow():
    limit = np.finfo(float).max
    running = RunningStats().update([limit, limit]).update([limit])
    assert snapshot(running) == (3, limit, 0)


def test_running_stats_merge_scales_before_squaring_delta():
    first = RunningStats().update([0.0])
    second = RunningStats().update([1.6e154])
    first.merge(second)
    assert first.mean == 8e153
    np.testing.assert_allclose(first.variance, 6.4e307, rtol=1e-15)


@pytest.mark.parametrize("dtype", [bool, np.int32, np.float32, np.longdouble])
def test_running_stats_numeric_batches_use_at_least_float64(dtype):
    data = np.array([0, 1, 1, 0], dtype=dtype)
    running = RunningStats().update(data)
    assert snapshot(running) == (4, 0.5, 0.25)
    assert running.mean.dtype == np.result_type(dtype, np.float64)


def test_running_stats_nan_and_empty_batches_are_noops_before_and_after_data():
    running = RunningStats()
    assert running.update([]) is running
    assert running.update([np.nan, np.nan]) is running
    assert running.count == 0 and np.isnan(running.mean)
    running.update([1, 3])
    before = snapshot(running)
    assert running.update([np.nan]) is running
    assert running.update([]) is running
    assert snapshot(running) == before


def test_running_stats_does_not_retain_or_modify_caller_data():
    data = np.arange(20, dtype=float)
    before = data.copy()
    data.flags.writeable = False
    view = data[::-3]
    running = RunningStats().update(view)
    np.testing.assert_allclose(running.variance, np.var(view))
    np.testing.assert_array_equal(data, before)
    mutable = np.array([1.0, 3.0])
    other = RunningStats().update(mutable)
    mutable[:] = 99
    assert snapshot(other) == (2, 2, 1)


def test_running_stats_merge_matches_full_data_and_preserves_source():
    data = np.random.default_rng(42).normal(size=103)
    data[::9] = np.nan
    first = RunningStats().update(data[:37])
    second = RunningStats().update(data[37:])
    source_before = snapshot(second)
    assert first.merge(second) is first
    assert snapshot(second) == source_before
    assert first.count == np.count_nonzero(~np.isnan(data))
    np.testing.assert_allclose(first.mean, np.nanmean(data), atol=1e-15)
    np.testing.assert_allclose(first.variance, np.nanvar(data), rtol=1e-12)


def test_running_stats_empty_merge_copy_and_self_merge_behaviour():
    source = RunningStats().update([1, 3])
    empty = RunningStats()
    assert empty.merge(RunningStats()) is empty
    assert empty.count == 0 and np.isnan(empty.variance)
    target = RunningStats().merge(source)
    before = snapshot(target)
    source.update([5])
    assert snapshot(target) == before
    assert target.merge(RunningStats()) is target
    assert snapshot(target) == before
    assert target.merge(target) is target
    assert snapshot(target) == (4, 2, 1)


def test_running_stats_merge_groupings_agree_within_roundoff():
    data = np.random.default_rng(42).normal(size=100)
    left = RunningStats().update(data[:20]).merge(RunningStats().update(data[20:]))
    right = RunningStats().update(data[20:]).merge(RunningStats().update(data[:20]))
    whole = RunningStats().update(data)
    np.testing.assert_allclose(snapshot(left), snapshot(whole), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(snapshot(right), snapshot(whole), rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("batch", [[np.inf], [-np.inf], [1, np.nan, np.inf]])
def test_running_stats_rejects_infinity_without_changing_state(batch):
    running = RunningStats().update([1, 3])
    before = snapshot(running)
    with pytest.raises(ValueError, match="infinite"):
        running.update(batch)
    assert snapshot(running) == before


@pytest.mark.parametrize("batch", [1, [[1, 2]], [[1], [2, 3]], ["1"], [1j], [None]])
def test_running_stats_bad_inputs_leave_state_unchanged(batch):
    running = RunningStats().update([1, 3])
    before = snapshot(running)
    with pytest.raises((ValueError, TypeError)):
        running.update(batch)
    assert snapshot(running) == before


def test_running_stats_batch_overflow_is_atomic():
    running = RunningStats().update([1, 3])
    before = snapshot(running)
    limit = np.finfo(float).max
    with pytest.raises(OverflowError, match="floating range"):
        running.update([-limit, limit])
    assert snapshot(running) == before


def test_running_stats_unrepresentable_centered_moment_is_rejected_atomically():
    running = RunningStats().update([1, 3])
    before = snapshot(running)
    with pytest.raises(OverflowError, match="centered moment"):
        running.update([-1e154, 1e154])
    assert snapshot(running) == before


def test_running_stats_merge_overflow_leaves_both_summaries_unchanged():
    limit = np.finfo(float).max
    first = RunningStats().update([limit])
    second = RunningStats().update([-limit])
    before_first, before_second = snapshot(first), snapshot(second)
    with pytest.raises(OverflowError, match="floating range"):
        first.merge(second)
    assert snapshot(first) == before_first
    assert snapshot(second) == before_second


def test_running_stats_rejects_other_merge_types_without_mutation():
    running = RunningStats().update([1, 3])
    before = snapshot(running)
    with pytest.raises(TypeError, match="other must be RunningStats; got list"):
        running.merge([1, 3])
    assert snapshot(running) == before


@pytest.mark.parametrize("name", ["count", "mean", "variance"])
def test_running_stats_public_summary_properties_are_readonly(name):
    running = RunningStats().update([1, 3])
    with pytest.raises(AttributeError):
        setattr(running, name, 99)
    assert snapshot(running) == (2, 2, 1)
