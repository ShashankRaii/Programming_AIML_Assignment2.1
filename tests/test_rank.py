"""Tie-handling contracts and independent reference checks for ranking."""

import numpy as np
import pytest

from numcompute.rank import _percentile, percentile, rank


@pytest.mark.parametrize(
    "method,expected",
    [
        ("average", [4, 1.5, 1.5, 3]),
        ("dense", [3, 1, 1, 2]),
        ("ordinal", [4, 1, 2, 3]),
        ("min", [4, 1, 1, 3]),
        ("max", [4, 2, 2, 3]),
    ],
)
def test_rank_hand_calculated_tie_methods(method, expected):
    np.testing.assert_array_equal(rank([30, 10, 10, 20], method), expected)


@pytest.mark.parametrize("method", ["average", "dense", "ordinal", "min", "max"])
def test_rank_random_input_matches_independent_counting_reference(method):
    data = np.random.default_rng(42).integers(-5, 6, size=60)
    expected = []
    for index, value in enumerate(data):
        lower = sum(int(other) < int(value) for other in data)
        equal = sum(int(other) == int(value) for other in data)
        if method == "dense":
            expected.append(1 + len({int(other) for other in data if other < value}))
        elif method == "ordinal":
            expected.append(lower + sum(other == value for other in data[:index]) + 1)
        elif method == "min":
            expected.append(lower + 1)
        elif method == "max":
            expected.append(lower + equal)
        else:
            expected.append(lower + (equal + 1) / 2)
    np.testing.assert_array_equal(rank(data, method), expected)


@pytest.mark.parametrize("method", ["average", "dense", "ordinal", "min", "max"])
def test_rank_nan_positions_do_not_consume_ranks(method):
    data = np.array([np.nan, 3, 1, np.nan, 1])
    result = rank(data, method)
    assert np.isnan(result[[0, 3]]).all()
    np.testing.assert_array_equal(result[[1, 2, 4]], rank([3, 1, 1], method))


@pytest.mark.parametrize("method", ["average", "dense", "ordinal", "min", "max"])
def test_rank_empty_all_nan_singleton_and_equal_inputs(method):
    assert rank([], method).shape == (0,)
    assert rank([], method).dtype == np.float64
    assert np.isnan(rank([np.nan, np.nan], method)).all()
    np.testing.assert_array_equal(rank([9], method), [1])
    expected = {
        "average": [2, 2, 2], "dense": [1, 1, 1], "ordinal": [1, 2, 3],
        "min": [1, 1, 1], "max": [3, 3, 3],
    }
    np.testing.assert_array_equal(rank([7, 7, 7], method), expected[method])


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_rank_large_integers_remain_distinct(dtype):
    limits = np.iinfo(dtype)
    data = np.array([limits.max, 2**53 + 1, 2**53, limits.min], dtype=dtype)
    np.testing.assert_array_equal(rank(data), [4, 3, 2, 1])


def test_rank_infinities_signed_zeros_and_boolean_values():
    np.testing.assert_array_equal(rank([np.inf, -np.inf, np.inf, 0]), [3.5, 1, 3.5, 2])
    np.testing.assert_array_equal(rank([-0.0, 0.0, -0.0]), [2, 2, 2])
    np.testing.assert_array_equal(rank([True, False, True]), [2.5, 1, 2.5])


def test_rank_preserves_readonly_strided_input_and_returns_independent_output():
    data = np.arange(12)
    before = data.copy()
    data.flags.writeable = False
    view = data[::-2]
    result = rank(view)
    np.testing.assert_array_equal(result, [6, 5, 4, 3, 2, 1])
    np.testing.assert_array_equal(data, before)
    assert not np.shares_memory(result, data)


def test_rank_rejects_unknown_or_nonstr_method():
    with pytest.raises(ValueError, match="unsupported rank method 'unknown'"):
        rank([1, 2], "unknown")
    with pytest.raises(TypeError, match="method must be a string"):
        rank([1, 2], ["average"])


def test_rank_rejects_invalid_shapes_and_nonnumeric_input():
    with pytest.raises(ValueError, match="shape"):
        rank(1)
    with pytest.raises(ValueError, match=r"1 dimension; got shape \(1, 2\)"):
        rank([[1, 2]])
    with pytest.raises(ValueError, match="rectangular"):
        rank([[1], [2, 3]])
    with pytest.raises(TypeError, match="real numeric"):
        rank(["one", "two"])


@pytest.mark.parametrize(
    "method,expected",
    [
        ("linear", [10, 17.5, 25, 32.5, 40]),
        ("lower", [10, 10, 20, 30, 40]),
        ("higher", [10, 20, 30, 40, 40]),
        ("midpoint", [10, 15, 25, 35, 40]),
        ("nearest", [10, 20, 30, 30, 40]),
    ],
)
def test_percentile_hand_calculated_interpolations(method, expected):
    np.testing.assert_allclose(percentile([40, 10, 30, 20], [0, 25, 50, 75, 100], method), expected)


@pytest.mark.parametrize("method", ["linear", "lower", "higher", "midpoint", "nearest"])
def test_percentile_random_nan_data_matches_numpy_reference(method):
    data = np.random.default_rng(42).normal(size=101)
    data[::7] = np.nan
    q = np.array([100, 0, 2.5, 25, 50, 75, 98, 50])
    expected = np.nanpercentile(data, q, method=method)
    np.testing.assert_allclose(percentile(data, q, method), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("method", ["linear", "lower", "higher", "midpoint", "nearest"])
@pytest.mark.parametrize("axis", [None, 0, 1, -1])
def test_percentile_shared_axis_kernel_matches_numpy_and_keeps_shapes(method, axis):
    # This kernel will be reused by stats.quantile in checkpoint 6.
    data = np.random.default_rng(42).normal(size=(5, 4))
    data[::2, 1] = np.nan
    q = [0, 12.5, 50, 100]
    expected = np.nanpercentile(data, q, axis=axis, method=method)
    result = _percentile(data, q, interpolation=method, axis=axis)
    assert result.shape == expected.shape
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
    scalar = _percentile(data, 50, interpolation=method, axis=axis)
    np.testing.assert_allclose(scalar, np.nanpercentile(data, 50, axis=axis, method=method))


def test_percentile_nearest_half_indices_round_to_even():
    assert percentile([10, 20], 50, "nearest") == 10
    assert percentile([10, 20, 30, 40], 50, "nearest") == 30


def test_percentile_scalar_vector_and_empty_query_shapes():
    result = percentile([1, 3], 50)
    assert isinstance(result, np.floating) and np.shape(result) == ()
    assert percentile([1, 3], [50]).shape == (1,)
    assert percentile([1, 3], []).shape == (0,)
    assert percentile([], []).shape == (0,)
    assert percentile([np.nan], []).shape == (0,)


@pytest.mark.parametrize("data", [[], [np.nan, np.nan]])
@pytest.mark.parametrize("q", [50, [0, 50, 100]])
def test_percentile_empty_or_all_nan_returns_nan_and_warns(data, q):
    with pytest.warns(RuntimeWarning, match="empty or all-NaN"):
        result = percentile(data, q)
    assert np.isnan(result).all()
    assert np.shape(result) == np.shape(q)


def test_percentile_kernel_handles_missing_slices_without_contaminating_others():
    data = np.array([[1, np.nan], [3, np.nan], [np.nan, np.nan]])
    with pytest.warns(RuntimeWarning, match="empty or all-NaN"):
        result = _percentile(data, [0, 50, 100], axis=0)
    np.testing.assert_array_equal(result[:, 0], [1, 2, 3])
    assert np.isnan(result[:, 1]).all()


def test_percentile_kernel_empty_dimensions_and_query_shapes():
    with pytest.warns(RuntimeWarning, match="empty or all-NaN"):
        result = _percentile(np.empty((0, 3)), [25, 50], axis=0)
    assert result.shape == (2, 3) and np.isnan(result).all()
    assert _percentile(np.empty((3, 0)), [25, 50], axis=0).shape == (2, 0)
    assert _percentile(np.empty((0, 3)), [], axis=0).shape == (0, 3)


@pytest.mark.parametrize("method", ["linear", "lower", "higher", "midpoint", "nearest"])
def test_percentile_singleton_equal_and_boolean_samples(method):
    np.testing.assert_array_equal(percentile([17], [0, 25, 100], method), [17, 17, 17])
    np.testing.assert_array_equal(percentile([7, 7, 7], [0, 50, 100], method), [7, 7, 7])
    expected = {"linear": 0.5, "lower": 0, "higher": 1, "midpoint": 0.5, "nearest": 0}
    assert percentile([False, True], 50, method) == expected[method]


def test_percentile_avoids_overflow_for_finite_extremes():
    limit = np.finfo(np.float64).max
    with np.errstate(over="raise", invalid="raise"):
        assert percentile([-limit, limit], 50) == 0
        assert percentile([-limit, limit], 50, "midpoint") == 0
        np.testing.assert_array_equal(percentile([limit, limit], [0, 50, 100]), [limit] * 3)
        assert np.isfinite(percentile([limit / 2, limit], 75))


@pytest.mark.parametrize("method", ["linear", "midpoint"])
def test_percentile_midpoint_retains_subnormal_precision(method):
    small = np.nextafter(0.0, 1.0)
    assert percentile([small, 2 * small], 50, method) == 2 * small


@pytest.mark.parametrize("method", ["linear", "midpoint"])
def test_percentile_infinities_have_explicit_endpoint_and_interior_semantics(method):
    with np.errstate(over="raise", invalid="raise"):
        np.testing.assert_array_equal(
            percentile([-np.inf, 0, np.inf], [0, 25, 50, 75, 100], method),
            [-np.inf, -np.inf, 0, np.inf, np.inf],
        )
        assert percentile([np.inf, np.inf], 50, method) == np.inf
        assert percentile([-np.inf, -np.inf], 50, method) == -np.inf
        np.testing.assert_array_equal(percentile([-np.inf, np.inf], [0, 100], method),
                                      [-np.inf, np.inf])
        with pytest.warns(RuntimeWarning, match="opposite infinities"):
            assert np.isnan(percentile([-np.inf, np.inf], 50, method))


@pytest.mark.parametrize("method", ["lower", "higher", "nearest"])
def test_percentile_discrete_methods_select_infinity_without_interpolation(method):
    expected = -np.inf if method in ("lower", "nearest") else np.inf
    assert percentile([-np.inf, np.inf], 50, method) == expected


@pytest.mark.parametrize("dtype", [np.int64, np.uint64, np.float32])
def test_percentile_preserves_input_and_documents_floating_output(dtype):
    data = np.array([8, 2, 4, 6], dtype=dtype)
    before = data.copy()
    result = percentile(data, [0, 50, 100])
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, [2, 5, 8])
    np.testing.assert_array_equal(data, before)


def test_percentile_readonly_strided_matrix_and_queries_are_preserved():
    data = np.arange(24).reshape(4, 6)
    data.flags.writeable = False
    view = data.T[::-2]
    q = np.array([0.0, 25.0, 50.0, 100.0])
    q.flags.writeable = False
    np.testing.assert_allclose(percentile(view, q), np.percentile(view, q))
    np.testing.assert_array_equal(data, np.arange(24).reshape(4, 6))
    np.testing.assert_array_equal(q, [0, 25, 50, 100])


@pytest.mark.parametrize("q", [-1, 101, np.nan, np.inf, [0, np.nan], [-1, 50]])
def test_percentile_rejects_invalid_percentages(q):
    with pytest.raises(ValueError, match="finite percentages between 0 and 100"):
        percentile([1, 2], q)


@pytest.mark.parametrize("q", ["50", ["50"], True, 1j, None])
def test_percentile_rejects_unsupported_percentage_types(q):
    with pytest.raises(TypeError, match="real numeric percentages"):
        percentile([1, 2], q)


def test_percentile_rejects_bad_shapes_and_methods():
    with pytest.raises(ValueError, match="q must be scalar or 1-D"):
        percentile([1, 2], [[50]])
    with pytest.raises(ValueError, match="shape"):
        percentile(1, 50)
    with pytest.raises(TypeError, match="real numeric"):
        percentile(["1"], 50)
    with pytest.raises(ValueError, match="unsupported interpolation 'unknown'"):
        percentile([1], 50, "unknown")
    with pytest.raises(TypeError, match="interpolation must be a string"):
        percentile([1], 50, ["linear"])


@pytest.mark.parametrize("axis", [True, 0.5, "0"])
def test_percentile_shared_kernel_rejects_bad_axis_types(axis):
    with pytest.raises(TypeError, match="axis must be an integer or None"):
        _percentile([1, 2], 50, axis=axis)


def test_percentile_shared_kernel_reports_out_of_bounds_axis():
    with pytest.raises(ValueError, match=r"axis 1 is invalid for data shape \(2,\)"):
        _percentile([1, 2], 50, axis=1)
