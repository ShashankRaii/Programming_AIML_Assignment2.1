"""Axis, shape, NaN, dtype and error contracts for descriptive statistics."""

import numpy as np
import pytest

from numcompute import stats


REDUCTIONS = [stats.mean, stats.median, stats.std, stats.var, stats.minimum, stats.maximum]


@pytest.mark.parametrize("function", REDUCTIONS)
@pytest.mark.parametrize("axis", [None, 0, 1, -1])
@pytest.mark.parametrize("ignore_nan", [False, True])
def test_reductions_axes_and_nan_policy_match_numpy(function, axis, ignore_nan):
    data = np.array([[1, np.nan, 3], [4, 5, 6]], dtype=float)
    name = {"minimum": "min", "maximum": "max"}.get(function.__name__, function.__name__)
    reference = getattr(np, ("nan" if ignore_nan else "") + name)
    expected = reference(data, axis=axis)
    result = function(data, axis=axis, ignore_nan=ignore_nan)
    assert np.shape(result) == np.shape(expected)
    np.testing.assert_allclose(result, expected, equal_nan=True)


def test_reductions_hand_calculated_values_and_3d_shape_semantics():
    data = np.array([[1, 2], [3, 4]])
    np.testing.assert_array_equal(stats.mean(data, axis=0), [2, 3])
    np.testing.assert_array_equal(stats.median(data, axis=1), [1.5, 3.5])
    np.testing.assert_array_equal(stats.var(data, axis=0), [1, 1])
    np.testing.assert_array_equal(stats.std(data, axis=1), [0.5, 0.5])
    np.testing.assert_array_equal(stats.minimum(data, axis=0), [1, 2])
    np.testing.assert_array_equal(stats.maximum(data, axis=1), [2, 4])
    cube = np.arange(24).reshape(2, 3, 4)
    for function in REDUCTIONS:
        assert np.shape(function(cube)) == ()
        assert function(cube, axis=0).shape == (3, 4)
        assert function(cube, axis=1).shape == (2, 4)
        assert function(cube, axis=-1).shape == (2, 3)


@pytest.mark.parametrize("function", REDUCTIONS)
def test_reductions_all_nan_slice_warns_without_contaminating_other_slices(function):
    data = np.array([[1, np.nan], [3, np.nan]])
    with pytest.warns(RuntimeWarning):
        result = function(data, axis=0)
    assert np.isnan(result[1])
    assert result[0] == function([1, 3])


@pytest.mark.parametrize("function", [stats.mean, stats.median, stats.std, stats.var])
def test_empty_averaging_reductions_warn_and_return_nan(function):
    with pytest.warns(RuntimeWarning):
        assert np.isnan(function([]))
    with pytest.warns(RuntimeWarning):
        result = function(np.empty((0, 3)), axis=0)
    assert result.shape == (3,) and np.isnan(result).all()


@pytest.mark.parametrize("function", [stats.minimum, stats.maximum])
def test_empty_extrema_raise_but_empty_output_axes_are_allowed(function):
    with pytest.raises(ValueError, match="zero-size"):
        function([])
    assert function(np.empty((3, 0)), axis=0).shape == (0,)


@pytest.mark.parametrize("function", [stats.std, stats.var])
@pytest.mark.parametrize("ignore_nan", [False, True])
def test_variance_ddof_matches_numpy_and_corrects_per_slice_count(function, ignore_nan):
    data = np.array([[1.0, 5], [3, 7], [5, 9]])
    reference = getattr(np, ("nan" if ignore_nan else "") + function.__name__)
    np.testing.assert_allclose(function(data, axis=0, ddof=1, ignore_nan=ignore_nan),
                               reference(data, axis=0, ddof=1))
    data[2, 1] = np.nan
    result = function(data, axis=0, ddof=np.int64(1), ignore_nan=ignore_nan)
    np.testing.assert_allclose(result, reference(data, axis=0, ddof=1), equal_nan=True)


@pytest.mark.parametrize("function", [stats.std, stats.var])
def test_insufficient_degrees_of_freedom_warn_and_follow_numpy(function):
    with pytest.warns(RuntimeWarning):
        assert np.isnan(function([7], ddof=1))
    with pytest.warns(RuntimeWarning):
        assert np.isinf(function([1, 3], ddof=2, ignore_nan=False))


def test_reductions_singleton_constant_boolean_and_promoted_float32():
    for function in REDUCTIONS:
        expected = 0 if function in (stats.std, stats.var) else 7
        assert function([7]) == expected
        assert function([7, 7, 7]) == expected
    assert stats.mean([False, True]) == 0.5
    assert stats.median([False, True]) == 0.5
    data = np.array([1e8, 1, -1e8], dtype=np.float32)
    assert stats.mean(data) == 1 / 3
    for function in [stats.mean, stats.median, stats.std, stats.var]:
        assert function(data).dtype == np.float64


@pytest.mark.parametrize("function", [stats.minimum, stats.maximum])
@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_extrema_preserve_extreme_integer_dtype_and_precision(function, dtype):
    limit = np.iinfo(dtype)
    data = np.array([limit.min, 2**53, 2**53 + 1, limit.max], dtype=dtype)
    result = function(data)
    assert result.dtype == dtype
    assert int(result) == (limit.min if function is stats.minimum else limit.max)
    assert function(data[1:3]) == (data[1] if function is stats.minimum else data[2])


@pytest.mark.parametrize("function", REDUCTIONS)
def test_reductions_readonly_strided_inputs_are_not_modified(function):
    data = np.arange(24).reshape(4, 6)
    before = data.copy()
    data.flags.writeable = False
    result = function(data.T[::-2], axis=0)
    assert result.shape == (4,)
    np.testing.assert_array_equal(data, before)


def test_reductions_infinity_semantics_are_not_silently_changed():
    assert stats.mean([1, np.inf]) == np.inf
    assert stats.minimum([-np.inf, np.nan, 1]) == -np.inf
    assert stats.maximum([np.nan, np.inf, 1]) == np.inf
    with pytest.warns(RuntimeWarning):
        assert np.isnan(stats.var([1, np.inf]))


@pytest.mark.parametrize("axis", [True, np.bool_(False), 1.5, "0", (0, 1)])
def test_reductions_reject_unsupported_axis_types(axis):
    with pytest.raises(TypeError, match="axis must be an integer or None"):
        stats.mean([[1, 2]], axis=axis)


def test_reductions_report_bad_axis_shape_and_invalid_data():
    with pytest.raises(ValueError, match=r"axis 2 is invalid for data shape \(1, 2\)"):
        stats.mean([[1, 2]], axis=2)
    with pytest.raises(ValueError, match="shape"):
        stats.mean(1)
    with pytest.raises(ValueError, match="rectangular"):
        stats.mean([[1], [2, 3]])
    with pytest.raises(TypeError, match="real numeric"):
        stats.mean(["1"])
    with pytest.raises(TypeError, match="ignore_nan must be boolean"):
        stats.mean([1], ignore_nan=1)


@pytest.mark.parametrize("ddof", [True, 0.5, None, "1"])
def test_variance_rejects_unsupported_ddof_types(ddof):
    with pytest.raises(TypeError, match="ddof must be a nonnegative integer"):
        stats.var([1, 2], ddof=ddof)


def test_variance_rejects_negative_ddof():
    with pytest.raises(ValueError, match="ddof must be nonnegative; got -1"):
        stats.var([1, 2], ddof=-1)


@pytest.mark.parametrize("method", ["linear", "lower", "higher", "midpoint", "nearest"])
@pytest.mark.parametrize("axis", [None, 0, 1, -1])
def test_quantile_modes_axes_and_shapes_match_numpy(method, axis):
    data = np.random.default_rng(42).normal(size=(3, 4, 5))
    data[0, 1, 2] = np.nan
    q = [0, 0.1, 0.25, 0.5, 1]
    expected = np.nanquantile(data, q, axis=axis, method=method)
    result = stats.quantile(data, q, axis=axis, interpolation=method)
    assert result.shape == expected.shape
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
    scalar = stats.quantile(data, 0.5, axis=axis, interpolation=method)
    np.testing.assert_allclose(scalar, np.nanquantile(data, 0.5, axis=axis, method=method))


def test_quantile_does_not_round_probabilities_through_percentages():
    # q*100/100 can move an exact half-index to its other side.
    data = np.arange(4)
    q = (np.arange(3) + 0.5) / 3
    np.testing.assert_array_equal(stats.quantile(data, q, interpolation="nearest"),
                                  np.quantile(data, q, method="nearest"))


def test_quantile_empty_all_nan_and_empty_queries_have_defined_shapes():
    with pytest.warns(RuntimeWarning, match="empty or all-NaN"):
        result = stats.quantile([[1, np.nan], [3, np.nan]], [0, 0.5, 1], axis=0)
    np.testing.assert_array_equal(result[:, 0], [1, 2, 3])
    assert np.isnan(result[:, 1]).all()
    with pytest.warns(RuntimeWarning):
        assert np.isnan(stats.quantile([], 0.5))
    assert stats.quantile(np.empty((0, 3)), [], axis=0).shape == (0, 3)
    assert stats.quantile(np.empty((3, 0)), [0.5], axis=0).shape == (1, 0)


def test_quantile_reuses_finite_extreme_and_subnormal_stability():
    limit = np.finfo(float).max
    small = np.nextafter(0.0, 1.0)
    with np.errstate(over="raise", invalid="raise"):
        assert stats.quantile([-limit, limit], 0.5) == 0
    assert stats.quantile([small, 2 * small], 0.5) == 2 * small
    assert stats.quantile([np.inf, np.inf], 0.5) == np.inf
    with pytest.warns(RuntimeWarning, match="opposite infinities"):
        assert np.isnan(stats.quantile([-np.inf, np.inf], 0.5))


def test_quantile_preserves_readonly_strided_data_and_queries():
    data = np.arange(24).reshape(4, 6)
    before = data.copy()
    data.flags.writeable = False
    q = np.array([0.25, 0.75])
    q.flags.writeable = False
    np.testing.assert_allclose(stats.quantile(data.T[::-2], q, axis=0),
                               np.quantile(data.T[::-2], q, axis=0))
    np.testing.assert_array_equal(data, before)
    np.testing.assert_array_equal(q, [0.25, 0.75])


@pytest.mark.parametrize("q", [-0.1, 1.1, 50, np.nan, np.inf])
def test_quantile_rejects_nonfinite_and_out_of_range_probabilities(q):
    with pytest.raises(ValueError, match="finite probabilities between 0 and 1"):
        stats.quantile([1, 2], q)


@pytest.mark.parametrize("q", [True, "0.5", None, 1j])
def test_quantile_rejects_unsupported_probability_types(q):
    with pytest.raises(TypeError, match="real numeric probabilities"):
        stats.quantile([1, 2], q)


def test_quantile_rejects_invalid_axis_shape_and_method():
    with pytest.raises(ValueError, match="q must be scalar or 1-D"):
        stats.quantile([1, 2], [[0.5]])
    with pytest.raises(ValueError, match="axis 1 is invalid"):
        stats.quantile([1, 2], 0.5, axis=1)
    with pytest.raises(TypeError, match="axis must be an integer or None"):
        stats.quantile([1, 2], 0.5, axis=True)
    with pytest.raises(ValueError, match="unsupported interpolation"):
        stats.quantile([1, 2], 0.5, interpolation="unknown")


@pytest.mark.parametrize("bins,bounds", [(1, None), (10, None), (4, (-1, 1)),
                                          (4, (0, 0)), ([-2, -0.5, 0, 2], None)])
def test_histogram_random_nan_data_matches_numpy(bins, bounds):
    data = np.random.default_rng(42).normal(size=(20, 10))
    data[::3, 2] = np.nan
    expected_counts, expected_edges = np.histogram(data[~np.isnan(data)], bins=bins, range=bounds)
    counts, edges = stats.histogram(data, bins=bins, range=bounds)
    np.testing.assert_array_equal(counts, expected_counts)
    np.testing.assert_array_equal(edges, expected_edges)


def test_histogram_edges_are_left_inclusive_with_final_right_edge_included():
    data = [-1, 0, 1, 2, 3, 4, np.nan]
    counts, edges = stats.histogram(data, bins=[0, 1, 3])
    np.testing.assert_array_equal(counts, [1, 3])
    np.testing.assert_array_equal(edges, [0, 1, 3])
    assert counts.sum() == 4


@pytest.mark.parametrize("data", [[], [np.nan, np.nan]])
def test_histogram_empty_or_all_nan_has_zero_counts_and_default_range(data):
    counts, edges = stats.histogram(data, bins=2)
    np.testing.assert_array_equal(counts, [0, 0])
    np.testing.assert_array_equal(edges, [0, 0.5, 1])


def test_histogram_singleton_constant_and_boolean_inputs_match_numpy():
    for data in [[3], [7, 7, 7], [False, True, True]]:
        source = np.asarray(data, dtype=float)
        expected = np.histogram(source, bins=3)
        actual = stats.histogram(data, bins=3)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])


def test_histogram_infinity_requires_bounds_but_explicit_infinite_edges_are_valid():
    data = [-np.inf, -1, 0, 1, np.inf, np.nan]
    with pytest.raises(ValueError, match="finite range or explicit edges"):
        stats.histogram(data)
    counts, _ = stats.histogram(data, bins=2, range=(-1, 1))
    np.testing.assert_array_equal(counts, [1, 2])
    counts, edges = stats.histogram(data, bins=[-np.inf, 0, np.inf])
    np.testing.assert_array_equal(counts, [2, 3])
    np.testing.assert_array_equal(edges, [-np.inf, 0, np.inf])


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_histogram_preserves_extreme_integer_bin_boundaries(dtype):
    limit = np.iinfo(dtype)
    edges = np.array([limit.min, 2**53, 2**53 + 1, limit.max], dtype=dtype)
    counts, actual_edges = stats.histogram(edges, bins=edges)
    np.testing.assert_array_equal(counts, [1, 1, 2])
    np.testing.assert_array_equal(actual_edges, edges)
    assert actual_edges.dtype == dtype


def test_histogram_mixed_large_integer_float_boundaries_do_not_round_into_wrong_bins():
    values = np.array([2**53 + 3, 2**53 + 4, 2**53 + 5], dtype=np.int64)
    edges = np.array([2**53, 2**53 + 4, 2**53 + 8], dtype=float)
    counts, _ = stats.histogram(values, bins=edges)
    np.testing.assert_array_equal(counts, [1, 2])


def test_histogram_readonly_strides_and_explicit_edges_are_not_modified_or_shared():
    data = np.arange(20)
    bins = np.array([0, 5, 10, 19])
    data.flags.writeable = bins.flags.writeable = False
    counts, edges = stats.histogram(data[::-2], bins=bins)
    np.testing.assert_array_equal(counts, np.histogram(data[::-2], bins=bins)[0])
    assert not np.shares_memory(edges, bins)
    assert not np.shares_memory(counts, data)
    np.testing.assert_array_equal(data, np.arange(20))
    np.testing.assert_array_equal(bins, [0, 5, 10, 19])


def test_histogram_explicit_edges_take_precedence_over_range():
    counts, edges = stats.histogram([0, 1, 2], bins=[0, 1, 2], range=(10, 20))
    np.testing.assert_array_equal(counts, [1, 2])
    np.testing.assert_array_equal(edges, [0, 1, 2])


@pytest.mark.parametrize("bins", [0, -1])
def test_histogram_rejects_nonpositive_bin_count(bins):
    with pytest.raises(ValueError, match="bins must be positive"):
        stats.histogram([1], bins=bins)


@pytest.mark.parametrize("bins", [True, 2.5, "auto", [True, False], ["0", "1"]])
def test_histogram_rejects_unsupported_bin_types(bins):
    with pytest.raises(TypeError, match="bins must"):
        stats.histogram([1], bins=bins)


@pytest.mark.parametrize("bins", [[], [1], [[0, 1]], [0, 0, 1], [1, 0], [0, np.nan, 1]])
def test_histogram_rejects_bad_edge_shapes_and_order(bins):
    with pytest.raises(ValueError, match="bin edges must"):
        stats.histogram([1], bins=bins)


@pytest.mark.parametrize("bounds", [(1, 0), (0, np.inf), (np.nan, 1)])
def test_histogram_rejects_nonfinite_or_reversed_range(bounds):
    with pytest.raises(ValueError, match="finite ordered bounds"):
        stats.histogram([1], range=bounds)


def test_histogram_rejects_invalid_range_shape_and_unsupported_data():
    with pytest.raises(ValueError, match=r"range must have shape \(2,\)"):
        stats.histogram([1], range=[0, 1, 2])
    with pytest.raises(TypeError, match="range must contain real numeric"):
        stats.histogram([1], range=["0", "1"])
    with pytest.raises(TypeError, match="real numeric"):
        stats.histogram(["1"])
    with pytest.raises(ValueError, match="shape"):
        stats.histogram(1)
