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
