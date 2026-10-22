"""Sorting correctness, stability, dtype, shape and validation tests."""

import numpy as np
import pytest

from numcompute.sort_search import stable_sort


@pytest.mark.parametrize("axis", [None, 0, 1, -1, -2, np.int64(1)])
def test_stable_sort_matches_numpy_axes(axis):
    data = np.array([[3, 1, 1], [0, 5, 2]])
    np.testing.assert_array_equal(stable_sort(data, axis), np.sort(data, axis=axis, kind="stable"))


def test_stable_sort_retains_order_of_equal_signed_zeros():
    data = np.array([0.0, -0.0, 2.0, 0.0, -1.0, -0.0])
    result = stable_sort(data)
    np.testing.assert_array_equal(result, [-1.0, 0.0, -0.0, 0.0, -0.0, 2.0])
    np.testing.assert_array_equal(np.signbit(result[1:5]), [False, True, False, True])


@pytest.mark.parametrize(
    "data",
    [np.array([], dtype=np.int16), np.empty((0, 3)), np.empty((3, 0)), np.array([7]),
     np.array([4, 4, 4]), np.array([True, False, True])],
)
def test_stable_sort_empty_singleton_equal_and_boolean_inputs(data):
    result = stable_sort(data)
    np.testing.assert_array_equal(result, np.sort(data, kind="stable"))
    assert result.shape == data.shape
    assert result.dtype == data.dtype


def test_stable_sort_places_nans_after_infinities():
    result = stable_sort([np.nan, np.inf, 2, -np.inf, np.nan, -1])
    np.testing.assert_array_equal(result, [-np.inf, -1, 2, np.inf, np.nan, np.nan])


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_stable_sort_preserves_extreme_integers(dtype):
    limits = np.iinfo(dtype)
    data = np.array([limits.max, 2**53 + 1, 2**53, limits.min], dtype=dtype)
    result = stable_sort(data)
    np.testing.assert_array_equal(result, [limits.min, 2**53, 2**53 + 1, limits.max])
    assert result.dtype == dtype


def test_stable_sort_does_not_modify_or_share_memory_with_strided_input():
    original = np.arange(24).reshape(4, 6)
    before = original.copy()
    view = original.T[::-2]
    result = stable_sort(view, axis=0)
    np.testing.assert_array_equal(result, np.sort(view, axis=0, kind="stable"))
    np.testing.assert_array_equal(original, before)
    assert not np.shares_memory(result, original)


@pytest.mark.parametrize("axis", [True, np.bool_(False), 1.0, "0", [0]])
def test_stable_sort_rejects_noninteger_axes(axis):
    with pytest.raises(TypeError, match="axis must be an integer or None"):
        stable_sort([1, 2], axis=axis)


@pytest.mark.parametrize("axis", [1, -2, 100])
def test_stable_sort_reports_out_of_bounds_axis_and_input_shape(axis):
    with pytest.raises(ValueError, match=r"invalid for input shape \(2,\)"):
        stable_sort([1, 2], axis=axis)


@pytest.mark.parametrize("data", [["a", "b"], [1j]])
def test_stable_sort_rejects_nonnumeric_inputs(data):
    with pytest.raises(TypeError, match="real numeric"):
        stable_sort(data)


def test_stable_sort_rejects_scalars_and_ragged_inputs():
    with pytest.raises(ValueError, match="shape"):
        stable_sort(1)
    with pytest.raises(ValueError, match="rectangular"):
        stable_sort([[1], [2, 3]])
