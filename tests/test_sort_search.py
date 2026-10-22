"""Sorting correctness, stability, dtype, shape and validation tests."""

import numpy as np
import pytest

from numcompute.sort_search import multi_key_sort, stable_sort


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


def test_multi_key_sort_first_key_is_primary_and_retains_complete_ties():
    # The last column is a row ID, deliberately excluded from the sort keys.
    data = np.array([[2, 1, 0], [1, 3, 1], [1, 2, 2], [1, 2, 3], [2, 0, 4]])
    np.testing.assert_array_equal(multi_key_sort(data, [0, 1]), data[[2, 3, 1, 4, 0]])
    np.testing.assert_array_equal(multi_key_sort(data, [1, 0]), data[[4, 0, 2, 3, 1]])


@pytest.mark.parametrize("descending", [[False, False], [True, False], [False, True], [True, True]])
def test_multi_key_sort_directions_match_independent_python_reference(descending):
    rng = np.random.default_rng(42)
    data = np.column_stack([rng.integers(-4, 5, size=(80, 2)), np.arange(80)])
    # Python's stable sort and unbounded ints form an independent reference.
    expected = sorted(
        range(len(data)),
        key=lambda row: tuple(
            -int(data[row, col]) if descending[col] else int(data[row, col])
            for col in [0, 1]
        ),
    )
    np.testing.assert_array_equal(multi_key_sort(data, [0, 1], descending), data[expected])


@pytest.mark.parametrize("descending", [[False, False], [True, False], [False, True], [True, True]])
def test_multi_key_sort_nan_and_infinity_order_matches_reference(descending):
    data = np.array(
        [[np.nan, 3, 0], [1, np.nan, 1], [np.inf, 2, 2], [-np.inf, 4, 3],
         [np.nan, 1, 4], [1, 2, 5], [1, 2, 6], [np.nan, np.nan, 7]]
    )

    def reference_key(row):
        return tuple(
            (True, 0) if np.isnan(data[row, col]) else
            (False, -data[row, col] if descending[col] else data[row, col])
            for col in [0, 1]
        )

    expected = sorted(range(len(data)), key=reference_key)
    np.testing.assert_array_equal(multi_key_sort(data, [0, 1], descending), data[expected])


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_multi_key_sort_descending_preserves_integer_extremes_and_ties(dtype):
    limits = np.iinfo(dtype)
    data = np.array(
        [[limits.min, 0], [limits.max, 1], [2**53, 2], [2**53 + 1, 3], [limits.max, 4]],
        dtype=dtype,
    )
    result = multi_key_sort(data, [0], [True])
    np.testing.assert_array_equal(result, data[[1, 4, 3, 2, 0]])
    assert result.dtype == dtype


def test_multi_key_sort_single_row_and_all_equal_keys():
    singleton = np.array([[2, 3]])
    np.testing.assert_array_equal(multi_key_sort(singleton, [0], [True]), singleton)
    data = np.column_stack([np.ones(6), np.arange(6)])
    np.testing.assert_array_equal(multi_key_sort(data, [0], [True]), data)


def test_multi_key_sort_all_nan_primary_uses_secondary_key():
    data = np.array([[np.nan, 1, 0], [np.nan, 3, 1], [np.nan, 1, 2]])
    np.testing.assert_array_equal(multi_key_sort(data, [0, 1], [True, True]), data[[1, 0, 2]])


def test_multi_key_sort_signed_zeros_remain_tied():
    data = np.array([[-0.0, 0], [0.0, 1], [-0.0, 2], [0.0, 3]])
    result = multi_key_sort(data, [0], [True])
    np.testing.assert_array_equal(result[:, 1], [0, 1, 2, 3])
    np.testing.assert_array_equal(np.signbit(result[:, 0]), [True, False, True, False])


def test_multi_key_sort_negative_keys_and_numpy_flags():
    data = np.array([[1, 3], [2, 1], [1, 2]])
    result = multi_key_sort(data, np.array([-2, -1]), np.array([False, True]))
    np.testing.assert_array_equal(result, data[[0, 2, 1]])


def test_multi_key_sort_supports_boolean_data():
    data = np.array([[False, True], [True, False], [False, False]])
    result = multi_key_sort(data, [0, 1], [True, False])
    np.testing.assert_array_equal(result, data[[1, 2, 0]])
    assert result.dtype == bool


@pytest.mark.parametrize("descending", [None, [True]])
def test_multi_key_sort_empty_rows_preserve_matrix_shape(descending):
    data = np.empty((0, 3), dtype=np.int16)
    result = multi_key_sort(data, [1], descending)
    assert result.shape == (0, 3)
    assert result.dtype == np.int16


def test_multi_key_sort_does_not_mutate_or_share_memory_with_strided_input():
    original = np.arange(48).reshape(8, 6)
    before = original.copy()
    view = original[::-2, ::2]
    result = multi_key_sort(view, [0, 1], [False, True])
    np.testing.assert_array_equal(result, view[::-1])
    np.testing.assert_array_equal(original, before)
    assert not np.shares_memory(result, original)


@pytest.mark.parametrize("keys", [[], 0, [[0, 1]]])
def test_multi_key_sort_rejects_invalid_key_shapes(keys):
    with pytest.raises(ValueError, match="nonempty 1-D"):
        multi_key_sort([[1, 2]], keys)


@pytest.mark.parametrize("keys", [[True], [0, True], [0.0], ["0"]])
def test_multi_key_sort_rejects_noninteger_keys(keys):
    with pytest.raises(TypeError, match="integer column indices"):
        multi_key_sort([[1, 2]], keys)


@pytest.mark.parametrize("keys", [[0, 0], [1, -1]])
def test_multi_key_sort_rejects_duplicate_columns(keys):
    with pytest.raises(ValueError, match="distinct columns"):
        multi_key_sort([[1, 2]], keys)


@pytest.mark.parametrize("key", [2, -3])
def test_multi_key_sort_reports_out_of_bounds_column_and_shape(key):
    with pytest.raises(ValueError, match=r"out of bounds for X shape \(1, 2\)"):
        multi_key_sort([[1, 2]], [key])


def test_multi_key_sort_rejects_keys_for_matrix_without_columns():
    with pytest.raises(ValueError, match="out of bounds"):
        multi_key_sort(np.empty((3, 0)), [0])


@pytest.mark.parametrize("flags", [True, [], [True, False], [[True]]])
def test_multi_key_sort_rejects_wrong_direction_shapes(flags):
    with pytest.raises(ValueError, match="one flag per key"):
        multi_key_sort([[1, 2]], [0], flags)


@pytest.mark.parametrize("flags", [[1], ["descending"], [None]])
def test_multi_key_sort_rejects_nonboolean_directions(flags):
    with pytest.raises(TypeError, match="boolean flags"):
        multi_key_sort([[1, 2]], [0], flags)


def test_multi_key_sort_rejects_vector_and_nonnumeric_data():
    with pytest.raises(ValueError, match="2 dimensions"):
        multi_key_sort([1, 2], [0])
    with pytest.raises(TypeError, match="real numeric"):
        multi_key_sort([["red", "blue"]], [0])
