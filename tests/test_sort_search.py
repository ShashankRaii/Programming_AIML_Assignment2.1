"""Sorting correctness, stability, dtype, shape and validation tests."""

import numpy as np
import pytest

from numcompute.sort_search import binary_search, multi_key_sort, stable_sort, topk


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


@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize("k", [0, 1, 2, 4, 7])
def test_topk_cutoff_ties_and_nans_match_independent_reference(largest, k):
    data = np.array([3.0, np.nan, 5.0, 3.0, 5.0, -np.inf, np.nan])
    expected = sorted(
        range(data.size),
        key=lambda i: (True, 0) if np.isnan(data[i]) else
        (False, -float(data[i]) if largest else float(data[i])),
    )[:k]
    values, indices = topk(data, k, largest=largest)
    np.testing.assert_array_equal(indices, expected)
    np.testing.assert_array_equal(values, data[expected])
    assert indices.dtype == np.intp
    assert values.dtype == data.dtype


@pytest.mark.parametrize("largest", [False, True])
def test_topk_repeated_values_at_cutoff_prefer_earliest_indices(largest):
    data = np.array([4, 4, 4, 9, 0, 4])
    _, indices = topk(data, 3, largest=largest)
    np.testing.assert_array_equal(indices, [3, 0, 1] if largest else [4, 0, 1])


@pytest.mark.parametrize("largest", [False, True])
def test_topk_all_equal_and_all_nan_inputs(largest):
    for data in [np.full(6, 3), np.full(6, np.nan)]:
        values, indices = topk(data, 4, largest=largest)
        np.testing.assert_array_equal(indices, [0, 1, 2, 3])
        np.testing.assert_array_equal(values, data[:4])


@pytest.mark.parametrize("largest", [False, True])
def test_topk_random_integer_data_matches_python_reference(largest):
    data = np.random.default_rng(42).integers(-10, 11, size=257)
    for k in [1, 8, 128, 256, 257]:
        expected = sorted(range(data.size), key=lambda i: int(data[i]), reverse=largest)[:k]
        values, indices = topk(data, k, largest)
        np.testing.assert_array_equal(indices, expected)
        np.testing.assert_array_equal(values, data[indices])


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
@pytest.mark.parametrize("largest", [False, True])
def test_topk_extreme_integers_remain_distinct(dtype, largest):
    limits = np.iinfo(dtype)
    data = np.array([limits.min, 2**53 + 1, limits.max, 2**53, limits.max], dtype=dtype)
    expected = sorted(range(data.size), key=lambda i: int(data[i]), reverse=largest)[:4]
    values, indices = topk(data, 4, largest)
    np.testing.assert_array_equal(indices, expected)
    np.testing.assert_array_equal(values, data[expected])
    assert values.dtype == dtype


def test_topk_nan_never_displaces_infinity_or_valid_value():
    data = [np.nan, -np.inf, np.inf, 1.0, np.nan]
    np.testing.assert_array_equal(topk(data, 3)[0], [np.inf, 1, -np.inf])
    np.testing.assert_array_equal(topk(data, 3, largest=False)[0], [-np.inf, 1, np.inf])


@pytest.mark.parametrize("largest", [False, True])
def test_topk_signed_zero_ties_preserve_input_order(largest):
    data = np.array([-0.0, 0.0, -0.0, 0.0])
    values, indices = topk(data, 3, largest)
    np.testing.assert_array_equal(indices, [0, 1, 2])
    np.testing.assert_array_equal(np.signbit(values), [True, False, True])


def test_topk_empty_and_singleton_inputs_and_value_only_result():
    values, indices = topk(np.array([], dtype=np.int16), 0)
    assert values.shape == indices.shape == (0,)
    assert values.dtype == np.int16
    np.testing.assert_array_equal(topk([8], np.int64(1), return_indices=False), [8])
    np.testing.assert_array_equal(topk([True, False, True], 2)[1], [0, 2])
    np.testing.assert_array_equal(topk([2], 0, return_indices=np.bool_(False)), [])


def test_topk_strided_readonly_input_is_not_modified_and_indices_are_local():
    original = np.arange(16)
    original.flags.writeable = False
    view = original[::-3]
    before = view.copy()
    values, indices = topk(view, 3, largest=False)
    np.testing.assert_array_equal(values, view[indices])
    np.testing.assert_array_equal(indices, [5, 4, 3])
    np.testing.assert_array_equal(view, before)
    assert not np.shares_memory(values, original)


@pytest.mark.parametrize("k", [-1, 4])
def test_topk_reports_invalid_k_and_length(k):
    with pytest.raises(ValueError, match=rf"0 <= k <= 3; got {k}"):
        topk([1, 2, 3], k)


@pytest.mark.parametrize("k", [True, np.bool_(False), 1.5, "2"])
def test_topk_rejects_noninteger_k(k):
    with pytest.raises(TypeError, match="k must be an integer"):
        topk([1, 2, 3], k)


@pytest.mark.parametrize("option", [{"largest": 1}, {"return_indices": 0}])
def test_topk_rejects_nonboolean_flags(option):
    with pytest.raises(TypeError, match="must be boolean"):
        topk([1], 1, **option)


def test_topk_rejects_wrong_shape_nonnumeric_and_nonzero_k_on_empty():
    with pytest.raises(ValueError, match=r"1 dimension; got shape \(1, 2\)"):
        topk([[1, 2]], 1)
    with pytest.raises(TypeError, match="real numeric"):
        topk(["1"], 1)
    with pytest.raises(ValueError, match="0 <= k <= 0"):
        topk([], 1)


@pytest.mark.parametrize(
    "query,expected",
    [(0, (0, False)), (1, (0, True)), (3, (1, True)), (4, (3, False)), (9, (4, False))],
)
def test_binary_search_scalar_boundaries_and_duplicate_leftmost(query, expected):
    result = binary_search([1, 3, 3, 5], query)
    assert result == expected
    assert type(result[0]) is int and type(result[1]) is bool


def test_binary_search_empty_source_scalar_and_shaped_queries():
    assert binary_search([], 4) == (0, False)
    queries = np.array([[np.nan, -np.inf], [0, np.inf]])
    positions, found = binary_search([], queries)
    np.testing.assert_array_equal(positions, np.zeros((2, 2), dtype=np.intp))
    np.testing.assert_array_equal(found, np.zeros((2, 2), dtype=bool))


def test_binary_search_empty_queries_preserve_shape_and_output_types():
    queries = np.empty((2, 0, 3))
    positions, found = binary_search([1, 2], queries)
    assert positions.shape == found.shape == queries.shape
    assert positions.dtype == np.intp and found.dtype == bool


def test_binary_search_singleton_and_all_equal_source():
    assert binary_search([7], 7) == (0, True)
    positions, found = binary_search([4, 4, 4], [3, 4, 5])
    np.testing.assert_array_equal(positions, [0, 0, 3])
    np.testing.assert_array_equal(found, [False, True, False])


def test_binary_search_vector_queries_match_independent_bisect_reference():
    from bisect import bisect_left

    rng = np.random.default_rng(42)
    data = np.sort(rng.integers(-20, 21, size=150))
    queries = rng.integers(-25, 26, size=(5, 6))
    expected_positions = np.array([bisect_left(data.tolist(), int(x)) for x in queries.flat])
    expected_found = np.array([int(x) in data.tolist() for x in queries.flat])
    positions, found = binary_search(data, queries)
    np.testing.assert_array_equal(positions, expected_positions.reshape(queries.shape))
    np.testing.assert_array_equal(found, expected_found.reshape(queries.shape))


def test_binary_search_nan_queries_insert_left_but_are_never_matches():
    data = np.array([-np.inf, 1, 1, np.inf, np.nan, np.nan])
    queries = np.array([np.nan, -np.inf, 1, np.inf, 2])
    positions, found = binary_search(data, queries)
    np.testing.assert_array_equal(positions, [4, 0, 1, 3, 3])
    np.testing.assert_array_equal(found, [False, True, True, True, False])
    assert binary_search([np.nan, np.nan], np.nan) == (0, False)
    assert binary_search([1.0, 2.0], np.nan) == (2, False)


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_binary_search_distinguishes_large_same_dtype_integers(dtype):
    limits = np.iinfo(dtype)
    data = np.array([limits.min, 2**53, 2**53 + 1, limits.max], dtype=dtype)
    positions, found = binary_search(data, data)
    np.testing.assert_array_equal(positions, [0, 1, 2, 3])
    assert found.all()


def test_binary_search_mixed_float_queries_do_not_round_integer_source():
    data = np.array([2**53 + 1, 2**53 + 3, 2**53 + 4], dtype=np.int64)
    queries = np.array([2**53, 2**53 + 2, 2**53 + 4], dtype=np.float64)
    positions, found = binary_search(data, queries)
    np.testing.assert_array_equal(positions, [0, 1, 2])
    np.testing.assert_array_equal(found, [False, False, True])
    assert binary_search([2**53 + 1], float(2**53)) == (0, False)


def test_binary_search_mixed_integer_queries_do_not_round_into_float_matches():
    data = np.array([float(2**53), float(2**53 + 2), np.inf, np.nan])
    queries = np.array([2**53 + 1, 2**53 + 2], dtype=np.int64)
    positions, found = binary_search(data, queries)
    np.testing.assert_array_equal(positions, [1, 1])
    np.testing.assert_array_equal(found, [False, True])


def test_binary_search_signed_unsigned_boundaries_are_exact():
    unsigned = np.array([2**63, 2**63 + 1], dtype=np.uint64)
    assert binary_search(unsigned, np.int64(2**63 - 1)) == (0, False)
    signed = np.array([-1, 2**63 - 1], dtype=np.int64)
    assert binary_search(signed, np.uint64(2**63)) == (2, False)


def test_binary_search_fallback_preserves_nan_insertion_and_infinity_matches():
    data = np.array([2**53 + 1], dtype=np.int64)
    queries = np.array([-np.inf, float(2**53), np.inf, np.nan])
    positions, found = binary_search(data, queries)
    np.testing.assert_array_equal(positions, [0, 0, 1, 1])
    assert not found.any()
    positions, found = binary_search([1.0, np.inf, np.nan], np.array([1, 2], dtype=np.int64))
    np.testing.assert_array_equal(positions, [0, 1])
    np.testing.assert_array_equal(found, [True, False])


def test_binary_search_all_nan_float_source_accepts_integer_queries():
    positions, found = binary_search([np.nan, np.nan], np.array([1, 2], dtype=np.int64))
    np.testing.assert_array_equal(positions, [0, 0])
    assert not found.any()


def test_binary_search_supports_boolean_and_fractional_queries():
    assert binary_search([False, True, True], True) == (1, True)
    assert binary_search([1, 3, 5], 2.5) == (1, False)
    assert binary_search([-0.0, 0.0], 0.0) == (0, True)


def test_binary_search_strided_readonly_inputs_are_not_modified():
    original = np.arange(20)
    original.flags.writeable = False
    data = original[::2]
    queries = original[1::3]
    data_before, query_before = data.copy(), queries.copy()
    positions, found = binary_search(data, queries)
    np.testing.assert_array_equal(positions, np.searchsorted(data, queries))
    np.testing.assert_array_equal(found, queries % 2 == 0)
    np.testing.assert_array_equal(data, data_before)
    np.testing.assert_array_equal(queries, query_before)


@pytest.mark.parametrize("source,query", [(["1"], 1), ([1], "1"), ([1], 1j), ([1], None)])
def test_binary_search_rejects_unsupported_types(source, query):
    with pytest.raises(TypeError, match="real numeric"):
        binary_search(source, query)


def test_binary_search_rejects_scalar_or_matrix_source_and_ragged_queries():
    with pytest.raises(ValueError, match="shape"):
        binary_search(1, 1)
    with pytest.raises(ValueError, match=r"1 dimension; got shape \(1, 2\)"):
        binary_search([[1, 2]], 1)
    with pytest.raises(ValueError):
        binary_search([1], [[1], [2, 3]])
