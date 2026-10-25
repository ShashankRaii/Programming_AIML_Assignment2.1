"""Contract tests for shared validation, independent of later algorithms."""

import numpy as np
import pytest

from numcompute.utils import (
    check_array,
    check_callable,
    check_categories,
    check_labels,
    check_same_length,
)


def test_numeric_validation_preserves_large_integers():
    data = np.array([2**53, 2**53 + 1], dtype=np.int64)
    result = check_array(data)
    assert result.dtype == data.dtype
    assert result[0] != result[1]
    np.testing.assert_array_equal(result, data)


@pytest.mark.parametrize("dtype", [np.int8, np.uint64, np.float32, np.float64, bool])
def test_numeric_validation_preserves_dtype(dtype):
    data = np.array([0, 1], dtype=dtype)
    assert check_array(data).dtype == data.dtype


def test_explicit_float_conversion_does_not_mutate_input():
    data = np.array([[1, 2], [3, 4]], dtype=np.int64)
    before = data.copy()
    result = check_array(data, ensure_2d=True, dtype=np.float64)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(data, before)
    np.testing.assert_array_equal(result, before)


def test_nan_and_infinity_policies_are_independent():
    data = np.array([np.nan, np.inf, -np.inf])
    np.testing.assert_array_equal(check_array(data), data)
    with pytest.raises(ValueError, match="NaN"):
        check_array([np.nan], allow_nan=False)
    with pytest.raises(ValueError, match="infinite"):
        check_array([np.inf], allow_inf=False)
    assert np.isnan(check_array([np.nan], allow_inf=False)[0])
    assert np.isinf(check_array([np.inf], allow_nan=False)[0])


def test_conversion_cannot_hide_forbidden_nan_or_introduce_forbidden_infinity():
    with pytest.raises(ValueError, match="NaN"):
        check_array([np.nan], allow_nan=False, dtype=np.int64)
    with pytest.raises(ValueError, match="conversion produced infinite"):
        check_array([1e100], dtype=np.float32, allow_inf=False)


@pytest.mark.parametrize("data", [["1", "2"], [1j], [None], np.array([1], dtype=object)])
def test_numeric_validation_rejects_unsupported_input_even_with_dtype(data):
    with pytest.raises(TypeError, match="real numeric"):
        check_array(data, dtype=float)


def test_numeric_validation_rejects_nonnumeric_output_dtype():
    with pytest.raises(TypeError, match="dtype must be real numeric"):
        check_array([1], dtype=str)


@pytest.mark.parametrize("data", [42, "value", None])
def test_numeric_validation_rejects_scalars_with_actual_shape(data):
    with pytest.raises(ValueError, match=r"shape \(\)"):
        check_array(data)


def test_matrix_validation_reports_wrong_shape():
    with pytest.raises(ValueError, match=r"2 dimensions; got shape \(3,\)"):
        check_array([1, 2, 3], ensure_2d=True)


def test_ragged_numeric_input_is_rejected():
    with pytest.raises(ValueError, match="rectangular"):
        check_array([[1], [2, 3]])


@pytest.mark.parametrize("shape", [(0,), (0, 3), (3, 0)])
def test_empty_shapes_are_preserved(shape):
    assert check_array(np.empty(shape)).shape == shape


def test_noncontiguous_view_is_accepted_without_mutation():
    original = np.arange(24).reshape(4, 6)
    before = original.copy()
    view = original.T[::2]
    assert not view.flags.c_contiguous
    result = check_array(view, ensure_2d=True)
    np.testing.assert_array_equal(result, view)
    np.testing.assert_array_equal(original, before)


def test_same_length_compares_samples_not_total_element_count():
    assert check_same_length(np.ones((3, 4)), ["a", "b", "c"]) is None
    assert check_same_length([], np.empty((0, 2))) is None


def test_length_mismatch_reports_both_lengths():
    with pytest.raises(ValueError, match="got 2 and 3"):
        check_same_length([1, 2], [1, 2, 3])


@pytest.mark.parametrize("first,second", [(1, [1]), ([1], 1)])
def test_same_length_rejects_scalar_inputs(first, second):
    with pytest.raises(ValueError, match="shape"):
        check_same_length(first, second)


def test_categories_preserve_strings_and_mixed_object_columns():
    strings = np.array([["red"], ["blue"]])
    np.testing.assert_array_equal(check_categories(strings), strings)
    mixed = np.array([["red", 1], ["blue", 2]], dtype=object)
    result = check_categories(mixed)
    assert result.dtype == object
    np.testing.assert_array_equal(result, mixed)


def test_categories_allow_nan_for_encoder_policy():
    assert np.isnan(check_categories([[np.nan]])[0, 0])
    assert np.isnan(check_categories(np.array([[np.nan]], dtype=object))[0, 0])


@pytest.mark.parametrize("data", [[[None]], [[1j]], [[object()]]])
def test_categories_reject_unsupported_atoms(data):
    with pytest.raises(TypeError, match="strings or real numbers"):
        check_categories(data)


@pytest.mark.parametrize("dtype", [float, object])
def test_categories_reject_infinity(dtype):
    with pytest.raises(ValueError, match="infinite"):
        check_categories(np.array([[np.inf]], dtype=dtype))


def test_categories_require_matrix_unless_explicitly_disabled():
    with pytest.raises(ValueError, match="2 dimensions"):
        check_categories(["red", "blue"])
    assert check_categories(["red", "blue"], ensure_2d=False).shape == (2,)


@pytest.mark.parametrize("data", [["cat", "dog"], [False, True], [10, 20], []])
def test_labels_accept_valid_vectors(data):
    np.testing.assert_array_equal(check_labels(data), np.asarray(data))


@pytest.mark.parametrize("dtype", [float, object])
def test_labels_reject_nan_and_infinity(dtype):
    for value in [np.nan, np.inf, -np.inf]:
        with pytest.raises(ValueError, match="NaN|infinite"):
            check_labels(np.array([value], dtype=dtype))


def test_labels_reject_missing_objects_and_wrong_dimensions():
    with pytest.raises(TypeError, match="NoneType"):
        check_labels([None])
    with pytest.raises(ValueError, match=r"1 dimension; got shape \(2, 1\)"):
        check_labels([[1], [2]])


def test_callable_validation_does_not_evaluate_function():
    def function():
        raise AssertionError("validation must not evaluate the function")

    assert check_callable(function) is function


def test_callable_object_is_accepted():
    class Function:
        def __call__(self, x):
            return x

    function = Function()
    assert check_callable(function) is function


def test_noncallable_reports_actual_type():
    with pytest.raises(TypeError, match="got int"):
        check_callable(42)
