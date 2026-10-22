"""Tie-handling contracts and independent reference checks for ranking."""

import numpy as np
import pytest

from numcompute.rank import rank


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
