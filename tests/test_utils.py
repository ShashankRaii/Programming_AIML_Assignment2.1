"""Test cases for the numerically stable helpers in numcompute.utils."""

import numpy as np
import pytest


def test_distances_preserve_small_separations_at_large_offsets():
    from numcompute.utils import euclidean_distances
    A = np.array([[1e8, 1e8], [1e8 + 3, 1e8 + 4]])
    np.testing.assert_array_equal(euclidean_distances(A, A), [[0, 5], [5, 0]])
    assert euclidean_distances(A[:1], [[1e8 + 1, 1e8]]).item() == 1


def test_distances_extremes_empty_features_and_validation():
    from numcompute.utils import euclidean_distances
    assert euclidean_distances([[0, 0]], [[3e200, 4e200]]).item() == pytest.approx(5e200)
    assert euclidean_distances([[0]], [[1e-200]]).item() == 1e-200
    assert np.isinf(euclidean_distances([[1e200]], [[0]], squared=True)).all()
    np.testing.assert_array_equal(euclidean_distances(np.empty((2, 0)), np.empty((3, 0))), 0)
    for bad in [np.nan, np.inf]:
        with pytest.raises(ValueError):
            euclidean_distances([[bad]], [[0]])


def test_distance_blocks_match_direct_small_reference_without_mutation():
    from numcompute.utils import euclidean_distances
    A = np.arange(2400.0).reshape(400, 6)[:, ::2]
    B = A[::100]
    before = A.copy()
    expected = np.sqrt(np.sum((A[:, None, :] - B[None, :, :])**2, axis=2))
    np.testing.assert_allclose(euclidean_distances(A, B), expected)
    np.testing.assert_array_equal(A, before)


def test_activation_infinities_nan_slices_and_empty_axis_shapes():
    from numcompute.utils import softmax, logsumexp
    X = np.array([[np.inf, 1000, np.inf], [-np.inf, -np.inf, -np.inf],
                  [np.nan, 1, 2]])
    np.testing.assert_array_equal(softmax(X)[0], [.5, 0, .5])
    assert np.isnan(softmax(X)[1:]).all()
    np.testing.assert_array_equal(logsumexp(X, axis=1), [np.inf, -np.inf, np.nan])
    assert logsumexp([np.inf, 1000]) == np.inf
    assert logsumexp(np.empty((2, 0)), axis=1).shape == (2,)
    assert np.isneginf(logsumexp(np.empty((2, 0)), axis=1)).all()


def test_activations_handle_extreme_finite_scores_without_warnings():
    from numcompute.utils import softmax, logsumexp
    x = np.array([np.finfo(float).max, -np.finfo(float).max])
    np.testing.assert_array_equal(softmax(x), [1, 0])
    assert logsumexp(x) == x[0]

from numcompute.utils import (
    batch_iterator,
    euclidean_distances,
    logsumexp,
    sigmoid,
    softmax,
)


# SOFTMAX

def test_softmax_matches_naive_formula():
    x = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(softmax(x), np.exp(x) / np.exp(x).sum())


def test_softmax_huge_values_do_not_overflow():
    p = softmax([1000.0, 1000.0, -1000.0])
    assert np.all(np.isfinite(p))
    np.testing.assert_allclose(p, [0.5, 0.5, 0.0], atol=1e-12)


def test_softmax_respects_axis():
    X = np.array([[1.0, 2.0], [3.0, 5.0]])
    np.testing.assert_allclose(softmax(X, axis=0).sum(axis=0), [1.0, 1.0])
    np.testing.assert_allclose(softmax(X, axis=1).sum(axis=1), [1.0, 1.0])


def test_softmax_neg_inf_empty_and_integer_input():
    np.testing.assert_allclose(softmax([0.0, -np.inf]), [1.0, 0.0])
    assert softmax([]).shape == (0,)
    assert softmax(np.array([1, 2], dtype=np.int64)).dtype == np.float64


def test_softmax_does_not_mutate_input():
    x = np.array([3.0, 1.0, 2.0])
    before = x.copy()
    softmax(x)
    np.testing.assert_array_equal(x, before)


# LOGSUMEXP


def test_logsumexp_matches_naive_for_small_values():
    x = np.array([0.5, 1.0, 2.0])
    np.testing.assert_allclose(logsumexp(x), np.log(np.exp(x).sum()))


@pytest.mark.parametrize("big", [1000.0, -1000.0])
def test_logsumexp_extreme_values(big):
    np.testing.assert_allclose(logsumexp([big, big]), big + np.log(2))


def test_logsumexp_all_neg_inf_and_empty():
    assert logsumexp([-np.inf, -np.inf]) == -np.inf
    assert logsumexp([]) == -np.inf


def test_logsumexp_axis_shape():
    X = np.arange(6.0).reshape(2, 3)
    out = logsumexp(X, axis=1)
    assert out.shape == (2,)
    np.testing.assert_allclose(out, np.log(np.exp(X).sum(axis=1)))


# SIGMOID


def test_sigmoid_known_values():
    np.testing.assert_allclose(sigmoid([0.0, 2.0]), [0.5, 1 / (1 + np.exp(-2.0))])


def test_sigmoid_extreme_values_do_not_overflow():
    with np.errstate(over="raise"):  # would raise if exp overflowed
        s = sigmoid([1000.0, -1000.0])
    np.testing.assert_allclose(s, [1.0, 0.0])


def test_sigmoid_nan_stays_nan_and_noncontiguous_input():
    assert np.isnan(sigmoid([np.nan])[0])
    X = np.arange(-6.0, 6.0).reshape(3, 4)
    np.testing.assert_allclose(sigmoid(X.T), 1 / (1 + np.exp(-X.T)))


# EUCLIDEAN DISTANCES


def test_euclidean_distances_matches_loop():
    rng = np.random.default_rng(42)
    A, B = rng.normal(size=(5, 3)), rng.normal(size=(4, 3))
    expected = np.array([[np.linalg.norm(a - b) for b in B] for a in A])
    np.testing.assert_allclose(euclidean_distances(A, B), expected)


def test_euclidean_distances_never_negative():
    A = np.array([[1e8, 1e8], [1.0, 2.0]])
    assert np.all(euclidean_distances(A, A, squared=True) >= 0)
    np.testing.assert_allclose(np.diag(euclidean_distances(A, A)), 0.0, atol=1e-6)


def test_euclidean_distances_errors():
    with pytest.raises(ValueError, match="same number of features"):
        euclidean_distances([[1, 2]], [[1, 2, 3]])
    with pytest.raises(ValueError, match="2 dimensions"):
        euclidean_distances([1, 2], [[1, 2]])


# BATCH ITERATOR


def test_batch_iterator_last_batch_smaller():
    batches = list(batch_iterator(np.arange(5), 2))
    assert [b.tolist() for b in batches] == [[0, 1], [2, 3], [4]]


def test_batch_iterator_rows_of_matrix():
    X = np.arange(12).reshape(6, 2)
    batches = list(batch_iterator(X, 4))
    assert [b.shape for b in batches] == [(4, 2), (2, 2)]
    np.testing.assert_array_equal(np.vstack(batches), X)


def test_batch_iterator_empty_and_errors_raised_immediately():
    assert list(batch_iterator([], 3)) == []
    with pytest.raises(ValueError, match="at least 1"):
        batch_iterator([1, 2], 0)
    with pytest.raises(TypeError, match="integer"):
        batch_iterator([1, 2], 1.5)


# SHARED VALIDATION PART


@pytest.mark.parametrize("func", [softmax, logsumexp, sigmoid])
def test_helpers_reject_non_numeric_and_scalar_input(func):
    with pytest.raises(TypeError, match="real numeric"):
        func(["a", "b"])
    with pytest.raises(ValueError, match=r"shape \(\)"):
        func(5.0)
