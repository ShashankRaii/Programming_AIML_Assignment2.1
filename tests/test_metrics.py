import numpy as np
import pytest

from numcompute.metrics import (
    UndefinedMetricWarning,
    accuracy,
    auc,
    confusion_matrix,
    f1,
    mae,
    mse,
    precision,
    r2,
    recall,
    rmse,
    roc_curve,
)


# confusion_matrix 


def test_confusion_matrix_binary_layout():
    cm = confusion_matrix([0, 0, 1, 1, 1], [0, 1, 1, 1, 0])
    # rows = true, cols = predicted
    np.testing.assert_array_equal(cm, [[1, 1], [1, 2]])
    assert cm.dtype == np.int64
    assert cm.sum() == 5


def test_confusion_matrix_labels_not_starting_at_zero():
    cm = confusion_matrix([3, 7, 7, 9], [3, 7, 9, 9])
    np.testing.assert_array_equal(cm, [[1, 0, 0], [0, 1, 1], [0, 0, 1]])


def test_confusion_matrix_string_labels_and_explicit_order():
    y_true = ["cat", "dog", "dog", "bird"]
    y_pred = ["cat", "cat", "dog", "bird"]
    cm = confusion_matrix(y_true, y_pred, labels=["dog", "cat", "bird"])
    np.testing.assert_array_equal(cm, [[1, 1, 0], [0, 1, 0], [0, 0, 1]])


def test_confusion_matrix_label_only_in_predictions():
    cm = confusion_matrix([0, 0], [0, 2])
    np.testing.assert_array_equal(cm, [[1, 1], [0, 0]])


def test_confusion_matrix_matches_loop_reference():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 4, size=200)
    y_pred = rng.integers(0, 4, size=200)
    expected = np.zeros((4, 4), dtype=int)
    for t, p in zip(y_true, y_pred):
        expected[t, p] += 1
    np.testing.assert_array_equal(confusion_matrix(y_true, y_pred), expected)


def test_confusion_matrix_errors():
    with pytest.raises(ValueError, match="got 2 and 3"):
        confusion_matrix([0, 1], [0, 1, 1])
    with pytest.raises(ValueError, match="at least one sample"):
        confusion_matrix([], [])
    with pytest.raises(ValueError, match="not in `labels`"):
        confusion_matrix([0, 1, 2], [0, 1, 2], labels=[0, 1])
    with pytest.raises(ValueError, match="duplicates"):
        confusion_matrix([0, 1], [0, 1], labels=[0, 0, 1])
    with pytest.raises(TypeError, match="both be numeric or both be strings"):
        confusion_matrix([0, 1], ["a", "b"])
    with pytest.raises(ValueError, match="NaN"):
        confusion_matrix([0.0, np.nan], [0.0, 1.0])


# accuracy 


def test_accuracy_basic_and_perfect():
    assert accuracy([1, 0, 1, 1], [1, 0, 0, 1]) == 0.75
    assert accuracy(["a", "b"], ["a", "b"]) == 1.0
    assert accuracy([True, False], [False, True]) == 0.0


def test_accuracy_noncontiguous_input():
    y = np.array([[0, 1], [1, 1], [0, 0]])
    assert accuracy(y[:, 0], y[:, 1]) == pytest.approx(2 / 3)


def test_mixed_numeric_labels_preserve_exact_classes_and_positive_label():
    large = 2**53
    true = np.array([large, large + 1], dtype=np.int64)
    pred = np.array([float(large), float(large)])
    assert accuracy(true, pred) == 0.5
    np.testing.assert_array_equal(confusion_matrix(true, pred), [[1, 0], [1, 0]])
    np.testing.assert_array_equal(
        confusion_matrix(true, pred, labels=true[::-1]), [[0, 1], [0, 1]]
    )
    assert recall(true, pred, pos_label=large + 1) == 0
    assert f1(true, pred, average="micro") == 0.5
    assert auc(*roc_curve(true, [.1, .9], pos_label=large + 1)[:2]) == 1


def test_signed_unsigned_labels_and_explicit_label_type_validation():
    true = np.array([-1, 2**63 - 1], dtype=np.int64)
    pred = np.array([0, 2**63], dtype=np.uint64)
    assert accuracy(true, pred) == 0
    assert confusion_matrix(true, pred).shape == (4, 4)
    with pytest.raises(TypeError, match="both be numeric or both be strings"):
        confusion_matrix([0, 1], [0, 1], labels=["0", "1"])


#  precision / recall / f1 


def test_binary_scores_hand_computed():
    y_true = [1, 1, 1, 0, 0, 0]
    y_pred = [1, 1, 0, 1, 0, 0]  # tp=2, fp=1, fn=1
    assert precision(y_true, y_pred) == pytest.approx(2 / 3)
    assert recall(y_true, y_pred) == pytest.approx(2 / 3)
    assert f1(y_true, y_pred) == pytest.approx(2 / 3)


def test_binary_pos_label_switches_class():
    y_true = ["spam", "spam", "ham", "ham"]
    y_pred = ["spam", "ham", "ham", "ham"]
    assert precision(y_true, y_pred, pos_label="spam") == 1.0
    assert recall(y_true, y_pred, pos_label="spam") == 0.5
    assert recall(y_true, y_pred, pos_label="ham") == 1.0


def test_f1_equals_harmonic_mean_of_precision_and_recall():
    y_true = [1, 0, 1, 1, 0, 1, 0, 1]
    y_pred = [1, 1, 1, 0, 0, 1, 1, 0]
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    assert f1(y_true, y_pred) == pytest.approx(2 * p * r / (p + r))


def test_macro_and_micro_multiclass():
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]
    # per-class precision: 1/2, 2/3, 1/1 ; recall: 1/2, 2/2, 1/2
    assert precision(y_true, y_pred, average="macro") == pytest.approx((0.5 + 2 / 3 + 1) / 3)
    assert recall(y_true, y_pred, average="macro") == pytest.approx((0.5 + 1 + 0.5) / 3)
    # micro scores equal accuracy for single-label multi-class data
    acc = accuracy(y_true, y_pred)
    for metric in (precision, recall, f1):
        assert metric(y_true, y_pred, average="micro") == pytest.approx(acc)


def test_perfect_predictions_score_one():
    y = [0, 1, 2, 1, 0]
    for metric in (precision, recall, f1):
        assert metric(y, y, average="macro") == 1.0
        assert metric([0, 1, 1], [0, 1, 1]) == 1.0


def test_zero_division_returns_zero_with_warning():
    # nothing predicted positive -> precision undefined
    with pytest.warns(UndefinedMetricWarning, match="precision"):
        assert precision([1, 0, 1], [0, 0, 0]) == 0.0
    # all one class and it is not pos_label -> recall undefined
    with pytest.warns(UndefinedMetricWarning, match="recall"):
        assert recall([0, 0, 0], [0, 0, 0]) == 0.0


def test_prf_errors():
    with pytest.raises(ValueError, match="average must be one of"):
        precision([0, 1], [0, 1], average="weighted")
    with pytest.raises(ValueError, match="at most 2 classes"):
        recall([0, 1, 2], [0, 1, 2])
    with pytest.raises(ValueError, match="pos_label"):
        f1(["a", "b"], ["a", "b"])  # default pos_label=1 is not a label


#  regression metrics 


def test_regression_metrics_hand_computed():
    y_true = [3.0, -0.5, 2.0, 7.0]
    y_pred = [2.5, 0.0, 2.0, 8.0]
    assert mse(y_true, y_pred) == pytest.approx(0.375)
    assert mae(y_true, y_pred) == pytest.approx(0.5)
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(0.375))
    assert r2(y_true, y_pred) == pytest.approx(0.9486081370449679)


def test_regression_perfect_and_mean_baseline():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert mse(y, y) == 0.0 and mae(y, y) == 0.0 and r2(y, y) == 1.0
    assert r2(y, np.full(4, y.mean())) == pytest.approx(0.0)
    assert r2(y, y[::-1]) < 0  # worse than predicting the mean


def test_regression_large_values_and_2d_input():
    y = np.array([[1e8, 2e8], [3e8, 4e8]])
    assert mse(y, y + 1.0) == pytest.approx(1.0)
    assert mae(y.T, y.T) == 0.0


def test_regression_extremes_reduce_before_rescaling():
    assert mse([1.5e154, 0], [0, 0]) == pytest.approx(1.125e308)
    assert rmse([1e200], [0]) == pytest.approx(1e200)
    assert mae([1e308, 1e308], [0, 0]) == 1e308
    assert np.isinf(mse([1e200], [0]))
    maximum = np.finfo(float).max
    assert rmse([maximum, 0, 0, 0], [-maximum, 0, 0, 0]) == maximum
    assert r2([1e200, 2e200, 3e200], [3e200, 2e200, 1e200]) == pytest.approx(-3)


def test_r2_constant_target():
    assert r2([5.0, 5.0], [5.0, 5.0]) == 1.0
    with pytest.warns(UndefinedMetricWarning, match="constant"):
        assert r2([5.0, 5.0], [4.0, 6.0]) == 0.0


@pytest.mark.parametrize("metric", [mse, mae, rmse, r2])
def test_regression_errors(metric):
    with pytest.raises(ValueError, match="same shape"):
        metric([1.0, 2.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="NaN"):
        metric([1.0, np.nan], [1.0, 2.0])
    with pytest.raises(ValueError, match="at least one value"):
        metric([], [])
    with pytest.raises(TypeError, match="real numeric"):
        metric(["a"], ["b"])


#  ROC / AUC 


def _roc_loop_reference(y_true, y_score):
    """Slow reference: evaluate every distinct threshold separately."""
    y_true = np.asarray(y_true) == 1
    y_score = np.asarray(y_score, dtype=float)
    thresholds = np.r_[np.inf, np.unique(y_score)[::-1]]
    fpr, tpr = [], []
    for t in thresholds:
        pred = y_score >= t
        tpr.append((pred & y_true).sum() / y_true.sum())
        fpr.append((pred & ~y_true).sum() / (~y_true).sum())
    return np.array(fpr), np.array(tpr), thresholds


def test_roc_curve_hand_computed():
    fpr, tpr, thr = roc_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    np.testing.assert_allclose(fpr, [0, 0, 0.5, 0.5, 1])
    np.testing.assert_allclose(tpr, [0, 0.5, 0.5, 1, 1])
    np.testing.assert_allclose(thr, [np.inf, 0.8, 0.4, 0.35, 0.1])
    assert auc(fpr, tpr) == pytest.approx(0.75)


def test_roc_curve_matches_threshold_loop():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=300)
    y_score = np.round(rng.random(300), 2)  # rounding creates many ties
    for got, expected in zip(roc_curve(y_true, y_score), _roc_loop_reference(y_true, y_score)):
        np.testing.assert_allclose(got, expected)


def test_roc_curve_ties_collapse_to_one_threshold():
    fpr, tpr, thr = roc_curve([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5])
    np.testing.assert_allclose(fpr, [0, 1])
    np.testing.assert_allclose(tpr, [0, 1])
    assert auc(fpr, tpr) == pytest.approx(0.5)  # ties give a random-guess diagonal


def test_roc_auc_perfect_and_inverted_scores():
    y = [0, 0, 1, 1]
    fpr, tpr, _ = roc_curve(y, [0.1, 0.2, 0.8, 0.9])
    assert auc(fpr, tpr) == 1.0
    fpr, tpr, _ = roc_curve(y, [0.9, 0.8, 0.2, 0.1])
    assert auc(fpr, tpr) == 0.0


def test_roc_curve_string_pos_label():
    fpr, tpr, _ = roc_curve(["no", "yes", "yes"], [0.2, 0.9, 0.6], pos_label="yes")
    assert auc(fpr, tpr) == 1.0


def test_roc_curve_single_class_warns():
    with pytest.warns(UndefinedMetricWarning, match="no negative"):
        fpr, tpr, _ = roc_curve([1, 1, 1], [0.2, 0.5, 0.9])
    assert np.all(np.isnan(fpr))
    np.testing.assert_allclose(tpr, [0, 1 / 3, 2 / 3, 1])


def test_roc_curve_errors():
    with pytest.raises(ValueError, match="binary"):
        roc_curve([0, 1, 2], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="NaN"):
        roc_curve([0, 1], [0.1, np.nan])
    with pytest.raises(ValueError, match="got 2 and 3"):
        roc_curve([0, 1], [0.1, 0.2, 0.3])


def test_auc_trapezoid_and_direction():
    x = np.linspace(0, 1, 101)
    assert auc(x, x**2) == pytest.approx(1 / 3, abs=1e-4)
    assert auc(x[::-1], (x**2)[::-1]) == pytest.approx(auc(x, x**2))


def test_auc_errors():
    with pytest.raises(ValueError, match="monotonic"):
        auc([0, 1, 0.5], [0, 1, 1])
    with pytest.raises(ValueError, match="at least 2 points"):
        auc([0.0], [1.0])
