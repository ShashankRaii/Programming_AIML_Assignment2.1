"""Classification and regression evaluation metrics.

All metrics validate their inputs with the shared helpers in ``utils``:
labels go through ``check_labels`` (so string, integer and boolean labels all
work) and continuous values go through ``check_array``. Counting is done with
``np.bincount`` on encoded labels, so there are no Python loops over samples.

When a metric is undefined because a denominator is zero (for example
precision when nothing was predicted positive), it returns 0.0 and emits an
``UndefinedMetricWarning`` instead of dividing by zero.
"""

import warnings

import numpy as np

from .utils import check_array, check_labels, check_same_length

__all__ = [
    "UndefinedMetricWarning",
    "confusion_matrix",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "mse",
    "mae",
    "rmse",
    "r2",
    "roc_curve",
    "auc",
]


class UndefinedMetricWarning(RuntimeWarning):
    """Warning for metrics that are undefined (zero denominator) and set to 0.0."""


# Internal helpers

def _check_label_pair(y_true, y_pred):
    """Validate two label vectors of equal, non-zero length and compatible types."""
    y_true = check_labels(y_true)
    y_pred = check_labels(y_pred)
    check_same_length(y_true, y_pred)
    if y_true.shape[0] == 0:
        raise ValueError("y_true and y_pred must contain at least one sample")
    true_is_text = y_true.dtype.kind in "USO"
    pred_is_text = y_pred.dtype.kind in "USO"
    if true_is_text != pred_is_text:
        raise TypeError(
            f"y_true and y_pred must both be numeric or both be strings; "
            f"got dtypes {y_true.dtype} and {y_pred.dtype}"
        )
    return y_true, y_pred


def _encode(y_true, y_pred, labels=None):
    """Map labels to integer codes 0..k-1.

    Returns ``(classes, true_codes, pred_codes)``. If ``labels`` is given, its
    order defines the codes and every observed label must appear in it.
    """
    if labels is None:
        classes = np.union1d(y_true, y_pred)
        return classes, np.searchsorted(classes, y_true), np.searchsorted(classes, y_pred)

    classes = check_labels(labels)
    if classes.shape[0] == 0:
        raise ValueError("labels must contain at least one label")
    if np.unique(classes).shape[0] != classes.shape[0]:
        raise ValueError("labels must not contain duplicates")
    order = np.argsort(classes, kind="stable")
    sorted_classes = classes[order]

    def lookup(y, name):
        pos = np.clip(np.searchsorted(sorted_classes, y), 0, sorted_classes.shape[0] - 1)
        missing = sorted_classes[pos] != y
        if np.any(missing):
            raise ValueError(f"{name} contains labels not in `labels`: {np.unique(y[missing])}")
        return order[pos]

    return classes, lookup(y_true, "y_true"), lookup(y_pred, "y_pred")


def _safe_divide(numerator, denominator, metric):
    """Element-wise division that returns 0.0 (and warns) where denominator is 0."""
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    zero = denominator == 0
    if np.any(zero):
        warnings.warn(
            f"{metric} is ill-defined (zero denominator) and is set to 0.0",
            UndefinedMetricWarning,
            stacklevel=3,
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(zero, 0.0, numerator / np.where(zero, 1.0, denominator))


# Confusion matrix and accuracy


def confusion_matrix(y_true, y_pred, labels=None):
    """Count true versus predicted labels.

    Labels are encoded to 0..k-1 .
    Parameters
    ----------
    y_true : array_like, shape (n_samples,)
        Ground-truth labels (numbers, booleans or strings).
    y_pred : array_like, shape (n_samples,)
        Predicted labels of the same kind as ``y_true``.
    labels : array_like, shape (n_classes,), optional
        Label order for the rows and columns. Defaults to the sorted union of
        the labels found in ``y_true`` and ``y_pred``.

    Returns
    -------
    cm : ndarray of int64, shape (n_classes, n_classes)
        ``cm[i, j]`` is the number of samples whose true label is class ``i``
        and whose predicted label is class ``j``.

    Raises
    ------
    TypeError
        If labels have unsupported types or mix numbers with strings.
    ValueError
        If inputs are empty, not 1-D, have different lengths, contain NaN, or
        contain a label missing from ``labels``.

    Complexity
    ----------
    Time O(n log n + k^2) for the label encoding and the output; space
    O(n + k^2), where k is the number of classes.

    Examples
    >>> confusion_matrix([0, 1, 1, 2], [0, 1, 0, 2])
    array([[1, 0, 0],
           [1, 1, 0],
           [0, 0, 1]])
    """
    y_true, y_pred = _check_label_pair(y_true, y_pred)
    classes, t, p = _encode(y_true, y_pred, labels)
    k = classes.shape[0]
    counts = np.bincount(t * k + p, minlength=k * k)
    return counts.reshape(k, k).astype(np.int64)


def accuracy(y_true, y_pred):
    """Fraction of predictions that exactly match the true labels.

    Parameters
    ----------
    y_true : array_like, shape (n_samples,)
        Ground-truth labels.
    y_pred : array_like, shape (n_samples,)
        Predicted labels.

    Returns
    -------
    score : float
        Value in [0, 1].

    Raises
    ------
    TypeError
        If labels have unsupported types or mix numbers with strings.
    ValueError
        If inputs are empty, not 1-D, have different lengths or contain NaN.

    Complexity
    ----------
    Time O(n); space O(n).

    Examples
    --------
    >>> accuracy([1, 0, 1, 1], [1, 0, 0, 1])
    0.75
    """
    y_true, y_pred = _check_label_pair(y_true, y_pred)
    return float(np.mean(y_true == y_pred))


# Precision, recall and F1

_AVERAGES = ("binary", "macro", "micro")


def _prf_counts(y_true, y_pred, average, pos_label):
    """Return per-class (or pooled) true-positive, false-positive and false-negative counts."""
    if average not in _AVERAGES:
        raise ValueError(f"average must be one of {_AVERAGES}; got {average!r}")
    y_true, y_pred = _check_label_pair(y_true, y_pred)

    if average == "binary":
        present = np.union1d(y_true, y_pred)
        if present.shape[0] > 2:
            raise ValueError(
                f"average='binary' needs at most 2 classes; got {present.shape[0]}. "
                "Use average='macro' or 'micro' for multi-class data."
            )
        if present.shape[0] == 2 and not np.any(present == pos_label):
            raise ValueError(f"pos_label={pos_label!r} is not one of the labels {present}")
        true_pos = y_true == pos_label
        pred_pos = y_pred == pos_label
        tp = np.count_nonzero(true_pos & pred_pos)
        fp = np.count_nonzero(~true_pos & pred_pos)
        fn = np.count_nonzero(true_pos & ~pred_pos)
        return np.array([tp]), np.array([fp]), np.array([fn])

    cm = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp  # predicted as the class but wrong
    fn = cm.sum(axis=1) - tp  # belong to the class but missed
    if average == "micro":
        return np.array([tp.sum()]), np.array([fp.sum()]), np.array([fn.sum()])
    return tp, fp, fn


def precision(y_true, y_pred, average="binary", pos_label=1):
    """Precision ``tp / (tp + fp)``: how many predicted positives are correct.

    Parameters
    ----------
    y_true : array_like, shape (n_samples,)
        Ground-truth labels.
    y_pred : array_like, shape (n_samples,)
        Predicted labels.
    average : {'binary', 'macro', 'micro'}, default 'binary'
        ``'binary'`` scores only ``pos_label``; ``'macro'`` is the unweighted
        mean of the per-class scores; ``'micro'`` pools the counts of all
        classes first.
    pos_label : label, default 1
        The positive class when ``average='binary'``.

    Returns
    -------
    score : float
        Value in [0, 1]. 0.0 (with ``UndefinedMetricWarning``) when there are
        no predicted positives.

    Raises
    ------
    TypeError
        If labels have unsupported types or mix numbers with strings.
    ValueError
        If ``average`` is invalid, inputs are empty or mismatched, binary
        averaging sees more than 2 classes, or ``pos_label`` is not a label.

    Complexity
    ----------
    Time O(n log n) for label encoding (O(n) for binary); space O(n + k^2).

    Examples
    >>> precision([1, 0, 1, 1], [1, 1, 1, 0])
    0.6666666666666666
    """
    tp, fp, _ = _prf_counts(y_true, y_pred, average, pos_label)
    return float(np.mean(_safe_divide(tp, tp + fp, "precision")))


def recall(y_true, y_pred, average="binary", pos_label=1):
    """Recall ``tp / (tp + fn)``: how many actual positives were found.

    Parameters
    ----------
    y_true : array_like, shape (n_samples,)
        Ground-truth labels.
    y_pred : array_like, shape (n_samples,)
        Predicted labels.
    average : {'binary', 'macro', 'micro'}, default 'binary'
        Averaging mode, as in :func:`precision`.
    pos_label : label, default 1
        The positive class when ``average='binary'``.

    Returns
    score : float
        Value in [0, 1]. 0.0 (with ``UndefinedMetricWarning``) when there are
        no actual positives.

    Raises
    ------
    TypeError
        If labels have unsupported types or mix numbers with strings.
    ValueError
        If ``average`` is invalid, inputs are empty or mismatched, binary
        averaging sees more than 2 classes, or ``pos_label`` is not a label.

    Complexity
    ----------
    Time O(n log n) for label encoding (O(n) for binary); space O(n + k^2).

    Examples
    >>> recall([1, 0, 1, 1], [1, 1, 1, 0])
    0.6666666666666666
    """
    tp, _, fn = _prf_counts(y_true, y_pred, average, pos_label)
    return float(np.mean(_safe_divide(tp, tp + fn, "recall")))


def f1(y_true, y_pred, average="binary", pos_label=1):
    """F1 score, the harmonic mean of precision and recall.

    Computed from counts as ``2 tp / (2 tp + fp + fn)``, which equals
    ``2 P R / (P + R)`` and avoids a second division by zero.

    Parameters
    ----------
    y_true : array_like, shape (n_samples,)
        Ground-truth labels.
    y_pred : array_like, shape (n_samples,)
        Predicted labels.
    average : {'binary', 'macro', 'micro'}, default 'binary'
        Averaging mode, as in :func:`precision`.
    pos_label : label, default 1
        The positive class when ``average='binary'``.

    Returns
    -------
    score : float
        Value in [0, 1]. 0.0 (with ``UndefinedMetricWarning``) when a class
        has no true or predicted samples.

    Raises
    ------
    TypeError
        If labels have unsupported types or mix numbers with strings.
    ValueError
        If ``average`` is invalid, inputs are empty or mismatched, binary
        averaging sees more than 2 classes, or ``pos_label`` is not a label.

    Complexity
    ----------
    Time O(n log n) for label encoding (O(n) for binary); space O(n + k^2).

    Examples
    >>> f1([1, 0, 1, 1], [1, 1, 1, 0])
    0.6666666666666666
    """
    tp, fp, fn = _prf_counts(y_true, y_pred, average, pos_label)
    return float(np.mean(_safe_divide(2 * tp, 2 * tp + fp + fn, "f1")))


# Regression metrics


def _check_regression(y_true, y_pred):
    """Validate two finite float arrays of identical, non-empty shape."""
    y_true = check_array(y_true, dtype=np.float64, allow_nan=False, allow_inf=False)
    y_pred = check_array(y_pred, dtype=np.float64, allow_nan=False, allow_inf=False)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape; got {y_true.shape} and {y_pred.shape}"
        )
    if y_true.size == 0:
        raise ValueError("y_true and y_pred must contain at least one value")
    return y_true, y_pred


def mse(y_true, y_pred):
    """Mean squared error ``mean((y_true - y_pred) ** 2)``.

    Parameters
    ----------
    y_true : array_like, shape (n_samples,) or (n_samples, n_outputs)
        True target values.
    y_pred : array_like, same shape as y_true
        Predicted values.

    Returns
    -------
    error : float
        Non-negative; 0.0 for perfect predictions.

    Raises
    ------
    TypeError
        If inputs are not real numeric.
    ValueError
        If inputs are empty, scalar, differ in shape, or contain NaN/inf.

    Complexity
    ----------
    Time O(n); space O(n).

    Examples
    >>> mse([3.0, 5.0], [2.0, 7.0])
    2.5
    """
    y_true, y_pred = _check_regression(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    """Mean absolute error ``mean(|y_true - y_pred|)``.

    Parameters
    ----------
    y_true : array_like, shape (n_samples,) or (n_samples, n_outputs)
        True target values.
    y_pred : array_like, same shape as y_true
        Predicted values.

    Returns
    -------
    error : float
        Non-negative; 0.0 for perfect predictions.

    Raises
    ------
    TypeError
        If inputs are not real numeric.
    ValueError
        If inputs are empty, scalar, differ in shape, or contain NaN/inf.

    Complexity
    ----------
    Time O(n); space O(n).

    Examples
    >>> mae([3.0, 5.0], [2.0, 7.0])
    1.5
    """
    y_true, y_pred = _check_regression(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    """Root mean squared error ``sqrt(mse)``, in the same units as y.

    Parameters
    ----------
    y_true : array_like, shape (n_samples,) or (n_samples, n_outputs)
        True target values.
    y_pred : array_like, same shape as y_true
        Predicted values.

    Returns
    -------
    error : float
        Non-negative; 0.0 for perfect predictions.

    Raises
    ------
    TypeError
        If inputs are not real numeric.
    ValueError
        If inputs are empty, scalar, differ in shape, or contain NaN/inf.

    Complexity
    ----------
    Time O(n); space O(n).

    Examples
    >>> rmse([0.0, 0.0], [3.0, 3.0])
    3.0
    """
    return float(np.sqrt(mse(y_true, y_pred)))


def r2(y_true, y_pred):
    """Coefficient of determination ``1 - SS_res / SS_tot``.

    Parameters
    ----------
    y_true : array_like, shape (n_samples,) or (n_samples, n_outputs)
        True target values (all outputs are pooled).
    y_pred : array_like, same shape as y_true
        Predicted values.

    Returns
    -------
    score : float
        1.0 for perfect predictions, 0.0 for always predicting the mean, and
        negative when worse than the mean. If ``y_true`` is constant
        (``SS_tot = 0``) it returns 1.0 for a perfect fit and otherwise 0.0
        with an ``UndefinedMetricWarning``.

    Raises
    ------
    TypeError
        If inputs are not real numeric.
    ValueError
        If inputs are empty, scalar, differ in shape, or contain NaN/inf.

    Complexity
    ----------
    Time O(n); space O(n).

    Examples
    >>> r2([1.0, 2.0, 3.0], [1.0, 2.0, 4.0])
    0.5
    """
    y_true, y_pred = _check_regression(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        if ss_res == 0:
            return 1.0
        warnings.warn(
            "r2 is ill-defined for constant y_true and is set to 0.0",
            UndefinedMetricWarning,
            stacklevel=2,
        )
        return 0.0
    return float(1.0 - ss_res / ss_tot)

# ROC curve and AUC (bonus)

def roc_curve(y_true, y_score, pos_label=1):
    """Receiver operating characteristic curve for a binary problem.

    Scores are sorted once in descending order; cumulative sums give the
    true- and false-positive counts at every threshold. Tied scores are
    collapsed into a single threshold so the curve does not depend on the
    order of tied samples.

    Parameters
    ----------
    y_true : array_like, shape (n_samples,)
        True binary labels.
    y_score : array_like, shape (n_samples,)
        Scores where higher means "more likely positive" (probabilities,
        decision values, ...).
    pos_label : label, default 1
        The label treated as positive.

    Returns
    -------
    fpr : ndarray of float64, shape (n_thresholds + 1,)
        False-positive rate for each threshold, starting at 0.
    tpr : ndarray of float64, shape (n_thresholds + 1,)
        True-positive rate for each threshold, starting at 0.
    thresholds : ndarray of float64, shape (n_thresholds + 1,)
        Decreasing thresholds; a sample is predicted positive when
        ``score >= threshold``. The first is ``inf`` (nothing positive).

    Raises
    ------
    TypeError
        If labels or scores have unsupported types.
    ValueError
        If inputs are empty or mismatched, ``y_true`` has more than two
        classes, or ``y_score`` contains NaN/inf.

    Warns
    -----
    UndefinedMetricWarning
        If ``y_true`` has no positives (tpr becomes NaN) or no negatives
        (fpr becomes NaN).

    Complexity
    ----------
    Time O(n log n) for the sort; space O(n).

    Examples
    >>> fpr, tpr, thr = roc_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    >>> fpr
    array([0. , 0. , 0.5, 0.5, 1. ])
    >>> tpr
    array([0. , 0.5, 0.5, 1. , 1. ])
    """
    y_true = check_labels(y_true)
    y_score = check_array(y_score, dtype=np.float64, allow_nan=False, allow_inf=False)
    if y_score.ndim != 1:
        raise ValueError(f"y_score must have 1 dimension; got shape {y_score.shape}")
    check_same_length(y_true, y_score)
    if y_true.shape[0] == 0:
        raise ValueError("y_true and y_score must contain at least one sample")
    if np.unique(y_true).shape[0] > 2:
        raise ValueError("roc_curve only supports binary y_true")

    positive = y_true == pos_label
    order = np.argsort(-y_score, kind="stable")
    score_sorted = y_score[order]
    positive_sorted = positive[order]

    # Last index of each run of equal scores = one threshold per distinct score.
    last_of_run = np.r_[np.flatnonzero(np.diff(score_sorted)), score_sorted.shape[0] - 1]
    tps = np.cumsum(positive_sorted)[last_of_run]
    fps = (last_of_run + 1) - tps

    tps = np.r_[0, tps].astype(np.float64)
    fps = np.r_[0, fps].astype(np.float64)
    thresholds = np.r_[np.inf, score_sorted[last_of_run]]

    if fps[-1] == 0:
        warnings.warn(
            "no negative samples in y_true; fpr is NaN", UndefinedMetricWarning, stacklevel=2
        )
        fpr = np.full_like(fps, np.nan)
    else:
        fpr = fps / fps[-1]
    if tps[-1] == 0:
        warnings.warn(
            "no positive samples in y_true; tpr is NaN", UndefinedMetricWarning, stacklevel=2
        )
        tpr = np.full_like(tps, np.nan)
    else:
        tpr = tps / tps[-1]
    return fpr, tpr, thresholds


def auc(x, y):
    """Area under a curve with the trapezoidal rule.

    Typically called as ``auc(fpr, tpr)`` on the output of :func:`roc_curve`.

    Parameters
    ----------
    x : array_like, shape (n_points,)
        x coordinates, monotonically increasing or decreasing.
    y : array_like, shape (n_points,)
        y coordinates.

    Returns
    -------
    area : float
        Area under the piecewise-linear curve through the points.

    Raises
    ------
    TypeError
        If inputs are not real numeric.
    ValueError
        If inputs are not 1-D, differ in length, have fewer than 2 points,
        contain NaN/inf, or ``x`` is not monotonic.

    Complexity
    ----------
    Time O(n); space O(n).

    Examples
    --------
    >>> auc([0.0, 0.5, 1.0], [0.0, 1.0, 1.0])
    0.75
    """
    x = check_array(x, dtype=np.float64, allow_nan=False, allow_inf=False)
    y = check_array(y, dtype=np.float64, allow_nan=False, allow_inf=False)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError(f"x and y must be 1-D; got shapes {x.shape} and {y.shape}")
    check_same_length(x, y)
    if x.shape[0] < 2:
        raise ValueError(f"auc needs at least 2 points; got {x.shape[0]}")
    dx = np.diff(x)
    if np.all(dx >= 0):
        direction = 1.0
    elif np.all(dx <= 0):
        direction = -1.0
    else:
        raise ValueError("x must be monotonically increasing or decreasing")
    return float(direction * np.sum(dx * (y[1:] + y[:-1]) / 2.0))
