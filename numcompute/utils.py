"""Shared input validation for NumCompute.

Validation never modifies caller data. Array results may share memory with their
inputs, so algorithms that write to their working arrays must first make a copy.
The second half of the module holds numerically stable helpers (softmax,
logsumexp, sigmoid, pairwise distances and batching) shared by other modules.
"""

import numpy as np


def _as_array(data, name):
    """Convert rectangular input and give ragged inputs an informative error."""
    try:
        return np.asarray(data)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a rectangular array: {exc}") from exc


def _check_dimensions(array, name, ensure_2d):
    """Reject scalars and optionally require a feature matrix."""
    if array.ndim == 0 or (ensure_2d and array.ndim != 2):
        expected = "2 dimensions" if ensure_2d else "at least 1 dimension"
        raise ValueError(f"{name} must have {expected}; got shape {array.shape}")


def check_array(X, ensure_2d=False, allow_nan=True, dtype=None, *, allow_inf=True):
    """Validate a real numeric array, preserving its dtype by default.

    Parameters
    ----------
    X : array_like, shape (...,)
        Rectangular numeric data with at least one dimension. Boolean data is
        accepted; strings, objects, complex values and datetime data are rejected.
    ensure_2d : bool, default False
        Require shape (n_samples, n_features) when True. Otherwise accept any
        positive number of dimensions. Empty dimensions are allowed.
    allow_nan : bool, default True
        Whether floating-point NaNs are accepted.
    dtype : numpy dtype or None, default None
        Optional explicit numeric output dtype. Original data must already be
        numeric; numeric strings are not silently parsed. None preserves dtype,
        including integer distinctions beyond floating-point precision.
    allow_inf : bool, default True
        Whether positive and negative infinity are accepted.

    Returns
    -------
    array : ndarray, shape matching X
        Validated data, possibly sharing memory with X. X is never modified.

    Raises
    ------
    TypeError
        If the input or requested output dtype is not real numeric or boolean.
    ValueError
        If input is ragged, has an invalid shape, contains forbidden NaNs or
        infinities, or cannot be converted to the requested dtype.

    Complexity
    ----------
    Time O(n); auxiliary space O(n) in the worst case for conversion or checks,
    where n is the number of elements.
    """
    array = _as_array(X, "X")
    _check_dimensions(array, "X", ensure_2d)
    if array.dtype.kind not in "biuf":
        raise TypeError(f"X must contain real numeric data; got dtype {array.dtype}")
    # Inspect the original values before casting, so casting cannot hide NaNs.
    if not allow_nan and np.isnan(array).any():
        raise ValueError("X contains NaN values; allow_nan=False")
    if not allow_inf and np.isinf(array).any():
        raise ValueError("X contains infinite values; allow_inf=False")
    if dtype is not None:
        target = np.dtype(dtype)
        if target.kind not in "biuf":
            raise TypeError(f"dtype must be real numeric or boolean; got {target}")
        with np.errstate(over="ignore", invalid="ignore"):
            array = array.astype(target, copy=False)
        # A narrowing conversion can itself introduce infinity.
        if not allow_nan and np.isnan(array).any():
            raise ValueError("dtype conversion produced NaN values; allow_nan=False")
        if not allow_inf and np.isinf(array).any():
            raise ValueError("dtype conversion produced infinite values; allow_inf=False")
    return array


def check_same_length(a, b):
    """Check that two arrays have the same number of samples.

    Parameters
    ----------
    a, b : array_like, shape (n_samples, ...)
        Arrays to compare along their first dimension; dtypes may differ.

    Returns
    -------
    None
        Successful validation. Neither input is modified.

    Raises
    ------
    ValueError
        If either input is scalar or ragged, or their first dimensions differ.
        Length mismatch messages include both actual lengths.

    Complexity
    ----------
    Time and auxiliary space O(n + m) for converting sequence inputs; O(1) for
    existing ndarrays, where n and m are their respective element counts.
    """
    first = _as_array(a, "a")
    second = _as_array(b, "b")
    _check_dimensions(first, "a", False)
    _check_dimensions(second, "b", False)
    if len(first) != len(second):
        raise ValueError(f"a and b must have the same length; got {len(first)} and {len(second)}")


def _check_discrete(data, name, ensure_2d, allow_nan):
    """Validate categorical atoms without converting strings to numbers."""
    array = _as_array(data, name)
    _check_dimensions(array, name, ensure_2d)
    if array.dtype.kind in "biufUS":
        if array.dtype.kind in "f" and not allow_nan and np.isnan(array).any():
            raise ValueError(f"{name} contains missing NaN values")
        if array.dtype.kind == "f" and np.isinf(array).any():
            raise ValueError(f"{name} contains infinite values")
        return array
    if array.dtype.kind != "O":
        raise TypeError(f"{name} must contain strings or real numbers; got dtype {array.dtype}")
    # Object arrays need element type inspection. This loop validates types only;
    # it is not a core numerical computation.
    for value in array.flat:
        if isinstance(value, (str, bytes, bool, int, float, np.integer, np.floating, np.bool_)):
            if isinstance(value, (float, np.floating)):
                if not allow_nan and np.isnan(value):
                    raise ValueError(f"{name} contains missing NaN values")
                if np.isinf(value):
                    raise ValueError(f"{name} contains infinite values")
        else:
            raise TypeError(
                f"{name} must contain strings or real numbers; got {type(value).__name__}"
            )
    return array


def check_categories(X, ensure_2d=True):
    """Validate categorical data without forcing numeric conversion.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        Strings, real numbers or booleans. Object arrays may mix these types
        across columns. NaNs are retained; infinities and None are rejected.
        Encoders are responsible for defining their missing-category handling
        and whether values within each feature are mutually comparable.
    ensure_2d : bool, default True
        Require a matrix; False permits any non-scalar shape.

    Returns
    -------
    array : ndarray, same shape and inferred dtype as X
        Validated data, possibly sharing memory with X. X is never modified.

    Raises
    ------
    TypeError
        If categorical atoms have unsupported types.
    ValueError
        If input has an invalid shape, is ragged or contains infinity.

    Complexity
    ----------
    Time O(n); auxiliary space O(n) worst case, where n is the element count.
    """
    return _check_discrete(X, "X", ensure_2d, allow_nan=True)


def check_labels(y):
    """Validate a one-dimensional classification label array.

    Parameters
    ----------
    y : array_like, shape (n_samples,)
        String, real numeric or boolean labels. Empty input is allowed for
        validation; individual metrics define whether it is meaningful. Missing
        values (NaN or None), infinity and unsupported objects are rejected.

    Returns
    -------
    labels : ndarray, shape (n_samples,)
        Validated labels with inferred dtype, possibly sharing memory with y.
        The input is never modified. Metrics must align label types across
        their true and predicted arrays.

    Raises
    ------
    TypeError
        If a label has an unsupported type, including None.
    ValueError
        If input is not one-dimensional, is ragged or contains NaN/infinity.

    Complexity
    ----------
    Time O(n); auxiliary space O(n) worst case for n labels.
    """
    labels = _check_discrete(y, "y", False, allow_nan=False)
    if labels.ndim != 1:
        raise ValueError(f"y must have 1 dimension; got shape {labels.shape}")
    return labels


def check_callable(function):
    """Validate a function or callable object without invoking it.

    Parameters
    ----------
    function : callable
        Function to be evaluated later by numerical differentiation. Its input
        and output shapes are checked by the differentiation routines.

    Returns
    -------
    function : callable
        The original callable, unchanged and unevaluated.

    Raises
    ------
    TypeError
        If function is not callable.

    Complexity
    ----------
    Time O(1); auxiliary space O(1).
    """
    if not callable(function):
        raise TypeError(f"function must be callable; got {type(function).__name__}")
    return function


# ---------------------------------------------------------------------------
# Numerically stable helpers
# ---------------------------------------------------------------------------


def softmax(x, axis=-1):
    """Numerically stable softmax along one axis.

    Subtracts the maximum along ``axis`` before exponentiating, so large
    inputs such as 1000 do not overflow.

    Parameters
    ----------
    x : array_like, shape 
        Real-valued scores with at least one dimension.
    axis : int, default -1
        Axis along which the output sums to 1.

    Returns
    -------
    p : ndarray of float, same shape as x

    Raises
    ------
    TypeError
        If x is not real numeric.
    ValueError
        If x is scalar or ragged.

    Complexity
    ----------
    Time O(n); auxiliary space O(n), where n is the number of elements.

    Examples
    --------
    >>> softmax([1000.0, 1000.0])
    array([0.5, 0.5])
    """
    x = check_array(x, dtype=np.float64)
    if x.size == 0:
        return x.copy()
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)  # avoids inf - inf = nan
    e = np.exp(x - m)
    with np.errstate(invalid="ignore"):
        return e / np.sum(e, axis=axis, keepdims=True)


def logsumexp(x, axis=None):
    """
    Parameters
    ----------
    x : array_like, shape (...,)
    axis : int or None, default None
        Axis to reduce. None reduces over all elements.

    Returns
    -------
    out : float scalar or ndarray
        0-d result when axis is None, otherwise x's shape with ``axis``
        removed. All ``-inf`` or empty input gives ``-inf``.

    Raises
    ------
    TypeError
        If x is not real numeric.
    ValueError
        If x is scalar or ragged.

    Complexity
    ----------
    Time O(n); auxiliary space O(n), where n is the number of elements.

    Examples
    float(logsumexp([1000.0, 1000.0]))  # 1000 + log(2)
    1000.6931471805599
    """
    x = check_array(x, dtype=np.float64)
    if x.size == 0:
        with np.errstate(divide="ignore"):
            return np.log(np.sum(np.exp(x), axis=axis))
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)  # handles all -inf and +inf
    with np.errstate(divide="ignore"):
        out = np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True)) + m
    return np.squeeze(out, axis=axis)


def sigmoid(x):
    """Numerically stable logistic sigmoid ``1 / (1 + exp(-x))

    Parameters
    ----------
    x : array_like, shape (...,)

    Returns
    -------
    s : ndarray of float, same shape as x
        Values in [0, 1]. NaN stays NaN.

    Raises
    ------
    TypeError
        If x is not real numeric.
    ValueError
        If x is scalar or ragged.

    Complexity
    ----------
    Time O(n); auxiliary space O(n), where n is the number of elements.

    Examples
    >>> sigmoid([0.0])
    array([0.5])
    """
    x = check_array(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    neg = ~pos  # NaN lands here and stays NaN
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out


def euclidean_distances(A, B, squared=False):
    """Pairwise Euclidean distances between the rows of A and B.

    Parameters
    ----------
    A : array_like, shape (n_samples_a, n_features)
        First set of points.
    B : array_like, shape (n_samples_b, n_features)
        Second set of points.
    squared : bool, default False
        Return squared distances.

    Returns
    -------
    D : ndarray of float, shape (n_samples_a, n_samples_b)

    Raises
    ------
    TypeError
        If A or B is not real numeric.
    ValueError
        If A or B is not 2-D, or their feature counts differ.

    Complexity
    ----------
    Time O(n_a * n_b * d); auxiliary space O(n_a * n_b).

    Examples
    >>> euclidean_distances([[0, 0]], [[3, 4]])
    array([[5.]])
    """
    A = check_array(A, ensure_2d=True, dtype=np.float64)
    B = check_array(B, ensure_2d=True, dtype=np.float64)
    if A.shape[1] != B.shape[1]:
        raise ValueError(
            f"A and B must have the same number of features; got {A.shape[1]} and {B.shape[1]}"
        )
    a2 = np.sum(A**2, axis=1)[:, None]  # (n_a, 1)
    b2 = np.sum(B**2, axis=1)[None, :]  # (1, n_b)
    d2 = np.maximum(a2 + b2 - 2.0 * (A @ B.T), 0.0)
    return d2 if squared else np.sqrt(d2)


def batch_iterator(X, batch_size):
    """Yield consecutive batches of samples from X.

    Parameters
    ----------
    X : array_like, shape (n_samples, ...)
        Numeric data split along axis 0. Its dtype is preserved.
    batch_size : int
        Samples per batch; must be at least 1.

    Returns
    -------
    batches : generator of ndarray, shape (<= batch_size, ...)
        Views (slices) of the validated X; X is never modified.

    Raises
    ------
    TypeError
        If batch_size is not an integer or X is not real numeric.
    ValueError
        If batch_size < 1, or X is scalar or ragged.

    Complexity
    ----------
    Time O(n_samples / batch_size) slices; auxiliary space O(1).

    Examples
    >>> [b.tolist() for b in batch_iterator([1, 2, 3, 4, 5], 2)]
    [[1, 2], [3, 4], [5]]
    """
    X = check_array(X)
    if isinstance(batch_size, (bool, np.bool_)) or not isinstance(batch_size, (int, np.integer)):
        raise TypeError(f"batch_size must be an integer; got {type(batch_size).__name__}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1; got {batch_size}")

    def _batches():
        # Loop over batch start positions only; slicing itself is vectorised.
        for start in range(0, X.shape[0], batch_size):
            yield X[start : start + batch_size]

    return _batches()
