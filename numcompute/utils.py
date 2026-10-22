"""Shared input validation for NumCompute.

Validation never modifies caller data. Array results may share memory with their
inputs, so algorithms that write to their working arrays must first make a copy.
Numerical utilities will be added in a later checkpoint.
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
