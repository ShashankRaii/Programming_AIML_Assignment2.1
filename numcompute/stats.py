"""Axis-aware descriptive statistics, quantiles and histograms.

Reductions use NumPy's numeric and warning semantics, with floating calculations
promoted to at least float64. Streaming statistics are scheduled for checkpoint 7.
"""

import operator

import numpy as np

from .utils import check_array


def _validate_axis(axis, shape):
    """Validate one reduction axis without relying on NumPy private APIs."""
    if axis is None:
        return None
    if isinstance(axis, (bool, np.bool_)):
        raise TypeError("axis must be an integer or None; got bool")
    try:
        axis = operator.index(axis)
    except TypeError as exc:
        raise TypeError("axis must be an integer or None") from exc
    if not -len(shape) <= axis < len(shape):
        raise ValueError(f"axis {axis} is invalid for data shape {shape}")
    return axis


def _reduce(data, operation, axis, ignore_nan, ddof=None):
    """Apply shared validation and dispatch to a NumPy reduction."""
    array = check_array(data)
    axis = _validate_axis(axis, array.shape)
    if not isinstance(ignore_nan, (bool, np.bool_)):
        raise TypeError("ignore_nan must be boolean")
    kwargs = {"axis": axis}
    if operation in ("var", "std"):
        if isinstance(ddof, (bool, np.bool_)):
            raise TypeError("ddof must be a nonnegative integer; got bool")
        try:
            ddof = operator.index(ddof)
        except TypeError as exc:
            raise TypeError("ddof must be a nonnegative integer") from exc
        if ddof < 0:
            raise ValueError(f"ddof must be nonnegative; got {ddof}")
        kwargs["ddof"] = ddof
    if operation not in ("min", "max"):
        array = array.astype(np.result_type(array.dtype, np.float64), copy=False)
    function = getattr(np, ("nan" if ignore_nan else "") + operation)
    return function(array, **kwargs)


def mean(data, axis=None, ignore_nan=True):
    """Return the arithmetic mean along one axis or over flattened data.

    Parameters
    ----------
    data : array_like, shape (...,)
        Real numeric or boolean data with at least one dimension. Read-only
        and non-contiguous input are accepted and never modified.
    axis : int or None, default None
        None reduces all elements; an integer (including negative indices)
        removes that axis. Boolean axes and tuples are rejected.
    ignore_nan : bool, default True
        Omit NaNs when True; False propagates NaNs in affected slices.

    Returns
    -------
    result : numpy scalar or ndarray
        A scalar for axis=None or when reducing a 1-D input. Otherwise, the input
        shape with the selected axis removed. Calculations and output use at
        least float64; wider floats are retained.

    Raises
    ------
    TypeError
        If data is unsupported or axis/ignore_nan has an invalid type.
    ValueError
        If data is scalar/ragged or axis is out of bounds.

    Warns
    -----
    RuntimeWarning
        Empty/all-NaN slices warn and return NaN. NumPy warnings and NaN/inf
        results apply to non-finite arithmetic and insufficient degrees of freedom.

    Complexity
    ----------
    Time O(n) for n elements. Auxiliary space O(n + r), including conversion
    and reduction workspace; r is the output element count.

    Examples
    --------
    >>> float(mean([0, 1, 2]))
    1.0
    """
    return _reduce(data, "mean", axis, ignore_nan)


def median(data, axis=None, ignore_nan=True):
    """Return the median along one axis or over flattened data.

    Parameters
    ----------
    data : array_like, shape (...,)
        Real numeric or boolean data with at least one dimension. Read-only
        and non-contiguous input are accepted and never modified.
    axis : int or None, default None
        None reduces all elements; an integer (including negative indices)
        removes that axis. Boolean axes and tuples are rejected.
    ignore_nan : bool, default True
        Omit NaNs when True; False propagates NaNs in affected slices.

    Returns
    -------
    result : numpy scalar or ndarray
        A scalar for axis=None or when reducing a 1-D input. Otherwise, the input
        shape with the selected axis removed. Calculations and output use at
        least float64; wider floats are retained.

    Raises
    ------
    TypeError
        If data is unsupported or axis/ignore_nan has an invalid type.
    ValueError
        If data is scalar/ragged or axis is out of bounds.

    Warns
    -----
    RuntimeWarning
        Empty/all-NaN slices warn and return NaN. NumPy warnings and NaN/inf
        results apply to non-finite arithmetic and insufficient degrees of freedom.

    Complexity
    ----------
    Time O(n) for n elements. Auxiliary space O(n + r), including conversion
    and reduction workspace; r is the output element count.

    Examples
    --------
    >>> float(median([1, 2, 7]))
    2.0
    """
    return _reduce(data, "median", axis, ignore_nan)


def std(data, axis=None, ddof=0, ignore_nan=True):
    """Return the standard deviation along one axis or over flattened data.

    Parameters
    ----------
    data : array_like, shape (...,)
        Real numeric or boolean data with at least one dimension. Read-only
        and non-contiguous input are accepted and never modified.
    axis : int or None, default None
        None reduces all elements; an integer (including negative indices)
        removes that axis. Boolean axes and tuples are rejected.
    ddof : int, default 0
        Nonnegative degrees-of-freedom correction; divisor is count-ddof,
        where count excludes NaNs when ignore_nan=True. Boolean ddof is rejected.
    ignore_nan : bool, default True
        Omit NaNs when True; False propagates NaNs in affected slices.

    Returns
    -------
    result : numpy scalar or ndarray
        A scalar for axis=None or when reducing a 1-D input. Otherwise, the input
        shape with the selected axis removed. Calculations and output use at
        least float64; wider floats are retained.

    Raises
    ------
    TypeError
        If data is unsupported or axis/ignore_nan has an invalid type.
        Also raised for a noninteger or boolean ddof.
    ValueError
        If data is scalar/ragged or axis is out of bounds.
        Also raised for negative ddof.

    Warns
    -----
    RuntimeWarning
        Empty/all-NaN slices warn and return NaN. NumPy warnings and NaN/inf
        results apply to non-finite arithmetic and insufficient degrees of freedom.

    Complexity
    ----------
    Time O(n) for n elements. Auxiliary space O(n + r), including conversion
    and reduction workspace; r is the output element count.

    Examples
    --------
    >>> float(std([1, 3]))
    1.0
    """
    return _reduce(data, "std", axis, ignore_nan, ddof=ddof)


def var(data, axis=None, ddof=0, ignore_nan=True):
    """Return the variance along one axis or over flattened data.

    Parameters
    ----------
    data : array_like, shape (...,)
        Real numeric or boolean data with at least one dimension. Read-only
        and non-contiguous input are accepted and never modified.
    axis : int or None, default None
        None reduces all elements; an integer (including negative indices)
        removes that axis. Boolean axes and tuples are rejected.
    ddof : int, default 0
        Nonnegative degrees-of-freedom correction; divisor is count-ddof,
        where count excludes NaNs when ignore_nan=True. Boolean ddof is rejected.
    ignore_nan : bool, default True
        Omit NaNs when True; False propagates NaNs in affected slices.

    Returns
    -------
    result : numpy scalar or ndarray
        A scalar for axis=None or when reducing a 1-D input. Otherwise, the input
        shape with the selected axis removed. Calculations and output use at
        least float64; wider floats are retained.

    Raises
    ------
    TypeError
        If data is unsupported or axis/ignore_nan has an invalid type.
        Also raised for a noninteger or boolean ddof.
    ValueError
        If data is scalar/ragged or axis is out of bounds.
        Also raised for negative ddof.

    Warns
    -----
    RuntimeWarning
        Empty/all-NaN slices warn and return NaN. NumPy warnings and NaN/inf
        results apply to non-finite arithmetic and insufficient degrees of freedom.

    Complexity
    ----------
    Time O(n) for n elements. Auxiliary space O(n + r), including conversion
    and reduction workspace; r is the output element count.

    Examples
    --------
    >>> float(var([1, 3]))
    1.0
    """
    return _reduce(data, "var", axis, ignore_nan, ddof=ddof)


def minimum(data, axis=None, ignore_nan=True):
    """Return the minimum along one axis or over flattened data.

    Parameters
    ----------
    data : array_like, shape (...,)
        Real numeric or boolean data with at least one dimension. Read-only
        and non-contiguous input are accepted and never modified.
    axis : int or None, default None
        None reduces all elements; an integer (including negative indices)
        removes that axis. Boolean axes and tuples are rejected.
    ignore_nan : bool, default True
        Omit NaNs when True; False propagates NaNs in affected slices.

    Returns
    -------
    result : numpy scalar or ndarray
        A scalar for axis=None or when reducing a 1-D input. Otherwise, the input
        shape with the selected axis removed. Input dtype is preserved.

    Raises
    ------
    TypeError
        If data is unsupported or axis/ignore_nan has an invalid type.
    ValueError
        If data is scalar/ragged or axis is out of bounds.
        Also raised when a minimum/maximum reduction has no elements.

    Warns
    -----
    RuntimeWarning
        All-NaN slices warn and return NaN. Empty reduced slices raise ValueError.

    Complexity
    ----------
    Time O(n) for n elements. Auxiliary space O(r) for ndarray input, O(n + r)
    including sequence conversion; r is the output element count.

    Examples
    --------
    >>> int(minimum([3, 1, 2]))
    1
    """
    return _reduce(data, "min", axis, ignore_nan)


def maximum(data, axis=None, ignore_nan=True):
    """Return the maximum along one axis or over flattened data.

    Parameters
    ----------
    data : array_like, shape (...,)
        Real numeric or boolean data with at least one dimension. Read-only
        and non-contiguous input are accepted and never modified.
    axis : int or None, default None
        None reduces all elements; an integer (including negative indices)
        removes that axis. Boolean axes and tuples are rejected.
    ignore_nan : bool, default True
        Omit NaNs when True; False propagates NaNs in affected slices.

    Returns
    -------
    result : numpy scalar or ndarray
        A scalar for axis=None or when reducing a 1-D input. Otherwise, the input
        shape with the selected axis removed. Input dtype is preserved.

    Raises
    ------
    TypeError
        If data is unsupported or axis/ignore_nan has an invalid type.
    ValueError
        If data is scalar/ragged or axis is out of bounds.
        Also raised when a minimum/maximum reduction has no elements.

    Warns
    -----
    RuntimeWarning
        All-NaN slices warn and return NaN. Empty reduced slices raise ValueError.

    Complexity
    ----------
    Time O(n) for n elements. Auxiliary space O(r) for ndarray input, O(n + r)
    including sequence conversion; r is the output element count.

    Examples
    --------
    >>> int(maximum([3, 1, 2]))
    3
    """
    return _reduce(data, "max", axis, ignore_nan)
