"""Vectorized tie-aware ranking and explicit NaN-aware percentile interpolation."""

import operator
import warnings

import numpy as np

from .utils import check_array


def rank(data, method="average"):
    """Return one-based ascending ranks with explicit tie handling.

    Parameters
    ----------
    data : array_like, shape (n,)
        Real numeric or boolean values. NaNs are excluded from ranking and
        remain NaN in the output. Infinities are valid; signed zeros are tied.
        Empty and non-contiguous input are accepted. Input is never modified.
    method : {'average', 'dense', 'ordinal', 'min', 'max'}, default 'average'
        For equal values, average uses the mean occupied rank; min/max use the
        first/last occupied rank. Dense ranks consecutive distinct groups
        without gaps. Ordinal assigns distinct ranks in original tie order.

    Returns
    -------
    ranks : ndarray of float64, shape (n,)
        Ranks in original input order. Empty input returns an empty array;
        all-NaN input returns all NaNs. Sorting preserves the input dtype before
        assigning ranks, so distinct large integers do not become false ties.

    Raises
    ------
    TypeError
        If data is not real numeric/boolean or method is not a string.
    ValueError
        If data is not one-dimensional/ragged, or method is unsupported.

    Complexity
    ----------
    Time O(n log(n + 1)); auxiliary space O(n), including output. A stable
    argsort and vectorized group boundaries avoid Python loops over elements.

    Examples
    --------
    >>> rank([30, 10, 10, 20])
    array([4. , 1.5, 1.5, 3. ])
    >>> rank([30, 10, 10, 20], method='ordinal')
    array([4., 1., 2., 3.])
    """
    array = check_array(data)
    if array.ndim != 1:
        raise ValueError(f"data must have 1 dimension; got shape {array.shape}")
    if not isinstance(method, str):
        raise TypeError("method must be a string")
    if method not in ("average", "dense", "ordinal", "min", "max"):
        raise ValueError(f"unsupported rank method {method!r}")
    output = np.full(array.shape, np.nan, dtype=np.float64)
    valid_indices = np.flatnonzero(~np.isnan(array))
    if valid_indices.size == 0:
        return output
    values = array[valid_indices]
    order = np.argsort(values, kind="stable")
    positions = np.arange(1, values.size + 1, dtype=np.float64)
    if method == "ordinal":
        output[valid_indices[order]] = positions
        return output

    sorted_values = values[order]
    starts_group = np.concatenate(([True], sorted_values[1:] != sorted_values[:-1]))
    groups = np.cumsum(starts_group) - 1
    starts = np.flatnonzero(starts_group)
    ends = np.concatenate((starts[1:], [values.size]))
    if method == "dense":
        sorted_ranks = groups + 1
    elif method == "min":
        sorted_ranks = (starts + 1)[groups]
    elif method == "max":
        sorted_ranks = ends[groups]
    else:
        sorted_ranks = ((starts.astype(np.float64) + 1 + ends) / 2)[groups]
    output[valid_indices[order]] = sorted_ranks
    return output


def _blend_percentile_values(lower, upper, weight):
    """Blend bounds with near-endpoint precision and a finite-overflow fallback."""
    result = lower.copy()
    result[weight == 1] = upper[weight == 1]
    interior = (weight > 0) & (weight < 1) & (lower != upper)
    finite = interior & np.isfinite(lower) & np.isfinite(upper)
    low, high, fraction = lower[finite], upper[finite], weight[finite]
    # Prefer the nearer endpoint to retain subnormal precision. If the finite
    # bounds' difference overflows, convex weights avoid that intermediate.
    with np.errstate(over="ignore"):
        difference = high - low
    safe = np.isfinite(difference)
    left = safe & (fraction < 0.5)
    right = safe & ~left
    blended = np.empty_like(low)
    blended[left] = low[left] + difference[left] * fraction[left]
    blended[right] = high[right] - difference[right] * (1 - fraction[right])
    blended[~safe] = (1 - fraction[~safe]) * low[~safe] + fraction[~safe] * high[~safe]
    result[finite] = blended
    result[interior & np.isposinf(upper)] = np.inf
    result[interior & np.isneginf(lower)] = -np.inf
    undefined = interior & np.isneginf(lower) & np.isposinf(upper)
    if undefined.any():
        result[undefined] = np.nan
        warnings.warn(
            "Percentile interpolation between opposite infinities is undefined",
            RuntimeWarning,
            stacklevel=4,
        )
    return result


def _percentile(data, q, interpolation="linear", axis=None):
    """Shared NaN-aware kernel; an integer axis is reserved for stats.quantile.

    Scalar q removes the reduction axis; vector q prepends its length. None
    flattens data. Input is sorted in its own dtype before bounds are gathered
    and converted to at least float64. No Python loops over data/slices are used.
    """
    array = check_array(data)
    if not isinstance(interpolation, str):
        raise TypeError("interpolation must be a string")
    if interpolation not in ("linear", "lower", "higher", "midpoint", "nearest"):
        raise ValueError(f"unsupported interpolation {interpolation!r}")
    levels = np.asarray(q)
    if levels.ndim > 1:
        raise ValueError(f"q must be scalar or 1-D; got shape {levels.shape}")
    if levels.dtype.kind not in "iuf":
        raise TypeError(f"q must contain real numeric percentages; got dtype {levels.dtype}")
    if not np.isfinite(levels).all() or np.any(levels < 0) or np.any(levels > 100):
        raise ValueError("q must contain finite percentages between 0 and 100 inclusive")
    scalar_q = levels.ndim == 0
    dtype = np.result_type(array.dtype, levels.dtype, np.float64)
    levels = levels.astype(dtype, copy=False).reshape(-1)

    if axis is None:
        matrix = array.reshape(-1, 1)
        remaining_shape = ()
    else:
        if isinstance(axis, (bool, np.bool_)):
            raise TypeError("axis must be an integer or None; got bool")
        try:
            axis = operator.index(axis)
        except TypeError as exc:
            raise TypeError("axis must be an integer or None") from exc
        if not -array.ndim <= axis < array.ndim:
            raise ValueError(f"axis {axis} is invalid for data shape {array.shape}")
        moved = np.moveaxis(array, axis, 0)
        remaining_shape = moved.shape[1:]
        columns = int(np.prod(remaining_shape, dtype=np.intp))
        matrix = moved.reshape(moved.shape[0], columns)

    shape = remaining_shape if scalar_q else (levels.size,) + remaining_shape
    result = np.full((levels.size, matrix.shape[1]), np.nan, dtype=dtype)
    if result.size:
        counts = np.count_nonzero(~np.isnan(matrix), axis=0)
        empty = counts == 0
        if empty.any():
            warnings.warn(
                "Percentile is undefined for an empty or all-NaN slice",
                RuntimeWarning,
                stacklevel=3,
            )
        if matrix.shape[0]:
            ordered = np.sort(matrix, axis=0, kind="stable")
            position = (levels[:, None] / 100) * np.maximum(counts - 1, 0)
            low_index = np.floor(position).astype(np.intp)
            high_index = np.ceil(position).astype(np.intp)
            if interpolation == "nearest":
                low_index = high_index = np.rint(position).astype(np.intp)
            elif interpolation == "lower":
                high_index = low_index
            elif interpolation == "higher":
                low_index = high_index
            lower = np.take_along_axis(ordered, low_index, axis=0).astype(dtype)
            upper = np.take_along_axis(ordered, high_index, axis=0).astype(dtype)
            if interpolation == "midpoint":
                weight = np.where(low_index == high_index, 0.0, 0.5)
            else:
                weight = position - low_index
            # Discrete methods gather a single bound; no interpolation is needed.
            if interpolation in ("lower", "higher", "nearest"):
                result = lower
            else:
                result = _blend_percentile_values(lower, upper, weight)
            result[:, empty] = np.nan
    result = result.reshape(shape)
    return result[()] if result.ndim == 0 else result


def percentile(data, q, interpolation="linear"):
    """Compute percentiles explicitly, ignoring NaNs and flattening input.

    Parameters
    ----------
    data : array_like, shape (...,)
        Real numeric or boolean data with at least one dimension. All elements
        are treated as one sample. Inputs, including strided/read-only arrays,
        are never modified. NaNs are excluded; infinities remain valid values.
    q : real scalar or one-dimensional array_like, shape (m,)
        Finite percentages in [0, 100]. Booleans and numeric strings are rejected.
        An empty sequence returns an empty array without warnings.
    interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}, default 'linear'
        At sorted index h=(n-1)*q/100, lower/higher select floor/ceil(h), linear
        blends the two values, and midpoint averages them. Nearest uses the
        nearest index, rounding exact half-indices to even (NumPy convention).

    Returns
    -------
    value : numpy floating scalar or ndarray, shape (m,)
        Scalar q returns a scalar; vector q returns a matching vector. Output
        uses at least float64, preserving wider floating inputs/query dtypes.
        Large integer output can round to floating precision, but sorting and
        selection occur before conversion. Empty/all-NaN samples return NaN.

    Raises
    ------
    TypeError
        If data/q is unsupported, or interpolation is not a string.
    ValueError
        If data is scalar/ragged, q is not scalar/1-D or outside [0, 100],
        q is non-finite, or interpolation is unsupported.

    Warns
    -----
    RuntimeWarning
        If a nonempty query has no non-NaN sample values, or an interior
        interpolation spans opposite infinities. The latter returns NaN.
        Exact endpoints and equal infinities return the selected infinity;
        interior finite-to-infinite interpolation returns that infinity.

    Complexity
    ----------
    Time O(n log(n + 1) + m); auxiliary space O(n + m), including sorted copy
    and output. Index/interpolation maths are vectorized NumPy operations;
    np.percentile and np.quantile are not used to implement the calculation.

    Examples
    --------
    >>> percentile([10, 20, 30, 40], [0, 25, 50, 100])
    array([10. , 17.5, 25. , 40. ])
    >>> float(percentile([1, np.nan, 3], 50))
    2.0
    """
    return _percentile(data, q, interpolation=interpolation)
