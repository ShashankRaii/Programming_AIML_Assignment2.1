"""Axis-aware descriptive statistics, quantiles, histograms and running moments.

Reductions use NumPy's numeric and warning semantics, with floating calculations
promoted to at least float64. RunningStats combines vectorized batch moments
using the parallel form of Welford's algorithm without retaining observations.
"""

import operator

import numpy as np

from .rank import _percentile
from .sort_search import binary_search
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
        Empty reduced slices warn and return NaN. With ignore_nan=True,
        all-NaN slices also warn and return NaN. Other arithmetic follows NumPy.

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
        Empty reduced slices warn and return NaN. With ignore_nan=True,
        all-NaN slices also warn and return NaN. Other arithmetic follows NumPy.

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
        With ignore_nan=True, empty/all-NaN slices or count<=ddof warn and
        return NaN. With ignore_nan=False, insufficient count warns and may
        return NaN or infinity. Non-finite arithmetic follows NumPy.

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
        With ignore_nan=True, empty/all-NaN slices or count<=ddof warn and
        return NaN. With ignore_nan=False, insufficient count warns and may
        return NaN or infinity. Non-finite arithmetic follows NumPy.

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
        With ignore_nan=True, all-NaN slices warn and return NaN. Empty reduced
        slices raise ValueError.

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
        With ignore_nan=True, all-NaN slices warn and return NaN. Empty reduced
        slices raise ValueError.

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


def quantile(data, q, axis=None, interpolation="linear"):
    """Return NaN-aware quantiles using the shared percentile calculation.

    Parameters
    ----------
    data : array_like, shape (...,)
        Real numeric or boolean data with at least one dimension. NaNs are
        omitted independently in each reduced slice. Inputs are never modified.
    q : real scalar or one-dimensional array_like, shape (m,)
        Finite probabilities in [0, 1], unlike percentile's [0, 100]. Numeric
        strings and booleans are rejected. Empty queries return empty output.
    axis : int or None, default None
        None flattens data; an integer, including a negative index, removes one
        axis. Tuples and boolean axes are rejected.
    interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}, default 'linear'
        Interpolate sorted indices (count-1)*q. Nearest rounds half-indices to
        even. Reuses rank.percentile's kernel without rounding q through percent.

    Returns
    -------
    result : numpy floating scalar or ndarray
        Scalar q returns the input shape with the reduction axis removed (a
        scalar for axis=None). Vector q prepends dimension m to that shape.
        Output uses at least float64; large integer outputs can round. Empty
        queries preserve a zero-length query dimension and do not warn.

    Raises
    ------
    TypeError
        If data/q is unsupported or axis/interpolation has an invalid type.
    ValueError
        If data is scalar/ragged, q is non-finite, not scalar/1-D or outside
        [0, 1], the axis is out of bounds, or interpolation is unsupported.

    Warns
    -----
    RuntimeWarning
        Empty/all-NaN reduced slices return NaN with a warning. Interior blends
        between opposite infinities also warn and return NaN. Other infinity
        handling and numerical safeguards match rank.percentile.

    Complexity
    ----------
    Time O(N log(L + 1) + m*S); auxiliary space O(N + m*S), where L is the
    reduced axis length, S is the remaining slice count and N=L*S. With
    axis=None, L=N and S=1. The sorting and interpolation are vectorized.

    Examples
    --------
    >>> quantile([[1, 2], [3, 4]], [0, 0.5, 1], axis=0)
    array([[1., 2.],
           [2., 3.],
           [3., 4.]])
    """
    return _percentile(data, q, interpolation=interpolation, axis=axis, _scale=1)


def histogram(data, bins=10, range=None):
    """Count flattened non-NaN values in bins with an inclusive final edge.

    Parameters
    ----------
    data : array_like, shape (...,)
        Real numeric or boolean data with at least one dimension. NaNs are
        omitted, not replaced. Input and bin edges are never modified.
    bins : positive int or real array_like, shape (b + 1,), default 10
        An integer generates equal-width edges using np.histogram_bin_edges.
        Explicit edges must be strictly increasing, contain at least two
        values, and have no NaNs; infinite endpoints are allowed. Boolean bin
        counts, boolean edges, and automatic string estimators are rejected.
    range : pair of finite real numbers or None, default None
        Inclusive bounds for integer bins. Values outside are ignored. Equal
        bounds are expanded by NumPy. None infers bounds from non-NaN data,
        using [0, 1] for empty/all-NaN input. Infinity in data requires a finite
        range or explicit edges. With explicit edges, range is validated but
        has no effect, matching NumPy's use of the explicit edges.

    Returns
    -------
    counts : ndarray of integers, shape (b,)
        Counts for [edge[i], edge[i+1]), except the final bin includes its right
        edge. Out-of-range values are excluded. No density normalization.
    edges : ndarray, shape (b + 1,)
        Independent copy of the edges. Explicit edge dtypes are preserved;
        generated edge dtypes follow NumPy. Returned counts/edges share no
        memory with caller input. Empty data returns zero counts.

    Raises
    ------
    TypeError
        If data, bins or range has an unsupported type.
    ValueError
        If shapes are invalid, bin count is not positive, edges are not strictly
        increasing/contain NaNs, range is not finite/ordered, or automatic
        edges cannot be represented as finite distinct values.

    Complexity
    ----------
    Time O(n log(b + 1) + b); auxiliary space O(n + b). NumPy generates edges;
    the shared binary_search locates bins with exact mixed numeric comparisons
    when needed, and np.bincount accumulates them without Python sample loops.

    Examples
    --------
    >>> histogram([0, 1, 2, 3, np.nan], bins=[0, 1, 3])
    (array([1, 3]), array([0, 1, 3]))
    """
    array = check_array(data)
    values = array.reshape(-1)
    values = values[~np.isnan(values)]
    if values.dtype.kind == "b":
        values = values.astype(np.uint8)

    bounds = None
    if range is not None:
        bounds = np.asarray(range)
        if bounds.dtype.kind not in "iuf":
            raise TypeError("range must contain real numeric bounds")
        if bounds.shape != (2,):
            raise ValueError(f"range must have shape (2,); got {bounds.shape}")
        if not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
            raise ValueError("range must have finite ordered bounds (low <= high)")

    supplied = np.asarray(bins)
    if supplied.ndim == 0:
        if supplied.dtype.kind not in "iu":
            raise TypeError("bins must be a positive integer or a 1-D real edge array")
        count = operator.index(bins)
        if count <= 0:
            raise ValueError(f"bins must be positive; got {count}")
        if bounds is None and np.isinf(values).any():
            raise ValueError("data contains infinity; provide a finite range or explicit edges")
        edges = np.histogram_bin_edges(values, bins=count, range=bounds)
        if not np.isfinite(edges).all():
            raise ValueError("automatic bin edges are not finite; provide explicit edges")
    else:
        if supplied.dtype.kind not in "iuf":
            raise TypeError("bins must contain real numeric edges")
        edges = supplied
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError(f"bin edges must be 1-D with at least 2 values; got shape {edges.shape}")
    # Direct comparisons avoid overflow in np.diff on extreme integer edges.
    if np.isnan(edges).any() or np.any(edges[1:] <= edges[:-1]):
        raise ValueError("bin edges must be strictly increasing and contain no NaNs")

    positions, matched = binary_search(edges, values)
    bin_indices = positions + matched - 1
    final_edge = matched & (positions == edges.size - 1)
    bin_indices[final_edge] -= 1
    inside = (bin_indices >= 0) & (bin_indices < edges.size - 1)
    counts = np.bincount(bin_indices[inside], minlength=edges.size - 1)
    return counts, edges.copy()


class RunningStats:
    """Accumulate one-dimensional streaming mean and population variance.

    Parameters
    ----------
    None
        Construct an empty accumulator; supply batches through update.

    Attributes
    ----------
    count : int
        Read-only count of observed non-NaN values, initially zero.
    mean : numpy floating scalar
        Read-only mean, or NaN for an empty stream without issuing a warning.
    variance : numpy floating scalar
        Read-only population variance (ddof=0), or NaN for an empty stream.
        A singleton or constant stream has variance zero.

    Notes
    -----
    Each batch is summarized with centered deviations; summaries are combined
    with the parallel Welford/Chan formula. State contains three scalars only.
    Calculations use at least float64, retaining wider floating batches. As
    with batch floating statistics, converting large integers can round values
    beyond the accumulator's floating precision. Merge/update results can
    differ slightly with grouping/order because of floating-point rounding.

    Infinity is rejected, while NaNs are omitted. Arithmetic that cannot keep
    a finite mean and centered sum of squares raises OverflowError atomically,
    even if dividing an unrepresentable moment could give a finite variance.

    Complexity
    ----------
    Construction/accessors and merges use O(1) time and scalar storage. Updating
    n values uses O(n) time and O(n) temporary space, with O(1) retained state.

    Examples
    --------
    >>> running = RunningStats().update([1, 3, np.nan])
    >>> running.count, float(running.mean), float(running.variance)
    (2, 2.0, 1.0)
    """

    __slots__ = ("_count", "_mean", "_m2")

    def __init__(self):
        """Initialize empty scalar state in O(1) time and space."""
        self._count = 0
        self._mean = np.float64(0)
        self._m2 = np.float64(0)

    @property
    def count(self):
        """Return the non-NaN sample count as a Python int.

        No parameters or exceptions. Time O(1), additional space O(1).
        Reading the property does not modify state.
        """
        return self._count

    @property
    def mean(self):
        """Return the floating scalar mean, or NaN for empty state.

        No parameters, warnings or exceptions. Time O(1), additional space O(1).
        Reading the property does not modify state.
        """
        return self._mean if self._count else np.float64(np.nan)

    @property
    def variance(self):
        """Return floating scalar population variance (M2/count).

        No parameters, warnings or exceptions. Empty state returns NaN and a
        singleton returns zero. Time O(1), additional space O(1). No mutation.
        """
        return self._m2 / self._count if self._count else np.float64(np.nan)

    def _combine(self, count, average, m2):
        """Combine snapshots, committing state only after finite checks pass."""
        if count == 0:
            return self
        if self._count == 0:
            total, combined_mean, combined_m2 = count, average, m2
        else:
            total = self._count + count
            dtype = np.result_type(self._mean.dtype, average.dtype, np.float64)
            first_mean = dtype.type(self._mean)
            second_mean = dtype.type(average)
            first_m2, second_m2 = dtype.type(self._m2), dtype.type(m2)
            try:
                with np.errstate(over="raise", invalid="raise"):
                    delta = second_mean - first_mean
                    weight = dtype.type(count / total)
                    combined_mean = first_mean + delta * weight
                    # Scale before squaring to avoid a needless delta**2
                    # overflow when the final correction is representable.
                    correction = delta * (delta * weight) * self._count
                    combined_m2 = first_m2 + second_m2 + correction
            except (FloatingPointError, OverflowError) as exc:
                raise OverflowError("running statistics merge exceeds floating range") from exc
        if not np.isfinite(combined_mean) or not np.isfinite(combined_m2):
            raise OverflowError("running statistics require finite mean and centered moment")
        self._count, self._mean, self._m2 = total, combined_mean, combined_m2
        return self

    def update(self, batch):
        """Accumulate a one-dimensional batch using vectorized centered moments.

        Parameters
        ----------
        batch : array_like, shape (n,)
            Real numeric/boolean observations. NaNs are skipped; infinity is
            rejected. Empty/all-NaN batches are no-ops. Scalars and matrices
            are rejected; wrap a single observation in a one-element list.
            Caller data is never modified or retained.

        Returns
        -------
        self : RunningStats
            Updated accumulator for chaining. Only this accumulator is mutated.

        Raises
        ------
        TypeError
            If batch is not real numeric or boolean.
        ValueError
            If batch is not one-dimensional, is ragged, or contains infinity.
        OverflowError
            If finite batch/combined moments cannot be represented in the
            accumulation dtype. Existing state is unchanged on any failure.

        Complexity
        ----------
        Time O(n); temporary space O(n), retained scalar state O(1). No Python
        loops over observations; batches are combined by the Welford formula.

        Examples
        --------
        >>> running = RunningStats().update([1, 3]).update([5])
        >>> running.count, float(running.mean)
        (3, 3.0)
        """
        array = check_array(batch, allow_inf=False)
        if array.ndim != 1:
            raise ValueError(f"batch must have 1 dimension; got shape {array.shape}")
        values = array[~np.isnan(array)]
        if values.size == 0:
            return self
        dtype = np.result_type(values.dtype, self._mean.dtype, np.float64)
        values = values.astype(dtype, copy=False)
        try:
            with np.errstate(over="raise", invalid="raise"):
                # Shift before averaging to retain variation around a large
                # offset, and avoid overflowing the sum of equal huge values.
                shifted = values - values[0]
                shift_mean = np.mean(shifted)
                average = values[0] + shift_mean
                deviations = shifted - shift_mean
                m2 = np.dot(deviations, deviations)
        except (FloatingPointError, OverflowError) as exc:
            raise OverflowError("running statistics batch exceeds floating range") from exc
        if not np.isfinite(average) or not np.isfinite(m2):
            raise OverflowError("batch must have representable finite mean and centered moment")
        return self._combine(int(values.size), average, m2)

    def merge(self, other):
        """Merge another accumulator's summary, leaving that source unchanged.

        Parameters
        ----------
        other : RunningStats
            Source summary. Empty sources are no-ops. Merging into empty state
            copies scalar moments without sharing mutable state. Merging self
            with itself counts the observations twice, retaining their mean
            and population variance.

        Returns
        -------
        self : RunningStats
            Combined accumulator for chaining. The source is not changed,
            except when source and destination are the same object.

        Raises
        ------
        TypeError
            If other is not a RunningStats instance.
        OverflowError
            If combined moments cannot remain finite. Both states are unchanged
            on failure. Grouping/order can introduce floating-point roundoff.

        Complexity
        ----------
        Time O(1); additional space O(1), using scalar snapshots only.

        Examples
        --------
        >>> first = RunningStats().update([1, 3])
        >>> second = RunningStats().update([5, 7])
        >>> combined = first.merge(second)
        >>> combined.count, float(combined.mean), float(combined.variance), second.count
        (4, 4.0, 5.0, 2)
        """
        if not isinstance(other, RunningStats):
            raise TypeError(f"other must be RunningStats; got {type(other).__name__}")
        return self._combine(other._count, other._mean, other._m2)
