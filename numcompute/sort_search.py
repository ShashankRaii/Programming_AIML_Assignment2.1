"""Stable sorting and row ordering for real numeric arrays.

Selection and searching functions will be added in later checkpoints.
"""

import operator

import numpy as np

from .utils import check_array


def stable_sort(a, axis=-1):
    """Return a stable ascending sort without modifying the input.

    Parameters
    ----------
    a : array_like, shape (...,)
        Real numeric or boolean data with at least one dimension. Empty arrays,
        strided arrays, NaNs and infinities are accepted. Dtype is preserved.
    axis : int or None, default -1
        Axis to sort, including negative indices. None flattens before sorting.
        Boolean axes are rejected.

    Returns
    -------
    sorted_array : ndarray
        Independent sorted copy with the same shape and dtype as a, or shape
        (a.size,) for axis=None. Equal values retain their original order.
        NaNs follow all numeric values, including positive infinity.

    Raises
    ------
    TypeError
        If a is not real numeric/boolean, or axis is not an integer or None.
    ValueError
        If a is scalar or ragged, or axis lies outside the valid dimensions.

    Complexity
    ----------
    Time O(N log(m + 1)) in the worst case, where N is the total element count
    and m is the sorted axis length (N for axis=None). Additional space O(N),
    including the output. NumPy may use faster dtype-specific stable algorithms.

    Examples
    --------
    >>> stable_sort([3, 1, 2, 1])
    array([1, 1, 2, 3])
    """
    array = check_array(a)
    if axis is not None:
        if isinstance(axis, (bool, np.bool_)):
            raise TypeError("axis must be an integer or None; got bool")
        try:
            axis = operator.index(axis)
        except TypeError as exc:
            raise TypeError("axis must be an integer or None") from exc
        if not -array.ndim <= axis < array.ndim:
            raise ValueError(f"axis {axis} is invalid for input shape {array.shape}")
    return np.sort(array, axis=axis, kind="stable")
