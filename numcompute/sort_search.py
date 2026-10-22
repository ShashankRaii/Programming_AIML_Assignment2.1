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


def multi_key_sort(X, keys, descending=None):
    """Return rows ordered by multiple columns with stable tie handling.

    Parameters
    ----------
    X : array_like, shape (n_rows, n_columns)
        Real numeric or boolean matrix. Dtype and all columns are preserved.
        Empty row sets, strided arrays, NaNs and infinities are supported.
    keys : sequence of int, shape (n_keys,)
        Nonempty, distinct column indices in priority order: the first key is
        primary. Negative indices count from the last column. Booleans are not
        column indices. Equivalent positive/negative duplicates are rejected.
    descending : sequence of bool or None, default None
        One flag per key; True sorts that key descending. None sorts every key
        ascending. NaNs follow all numeric values in either direction. Rows
        whose primary values tie (including NaNs) are ordered by the next key.

    Returns
    -------
    sorted_rows : ndarray, shape (n_rows, n_columns)
        Independent row-permuted copy of X with unchanged dtype. Complete ties
        retain their input row order. Caller data is never modified.

    Raises
    ------
    TypeError
        If X is not real numeric/boolean, a key is not an integer, or a
        descending flag is not boolean.
    ValueError
        If X is not a matrix, keys are empty/not one-dimensional, a key is out
        of bounds or duplicated, or descending does not have one flag per key.

    Complexity
    ----------
    Time O(n*p + k*n*log(n + 1) + k); additional space O(n*p + k*n + k), including
    output, for n rows, p columns and k keys. Descending keys use integer order
    codes so unsigned values and minimum signed integers cannot overflow.

    Examples
    --------
    >>> multi_key_sort([[2, 1], [1, 2], [1, 3]], [0, 1], [False, True])
    array([[1, 3],
           [1, 2],
           [2, 1]])
    """
    array = check_array(X, ensure_2d=True)
    # Object conversion preserves bool/int distinctions for validation rather
    # than silently converting a mixed [0, True] sequence to integer indices.
    key_array = np.asarray(keys, dtype=object)
    if key_array.ndim != 1 or key_array.size == 0:
        raise ValueError(f"keys must be a nonempty 1-D sequence; got shape {key_array.shape}")
    normalized = []
    # Loops visit key metadata only. Row comparisons and ordering are NumPy
    # operations, with no Python loops over rows or individual numerical values.
    for key in key_array:
        if isinstance(key, (bool, np.bool_)) or not isinstance(key, (int, np.integer)):
            raise TypeError(f"keys must contain integer column indices; got {type(key).__name__}")
        key = int(key)
        if not -array.shape[1] <= key < array.shape[1]:
            raise ValueError(f"column key {key} is out of bounds for X shape {array.shape}")
        normalized.append(key % array.shape[1])
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"keys must identify distinct columns; got {key_array.tolist()}")

    if descending is None:
        flags = np.zeros(len(normalized), dtype=bool)
    else:
        flags = np.asarray(descending, dtype=object)
        if flags.ndim != 1 or flags.size != len(normalized):
            raise ValueError(
                f"descending must have one flag per key ({len(normalized)}); "
                f"got shape {flags.shape}"
            )
        if not all(isinstance(flag, (bool, np.bool_)) for flag in flags):
            raise TypeError("descending must contain boolean flags")

    sort_columns = []
    for key, reverse in zip(normalized, flags):
        column = array[:, key]
        if reverse:
            # Negate bounded order codes, never the original numeric data.
            # All NaNs share a code larger than the codes of valid values.
            _, codes = np.unique(column, return_inverse=True)
            column = np.where(np.isnan(column), array.shape[0], -codes)
        sort_columns.append(column)
    # lexsort is stable and uses its last key as primary.
    order = np.lexsort(tuple(sort_columns[::-1]))
    return array[order]
