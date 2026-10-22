"""Stable sorting, partial selection and searching for real numeric arrays."""

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


def topk(values, k, largest=True, return_indices=True):
    """Select and order k values using partial partitioning.

    Parameters
    ----------
    values : array_like, shape (n,)
        Real numeric or boolean values. NaNs are ranked after every valid
        value in either direction; infinities are valid values. Input data and
        dtype are preserved. Non-contiguous inputs are accepted.
    k : int
        Number of values to return, with 0 <= k <= n. Booleans are rejected.
    largest : bool, default True
        Return largest values in descending order; False selects smallest
        values in ascending order. Equal values prefer earlier input indices,
        both at the selection cutoff and in the returned order.
    return_indices : bool, default True
        Also return the selected original input indices when True.

    Returns
    -------
    top : ndarray, shape (k,)
        Independent copy of selected values, with the input dtype. If fewer
        than k non-NaN values exist, earliest NaNs fill the remaining positions.
    indices : ndarray of numpy.intp, shape (k,)
        Returned only when return_indices=True. Satisfies top == values[indices]
        with NaN-aware equality. Empty input is valid only for k=0.

    Raises
    ------
    TypeError
        If values is not real numeric/boolean, k is not an integer, or either
        option flag is not boolean.
    ValueError
        If values is not one-dimensional or k is outside [0, n].

    Complexity
    ----------
    Time O(n + k*log(k + 1)); auxiliary space O(n), including indices and output.
    Uses np.argpartition to find the cutoff, scans boundary ties in original
    order, and sorts only the selected non-NaN values. For k=n every value must
    be ordered. No Python loops over elements are used.

    Examples
    --------
    >>> topk([3, 1, 4, 1, 5], 2)
    (array([5, 4]), array([4, 2]))
    >>> topk([5, 5, 1], 1, return_indices=False)
    array([5])
    """
    array = check_array(values)
    if array.ndim != 1:
        raise ValueError(f"values must have 1 dimension; got shape {array.shape}")
    if isinstance(k, (bool, np.bool_)):
        raise TypeError("k must be an integer; got bool")
    try:
        k = operator.index(k)
    except TypeError as exc:
        raise TypeError("k must be an integer") from exc
    if not 0 <= k <= array.size:
        raise ValueError(f"k must satisfy 0 <= k <= {array.size}; got {k}")
    if not isinstance(largest, (bool, np.bool_)):
        raise TypeError("largest must be boolean")
    if not isinstance(return_indices, (bool, np.bool_)):
        raise TypeError("return_indices must be boolean")
    if k == 0:
        indices = np.empty(0, dtype=np.intp)
        top = array[indices]
        return (top, indices) if return_indices else top

    nan_mask = np.isnan(array)
    valid_indices = np.flatnonzero(~nan_mask)
    valid_values = array[valid_indices]
    count = min(k, valid_indices.size)
    if count == valid_indices.size:
        selected = valid_indices
    else:
        position = valid_indices.size - count if largest else count - 1
        partition = np.argpartition(valid_values, position)
        cutoff = valid_values[partition[position]]
        better = valid_values > cutoff if largest else valid_values < cutoff
        better_indices = valid_indices[better]
        # argpartition itself is unstable: choose boundary ties independently.
        tied_indices = valid_indices[valid_values == cutoff][:count - better_indices.size]
        selected = np.concatenate((better_indices, tied_indices))

    selected_values = array[selected]
    if largest:
        # Reverse before AND after a stable sort to keep ties in original order,
        # without negating data (which would overflow unsigned/minimum integers).
        order = np.argsort(selected_values[::-1], kind="stable")
        order = (selected.size - 1 - order)[::-1]
    else:
        order = np.argsort(selected_values, kind="stable")
    indices = selected[order]
    if count < k:
        indices = np.concatenate((indices, np.flatnonzero(nan_mask)[:k - count]))
    top = array[indices]
    return (top, indices) if return_indices else top


def binary_search(sorted_array, x):
    """Return left insertion positions and exact existence flags.

    Parameters
    ----------
    sorted_array : array_like, shape (n,)
        Real numeric or boolean values, already sorted ascending with NaNs at
        the end. Empty arrays, duplicates, infinities and strides are accepted.
        Sorting is a precondition: no linear sortedness check is performed.
    x : real numeric scalar or array_like, shape (...,)
        Queries, including empty/multidimensional arrays. NaN queries insert
        before existing NaNs (or after all values if there are no NaNs) and
        always have found=False. Numeric dtypes may differ between inputs.

    Returns
    -------
    index : int or ndarray of numpy.intp, shape matching x
        Leftmost matching index, or insertion position in [0, n] if absent.
        Scalar queries return a Python int.
    found : bool or ndarray of bool, shape matching x
        Whether the queried numeric value exists. Scalar queries return a
        Python bool. Large integers are compared exactly even when combining
        signed/unsigned or integer/floating dtypes. Inputs are never modified.

    Raises
    ------
    TypeError
        If either input contains unsupported nonnumeric or complex values.
    ValueError
        If sorted_array is not one-dimensional or either input is ragged.

    Complexity
    ----------
    For existing ndarrays, time O(m*log(n + 1) + m) and auxiliary space O(m) for
    m queries. Converting input sequences adds O(n + m) time/space. When NumPy's
    common dtype cannot represent an integer dtype exactly, an object-array
    fallback adds O(n + m) time/space to preserve exact Python numeric
    comparisons. Search itself uses np.searchsorted in all cases.

    Examples
    --------
    >>> binary_search([1, 3, 3, 5], 3)
    (1, True)
    >>> binary_search([1, 3, 3, 5], [0, 3, 6])
    (array([0, 1, 4]), array([False,  True, False]))
    """
    array = check_array(sorted_array)
    if array.ndim != 1:
        raise ValueError(f"sorted_array must have 1 dimension; got shape {array.shape}")
    queries = np.asarray(x)
    # Reuse numeric validation while allowing scalar queries and preserving
    # their original shape in the returned results.
    check_array(queries.reshape(-1))
    comparison_array, comparison_queries = array, queries
    common_dtype = np.result_type(array.dtype, queries.dtype)
    if common_dtype.kind == "f":
        precision = np.finfo(common_dtype).nmant + 1
        # This inspects only two dtype descriptors, never loops over samples.
        needs_exact_comparison = any(
            dtype.kind in "iu" and np.iinfo(dtype).bits - (dtype.kind == "i") > precision
            for dtype in (array.dtype, queries.dtype)
        )
        if needs_exact_comparison:
            # Python int/float comparisons avoid lossy float promotion of int64
            # and uint64. Remove trailing NaNs before object search because
            # Python NaN comparisons do not define NumPy's NaNs-last ordering.
            end = int(np.searchsorted(array, np.nan)) if array.dtype.kind == "f" else array.size
            comparison_array = array[:end].astype(object)
            comparison_queries = queries.astype(object)

    indices = np.searchsorted(comparison_array, comparison_queries, side="left")
    query_nan = np.isnan(queries)
    if comparison_array.dtype.kind == "O":
        indices = np.where(query_nan, comparison_array.size, indices)
    found = np.zeros(queries.shape, dtype=bool)
    if comparison_array.size:
        # Bound the gather before indexing, including insertion positions n.
        candidates = np.minimum(indices, comparison_array.size - 1)
        found = (
            (indices < comparison_array.size)
            & np.equal(comparison_array[candidates], comparison_queries)
            & ~query_nan
        )
    if queries.ndim == 0:
        return int(indices), bool(found)
    return np.asarray(indices, dtype=np.intp), found


def quickselect(a, k):
    """Return the zero-based kth smallest value using explicit quickselect.

    Parameters
    ----------
    a : array_like, shape (n,)
        Nonempty real numeric or boolean data. NaNs rank after all valid
        values; infinities are valid values. A private working copy preserves
        caller data, including read-only arrays and non-contiguous views.
    k : int
        Zero-based ascending order statistic, with 0 <= k < n. This is an
        index, unlike topk's count. Booleans are rejected.

    Returns
    -------
    value : numpy scalar, shape ()
        kth smallest value with the input array's inferred dtype. Returns NaN
        when k falls in the trailing NaN ranks. Equal values are equivalent;
        the original occurrence/sign of tied zeros is not guaranteed.

    Raises
    ------
    TypeError
        If a is not real numeric/boolean or k is not an integer.
    ValueError
        If a is not one-dimensional, is empty/ragged, or k is outside [0, n).

    Complexity
    ----------
    Average time O(n), worst-case O(n**2). Additional space O(n) for the private
    copy and NaN mask; the iterative partition loop uses O(1) workspace and no
    recursion. A local generator seeded with 42 selects reproducible pivots
    without affecting global random state. Three-way partitioning processes
    all values equal to a pivot together, avoiding repeated work on tie groups.
    This educational algorithm deliberately uses Python partition loops rather
    than delegating to sorting, np.partition, or np.argpartition.

    Examples
    --------
    >>> int(quickselect([3, 1, 4, 1, 5], 2))
    3
    >>> bool(np.isnan(quickselect([2.0, np.nan, 1.0], 2)))
    True
    """
    array = check_array(a)
    if array.ndim != 1:
        raise ValueError(f"a must have 1 dimension; got shape {array.shape}")
    if isinstance(k, (bool, np.bool_)):
        raise TypeError("k must be an integer; got bool")
    try:
        k = operator.index(k)
    except TypeError as exc:
        raise TypeError("k must be an integer") from exc
    if array.size == 0:
        raise ValueError("a must be nonempty; got shape (0,)")
    if not 0 <= k < array.size:
        raise ValueError(f"k must satisfy 0 <= k < {array.size}; got {k}")

    valid = ~np.isnan(array)
    working = array[valid]  # Boolean indexing creates the private working copy.
    if k >= working.size:
        return array[~valid][0]

    rng = np.random.default_rng(42)
    left, right = 0, working.size - 1
    # Explicit loops expose the educational partition algorithm. NaNs have
    # already been removed so that every pivot comparison has a total order.
    while left < right:
        pivot = working[int(rng.integers(left, right + 1))]
        lower, current, upper = left, left, right
        # Invariant: [left, lower) < pivot, [lower, current) == pivot,
        # (upper, right] > pivot; [current, upper] is still unclassified.
        while current <= upper:
            if working[current] < pivot:
                working[lower], working[current] = working[current], working[lower]
                lower += 1
                current += 1
            elif working[current] > pivot:
                working[current], working[upper] = working[upper], working[current]
                upper -= 1  # Inspect the newly swapped-in value next.
            else:
                current += 1
        if k < lower:
            right = lower - 1
        elif k > upper:
            left = upper + 1
        else:
            return working[k]
    return working[left]
