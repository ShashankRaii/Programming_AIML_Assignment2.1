"""Vectorized tie-aware ranking and explicit NaN-aware percentile interpolation."""

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
