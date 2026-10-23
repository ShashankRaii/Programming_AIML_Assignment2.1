"""CSV loading utilities (delimiters, missing values, chunked streaming).

Missing cells can be filled with a placeholder or the row that contain them can be dropped. Large files can be read in chunks with load_csv_chunks.

Functions:-
read_header
    Return the column names from the first line of a CSV file.
load_csv
    Load a whole CSV file into a 2-D NumPy array.
load_csv_chunks
    Stream a large CSV file as a sequence of 2-D NumPy arrays.
"""

import os
import warnings
from itertools import islice

import numpy as np

__all__ = ["read_header", "load_csv", "load_csv_chunks"]

_MISSING_OPTIONS = ("fill", "skip")


def _is_int(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _validate_common(path, delimiter, skip_header, missing):
    # Check the arguments every reader shares.
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV file not found: {path!r}")
    if not isinstance(delimiter, str):
        raise TypeError(f"delimiter must be a string, got {type(delimiter).__name__}")
    if delimiter == "":
        raise ValueError("delimiter must be a non-empty string")
    if not _is_int(skip_header):
        raise TypeError(f"skip_header must be an int, got {type(skip_header).__name__}")
    if skip_header < 0:
        raise ValueError(f"skip_header must be >= 0, got {skip_header}")
    if missing not in _MISSING_OPTIONS:
        raise ValueError(f"missing must be one of {_MISSING_OPTIONS}, got {missing!r}")


def _missing_row_mask(data):
    # True for rows with a NaN (numeric) or an empty cell (strings).
    if data.size == 0:
        return np.zeros(data.shape[0], dtype=bool)
    if np.issubdtype(data.dtype, np.number):
        return np.isnan(data).any(axis=1)
    return (np.char.strip(data.astype(str)) == "").any(axis=1)


def _parse(source, delimiter, skip_header, missing, fill_value, usecols, dtype):
    # Run genfromtxt on a path or a list of lines and return a 2-D array.
    is_numeric = np.issubdtype(np.dtype(dtype), np.number)
    # In skip mode we need NaN/"" in the gaps so we can find those rows later,
    # whatever fill_value the caller passed.
    if missing == "skip" or fill_value is None:
        filling = np.nan if is_numeric else ""
    else:
        filling = fill_value

    with warnings.catch_warnings():
        # genfromtxt warns about empty files; we return an empty array instead
        warnings.simplefilter("ignore", UserWarning)
        data = np.genfromtxt(
            source,
            delimiter=delimiter,
            skip_header=skip_header,
            dtype=dtype,
            usecols=usecols,
            filling_values=filling,
            autostrip=True,
            ndmin=2,
            encoding="utf-8",
        )

    if data.size == 0:
        return np.empty((0, 0), dtype=dtype)
    if missing == "skip":
        data = data[~_missing_row_mask(data)]
    return data


def read_header(path, delimiter=","):
    """Return the column names from the first line of a CSV file.
    Parameters
    ----------
    path : str or os.PathLike
    delimiter : str, default ","

    Returns
    -------
    names : list of str, length n_cols
        Stripped column names; an empty list for an empty file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    TypeError, ValueError
        If the delimiter is invalid.

    Complexity
    ----------
    Time and space O(L), L = length of the first line.
    """
    _validate_common(path, delimiter, 0, "fill")
    with open(path, "r", encoding="utf-8") as fh:
        first = fh.readline().rstrip("\r\n")
    if first == "":
        return []
    return [name.strip() for name in first.split(delimiter)]


def load_csv(
    path,
    delimiter=",",
    skip_header=0,
    missing="fill",
    fill_value=np.nan,
    usecols=None,
    dtype=float,
):
    """Load a CSV file into a 2-D NumPy array.
    Parameters
    ----------
    path : str or os.PathLike
    delimiter : str, default ","
        Use "\\t" for tab-separated files.
    skip_header : int, default 0
        Lines to skip at the top, e.g. 1 for a header row.
    missing : {"fill", "skip"}, default "fill"
        "fill" puts fill_value in empty cells, "skip" drops rows that have any.
    fill_value : scalar, default np.nan
        Only used when missing="fill".
    usecols : int or sequence of int, optional
        Columns to read; all by default.
    dtype : data-type, default float
        Use str for categorical columns.

    Returns
    -------
    data : ndarray, shape (n_rows, n_cols)
        Always 2-D. (0, 0) for an empty file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    TypeError
        If delimiter is not a string or skip_header is not an int.
    ValueError
        Empty delimiter, negative skip_header, unknown `missing` option, or
        rows with different numbers of columns.

    Complexity
    ----------
    Time and space O(n_rows * n_cols).

    Examples
    --------
    >>> X = load_csv("data/sample.csv", skip_header=1)  # doctest: +SKIP
    >>> X.shape  # doctest: +SKIP
    (100, 4)
    """
    _validate_common(path, delimiter, skip_header, missing)
    return _parse(path, delimiter, skip_header, missing, fill_value, usecols, dtype)


def load_csv_chunks(
    path,
    chunk_size,
    delimiter=",",
    skip_header=0,
    missing="fill",
    fill_value=np.nan,
    usecols=None,
    dtype=float,
):
    """Read a CSV file in blocks of at most chunk_size rows.

    Useful when the file is too big to load at once, e.g. together with
    stats.RunningStats.

    Parameters
    ----------
    path : str or os.PathLike
    chunk_size : int
        Maximum lines per chunk, >= 1.
    delimiter, skip_header, missing, fill_value, usecols, dtype
        Same as load_csv. Header lines are skipped once at the start.

    Yields
    ------
    chunk : ndarray, shape (m, n_cols), m <= chunk_size
        With missing="skip" a chunk can be shorter; empty chunks are not yielded.

    Raises
    ------
    FileNotFoundError, TypeError, ValueError
        Same as load_csv, plus a bad chunk_size.

    Complexity
    ----------
    Time O(n_rows * n_cols) in total; space O(chunk_size * n_cols).

    Examples
    --------
    >>> for chunk in load_csv_chunks("big.csv", 10_000, skip_header=1):
    ...     process(chunk)  # doctest: +SKIP
    """
    _validate_common(path, delimiter, skip_header, missing)
    if not _is_int(chunk_size):
        raise TypeError(f"chunk_size must be an int, got {type(chunk_size).__name__}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    with open(path, "r", encoding="utf-8") as fh:
        for _ in islice(fh, skip_header):
            pass
        # Looping over the file is the point of streaming; each chunk itself
        # is parsed by genfromtxt in one go.
        while True:
            raw = list(islice(fh, chunk_size))
            if not raw:
                break
            lines = [ln for ln in raw if ln.strip()]
            if not lines:
                continue
            chunk = _parse(lines, delimiter, 0, missing, fill_value, usecols, dtype)
            if chunk.shape[0] > 0:
                yield chunk