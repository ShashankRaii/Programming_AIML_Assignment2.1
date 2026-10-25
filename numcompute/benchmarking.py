"""Micro-benchmark harness for comparing loop and vectorised implementations.

``benchmark`` times one function with ``time.perf_counter``

The timing loops below are intentional: benchmarking means calling the
function repeatedly, so they are not core numerical computation.
"""

import os
import platform
import sys
import time

import numpy as np

from .utils import check_callable

__all__ = [
    "SEED",
    "benchmark",
    "compare",
    "environment_info",
]

SEED = 42

def _check_count(value, name, minimum):
    """Validate an integer timing parameter."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer; got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}; got {value}")
    return int(value)


def benchmark(func, *args, repeats=7, number=10, warmup=1):
    """Time ``func(*args)`` and summarise the per-call duration.

    Parameters
    ----------
    func : callable
        Function to time.
    *args
        Positional arguments passed to ``func`` on every call.
    repeats : int, default 7
        Number of timed rounds; statistics are taken over rounds.
    number : int, default 10
        Calls per round. Use more calls for very fast functions.
    warmup : int, default 1
        Untimed calls made first (caches, lazy imports, ...).

    Returns
    -------
    result : dict
        ``{'mean', 'std', 'min'}`` in seconds per call, plus ``'repeats'``
        and ``'number'``.

    Raises
    ------
    TypeError
        If ``func`` is not callable or a count is not an integer.
    ValueError
        If ``repeats`` or ``number`` is below 1, or ``warmup`` is negative.

    Complexity
    ----------
    ``warmup + repeats * number`` calls of ``func``; O(repeats) extra space.

    Examples
    >>> result = benchmark(sum, [1, 2, 3], repeats=3, number=5)
    >>> sorted(result)
    ['mean', 'min', 'number', 'repeats', 'std']
    """
    check_callable(func)
    repeats = _check_count(repeats, "repeats", 1)
    number = _check_count(number, "number", 1)
    warmup = _check_count(warmup, "warmup", 0)

    for _ in range(warmup):
        func(*args)
    times = np.empty(repeats)
    for r in range(repeats):
        start = time.perf_counter()
        for _ in range(number):
            func(*args)
        times[r] = (time.perf_counter() - start) / number
    return {
        "mean": float(times.mean()),
        "std": float(times.std()),
        "min": float(times.min()),
        "repeats": repeats,
        "number": number,
    }


def environment_info():
    """Describe the software and hardware the benchmarks ran on.

    Returns
    -------
    info : dict of str
        Keys ``python``, ``numpy``, ``os``, ``machine``, ``processor`` and
        ``cpu_count``.

    Complexity
    ----------
    Time O(1); space O(1).

    Examples
    >>> sorted(environment_info())
    ['cpu_count', 'machine', 'numpy', 'os', 'processor', 'python']
    """
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": str(os.cpu_count()),
    }


def compare(funcs_dict, *args, repeats=7, number=10, warmup=1, check_equal=True):
    """Benchmark several implementations of the same computation.

    Parameters
    ----------
    funcs_dict : dict of {str: callable}
        Implementations to compare, e.g. ``{'loop': f1, 'vectorised': f2}``.
    *args
        Positional arguments passed to every implementation.
    repeats, number, warmup : int
        Passed to :func:`benchmark`.
    check_equal : bool, default True
        First check that every implementation returns the same result as the
        first one (``np.allclose`` with NaNs treated as equal).

    Returns
    -------
    results : list of dict
        One row per implementation, fastest first, with keys ``name``,
        ``mean``, ``std``, ``min`` (seconds per call) and ``speedup``
        (slowest mean divided by this mean, so the slowest is 1.0).

    Raises
    ------
    TypeError
        If ``funcs_dict`` is not a dict of callables.
    ValueError
        If ``funcs_dict`` is empty or, with ``check_equal``, the outputs
        disagree.

    Complexity
    ----------
    ``k * (warmup + repeats * number)`` calls for k implementations, plus k
    calls for the equality check; O(k) extra space.

    Examples
    >>> rows = compare({"builtin": sum, "numpy": np.sum}, [1, 2, 3], repeats=2, number=2)
    >>> sorted(row["name"] for row in rows)
    ['builtin', 'numpy']
    """
    if not isinstance(funcs_dict, dict):
        raise TypeError(f"funcs_dict must be a dict; got {type(funcs_dict).__name__}")
    if not funcs_dict:
        raise ValueError("funcs_dict must contain at least one function")
    for func in funcs_dict.values():
        check_callable(func)

    if check_equal:
        names = list(funcs_dict)
        reference = np.asarray(funcs_dict[names[0]](*args), dtype=np.float64)
        for name in names[1:]:
            output = np.asarray(funcs_dict[name](*args), dtype=np.float64)
            if output.shape != reference.shape or not np.allclose(
                output, reference, rtol=1e-6, atol=1e-9, equal_nan=True
            ):
                raise ValueError(f"{name!r} returned a different result from {names[0]!r}")

    rows = []
    for name, func in funcs_dict.items():
        timing = benchmark(func, *args, repeats=repeats, number=number, warmup=warmup)
        rows.append(
            {"name": name, "mean": timing["mean"], "std": timing["std"], "min": timing["min"]}
        )
    slowest = max(row["mean"] for row in rows)
    for row in rows:
        row["speedup"] = slowest / row["mean"] if row["mean"] > 0 else float("inf")
    return sorted(rows, key=lambda row: row["mean"])
