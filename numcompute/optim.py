"""Finite-difference gradient and Jacobian estimation.

Derivatives are approximated by perturbing every coordinate of ``x`` at once
with the perturbation matrix ``h * np.eye(n)``: row ``i`` of ``x + E`` is ``x``
moved by ``h`` along coordinate ``i``.

The user's function is a black box, so it must be evaluated once per
perturbed point. If the function can evaluate many points in one call, pass
``vectorized=True`` and all rows are evaluated together with no Python loop.
"""

import numpy as np

from .utils import check_array, check_callable

__all__ = [
    "grad",
    "jacobian",
    "line_search",
]

_METHODS = ("central", "forward")


# Internal helpers


def _check_point(x):
    """Return x as a finite 1-D float array and whether the input was a scalar."""
    is_scalar = np.ndim(x) == 0
    point = check_array(
        np.atleast_1d(x) if is_scalar else x,
        dtype=np.float64,
        allow_nan=False,
        allow_inf=False,
    )
    if point.ndim != 1:
        raise ValueError(f"x must be a scalar or 1-D array; got shape {point.shape}")
    if point.shape[0] == 0:
        raise ValueError("x must contain at least one coordinate")
    return point, is_scalar


def _check_step(h, method):
    """Validate the step size and difference method."""
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}; got {method!r}")
    if isinstance(h, bool) or not isinstance(h, (int, float, np.integer, np.floating)):
        raise TypeError(f"h must be a real number; got {type(h).__name__}")
    if not np.isfinite(h) or h <= 0:
        raise ValueError(f"h must be a positive finite number; got {h}")
    return float(h)


def _evaluate_rows(function, points, vectorized):
    """Evaluate function at every row of ``points``; return shape (k, m)."""
    if vectorized:
        values = np.asarray(function(points), dtype=np.float64)
        if values.shape[0:1] != points.shape[0:1]:
            raise ValueError(
                f"vectorized function must return one result per row; "
                f"got shape {values.shape} for {points.shape[0]} rows"
            )
        return values.reshape(points.shape[0], -1)
    
    # Black-box function: one call per perturbed point is unavoidable.

    values = [np.asarray(function(row), dtype=np.float64).ravel() for row in points]
    sizes = {v.shape[0] for v in values}
    if len(sizes) != 1:
        raise ValueError(f"function returned outputs of different sizes: {sorted(sizes)}")
    return np.stack(values)


def _differences(function, point, h, method, vectorized):
    """Finite-difference matrix of shape (m, n): column i is d f / d x_i."""
    n = point.shape[0]
    steps = h * np.eye(n)
    if method == "central":
        both = np.vstack([point + steps, point - steps])
        values = _evaluate_rows(function, both, vectorized)
        diff = (values[:n] - values[n:]) / (2.0 * h)
    else:
        both = np.vstack([point[None, :], point + steps])
        values = _evaluate_rows(function, both, vectorized)
        diff = (values[1:] - values[0]) / h
    if not np.all(np.isfinite(diff)):
        raise ValueError("function returned NaN or inf near x; try a different h")
    return diff.T



# Gradient

def grad(f, x, h=1e-5, method="central", vectorized=False):
    """Numerical gradient of a scalar function ``f: R^n -> R``.

    Parameters
    ----------
    f : callable
        Function taking an array of shape (n,) and returning a scalar. With
        ``vectorized=True`` it must take shape (k, n) and return shape (k,).
    x : float or array_like, shape (n,)
        Point at which to evaluate the gradient.
    h : float, default 1e-5
        Step size. Central differences are accurate to O(h^2), forward to O(h).
    method : {'central', 'forward'}, default 'central'
        Finite-difference scheme.
    vectorized : bool, default False
        Evaluate all perturbed points in a single call to ``f``.

    Returns
    -------
    g : float or ndarray of float64, shape (n,)
        The gradient; a float when ``x`` is a scalar.

    Raises
    ------
    TypeError
        If ``f`` is not callable, or ``x`` or ``h`` is not real numeric.
    ValueError
        If ``x`` is not a scalar/1-D finite array, ``h`` is not positive,
        ``method`` is unknown, or ``f`` does not return one finite scalar.

    Complexity
    ----------
    2n evaluations of ``f`` (central) or n + 1 (forward); O(n^2) extra space
    for the perturbation matrix.

    Examples
    >>> round(grad(lambda v: v[0] ** 2, 3.0), 6)
    6.0
    >>> np.round(grad(lambda v: v[0] * v[1], [2.0, 5.0]), 6)
    array([5., 2.])
    """
    check_callable(f)
    point, is_scalar = _check_point(x)
    h = _check_step(h, method)
    diff = _differences(f, point, h, method, vectorized)
    if diff.shape[0] != 1:
        raise ValueError(f"f must return a scalar; got {diff.shape[0]} values")
    g = diff[0]
    return float(g[0]) if is_scalar else g



# Jacobian

def jacobian(F, x, h=1e-5, method="central", vectorized=False):
    """Numerical Jacobian of a vector function ``F: R^n -> R^m``.

    Parameters
    ----------
    F : callable
        Function taking an array of shape (n,) and returning shape (m,) (a
        scalar counts as m = 1). With ``vectorized=True`` it must take shape
        (k, n) and return shape (k, m).
    x : float or array_like, shape (n,)
        Point at which to evaluate the Jacobian.
    h : float, default 1e-5
        Step size.
    method : {'central', 'forward'}, default 'central'
        Finite-difference scheme.
    vectorized : bool, default False
        Evaluate all perturbed points in a single call to ``F``.

    Returns
    -------
    J : ndarray of float64, shape (m, n)
        ``J[i, j] = d F_i / d x_j``.

    Raises
    ------
    TypeError
        If ``F`` is not callable, or ``x`` or ``h`` is not real numeric.
    ValueError
        If ``x`` is not a scalar/1-D finite array, ``h`` is not positive,
        ``method`` is unknown, or ``F`` returns inconsistent or non-finite
        outputs.

    Complexity
    ----------
    2n evaluations of ``F`` (central) or n + 1 (forward); O(n^2 + m n) extra
    space.

    Examples
    >>> A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    >>> np.round(jacobian(lambda v: A @ v, [0.5, -1.0]), 6)
    array([[1., 2.],
           [3., 4.],
           [5., 6.]])
    """
    check_callable(F)
    point, _ = _check_point(x)
    h = _check_step(h, method)
    return _differences(F, point, h, method, vectorized)


# Backtracking line search (optional extra)

def line_search(f, x, direction, gradient=None, alpha0=1.0, rho=0.5, c=1e-4, max_iter=50):
    """Backtracking line search satisfying the Armijo condition.

    Starting from ``alpha0``, the step is multiplied by ``rho`` until
    ``f(x + alpha d) <= f(x) + c * alpha * (g . d)``, i.e. until the step
    gives a sufficient decrease.

    Parameters
    ----------
    f : callable
        Scalar function taking shape (n,).
    x : float or array_like, shape (n,)
        Current point.
    direction : float or array_like, shape (n,)
        Search direction ``d``; must be a descent direction (``g . d < 0``).
    gradient : array_like, shape (n,), optional
        Gradient of ``f`` at ``x``. Estimated with :func:`grad` if omitted.
    alpha0 : float, default 1.0
        Initial step length.
    rho : float, default 0.5
        Shrink factor, in (0, 1).
    c : float, default 1e-4
        Sufficient-decrease constant, in (0, 1).
    max_iter : int, default 50
        Maximum number of step reductions.

    Returns
    -------
    alpha : float
        Accepted step length.

    Raises
    ------
    TypeError
        If ``f`` is not callable or inputs are not real numeric.
    ValueError
        If shapes differ, parameters are out of range, ``direction`` is not a
        descent direction, or no step is accepted within ``max_iter``.

    Complexity
    ----------
    At most ``max_iter + 1`` evaluations of ``f`` (plus 2n for the gradient
    if it is estimated); O(n) space.

    Examples
    >>> line_search(lambda v: float(v @ v), [1.0, 1.0], [-1.0, -1.0])
    1.0
    """
    check_callable(f)
    point, _ = _check_point(x)
    d, _ = _check_point(direction)
    if d.shape != point.shape:
        raise ValueError(f"direction must have shape {point.shape}; got {d.shape}")
    g = grad(f, point) if gradient is None else _check_point(gradient)[0]
    if g.shape != point.shape:
        raise ValueError(f"gradient must have shape {point.shape}; got {g.shape}")
    if not 0 < rho < 1 or not 0 < c < 1 or alpha0 <= 0:
        raise ValueError("need alpha0 > 0, 0 < rho < 1 and 0 < c < 1")
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError(f"max_iter must be a positive integer; got {max_iter!r}")

    slope = float(g @ d)
    if slope >= 0:
        raise ValueError(f"direction is not a descent direction (g . d = {slope})")
    f0 = float(f(point))
    alpha = float(alpha0)
    for _ in range(max_iter + 1):  # iterative by nature; each step needs f
        if float(f(point + alpha * d)) <= f0 + c * alpha * slope:
            return alpha
        alpha *= rho
    raise ValueError(f"no step satisfied the Armijo condition in {max_iter} reductions")
