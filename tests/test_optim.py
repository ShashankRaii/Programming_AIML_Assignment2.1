import numpy as np
import pytest

from numcompute.optim import grad, jacobian, line_search


# grad


def test_grad_scalar_x_squared_returns_float():
    g = grad(lambda v: v[0] ** 2, 3.0)
    assert isinstance(g, float)
    assert g == pytest.approx(6.0, abs=1e-8)


def test_grad_sin_matches_cos():
    x = np.linspace(-3, 3, 7)
    g = grad(lambda v: np.sum(np.sin(v)), x)
    np.testing.assert_allclose(g, np.cos(x), atol=1e-8)


def test_grad_quadratic_form_matches_analytic():
    rng = np.random.default_rng(42)
    A = rng.normal(size=(4, 4))
    Q = A + A.T
    x = rng.normal(size=4)
    g = grad(lambda v: 0.5 * v @ Q @ v, x)
    np.testing.assert_allclose(g, Q @ x, atol=1e-6)


def test_central_is_more_accurate_than_forward():
    f, x, exact = (lambda v: np.exp(v[0])), 1.0, np.exp(1.0)
    central_err = abs(grad(f, x, h=1e-3, method="central") - exact)
    forward_err = abs(grad(f, x, h=1e-3, method="forward") - exact)
    assert central_err < forward_err
    assert central_err < 1e-5 and forward_err < 1e-2  # O(h^2) vs O(h)


def test_grad_vectorized_matches_loop():
    def f_rows(X):
        return np.sum(X**2, axis=-1)

    x = np.arange(1.0, 6.0)
    np.testing.assert_allclose(grad(f_rows, x, vectorized=True), grad(f_rows, x), rtol=1e-10)
    np.testing.assert_allclose(grad(f_rows, x, vectorized=True), 2 * x, atol=1e-6)


def test_grad_noncontiguous_input_unchanged():
    base = np.arange(10.0)
    x = base[::2]
    before = x.copy()
    np.testing.assert_allclose(grad(lambda v: np.sum(v**3), x), 3 * x**2, rtol=1e-6)
    np.testing.assert_array_equal(x, before)


def test_grad_errors():
    with pytest.raises(TypeError, match="callable"):
        grad(42, [1.0])
    with pytest.raises(ValueError, match="method must be one of"):
        grad(np.sum, [1.0], method="backward")
    with pytest.raises(ValueError, match="positive"):
        grad(np.sum, [1.0], h=0)
    with pytest.raises(ValueError, match="1-D"):
        grad(np.sum, [[1.0, 2.0]])
    with pytest.raises(ValueError, match="scalar"):
        grad(lambda v: v, [1.0, 2.0])  # returns a vector
    with pytest.raises(ValueError, match="NaN"):
        grad(np.sum, [np.nan])
    with pytest.raises(ValueError, match="NaN or inf"):
        grad(lambda v: np.nan if v[0] < 1.0 else v[0], [1.0])  # undefined left of x


#  jacobian 


def test_jacobian_of_linear_map_is_the_matrix():
    A = np.array([[1.0, 2.0, 0.0], [-3.0, 4.0, 5.0]])
    J = jacobian(lambda v: A @ v, [0.3, -1.2, 2.0])
    assert J.shape == (2, 3)
    np.testing.assert_allclose(J, A, atol=1e-8)


def test_jacobian_nonlinear_matches_analytic():
    def F(v):
        x, y = v
        return np.array([x**2 * y, 5 * x + np.sin(y)])

    x, y = 1.5, 0.7
    expected = np.array([[2 * x * y, x**2], [5.0, np.cos(y)]])
    np.testing.assert_allclose(jacobian(F, [x, y]), expected, atol=1e-7)
    np.testing.assert_allclose(jacobian(F, [x, y], method="forward"), expected, atol=1e-4)


def test_jacobian_of_scalar_function_is_gradient_row():
    f = lambda v: np.sum(v**2)  # noqa: E731
    x = np.array([1.0, -2.0])
    np.testing.assert_allclose(jacobian(f, x), grad(f, x)[None, :])


def test_jacobian_vectorized_and_scalar_x():
    A = np.array([[2.0], [3.0]])
    J = jacobian(lambda X: X @ A.T, [4.0], vectorized=True)
    np.testing.assert_allclose(J, A, atol=1e-8)
    J = jacobian(lambda v: [v[0], v[0] ** 2], 3.0)
    np.testing.assert_allclose(J, [[1.0], [6.0]], atol=1e-6)


def test_jacobian_errors():
    with pytest.raises(ValueError, match="different sizes"):
        jacobian(lambda v: np.ones(1 + int(v[0] > 0)), [0.0])
    with pytest.raises(ValueError, match="one result per row"):
        jacobian(lambda X: np.ones(3), [1.0, 2.0], vectorized=True)
    with pytest.raises(TypeError, match="h must be a real number"):
        jacobian(np.sum, [1.0], h="small")


# line_search 


def test_line_search_accepts_full_step_on_simple_quadratic():
    assert line_search(lambda v: float(v @ v), [1.0, 1.0], [-1.0, -1.0]) == 1.0


def test_line_search_backtracks_and_decreases_f():
    f = lambda v: float(10 * v @ v)  # noqa: E731
    x = np.array([1.0, -2.0])
    g = 20 * x
    alpha = line_search(f, x, -g, gradient=g)
    assert alpha < 1.0
    assert f(x - alpha * g) < f(x)


def test_line_search_errors():
    f = lambda v: float(v @ v)  # noqa: E731
    with pytest.raises(ValueError, match="descent direction"):
        line_search(f, [1.0, 1.0], [1.0, 1.0])
    with pytest.raises(ValueError, match="shape"):
        line_search(f, [1.0, 1.0], [-1.0])
    with pytest.raises(ValueError, match="rho"):
        line_search(f, [1.0], [-1.0], rho=1.5)
