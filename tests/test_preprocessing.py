"""Unit tests for numcompute.preprocessing."""

import numpy as np
import pytest

from numcompute.pipeline import NotFittedError
from numcompute.preprocessing import MinMaxScaler, OneHotEncoder, SimpleImputer, StandardScaler

RNG = np.random.default_rng(42)
X_RAND = RNG.normal(loc=5.0, scale=3.0, size=(50, 4))

ALL_TRANSFORMERS = [
    StandardScaler,
    MinMaxScaler,
    SimpleImputer,
]


# Shared API contract
@pytest.mark.parametrize("cls", ALL_TRANSFORMERS + [OneHotEncoder])
def test_fit_returns_self_and_fit_transform_matches(cls):
    X = np.array([[1.0, 2.0], [3.0, 2.0], [1.0, 5.0]])
    model = cls()
    assert model.fit(X) is model
    np.testing.assert_array_equal(cls().fit_transform(X), model.transform(X))


@pytest.mark.parametrize("cls", ALL_TRANSFORMERS + [OneHotEncoder])
def test_transform_before_fit_raises(cls):
    with pytest.raises(NotFittedError, match="not fitted"):
        cls().transform(np.array([[1.0]]))


@pytest.mark.parametrize("cls", ALL_TRANSFORMERS + [OneHotEncoder])
def test_feature_count_mismatch_raises(cls):
    model = cls().fit(np.array([[1.0, 2.0], [3.0, 4.0]]))
    with pytest.raises(ValueError, match="features"):
        model.transform(np.array([[1.0, 2.0, 3.0]]))


@pytest.mark.parametrize("cls", ALL_TRANSFORMERS + [OneHotEncoder])
def test_empty_fit_raises(cls):
    with pytest.raises(ValueError, match="at least one sample"):
        cls().fit(np.empty((0, 2)))


@pytest.mark.parametrize("cls", ALL_TRANSFORMERS + [OneHotEncoder])
def test_1d_input_rejected(cls):
    with pytest.raises(ValueError, match="2 dimensions"):
        cls().fit(np.array([1.0, 2.0]))


@pytest.mark.parametrize("cls", ALL_TRANSFORMERS)
def test_input_not_modified(cls):
    X = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
    original = X.copy()
    cls().fit_transform(X)
    np.testing.assert_array_equal(X, original)


@pytest.mark.parametrize("cls", [StandardScaler, MinMaxScaler])
def test_non_contiguous_input_matches_contiguous(cls):
    big = RNG.normal(size=(20, 6))
    strided = big[::2, ::3]
    assert not strided.flags["C_CONTIGUOUS"]
    np.testing.assert_allclose(
        cls().fit_transform(strided), cls().fit_transform(np.ascontiguousarray(strided))
    )


def test_repr_shows_parameters():
    expected = "MinMaxScaler(feature_range=(-1, 1), clip=False)"
    assert repr(MinMaxScaler(feature_range=(-1, 1))) == expected


# StandardScaler
def test_standard_scaler_zero_mean_unit_std():
    Z = StandardScaler().fit_transform(X_RAND)
    np.testing.assert_allclose(Z.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(Z.std(axis=0), 1, atol=1e-12)


def test_standard_scaler_matches_numpy_formula():
    scaler = StandardScaler().fit(X_RAND)
    np.testing.assert_allclose(scaler.mean_, X_RAND.mean(axis=0))
    np.testing.assert_allclose(scaler.scale_, X_RAND.std(axis=0))
    np.testing.assert_allclose(scaler.var_, X_RAND.var(axis=0))


def test_standard_scaler_constant_column_has_unit_scale():
    X = np.array([[0.1, 1.0], [0.1, 2.0], [0.1, 3.0]])
    scaler = StandardScaler().fit(X)
    assert scaler.scale_[0] == 1.0
    np.testing.assert_array_equal(scaler.transform(X)[:, 0], 0.0)


def test_standard_scaler_ignores_and_keeps_nan():
    X = np.array([[1.0], [np.nan], [3.0]])
    scaler = StandardScaler().fit(X)
    assert scaler.mean_[0] == 2.0 and scaler.n_samples_seen_[0] == 2
    out = scaler.transform(X)
    assert np.isnan(out[1, 0])
    np.testing.assert_array_equal(out[[0, 2], 0], [-1.0, 1.0])


def test_standard_scaler_all_nan_column_passes_through():
    out = StandardScaler().fit_transform(np.array([[np.nan, 1.0], [np.nan, 3.0]]))
    assert np.isnan(out[:, 0]).all()
    np.testing.assert_array_equal(out[:, 1], [-1.0, 1.0])


def test_standard_scaler_near_float_limit_does_not_overflow():
    X = np.array([[1.7e308], [-1.7e308]])
    np.testing.assert_allclose(StandardScaler().fit_transform(X).ravel(), [1.0, -1.0])


def test_standard_scaler_tiny_values_are_not_treated_as_constant():
    X = np.array([[1e-20], [3e-20]])
    np.testing.assert_allclose(StandardScaler().fit_transform(X).ravel(), [-1.0, 1.0])


def test_standard_scaler_flags_and_inverse():
    X = np.array([[1.0], [3.0]])
    np.testing.assert_array_equal(StandardScaler(with_mean=False).fit_transform(X).ravel(), [1, 3])
    np.testing.assert_array_equal(StandardScaler(with_std=False).fit_transform(X).ravel(), [-1, 1])
    scaler = StandardScaler().fit(X_RAND)
    np.testing.assert_allclose(scaler.inverse_transform(scaler.transform(X_RAND)), X_RAND)


def test_standard_scaler_rejects_inf_and_strings():
    with pytest.raises(ValueError, match="infinite"):
        StandardScaler().fit(np.array([[1.0], [np.inf]]))
    with pytest.raises(TypeError, match="numeric"):
        StandardScaler().fit(np.array([["a"], ["b"]]))


def test_standard_scaler_integer_input_returns_float():
    out = StandardScaler().fit_transform(np.array([[1], [3]]))
    assert out.dtype == np.float64


# MinMaxScaler
def test_minmax_default_range():
    out = MinMaxScaler().fit_transform(X_RAND)
    np.testing.assert_allclose(out.min(axis=0), 0)
    np.testing.assert_allclose(out.max(axis=0), 1)


def test_minmax_custom_range_and_attributes():
    X = np.array([[0.0, 10.0], [5.0, 20.0], [10.0, 30.0]])
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X)
    np.testing.assert_array_equal(scaler.data_min_, [0, 10])
    np.testing.assert_array_equal(scaler.data_range_, [10, 20])
    np.testing.assert_allclose(scaler.transform(X), [[-1, -1], [0, 0], [1, 1]])


def test_minmax_constant_column_maps_to_lower_bound():
    out = MinMaxScaler(feature_range=(2, 3)).fit_transform(np.array([[7.0], [7.0]]))
    np.testing.assert_array_equal(out.ravel(), [2.0, 2.0])


def test_minmax_unseen_data_and_clip():
    scaler = MinMaxScaler().fit(np.array([[0.0], [10.0]]))
    np.testing.assert_allclose(scaler.transform(np.array([[20.0]])), [[2.0]])
    clipped = MinMaxScaler(clip=True).fit(np.array([[0.0], [10.0]]))
    np.testing.assert_allclose(clipped.transform(np.array([[20.0], [-5.0]])), [[1.0], [0.0]])


def test_minmax_nan_and_extremes():
    out = MinMaxScaler().fit_transform(np.array([[1e-20], [2e-20], [np.nan]]))
    np.testing.assert_allclose(out[:2, 0], [0.0, 1.0])
    assert np.isnan(out[2, 0])
    extreme = MinMaxScaler().fit_transform(np.array([[1.7e308], [-1.7e308], [0.0]]))
    np.testing.assert_allclose(extreme.ravel(), [1.0, 0.0, 0.5])


@pytest.mark.parametrize("bad", [(1, 0), (0, 0), (0, np.inf), "ab", (1,)])
def test_minmax_invalid_feature_range(bad):
    with pytest.raises(ValueError, match="feature_range"):
        MinMaxScaler(feature_range=bad).fit(np.array([[1.0], [2.0]]))


# SimpleImputer
X_MISSING = np.array([[1.0, np.nan], [3.0, 4.0], [np.nan, 8.0], [3.0, 9.0]])


def test_imputer_mean_and_median():
    mean_out = SimpleImputer(strategy="mean").fit_transform(X_MISSING)
    np.testing.assert_allclose(mean_out[2, 0], 7 / 3)
    np.testing.assert_allclose(mean_out[0, 1], 7.0)
    median_out = SimpleImputer(strategy="median").fit(X_MISSING)
    np.testing.assert_array_equal(median_out.statistics_, [3.0, 8.0])
    assert not np.isnan(median_out.transform(X_MISSING)).any()


def test_imputer_most_frequent_numeric_ties_pick_smallest():
    X = np.array([[2.0], [1.0], [2.0], [1.0], [np.nan]])
    imputer = SimpleImputer(strategy="most_frequent").fit(X)
    assert imputer.statistics_[0] == 1.0


def test_imputer_constant_numeric_and_string():
    out = SimpleImputer(strategy="constant", fill_value=-1).fit_transform(X_MISSING)
    assert out[0, 1] == -1 and out[2, 0] == -1
    cities = np.array([["Adelaide"], [""], ["Perth"]])
    filled = SimpleImputer(strategy="constant", missing_values="").fit_transform(cities)
    assert filled[1, 0] == "missing"


def test_imputer_most_frequent_strings():
    cities = np.array([["Perth"], [""], ["Perth"], ["Adelaide"]])
    filled = SimpleImputer(strategy="most_frequent", missing_values="").fit_transform(cities)
    assert filled.ravel().tolist() == ["Perth", "Perth", "Perth", "Adelaide"]


def test_imputer_custom_numeric_marker():
    X = np.array([[1.0], [-999.0], [3.0]])
    out = SimpleImputer(missing_values=-999.0).fit_transform(X)
    np.testing.assert_array_equal(out.ravel(), [1.0, 2.0, 3.0])


def test_imputer_all_missing_column_raises():
    with pytest.raises(ValueError, match="only missing"):
        SimpleImputer().fit(np.array([[np.nan, 1.0], [np.nan, 2.0]]))


def test_imputer_errors():
    with pytest.raises(ValueError, match="strategy"):
        SimpleImputer(strategy="mode").fit(X_MISSING)
    with pytest.raises(TypeError, match="numeric"):
        SimpleImputer(strategy="mean").fit(np.array([["a"], ["b"]]))


def test_imputer_no_missing_is_identity():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(SimpleImputer().fit_transform(X), X)


# OneHotEncoder
def test_onehot_strings_multiple_columns():
    X = np.array([["red", "S"], ["blue", "M"], ["red", "M"]])
    enc = OneHotEncoder().fit(X)
    assert [c.tolist() for c in enc.categories_] == [["blue", "red"], ["M", "S"]]
    np.testing.assert_array_equal(
        enc.transform(X), [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 1, 0]]
    )


def test_onehot_rows_sum_to_number_of_features():
    X = RNG.integers(0, 5, size=(30, 3))
    out = OneHotEncoder().fit_transform(X)
    np.testing.assert_array_equal(out.sum(axis=1), 3)
    assert set(np.unique(out)) <= {0.0, 1.0}


def test_onehot_numeric_categories_and_single_category():
    out = OneHotEncoder().fit_transform(np.array([[3], [1], [3]]))
    np.testing.assert_array_equal(out, [[0, 1], [1, 0], [0, 1]])
    single = OneHotEncoder().fit_transform(np.array([["x"], ["x"]]))
    np.testing.assert_array_equal(single, [[1], [1]])


def test_onehot_unknown_error_and_ignore():
    enc = OneHotEncoder().fit(np.array([["a"], ["b"]]))
    with pytest.raises(ValueError, match="unseen"):
        enc.transform(np.array([["c"]]))
    ignore = OneHotEncoder(handle_unknown="ignore").fit(np.array([["a"], ["b"]]))
    np.testing.assert_array_equal(ignore.transform(np.array([["c"], ["a"]])), [[0, 0], [1, 0]])


def test_onehot_rejects_nan_and_bad_option():
    with pytest.raises(ValueError, match="impute"):
        OneHotEncoder().fit(np.array([[1.0], [np.nan]]))
    with pytest.raises(ValueError, match="handle_unknown"):
        OneHotEncoder(handle_unknown="skip").fit(np.array([["a"]]))


def test_onehot_feature_names_and_empty_transform():
    enc = OneHotEncoder().fit(np.array([["Adelaide", "S"], ["Perth", "M"]]))
    assert enc.get_feature_names_out(["city", "size"]).tolist() == [
        "city_Adelaide",
        "city_Perth",
        "size_M",
        "size_S",
    ]
    assert enc.get_feature_names_out().tolist()[0] == "x0_Adelaide"
    with pytest.raises(ValueError, match="input_features"):
        enc.get_feature_names_out(["only_one"])
    assert enc.transform(np.empty((0, 2), dtype=str)).shape == (0, 4)


def test_imputer_then_onehot_on_csv_style_strings():
    raw = np.array([["Adelaide"], [""], ["Perth"], ["Adelaide"]])
    filled = SimpleImputer(strategy="most_frequent", missing_values="").fit_transform(raw)
    out = OneHotEncoder().fit_transform(filled)
    np.testing.assert_array_equal(out, [[1, 0], [1, 0], [0, 1], [1, 0]])